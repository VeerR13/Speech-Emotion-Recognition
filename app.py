import os
import io
import json
import logging
import tempfile
import traceback

import numpy as np
import librosa
import pickle
import soundfile as sf
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]

SR = 22050
DURATION = 3
N_MELS = 128
MEL_TIME_STEPS = 128
N_MFCC = 40

model = None
scaler = None
label_encoder = None


def load_artifacts():
    global model, scaler, label_encoder
    try:
        import tensorflow as tf
        import keras

        # Enable unsafe deserialization for Lambda layers with Python functions
        keras.config.enable_unsafe_deserialization()

        # ── Custom layers required for deserialization ──────────────────────
        class ChannelMean(tf.keras.layers.Layer):
            def call(self, x):
                return tf.reduce_mean(x, axis=-1, keepdims=True)

        class ChannelMax(tf.keras.layers.Layer):
            def call(self, x):
                return tf.reduce_max(x, axis=-1, keepdims=True)

        class AttnWeightedSum(tf.keras.layers.Layer):
            def call(self, inputs):
                return tf.reduce_sum(inputs[0] * inputs[1], axis=1)

        class FocalLoss(tf.keras.losses.Loss):
            def __init__(self, gamma=2.0, smoothing=0.1, **kwargs):
                super().__init__(**kwargs)
                self.gamma = gamma
                self.smoothing = smoothing

            def call(self, y_true, y_pred):
                y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
                ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
                p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
                focal_weight = tf.pow(1.0 - p_t, self.gamma)
                return tf.reduce_mean(focal_weight * ce)

            def get_config(self):
                config = super().get_config()
                config.update({"gamma": self.gamma, "smoothing": self.smoothing})
                return config

        class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, warmup_steps, total_steps, peak_lr=1e-3, min_lr=1e-6, **kwargs):
                super().__init__(**kwargs)
                self.warmup_steps = warmup_steps
                self.total_steps = total_steps
                self.peak_lr = peak_lr
                self.min_lr = min_lr

            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                warmup = self.peak_lr * (step / self.warmup_steps)
                cosine = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                    1 + tf.cos(np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
                )
                return tf.where(step < self.warmup_steps, warmup, cosine)

            def get_config(self):
                return {
                    "warmup_steps": self.warmup_steps,
                    "total_steps": self.total_steps,
                    "peak_lr": self.peak_lr,
                    "min_lr": self.min_lr,
                }

        # ── Patched Lambda to handle missing output_shape ───────────────────
        class PatchedLambda(tf.keras.layers.Lambda):
            def __init__(self, *args, **kwargs):
                if "output_shape" not in kwargs:
                    kwargs["output_shape"] = lambda s: s
                super().__init__(*args, **kwargs)

        custom_objects = {
            "ChannelMean": ChannelMean,
            "ChannelMax": ChannelMax,
            "AttnWeightedSum": AttnWeightedSum,
            "FocalLoss": FocalLoss,
            "WarmupCosine": WarmupCosine,
            "Lambda": PatchedLambda,
        }

        model_path = os.path.join(MODEL_DIR, "best_ser_v2.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, "best_ser_model.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, "best_ser.keras")

        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
        logger.info(f"Model loaded from {model_path}")

    except Exception as e:
        logger.error(f"Model load failed: {e}")
        model = None

    try:
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded")
    except Exception as e:
        logger.error(f"Scaler load failed: {e}")
        scaler = None

    try:
        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        logger.info("Label encoder loaded")
    except Exception as e:
        logger.error(f"Label encoder load failed: {e}")
        label_encoder = None


def load_audio(file_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=SR, duration=DURATION, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)
        target_len = SR * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
    finally:
        os.unlink(tmp_path)
    return y


def extract_flat_features(y, sr=SR):
    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    for feat_2d in [mfcc, delta_mfcc, delta2_mfcc]:
        feats.extend(feat_2d.mean(axis=1))
        feats.extend(feat_2d.std(axis=1))
        feats.extend(feat_2d.min(axis=1))
        feats.extend(feat_2d.max(axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(chroma.mean(axis=1))
    feats.extend(chroma.std(axis=1))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats.extend([mel_db.mean(), mel_db.std(), mel_db.min(), mel_db.max()])

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats.extend(contrast.mean(axis=1))
    feats.extend(contrast.std(axis=1))

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    feats.extend(tonnetz.mean(axis=1))
    feats.extend(tonnetz.std(axis=1))

    zcr = librosa.feature.zero_crossing_rate(y)
    feats.extend([zcr.mean(), zcr.std()])

    rms = librosa.feature.rms(y=y)
    feats.extend([rms.mean(), rms.std()])

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats.extend([cent.mean(), cent.std()])

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats.extend([bw.mean(), bw.std()])

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats.extend([rolloff.mean(), rolloff.std()])

    try:
        f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[voiced]
        if len(f0_clean) > 0:
            feats.extend([f0_clean.mean(), f0_clean.std(), f0_clean.min(), f0_clean.max()])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])
    except Exception:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)


def extract_mel_multiscale(y, sr=SR):
    """3-channel multi-scale mel spectrogram matching training pipeline."""
    channels = []
    for n_fft in [512, 1024, 2048]:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=n_fft)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :128]
        channels.append(mel_db)
    return np.stack(channels, axis=-1)  # (128, 128, 3)


def predict(file_bytes):
    y = load_audio(file_bytes)

    flat = extract_flat_features(y)
    mel_3ch = extract_mel_multiscale(y)

    if scaler is not None:
        flat = scaler.transform(flat.reshape(1, -1))[0]

    flat_input = flat.reshape(1, -1)
    mel_input = mel_3ch[np.newaxis, ...]  # (1, 128, 128, 3)

    classes = label_encoder.classes_.tolist() if label_encoder is not None else EMOTIONS

    if model is None:
        probs = np.random.dirichlet(np.ones(len(classes)) * 2).tolist()
        emotion_idx = int(np.argmax(probs))
        return {
            "emotion": classes[emotion_idx],
            "confidence": round(probs[emotion_idx] * 100, 1),
            "probabilities": {e: round(p * 100, 1) for e, p in zip(classes, probs)},
            "demo_mode": True,
        }

    preds = model.predict({"spectrogram": mel_input, "features": flat_input}, verbose=0)
    probs = preds[0].tolist()

    emotion_idx = int(np.argmax(probs))
    emotion = classes[emotion_idx]
    confidence = probs[emotion_idx]

    prob_dict = {c: round(p * 100, 1) for c, p in zip(classes, probs)}

    return {
        "emotion": emotion.lower(),
        "confidence": round(confidence * 100, 1),
        "probabilities": prob_dict,
        "demo_mode": False,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        file_bytes = audio_file.read()
        result = predict(file_bytes)
        return jsonify(result)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    })


load_artifacts()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
