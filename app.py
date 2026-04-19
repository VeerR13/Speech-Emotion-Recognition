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
EMOTIONS = ["angry", "happy", "neutral", "sad"]

SR = 22050
DURATION = 3
N_MELS = 128
MEL_TIME_STEPS = 130
N_MFCC = 40

model = None
scaler = None
label_encoder = None


def load_artifacts():
    global model, scaler, label_encoder
    try:
        import tensorflow as tf
        model_path = os.path.join(MODEL_DIR, "best_ser_model.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, "best_ser.keras")
        model = tf.keras.models.load_model(model_path)
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
        y, _ = librosa.effects.trim(y, top_db=20)
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


def extract_mel_2d(y, sr=SR):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    if mel_db.shape[1] < MEL_TIME_STEPS:
        pad_width = MEL_TIME_STEPS - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)))
    else:
        mel_db = mel_db[:, :MEL_TIME_STEPS]
    return mel_db[..., np.newaxis]


def predict(file_bytes):
    y = load_audio(file_bytes)

    flat = extract_flat_features(y)
    mel_2d = extract_mel_2d(y)

    if scaler is not None:
        flat = scaler.transform(flat.reshape(1, -1))[0]

    flat_input = flat.reshape(1, -1)
    mel_input = mel_2d[np.newaxis, ...]

    if model is None:
        # Demo mode: return random plausible predictions
        probs = np.random.dirichlet(np.ones(len(EMOTIONS)) * 2).tolist()
        emotion_idx = int(np.argmax(probs))
        emotion = EMOTIONS[emotion_idx]
        confidence = probs[emotion_idx]
        return {
            "emotion": emotion,
            "confidence": round(confidence * 100, 1),
            "probabilities": {e: round(p * 100, 1) for e, p in zip(EMOTIONS, probs)},
            "demo_mode": True,
        }

    preds = model.predict([mel_input, flat_input], verbose=0)
    probs = preds[0].tolist()

    if label_encoder is not None:
        classes = label_encoder.classes_.tolist()
    else:
        classes = EMOTIONS

    emotion_idx = int(np.argmax(probs))
    emotion = classes[emotion_idx] if emotion_idx < len(classes) else EMOTIONS[emotion_idx]
    confidence = probs[emotion_idx]

    prob_dict = {}
    for i, e in enumerate(EMOTIONS):
        matched = next((p for c, p in zip(classes, probs) if c.lower() == e.lower()), probs[i] if i < len(probs) else 0.0)
        prob_dict[e] = round(matched * 100, 1)

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
