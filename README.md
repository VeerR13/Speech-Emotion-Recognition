# Speech Emotion Recognition

A deep learning system for classifying speech into 8 emotions using a dual-input fusion architecture.  
Trained on 4 public datasets — achieves **89.3% test accuracy** and **91.1% macro F1**.

## Live Demo

**[speech-emotion-app.hf.space](https://veerr13-speech-emotion-app.hf.space)**

Record 3 seconds of audio or upload a file and watch the signal race through the pipeline in real time.  
Hosted on HuggingFace Spaces · Docker · Keras 3.10 · TensorFlow 2.18

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **89.29%** |
| Macro F1 | **91.09%** |
| Best Val Accuracy (training) | 81.41% (epoch 67/80) |
| TTA Test Accuracy (15 passes) | ~82–83% on Kaggle |

> Full evaluation on held-out test data in Google Colab (CPU, no TTA).  
> Pre-trained model: [huggingface.co/VeerR13/Speech-Emotion-Recognition](https://huggingface.co/VeerR13/Speech-Emotion-Recognition/tree/main)

### Confusion Matrix

![Confusion Matrix](https://huggingface.co/VeerR13/Speech-Emotion-Recognition/resolve/main/results.png)

### Per-Class Performance

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| calm | 1.00 | 1.00 | 1.00 |
| surprise | 1.00 | 1.00 | 1.00 |
| angry | 0.98 | 0.98 | 0.98 |
| fear | 0.97 | 0.97 | 0.97 |
| happy | 0.94 | 0.97 | 0.95 |
| disgust | 0.93 | 0.87 | 0.90 |
| neutral | 0.82 | 0.79 | 0.81 |
| sad | 0.79 | 0.81 | 0.80 |

---

## Web App

The live demo is a premium dark-mode SPA built with Flask and vanilla JS:

- **3-second WAV recorder** — uses `AudioContext + ScriptProcessor` for cross-browser compatibility (works on iOS Safari, Chrome, Firefox)
- **Speed-of-sound analysis overlay** — when audio is submitted, a waveform blasts through 5 labeled pipeline stages at full speed, then the overlay whooshes off-screen left to reveal results
- **Live neural architecture canvas** — section 03 shows the dual-branch CNN + MLP model as a live-animated flow diagram with data packets racing through processing nodes
- **Three.js 3D audio sphere** — hero section Fibonacci-distributed frequency bars that respond to scroll
- **Result panel** — emotion label, confidence %, animated probability bars per class

Source: [`VeerR13/Speech-Emotion-App`](https://huggingface.co/spaces/VeerR13/Speech-Emotion-App) on HuggingFace Spaces

---

## Emotions Detected

`angry` · `happy` · `neutral` · `sad`

> The web app is trained on a 4-class subset. The full model supports 8 classes — see per-class table above.

---

## Architecture

Dual-input fusion — CNN branch processes mel spectrograms, MLP branch processes handcrafted acoustic features. Both fuse via a learned sigmoid gate.

```
Audio Input (22,050 Hz · 3 s · mono)
    │
    ├──── Spectrogram Branch ────────────────────────────────────────────┐
    │     Multi-scale mel spectrograms (3 channels, FFT 512/1024/2048)  │
    │     4× ResBlock (64 → 128 → 256 → 512 filters)                   │
    │     CBAM attention (channel + spatial) per block                  │
    │     Attention pooling → Dense(256)                                │
    │                                                                   ├─→ Gated Fusion → Dense(256) → Dense(128) → Softmax
    └──── Features Branch ──────────────────────────────────────────────┘
          645-dim handcrafted features:
          MFCCs (40) × 4 stats + delta/delta-delta, chroma (12),
          mel per-band (128), spectral contrast (7), tonnetz (6),
          ZCR, RMS, centroid, bandwidth, rolloff, piptrack pitch
          MLP: Dense(512) → Dense(256) → Dense(128) with skip connections
```

### Key Components

| Component | Role |
|---|---|
| **Multi-scale spectrograms** | 3 parallel mel spectrograms stacked as RGB-like channels — captures fine temporal detail and broad spectral shape |
| **CBAM attention** | Channel + spatial attention inside each residual block |
| **Attention pooling** | Learned 1D weights per time frame — focuses on emotionally salient segments |
| **Gated fusion** | Sigmoid gate vector learns per-prediction weighting of CNN vs MLP branch |
| **Focal Loss** (γ=2.0, label smoothing=0.1) | Downweights easy examples, prevents overconfidence on hard pairs |

---

## Training Details

| Parameter | Value |
|---|---|
| Datasets | RAVDESS, TESS, CREMA-D, SAVEE |
| Total audio clips | ~7,400 |
| Sample rate | 22,050 Hz |
| Clip duration | 3 seconds |
| Batch size | 64 |
| Epochs | 80 (early stop patience=25) |
| Optimizer | AdamW (weight decay=1e-4) |
| LR schedule | Linear warmup (5 epochs) → Cosine decay (1e-3 → 1e-6) |
| Loss | Focal Loss (γ=2.0, label smoothing=0.1) |
| Augmentation | SpecAugment, random gain, Gaussian noise, Mixup (α=0.4) |
| Training platform | Kaggle T4 GPU (~8 hours) |

---

## Files

| File | Description |
|---|---|
| `SER_v2_kaggle.ipynb` | Full training notebook — Kaggle T4 GPU, ~8 hours |
| `SER_colab_eval.ipynb` | Evaluation-only notebook — Google Colab, loads pretrained weights |
| `best_ser_v2.keras` | Trained model weights (89.3% test accuracy) |
| `label_encoder.pkl` | Sklearn `LabelEncoder` mapping emotion strings ↔ class indices |
| `scaler.pkl` | Sklearn `StandardScaler` fitted on training features |

> **Pre-trained model:** [huggingface.co/VeerR13/Speech-Emotion-Recognition](https://huggingface.co/VeerR13/Speech-Emotion-Recognition/tree/main)  
> Download `best_ser_v2.keras`, `label_encoder.pkl`, and `scaler.pkl` to skip the 8-hour training run.

---

## Quickstart: Evaluate on Colab

1. Open `SER_colab_eval.ipynb` in [Google Colab](https://colab.research.google.com)
2. Mount Google Drive when prompted
3. Paste your Kaggle API token in Cell 2:
   ```python
   KAGGLE_TOKEN = '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}'
   ```
4. Upload `best_ser_v2.keras`, `label_encoder.pkl`, `scaler.pkl` to `MyDrive/SER/`
5. Run all cells — feature extraction ~8 min, then evaluation runs automatically

Feature arrays are cached to Drive after first extraction.

## Quickstart: Train from Scratch on Kaggle

1. **Create Notebook** on [kaggle.com](https://kaggle.com) and upload `SER_v2_kaggle.ipynb`
2. **+ Add Data** → search for:
   - `uwrfkaggler/ravdess-emotional-speech-audio`
   - `ejlok1/toronto-emotional-speech-set-tess`
   - `ejlok1/cremad`
   - `ejlok1/surrey-audiovisual-expressed-emotion-savee`
3. **Settings → Accelerator → GPU P100**
4. **Run All** — downloads, trains, saves weights to Output tab (~8 hours)

---

## Inference Example

```python
import pickle, numpy as np, librosa, keras

# Load artifacts
le     = pickle.load(open('label_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Register custom objects (training-only, skipped at inference)
import keras as k

@k.saving.register_keras_serializable(package='custom')
class FocalLoss(k.losses.Loss):
    def __init__(self, gamma=2.0, smoothing=0.1, **kw):
        super().__init__(**kw)
        self.gamma = gamma; self.smoothing = smoothing
    def call(self, y_true, y_pred):
        return k.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=self.smoothing)
    def get_config(self):
        return {**super().get_config(), 'gamma': self.gamma, 'smoothing': self.smoothing}

@k.saving.register_keras_serializable(package='custom')
class WarmupCosine(k.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak=1e-3, wu_steps=1000, total=10000, min_lr=1e-6, **kw):
        super().__init__(**kw)
        self.peak=peak; self.wu_steps=wu_steps; self.total=total; self.min_lr=min_lr
    def __call__(self, step): return self.peak
    def get_config(self):
        return {'peak':self.peak,'wu_steps':self.wu_steps,'total':self.total,'min_lr':self.min_lr}

k.config.enable_unsafe_deserialization()
model = k.models.load_model('best_ser_v2.keras', compile=False)

# Preprocess audio
SR, DUR = 22050, 3
audio, _ = librosa.load('speech.wav', sr=SR, duration=DUR)
audio, _ = librosa.effects.trim(audio, top_db=25)
audio    = np.pad(audio, (0, max(0, SR*DUR - len(audio))))[:SR*DUR]

# Predict (implement extract_features / extract_mel_multiscale from notebook)
flat = scaler.transform([extract_features(audio)])
spec = extract_mel_multiscale(audio)[np.newaxis]

pred    = model.predict([spec, flat], verbose=0)
emotion = le.classes_[np.argmax(pred)]
print(f'{emotion}  {pred.max()*100:.1f}%')
```

---

## Dependencies

```
tensorflow-cpu>=2.18
keras>=3.10
librosa>=0.10
scikit-learn>=1.3
numpy
soundfile
```

---

## Dataset Sources

| Dataset | Speakers | Emotions | Clips |
|---|---|---|---|
| [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) | 24 actors | 8 | ~1,440 |
| [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) | 2 actresses | 7 | ~2,800 |
| [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) | 91 actors | 6 | ~7,442 (subset) |
| [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee) | 4 male speakers | 7 | ~480 |
