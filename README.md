# Speech Emotion Recognition (SER)

A deep learning system for classifying speech into 8 emotions using a dual-input fusion architecture.
Trained on 4 public datasets — achieves **89.3% test accuracy** and **91.1% macro F1**.

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **89.29%** |
| Macro F1 | **91.09%** |
| Best Val Accuracy (training) | 81.41% (epoch 67/80) |
| TTA Test Accuracy (15 passes) | ~82–83% on Kaggle |

> The 89.3% figure is from full evaluation on held-out test data in Google Colab (CPU, no TTA).
> Pre-trained model: [huggingface.co/VeerR13/Speech-Emotion-Recognition](https://huggingface.co/VeerR13/Speech-Emotion-Recognition/tree/main)

### Confusion Matrix

![Confusion Matrix](https://huggingface.co/VeerR13/Speech-Emotion-Recognition/resolve/main/results.png)
> Pre-trained model: [huggingface.co/VeerR13/Speech-Emotion-Recognition](https://huggingface.co/VeerR13/Speech-Emotion-Recognition/tree/main)

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

## Emotions Detected

`neutral` · `calm` · `happy` · `sad` · `angry` · `fear` · `disgust` · `surprise`

---

## Architecture

The model uses a dual-input fusion design — a convolutional branch processes mel spectrograms while an MLP branch processes handcrafted audio features. Both branches are fused via a learned sigmoid gate.

```
Audio Input
    │
    ├──── Spectrogram Branch ────────────────────────────────────────────┐
    │     Multi-scale mel spectrograms (3 channels)                      │
    │     4× ResBlock (64→128→256→512 filters)                           │
    │     CBAM attention (channel + spatial) per block                   │
    │     Attention pooling → Dense(256)                                 │
    │                                                                    ├─→ Gated Fusion → Dense(256) → Dense(128) → Softmax(8)
    └──── Features Branch ───────────────────────────────────────────────┘
          193-dim handcrafted features (MFCCs×4 stats, deltas, chroma,
          spectral contrast, tonnetz, ZCR, RMS, centroid, bandwidth,
          rolloff, pitch)
          MLP: Dense(512) → Dense(256) → Dense(128) with skip connections
```

### Key Components

- **Multi-scale spectrograms** — 3 parallel mel spectrograms with different FFT window sizes (512, 1024, 2048), stacked as RGB-like channels. Captures fine temporal detail and broad spectral shape simultaneously.
- **CBAM attention** — applied inside each residual block. Channel attention re-weights which feature maps matter; spatial attention re-weights which time-frequency regions matter.
- **Attention pooling** — replaces global average pooling. A learned 1D attention vector weights each time frame before summing, so the model focuses on emotionally salient segments.
- **Gated fusion** — a sigmoid gate vector (same dimension as the concatenated embeddings) learns how much to weight each branch per prediction.
- **Focal Loss** (γ=2.0) with label smoothing=0.1 — downweights easy examples and prevents overconfident predictions on hard-to-distinguish pairs like *neutral/calm* and *sad/neutral*.

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
| `SER_v2_kaggle.ipynb` | Full training notebook — runs on Kaggle T4 GPU in ~8 hours |
| `SER_colab_eval.ipynb` | Evaluation-only notebook — runs on Google Colab, loads pretrained model |
| `best_ser_v2.keras` | Trained model weights (89.3% test accuracy) |
| `label_encoder.pkl` | Sklearn `LabelEncoder` mapping emotion strings ↔ class indices |
| `scaler.pkl` | Sklearn `StandardScaler` fitted on training features |

> **Pre-trained model on Hugging Face:** [VeerR13/Speech-Emotion-Recognition](https://huggingface.co/VeerR13/Speech-Emotion-Recognition/tree/main)
> Download `best_ser_v2.keras`, `label_encoder.pkl`, and `scaler.pkl` from there to skip the 8-hour training run.

---

## Quickstart: Evaluate on Google Colab

No training required. Use the saved weights directly.

### Prerequisites

1. A [Kaggle account](https://kaggle.com) (free) — needed to download the 4 audio datasets
2. A [Google account](https://drive.google.com) with Google Drive — results and features are cached there
3. Download `best_ser_v2.keras`, `label_encoder.pkl`, `scaler.pkl` from this repo

### Steps

1. Open `SER_colab_eval.ipynb` in Google Colab
2. Mount Google Drive when Cell 1 prompts you
3. In Cell 2, paste your Kaggle API token (JSON string from `kaggle.com → Settings → API → Create New Token`):
   ```python
   KAGGLE_TOKEN = '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}'
   ```
4. Upload `best_ser_v2.keras`, `label_encoder.pkl`, `scaler.pkl` to `MyDrive/SER/` in your Drive
5. Run all cells — feature extraction takes ~8 minutes, then evaluation runs automatically

Feature arrays are cached to Drive after first extraction — subsequent runs skip re-extraction.

---

## Quickstart: Train from Scratch on Kaggle

1. Go to [kaggle.com](https://kaggle.com) → **Create Notebook**
2. Upload `SER_v2_kaggle.ipynb`
3. Click **+ Add Data** and add these datasets by searching:
   - `uwrfkaggler/ravdess-emotional-speech-audio`
   - `ejlok1/toronto-emotional-speech-set-tess`
   - `ejlok1/cremad`
   - `ejlok1/surrey-audiovisual-expressed-emotion-savee`
4. Set accelerator to **GPU P100** (Settings → Accelerator)
5. Click **Run All**

Training runs fully unattended (~8 hours). Download `best_ser_v2.keras`, `label_encoder.pkl`, `scaler.pkl` from the Output tab when done.

---

## Inference Example

```python
import pickle, numpy as np, librosa, tensorflow as tf

# Define custom layers (required for model deserialization)
class ChannelMean(tf.keras.layers.Layer):
    def call(self, x): return tf.reduce_mean(x, axis=-1, keepdims=True)

class ChannelMax(tf.keras.layers.Layer):
    def call(self, x): return tf.reduce_max(x, axis=-1, keepdims=True)

class AttnWeightedSum(tf.keras.layers.Layer):
    def call(self, inputs): return tf.reduce_sum(inputs[0] * inputs[1], axis=1)

# ── Paste FocalLoss, WarmupCosine, build_model, extract_features,
#    extract_mel_multiscale definitions from SER_colab_eval.ipynb ────────────

# Load saved artifacts
le     = pickle.load(open('label_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
NC     = len(le.classes_)   # 8

model = build_model(193, (128, 128, 3), NC)
model.load_weights('best_ser_v2.keras')
model.compile(loss=FocalLoss(gamma=2.0, smoothing=0.1), metrics=['accuracy'])

# Load and preprocess audio
SAMPLE_RATE = 22050
N_SAMPLES   = 66150   # 3 seconds

audio, _ = librosa.load('speech.wav', sr=SAMPLE_RATE, duration=3.0)
audio, _ = librosa.effects.trim(audio, top_db=25)
audio    = np.pad(audio, (0, max(0, N_SAMPLES - len(audio))))[:N_SAMPLES]

# Extract features
flat = scaler.transform([extract_features(audio)])
spec = extract_mel_multiscale(audio)[np.newaxis]

# Predict
pred    = model.predict({'spectrogram': spec, 'features': flat}, verbose=0)
emotion = le.classes_[np.argmax(pred)]
print(f'Predicted: {emotion}  ({pred.max()*100:.1f}% confidence)')
```

---

## Dependencies

```
tensorflow>=2.15
librosa>=0.10
scikit-learn>=1.3
numpy
matplotlib
seaborn
kaggle
```

---

## Dataset Sources

| Dataset | Speakers | Emotions | Clips |
|---|---|---|---|
| [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) | 24 actors | 8 | ~1,440 |
| [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) | 2 actresses | 7 | ~2,800 |
| [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) | 91 actors | 6 | ~7,442 (subset used) |
| [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee) | 4 male speakers | 7 | ~480 |
