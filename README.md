# Speech Emotion Recognition (SER)

A deep learning model for classifying speech into 8 emotions using a dual-input fusion architecture trained on 4 public datasets. Achieves **81.4% validation accuracy** with Test-Time Augmentation (TTA).

---

## Emotions Detected

`neutral` · `calm` · `happy` · `sad` · `angry` · `fear` · `disgust` · `surprise`

---

## Architecture

The model fuses two parallel input branches:

**Spectrogram Branch**
- Multi-scale mel spectrograms (3 channels: different FFT window sizes)
- 4 residual blocks (64 → 128 → 256 → 512 filters)
- CBAM attention (channel + spatial) in each residual block
- Attention pooling → Dense(256)

**Features Branch**
- 193-dimensional handcrafted features: MFCCs (×4 stats), deltas, chroma, spectral contrast, tonnetz, ZCR, RMS, spectral centroid/bandwidth/rolloff, pitch
- 3-layer MLP with skip connections: Dense(512) → Dense(256) → Dense(128)

**Fusion**
- Concatenate both branches → gated fusion (sigmoid gate)
- Dense(256) → Dense(128) → Softmax(8)

---

## Results

| Metric | Score |
|---|---|
| Best Val Accuracy | **81.41%** (epoch 67/80) |
| Standard Test Accuracy | ~80% |
| TTA Test Accuracy (15 passes) | ~82-83% |
| TTA Macro F1 | ~82% |

---

## Training Details

| Parameter | Value |
|---|---|
| Datasets | RAVDESS, TESS, CREMA-D, SAVEE |
| Total files | ~7,400 audio clips |
| Sample rate | 22,050 Hz |
| Duration | 3 seconds |
| Batch size | 64 |
| Epochs | 80 (early stop patience=25) |
| Optimizer | AdamW (weight_decay=1e-4) |
| LR schedule | Warmup (5 epochs) → Cosine decay (1e-3 → 1e-6) |
| Loss | Focal Loss (γ=2.0, label smoothing=0.1) |
| Augmentation | SpecAugment + random gain + Gaussian noise + Mixup (α=0.4) |

---

## Files

| File | Description |
|---|---|
| `SER_v2_kaggle.ipynb` | Full training notebook — runs on Kaggle T4 GPU in ~8 hours |
| `best_ser_v2.keras` | Trained model weights (81.4% val accuracy) — download from Kaggle output after training |
| `label_encoder.pkl` | Sklearn LabelEncoder mapping emotion strings to class indices |
| `scaler.pkl` | Sklearn StandardScaler fitted on training features |

> Pre-trained files are included in this repo. Download `best_ser_v2.keras`, `label_encoder.pkl`, and `scaler.pkl` directly from GitHub to skip the 8-hour training run.

---

## How to Run Training (Kaggle)

1. Go to [kaggle.com](https://kaggle.com) → **Create Notebook**
2. Upload `SER_v2_kaggle.ipynb`
3. Click **+ Add Data** and add these 4 datasets by searching:
   - `uwrfkaggler/ravdess-emotional-speech-audio`
   - `ejlok1/toronto-emotional-speech-set-tess`
   - `ejlok1/cremad`
   - `ejlok1/surrey-audiovisual-expressed-emotion-savee`
4. Set accelerator to **GPU T4 x2** (Settings → Accelerator)
5. Click **Run All**
6. When done, download `best_ser_v2.keras`, `label_encoder.pkl`, `scaler.pkl` from the Output tab

Training runs fully unattended — safe to close the browser.

---

## How to Run Evaluation Only (if you have the saved files)

If you already have `best_ser_v2.keras`, `label_encoder.pkl`, and `scaler.pkl`:

1. Create a new Kaggle notebook and upload `SER_v2_kaggle.ipynb`
2. Add the 4 audio datasets (same as above)
3. Click **+ Add Data → Upload** and upload the 3 saved files as a dataset titled `ser-saved-model`
4. Run all cells — feature extraction takes ~8 min, then it loads the saved model and runs TTA evaluation directly without retraining

---

## Inference Example

```python
import pickle, numpy as np, librosa, tensorflow as tf
from tensorflow.keras import regularizers

# ── Paste FocalLoss and WarmupCosine class definitions from the notebook ──────

# Load saved artifacts
model  = tf.keras.models.load_model('best_ser_v2.keras',
             custom_objects={'FocalLoss': FocalLoss, 'WarmupCosine': WarmupCosine})
le     = pickle.load(open('label_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load and preprocess audio
SAMPLE_RATE = 22050
N_SAMPLES   = 66150   # 3 seconds

audio, _ = librosa.load('speech.wav', sr=SAMPLE_RATE, duration=3.0)
audio, _ = librosa.effects.trim(audio, top_db=25)
audio    = np.pad(audio, (0, max(0, N_SAMPLES - len(audio))))[:N_SAMPLES]

# Extract features (copy extract_features and extract_mel_multiscale from notebook)
flat  = scaler.transform([extract_features(audio)])
spec  = extract_mel_multiscale(audio)[np.newaxis]

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
```
