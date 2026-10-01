# SoundRoot

SoundRoot is a final-year Data Science capstone prototype for automatic classification of five South African music genres: **amapiano, maskandi, lekompo, gospel, and afro house**.

## Project pipeline

```text
Audio clip
   ↓
Mono, 22,050 Hz, up to 30 seconds
   ↓
85 handcrafted audio features (Librosa)
   ↓
StandardScaler
   ↓
Random Forest classifier
   ↓
Genre prediction + class probabilities
```

## Recorded dataset and model results

The submitted notebook records **108 samples** with **85 numerical features**. The recorded genre counts are:

| Genre | Samples |
|---|---:|
| maskandi | 24 |
| afro_house | 23 |
| gospel | 22 |
| amapiano | 20 |
| lekompo | 19 |

The Random Forest used an 80/20 stratified split (`random_state=42`). This produced **86 training samples** and **22 test samples**. The recorded held-out test accuracy was **81.8%**.

Recorded Random Forest settings:
- `n_estimators=200`
- `max_depth=15`
- `min_samples_split=3`
- `random_state=42`
- `n_jobs=-1`

The notebook also records 5-fold cross-validation of **91.8% ± 6.0%**. This should be interpreted cautiously because the `StandardScaler` was fitted before the folds were created. A cleaner cross-validation design would fit preprocessing inside each fold.

XGBoost was also tested with the settings in the notebook and recorded the same 81.8% held-out accuracy.

## Feature extraction

The reusable implementation is in `src/feature_extraction.py`. It follows the notebook's feature extraction process and returns the same feature dictionary used during training. The extracted groups include tempo/rhythm, spectral characteristics, harmonic/percussive information, chroma, MFCCs, spectral contrast, energy, temporal behaviour, and other signal characteristics.

Important recorded features include spectral rolloff, spectral flatness, zero-crossing rate, harmonic/percussive energy ratios, chroma statistics, Tonnetz, 20 MFCCs with mean and standard deviation, spectral contrast, RMS, spectral centroid, spectral bandwidth, tempogram, onset strength, beat statistics, chroma CQT, RMS fluctuation, crest factor, attack and decay measures, inharmonicity, transient ratio, spectral slope, rhythm regularity, and pulse clarity.

## Streamlit application

The application is in `app/soundroot_app.py`. It:

1. accepts an uploaded audio file;
2. loads it as mono audio at 22,050 Hz for up to 30 seconds;
3. extracts the same feature set used during training;
4. orders the features using `feature_names.txt`;
5. applies the saved scaler;
6. uses the saved Random Forest model; and
7. displays the predicted genre, probabilities, selected audio features, waveform, and spectrogram.

### Run the app

Create and activate a virtual environment, then install the requirements:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate with:

```bash
source .venv/bin/activate
```

Then run:

```bash
streamlit run app/soundroot_app.py
```

## Model files

The Streamlit app expects these trained artifacts in `app/models/`:

```text
app/models/
├── genre_classifier.pkl
├── scaler.pkl
├── label_encoder.pkl
└── feature_names.txt
```

The binary model artifacts were not embedded in the notebook itself, so they must be copied into the repository from the training environment after running the model-persistence section of the notebook.

## Repository structure

```text
SoundRoot/
├── app/
│   ├── soundroot_app.py
│   └── models/
│       └── README.md
├── data/
│   └── features.csv              # generated dataset; add if appropriate for submission
├── notebooks/
│   └── SoundRoot2.ipynb
├── src/
│   ├── feature_extraction.py
│   └── extract_dataset_features.py
├── requirements.txt
└── README.md
```

## Reproducing the feature table

If the 30-second WAV clips are available under `data/clips_30sec/<genre>/`, run:

```bash
python src/extract_dataset_features.py
```

This writes `data/features.csv`.

## Limitations

The dataset is small (108 samples), so the results should be treated as a prototype evaluation rather than evidence of broad real-world genre classification performance. The test set contains only 22 samples. The recorded cross-validation score also needs cautious interpretation because scaling was performed before the folds were created.

## Development environment

The notebook recorded these development versions: librosa 0.11.0, soundfile 0.14.0, numpy 1.26.4, pandas 2.2.2, matplotlib 3.8.4, scikit-learn 1.4.2, xgboost 3.2.0, Streamlit 1.32.0, TensorFlow 2.19.1, yt-dlp 2026.6.9, and ffmpeg 8.1.1. The Streamlit app itself does not require TensorFlow or XGBoost because it loads the saved Random Forest model.

## Academic project note

This repository contains work for a final-year Data Science capstone at Sol Plaatje University. Audio-source permissions, copyright, and platform terms should be checked for any tracks included in a public repository or demonstration dataset.
