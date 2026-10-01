import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import tempfile
import os
import shutil

st.set_page_config(page_title="SoundRoot", page_icon="🎵", layout="wide")

# ========== PATH SETUP ==========
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    @keyframes equalizer {
        0% { height: 10px; }
        50% { height: 40px; }
        100% { height: 10px; }
    }
    .music-bars {
        display: flex;
        align-items: flex-end;
        justify-content: center;
        gap: 4px;
        height: 50px;
        margin: 20px 0;
    }
    .bar {
        width: 6px;
        background: linear-gradient(to top, #1DB954, #1ed760);
        border-radius: 3px;
        animation: equalizer 1s ease-in-out infinite;
    }
    .bar:nth-child(1) { animation-delay: 0s; height: 15px; }
    .bar:nth-child(2) { animation-delay: 0.1s; height: 30px; }
    .bar:nth-child(3) { animation-delay: 0.2s; height: 20px; }
    .bar:nth-child(4) { animation-delay: 0.3s; height: 35px; }
    .bar:nth-child(5) { animation-delay: 0.4s; height: 25px; }
    .bar:nth-child(6) { animation-delay: 0.5s; height: 40px; }
    .bar:nth-child(7) { animation-delay: 0.6s; height: 18px; }
    .bar:nth-child(8) { animation-delay: 0.7s; height: 32px; }
    @keyframes glow {
        0% { box-shadow: 0 0 5px #1DB954, 0 0 10px #1DB954; }
        50% { box-shadow: 0 0 20px #1DB954, 0 0 30px #1ed760; }
        100% { box-shadow: 0 0 5px #1DB954, 0 0 10px #1DB954; }
    }
    @keyframes textGlow {
        0% { text-shadow: 0 0 10px #1DB954; }
        50% { text-shadow: 0 0 20px #1ed760, 0 0 30px #1DB954; }
        100% { text-shadow: 0 0 10px #1DB954; }
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    @keyframes smoke {
        0% { opacity: 0.3; transform: translateY(0) scale(1); }
        50% { opacity: 0.6; transform: translateY(-5px) scale(1.05); }
        100% { opacity: 0.3; transform: translateY(0) scale(1); }
    }
    .soundroot-title {
        font-family: 'Courier New', monospace;
        font-size: 4em;
        font-weight: 900;
        text-align: center;
        color: #1DB954;
        animation: textGlow 2s ease-in-out infinite, float 3s ease-in-out infinite;
        letter-spacing: 8px;
        margin: 0;
        text-transform: uppercase;
    }
    .soundroot-subtitle {
        font-family: 'Courier New', monospace;
        text-align: center;
        color: #888;
        font-size: 1.1em;
        letter-spacing: 4px;
        margin-top: 5px;
    }
    .blxck-signature {
        font-family: 'Courier New', monospace;
        text-align: center;
        color: #666;
        font-size: 0.9em;
        letter-spacing: 3px;
        margin-top: 30px;
        padding: 15px;
        border: 1px solid #333;
        border-radius: 10px;
        background: rgba(0,0,0,0.3);
        animation: smoke 4s ease-in-out infinite;
    }
    .blxck-text {
        color: #1DB954;
        font-weight: bold;
        font-size: 1.2em;
        text-shadow: 0 0 10px rgba(29, 185, 84, 0.5);
    }
    .prediction-box {
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.2) 0%, rgba(30, 215, 96, 0.1) 100%);
        border: 2px solid #1DB954;
        color: #fff;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        animation: glow 3s ease-in-out infinite;
        backdrop-filter: blur(10px);
    }
    .prediction-genre {
        font-family: 'Courier New', monospace;
        font-size: 3em;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 5px;
        margin: 10px 0;
    }
    .genre-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        transition: all 0.3s;
        color: #e0e0e0;
    }
    .genre-card:hover {
        transform: translateX(10px);
        background: rgba(255,255,255,0.1);
        border-color: #1DB954;
    }
    .feature-metric {
        background: rgba(0,0,0,0.4);
        border: 1px solid rgba(29, 185, 84, 0.3);
        color: #1DB954;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-family: 'Courier New', monospace;
    }
    .warning-box {
        background: rgba(255, 107, 107, 0.2);
        border: 2px solid #ff6b6b;
        color: #ff6b6b;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        animation: glow 2s ease-in-out infinite;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1DB954, #1ed760);
        color: #000;
        border-radius: 25px;
        padding: 12px 35px;
        font-weight: 900;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(29, 185, 84, 0.6);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px 10px 0 0;
        color: #888;
        font-family: 'Courier New', monospace;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(29, 185, 84, 0.2) !important;
        color: #1DB954 !important;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #1DB954, #1ed760);
        border-radius: 10px;
    }
    .stFileUploader {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(29, 185, 84, 0.3);
        border-radius: 15px;
        padding: 20px;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(29, 185, 84, 0.3);
        color: #e0e0e0;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
    }
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #1DB954, transparent);
        margin: 30px 0;
    }
    .footer {
        text-align: center;
        color: #555;
        font-family: 'Courier New', monospace;
        font-size: 0.8em;
        letter-spacing: 2px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "genre_classifier.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    with open(MODELS_DIR / "feature_names.txt", "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f if line.strip()]
    return model, scaler, le, feature_names

model, scaler, le, feature_names = load_model()

GENRE_INFO = {
    "amapiano": {"color": "#FF6B6B", "desc": "Log drums, shuffles, South African house music"},
    "maskandi": {"color": "#4ECDC4", "desc": "Zulu guitar, concertina, traditional vocals"},
    "lekompo": {"color": "#45B7D1", "desc": "Accordion bounce, fast 4/4, bass-heavy"},
    "gospel": {"color": "#96CEB4", "desc": "Choir harmonies, organ, spiritual vocals"},
    "afro_house": {"color": "#FFEAA7", "desc": "Four-on-floor kick, tribal percussion, electronic"}
}

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)
    features = {}
    
    # Tempo with correction for half-tempo detection
    beat_result = librosa.beat.beat_track(y=y, sr=sr)
    
    # Handle both old and new librosa return formats
    if isinstance(beat_result, tuple):
        tempo = beat_result[0]
        beat_frames = beat_result[1]
    else:
        tempo = beat_result
        beat_frames = librosa.beat.beat_track(y=y, sr=sr)[1]
    
    # Convert tempo to float safely
    if hasattr(tempo, 'item'):
        tempo = float(tempo.item())
    elif isinstance(tempo, (list, tuple, np.ndarray)):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)
    
    # Correct for half-tempo in fast electronic genres
    if tempo < 90:
        tempo_doubled = tempo * 2
        if tempo_doubled < 180:
            tempo = tempo_doubled
    
    features["tempo"] = tempo
    
    # Convert beat_frames to numpy array if needed
    if isinstance(beat_frames, (list, tuple)):
        beat_frames = np.array(beat_frames)
    
    # 1. SPECTRAL ROLLOFF
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))
    
    # 2. SPECTRAL FLATNESS
    flatness = librosa.feature.spectral_flatness(y=y)
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["spectral_flatness_std"] = float(np.std(flatness))
    
    # 3. ZERO CROSSING RATE
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))
    
    # 4. HARMONIC vs PERCUSSIVE
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    features["harmonic_energy_ratio"] = float(np.sum(y_harmonic**2) / (np.sum(y**2) + 1e-8))
    features["percussive_energy_ratio"] = float(np.sum(y_percussive**2) / (np.sum(y**2) + 1e-8))
    
    # 5. CHROMA
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
    features["chroma_mean"] = float(np.mean(chroma))
    features["chroma_std"] = float(np.std(chroma))
    chroma_sum = np.sum(chroma, axis=0, keepdims=True)
    chroma_norm = chroma / (chroma_sum + 1e-8)
    features["chroma_entropy"] = float(-np.sum(chroma_norm * np.log2(chroma_norm + 1e-8)))
    
    # 6. TONNETZ
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    features["tonnetz_mean"] = float(np.mean(tonnetz))
    features["tonnetz_std"] = float(np.std(tonnetz))
    
    # 7. MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f"mfcc_{i+1}"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))
    
    # 8. SPECTRAL CONTRAST
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["spectral_contrast_mean"] = float(np.mean(contrast))
    features["spectral_contrast_std"] = float(np.std(contrast))
    
    # 9. RMS ENERGY
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    
    # 10. SPECTRAL CENTROID
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))
    
    # 11. SPECTRAL BANDWIDTH
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))
    
    # 12. TEMPOGRAM
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    features["tempogram_mean"] = float(np.mean(tempogram))
    features["tempogram_std"] = float(np.std(tempogram))
    
    # 13. ONSET STRENGTH
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features["onset_strength_mean"] = float(np.mean(onset_env))
    features["onset_strength_std"] = float(np.std(onset_env))
    
    # 14. BEAT INTERVALS
    features["num_beats"] = int(len(beat_frames))
    if len(beat_frames) > 1:
        diffs = np.diff(beat_frames)
        features["beat_interval_mean"] = float(np.mean(diffs))
        features["beat_interval_std"] = float(np.std(diffs))
    else:
        features["beat_interval_mean"] = 0.0
        features["beat_interval_std"] = 0.0
    
    # 15. CHROMA CQT
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    features["chroma_cqt_mean"] = float(np.mean(chroma_cqt))
    features["chroma_cqt_std"] = float(np.std(chroma_cqt))
    
    # 16. RMS FLUCTUATION
    rms_flat = rms.flatten()
    if len(rms_flat) > 1:
        rms_diff = np.diff(rms_flat)
        features["rms_fluctuation"] = float(np.std(rms_diff))
    else:
        features["rms_fluctuation"] = 0.0
    
    # 17. SPECTRAL ROLLOFF 50%
    rolloff_50 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.5)
    features["spectral_rolloff_50_mean"] = float(np.mean(rolloff_50))
    features["spectral_rolloff_50_std"] = float(np.std(rolloff_50))
    
    # 18. CREST FACTOR
    peak = float(np.max(np.abs(y)))
    rms_val = float(np.sqrt(np.mean(y**2)))
    features["crest_factor"] = peak / (rms_val + 1e-8)
    
    # 19. ATTACK TIME
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if len(onset_env) > 0:
        onset_threshold = float(np.max(onset_env)) * 0.5
        attack_frames = np.where(onset_env > onset_threshold)[0]
        if len(attack_frames) > 0:
            features["attack_time"] = float(attack_frames[0]) / sr
        else:
            features["attack_time"] = 0.0
    else:
        features["attack_time"] = 0.0
    
    # 20. DECAY TIME
    env = np.abs(y)
    env_smooth = np.convolve(env, np.ones(1000)/1000, mode='same')
    peak_idx = int(np.argmax(env_smooth))
    decay_threshold = env_smooth[peak_idx] * 0.1
    post_peak = env_smooth[peak_idx:]
    if len(post_peak) > 0 and np.any(post_peak < decay_threshold):
        decay_idx = peak_idx + int(np.argmax(post_peak < decay_threshold))
        features["decay_time"] = float(decay_idx - peak_idx) / sr
    else:
        features["decay_time"] = float(len(env_smooth) - peak_idx) / sr
    
    # 21. INHARMONICITY
    fft = np.fft.rfft(y_harmonic)
    freqs = np.fft.rfftfreq(len(y_harmonic), 1/sr)
    magnitudes = np.abs(fft)
    peak_indices = np.argsort(magnitudes)[-20:]
    peak_freqs = freqs[peak_indices]
    peak_mags = magnitudes[peak_indices]
    if len(peak_freqs) > 1 and peak_freqs[0] > 0:
        fundamental = float(peak_freqs[0])
        inharmonicity = 0.0
        for i in range(1, len(peak_freqs)):
            f = float(peak_freqs[i])
            harmonic_num = round(f / fundamental)
            if harmonic_num > 0:
                deviation = abs(f - harmonic_num * fundamental) / fundamental
                inharmonicity += deviation * float(peak_mags[i])
        features["inharmonicity"] = inharmonicity / (float(np.sum(peak_mags[1:])) + 1e-8)
    else:
        features["inharmonicity"] = 0.0
    
    # 22. TRANSIENT RATIO
    transient_env = np.abs(y_percussive)
    sustained_env = np.abs(y_harmonic)
    features["transient_ratio"] = float(np.mean(transient_env)) / (float(np.mean(sustained_env)) + 1e-8)
    
    # 23. SPECTRAL SLOPE
    freqs_spec = np.linspace(0, sr/2, len(fft))
    spec = np.abs(fft)
    if len(freqs_spec) > 1 and len(spec) > 1:
        log_freqs = np.log(freqs_spec[1:] + 1e-8)
        log_spec = np.log(spec[1:] + 1e-8)
        valid_mask = np.isfinite(log_freqs) & np.isfinite(log_spec)
        if np.sum(valid_mask) > 1:
            slope = np.polyfit(log_freqs[valid_mask], log_spec[valid_mask], 1)[0]
            features["spectral_slope"] = float(slope)
        else:
            features["spectral_slope"] = 0.0
    else:
        features["spectral_slope"] = 0.0
    
    # 24. SPECTRAL FLATNESS BAND
    spec_flat = librosa.feature.spectral_flatness(y=y)
    features["spectral_flatness_band_mean"] = float(np.mean(spec_flat))
    features["spectral_flatness_band_std"] = float(np.std(spec_flat))
    
    # 25. RHYTHM REGULARITY
    if len(beat_frames) > 1:
        diffs = np.diff(beat_frames)
        if features["beat_interval_mean"] > 0:
            features["rhythm_regularity"] = features["beat_interval_std"] / (features["beat_interval_mean"] + 1e-8)
        else:
            features["rhythm_regularity"] = 0.0
    else:
        features["rhythm_regularity"] = 0.0
    
    # 26. PULSE CLARITY
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    if tempogram.shape[1] > 0:
        tempogram_max = np.max(tempogram, axis=0)
        tempogram_mean = np.mean(tempogram, axis=0)
        pulse_clarity = tempogram_max / (tempogram_mean + 1e-8)
        features["pulse_clarity_mean"] = float(np.mean(pulse_clarity))
        features["pulse_clarity_std"] = float(np.std(pulse_clarity))
    else:
        features["pulse_clarity_mean"] = 0.0
        features["pulse_clarity_std"] = 0.0
    
    return features, y, sr

def download_youtube_audio(url):
    tmpdir = tempfile.mkdtemp()
    try:
        output_path = os.path.join(tmpdir, "audio.%(ext)s")
        cmd = [
            "yt-dlp", "--format", "bestaudio/best",
            "-o", output_path, "--no-playlist",
            "--no-check-certificates", "--no-warnings", url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, f"Download failed: {result.stderr[:300]}"
        files = [f for f in os.listdir(tmpdir) if f.startswith("audio.")]
        if not files:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, "No audio file found"
        downloaded = os.path.join(tmpdir, files[0])
        wav_path = os.path.join(tmpdir, "converted.wav")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", downloaded, "-ar", "22050", "-ac", "1", wav_path
        ]
        ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
        if ffmpeg_result.returncode != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, f"FFmpeg failed: {ffmpeg_result.stderr[:200]}"
        return wav_path, None
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None, str(e)

st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <div class="music-bars">
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
    </div>
    <h1 class="soundroot-title">SOUNDROOT</h1>
    <p class="soundroot-subtitle">SOUTH AFRICAN MUSIC GENRE CLASSIFIER</p>
    <div class="music-bars">
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="blxck-signature">
    <span class="blxck-text">BY BLXCK 420</span>
    <br>
    <small>Data Science Capstone | 2026</small>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("Upload or Link")
tab1, tab2 = st.tabs([":open_file_folder: Upload File", ":link: YouTube Link"])

audio_path = None
uploaded_file = None

with tab1:
    uploaded_file = st.file_uploader("Drop your audio file here", type=["wav", "mp3"], 
                                     help="Upload a 30-60 second clip for best results")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.getvalue())
            audio_path = tmp.name

with tab2:
    youtube_url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    if youtube_url and st.button("Download & Analyze", key="youtube_btn"):
        with st.spinner("Downloading from YouTube..."):
            audio_path, error = download_youtube_audio(youtube_url)
            if error:
                st.error(error)
                st.info("Try uploading a file instead. YouTube download requires yt-dlp setup.")
                audio_path = None
            else:
                st.success("Downloaded successfully!")

if audio_path is not None:
    with st.spinner("Analyzing audio..."):
        features, y, sr = extract_features(audio_path)
        feature_vector = np.array([[features[f] for f in feature_names]])
        feature_vector_scaled = scaler.transform(feature_vector)
        
        prediction = model.predict(feature_vector_scaled)[0]
        pred_label = le.inverse_transform([prediction])[0]
        probs = model.predict_proba(feature_vector_scaled)[0] if hasattr(model, "predict_proba") else None
    
    st.markdown("---")
    
    pred_color = GENRE_INFO.get(pred_label, {}).get("color", "#1DB954")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="prediction-box" style="border-color: {pred_color};">
            <h2 style="margin: 0; font-size: 1.2em; color: #888; font-family: Courier New;">Detected Genre</h2>
            <h1 class="prediction-genre" style="color: {pred_color};">{pred_label.replace('_', ' ')}</h1>
            <p style="margin: 0; color: #aaa; font-family: Courier New;">{GENRE_INFO.get(pred_label, {}).get('desc', '')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    max_conf = max(probs) if probs is not None else 0
    if max_conf < 0.5:
        st.markdown("""
        <div class="warning-box">
            :warning: LOW CONFIDENCE PREDICTION<br>
            The model is uncertain about this classification.<br>
            Try a clearer example of the genre.
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("Top Predictions")
    if probs is not None:
        sorted_indices = np.argsort(probs)[::-1][:3]
        for idx in sorted_indices:
            genre = le.classes_[idx]
            prob = probs[idx]
            color = GENRE_INFO.get(genre, {}).get("color", "#666")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div class="genre-card" style="border-left: 5px solid {color};">
                    <strong style="color: {color};">{genre.replace('_', ' ').title()}</strong> 
                    <span style="color: #888;">- {prob:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.progress(float(prob))
    
    st.subheader("Key Features")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:5]
        
        feat_cols = st.columns(5)
        for i, idx in enumerate(top_indices):
            with feat_cols[i]:
                feat_name = feature_names[idx]
                feat_val = features[feat_name]
                st.markdown(f"""
                <div class="feature-metric">
                    <small style="color: #888;">{feat_name.replace('_', ' ').upper()}</small>
                    <h3 style="margin: 5px 0;">{feat_val:.2f}</h3>
                </div>
                """, unsafe_allow_html=True)
    
    st.subheader("Audio Features")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tempo", f"{features['tempo']:.0f} BPM")
    with col2:
        st.metric("Spectral Centroid", f"{features['spectral_centroid_mean']:.0f} Hz")
    with col3:
        st.metric("ZCR", f"{features['zcr_mean']:.4f}")
    with col4:
        st.metric("RMS Energy", f"{features['rms_mean']:.4f}")
    
    # =========================================================
    # ENHANCED AUDIO PREVIEW SECTION WITH ANIMATIONS
    # =========================================================
    st.subheader("Audio Preview")
    
    st.markdown(f"""
    <style>
        @keyframes spin-vinyl {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes pulse-glow {{
            0%, 100% {{ box-shadow: 0 0 20px {pred_color}40; }}
            50% {{ box-shadow: 0 0 40px {pred_color}80, 0 0 60px {pred_color}40; }}
        }}
        @keyframes sound-bars {{
            0% {{ height: 5px; opacity: 0.4; }}
            50% {{ height: 35px; opacity: 1; }}
            100% {{ height: 5px; opacity: 0.4; }}
        }}
        @keyframes slide-light {{
            0% {{ left: -100%; }}
            100% {{ left: 100%; }}
        }}
        @keyframes bounce-in {{
            0% {{ transform: scale(0); opacity: 0; }}
            60% {{ transform: scale(1.1); }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        
        .audio-box {{
            background: linear-gradient(135deg, rgba(0,0,0,0.6) 0%, {pred_color}10 100%);
            border: 2px solid {pred_color}40;
            border-radius: 25px;
            padding: 30px;
            margin: 20px 0;
            position: relative;
            overflow: hidden;
            animation: pulse-glow 3s ease-in-out infinite;
        }}
        .audio-box::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, transparent, {pred_color}, {pred_color}80, transparent);
            animation: slide-light 4s linear infinite;
        }}
        
        .vinyl {{
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 35%, #444 0%, #111 30%, #000 45%, #222 50%, #000 100%);
            border: 3px solid {pred_color};
            margin: 0 auto 15px;
            position: relative;
            box-shadow: 0 0 20px {pred_color}40;
            animation: spin-vinyl 3s linear infinite;
        }}
        .vinyl::after {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 30px;
            height: 30px;
            background: {pred_color};
            border-radius: 50%;
            border: 3px solid #000;
        }}
        
        .now-playing {{
            display: inline-block;
            background: linear-gradient(135deg, {pred_color}, {pred_color}CC);
            color: #000;
            padding: 8px 20px;
            border-radius: 20px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            font-size: 13px;
            letter-spacing: 2px;
            animation: bounce-in 0.6s ease-out;
            margin-bottom: 10px;
        }}
        
        .genre-pill {{
            display: inline-block;
            background: {pred_color}20;
            color: {pred_color};
            border: 1px solid {pred_color};
            padding: 5px 15px;
            border-radius: 15px;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 8px 5px;
        }}
        
        .equalizer {{
            display: flex;
            align-items: flex-end;
            justify-content: center;
            height: 50px;
            gap: 3px;
            margin: 20px 0;
            padding: 10px;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
        }}
        .eq-bar {{
            width: 5px;
            background: linear-gradient(to top, {pred_color}, {pred_color}80, #fff);
            border-radius: 3px;
            animation: sound-bars 1s ease-in-out infinite;
        }}
        .eq-bar:nth-child(1) {{ animation-delay: 0s; height: 12px; }}
        .eq-bar:nth-child(2) {{ animation-delay: 0.1s; height: 25px; }}
        .eq-bar:nth-child(3) {{ animation-delay: 0.2s; height: 18px; }}
        .eq-bar:nth-child(4) {{ animation-delay: 0.15s; height: 30px; }}
        .eq-bar:nth-child(5) {{ animation-delay: 0.3s; height: 22px; }}
        .eq-bar:nth-child(6) {{ animation-delay: 0.25s; height: 35px; }}
        .eq-bar:nth-child(7) {{ animation-delay: 0.1s; height: 20px; }}
        .eq-bar:nth-child(8) {{ animation-delay: 0.35s; height: 28px; }}
        .eq-bar:nth-child(9) {{ animation-delay: 0.05s; height: 32px; }}
        .eq-bar:nth-child(10) {{ animation-delay: 0.4s; height: 15px; }}
        .eq-bar:nth-child(11) {{ animation-delay: 0.2s; height: 38px; }}
        .eq-bar:nth-child(12) {{ animation-delay: 0.45s; height: 22px; }}
        .eq-bar:nth-child(13) {{ animation-delay: 0.15s; height: 30px; }}
        .eq-bar:nth-child(14) {{ animation-delay: 0.5s; height: 18px; }}
        .eq-bar:nth-child(15) {{ animation-delay: 0.3s; height: 35px; }}
        .eq-bar:nth-child(16) {{ animation-delay: 0.55s; height: 25px; }}
        
        .demo-tip {{
            text-align: center;
            color: {pred_color}AA;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            margin-top: 15px;
            letter-spacing: 1px;
        }}
    </style>
    
    <div class="audio-box">
        <div class="vinyl"></div>
        <div style="text-align: center;">
            <span class="now-playing">▶ NOW PLAYING</span>
        </div>
        <div style="text-align: center;">
            <span class="genre-pill">{pred_label.replace('_', ' ').upper()}</span>
        </div>
        <div class="equalizer">
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
        </div>
        <div class="demo-tip">
            💡 Press play below — the vinyl spins and bars dance automatically!
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.audio(audio_path)
    
    st.subheader("Visualizations")
    tab1, tab2 = st.tabs([":bar_chart: Waveform", ":art: Spectrogram"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(y, color=pred_color, linewidth=0.5, alpha=0.8)
        ax.set_title("Waveform", fontsize=14, fontweight='bold', color='white')
        ax.set_xlabel("Sample", color='white')
        ax.set_ylabel("Amplitude", color='white')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('#0a0a0a')
        fig.patch.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title("Spectrogram", fontsize=14, fontweight='bold', color='white')
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.ax.tick_params(colors='white')
        st.pyplot(fig)

st.markdown("---")
st.markdown("""
<div class="footer">
    <p>SOUNDROOT - SOUTH AFRICAN MUSIC GENRE CLASSIFICATION</p>
    <p>Capstone Project | Data Science Final Year | 108 Samples | 81.8% Held-Out Test Accuracy</p>
    <p style="color: #1DB954; font-weight: bold;">BY BLXCK 420</p>
</div>
""", unsafe_allow_html=True)
