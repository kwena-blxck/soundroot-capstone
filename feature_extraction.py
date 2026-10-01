"""SoundRoot audio feature extraction.

This module mirrors the feature extraction used in the SoundRoot training notebook.
It extracts the same 85 numerical features from up to 30 seconds of mono audio at
22,050 Hz.
"""

import librosa
import numpy as np
SAMPLE_RATE = 22050

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=30)
        features = {}

        # Tempo with correction for half-tempo detection
        # Handle both old and new librosa return formats
        beat_result = librosa.beat.beat_track(y=y, sr=sr)

        # New librosa returns (tempo, beat_frames) as a tuple
        if isinstance(beat_result, tuple):
            tempo = beat_result[0]
            beat_frames = beat_result[1]
        else:
            # Old librosa returns tempo as float, beat_frames separately
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

        # --- GUITAR vs ELECTRONIC DISCRIMINATORS ---

        # 1. SPECTRAL ROLLOFF (guitar has lower rolloff than synths)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))
        features["spectral_rolloff_std"] = float(np.std(rolloff))

        # 2. SPECTRAL FLATNESS (electronic = more noise-like = higher flatness)
        flatness = librosa.feature.spectral_flatness(y=y)
        features["spectral_flatness_mean"] = float(np.mean(flatness))
        features["spectral_flatness_std"] = float(np.std(flatness))

        # 3. ZERO CROSSING RATE (percussion = higher ZCR)
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        # 4. HARMONIC vs PERCUSSIVE separation (crucial for guitar vs electronic)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features["harmonic_energy_ratio"] = float(np.sum(y_harmonic**2) / (np.sum(y**2) + 1e-8))
        features["percussive_energy_ratio"] = float(np.sum(y_percussive**2) / (np.sum(y**2) + 1e-8))

        # 5. CHROMA features (guitar vs piano chords)
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
        features["chroma_mean"] = float(np.mean(chroma))
        features["chroma_std"] = float(np.std(chroma))
        # Guitar often uses fewer distinct chords - measure entropy
        chroma_sum = np.sum(chroma, axis=0, keepdims=True)
        chroma_norm = chroma / (chroma_sum + 1e-8)
        features["chroma_entropy"] = float(-np.sum(chroma_norm * np.log2(chroma_norm + 1e-8)))

        # 6. TONNETZ (harmonic relationships - guitar vs electronic harmony)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        features["tonnetz_mean"] = float(np.mean(tonnetz))
        features["tonnetz_std"] = float(np.std(tonnetz))

        # 7. MFCCs (timbre - guitar vs synth texture)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f"mfcc_{i+1}"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))

        # 8. SPECTRAL CONTRAST (guitar has different peak/valley structure)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features["spectral_contrast_mean"] = float(np.mean(contrast))
        features["spectral_contrast_std"] = float(np.std(contrast))

        # 9. RMS ENERGY (electronic has more consistent energy)
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        # 10. SPECTRAL CENTROID (brightness - synths are brighter)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = float(np.mean(centroid))
        features["spectral_centroid_std"] = float(np.std(centroid))

        # 11. SPECTRAL BANDWIDTH
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(bandwidth))

        # 12. TEMPOGRAM (rhythmic structure)
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        features["tempogram_mean"] = float(np.mean(tempogram))
        features["tempogram_std"] = float(np.std(tempogram))

        # 13. ONSET STRENGTH (percussive attacks)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features["onset_strength_mean"] = float(np.mean(onset_env))
        features["onset_strength_std"] = float(np.std(onset_env))

        # 14. BEAT INTERVALS (rhythmic regularity - amapiano is more regular)
        features["num_beats"] = int(len(beat_frames))

        # Safe diff calculation
        if len(beat_frames) > 1:
            diffs = np.diff(beat_frames)
            features["beat_interval_mean"] = float(np.mean(diffs))
            features["beat_interval_std"] = float(np.std(diffs))
        else:
            features["beat_interval_mean"] = 0.0
            features["beat_interval_std"] = 0.0

        # 15. CHROMA CQT (constant-Q transform - better for guitar)
        chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        features["chroma_cqt_mean"] = float(np.mean(chroma_cqt))
        features["chroma_cqt_std"] = float(np.std(chroma_cqt))

        # 16. TEMPORAL FEATURES - how energy changes over time
        # Amapiano has more "pumping" (sidechain compression)
        rms_flat = rms.flatten()
        if len(rms_flat) > 1:
            rms_diff = np.diff(rms_flat)
            features["rms_fluctuation"] = float(np.std(rms_diff))
        else:
            features["rms_fluctuation"] = 0.0

        # 17. Spectral Rolloff at 50% (median frequency)
        rolloff_50 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.5)
        features["spectral_rolloff_50_mean"] = float(np.mean(rolloff_50))
        features["spectral_rolloff_50_std"] = float(np.std(rolloff_50))

        # 18. Crest Factor (peak vs RMS - electronic has higher crest)
        peak = float(np.max(np.abs(y)))
        rms_val = float(np.sqrt(np.mean(y**2)))
        features["crest_factor"] = peak / (rms_val + 1e-8)

        # 19. Attack Time (how fast sound reaches peak - percussion vs guitar pluck)
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

        # 20. Decay Time (how fast sound fades - guitar sustains longer)
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

        # 21. Inharmonicity (how much partials deviate from harmonic series)
        fft = np.fft.rfft(y_harmonic)
        freqs = np.fft.rfftfreq(len(y_harmonic), 1/sr)
        magnitudes = np.abs(fft)

        # Find peaks in spectrum
        peak_indices = np.argsort(magnitudes)[-20:]  # Top 20 peaks
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

        # 22. Transient Ratio (sharp attacks vs sustained tones)
        transient_env = np.abs(y_percussive)
        sustained_env = np.abs(y_harmonic)
        features["transient_ratio"] = float(np.mean(transient_env)) / (float(np.mean(sustained_env)) + 1e-8)

        # 23. Spectral Slope (brightness trend - electronic has flatter slope)
        freqs_spec = np.linspace(0, sr/2, len(fft))
        spec = np.abs(fft)
        if len(freqs_spec) > 1 and len(spec) > 1:
            log_freqs = np.log(freqs_spec[1:] + 1e-8)
            log_spec = np.log(spec[1:] + 1e-8)
            # Filter out invalid values
            valid_mask = np.isfinite(log_freqs) & np.isfinite(log_spec)
            if np.sum(valid_mask) > 1:
                slope = np.polyfit(log_freqs[valid_mask], log_spec[valid_mask], 1)[0]
                features["spectral_slope"] = float(slope)
            else:
                features["spectral_slope"] = 0.0
        else:
            features["spectral_slope"] = 0.0

        # 24. Tonal vs Noise-like (guitar is tonal, amapiano has noise)
        spec_flat = librosa.feature.spectral_flatness(y=y)
        features["spectral_flatness_band_mean"] = float(np.mean(spec_flat))
        features["spectral_flatness_band_std"] = float(np.std(spec_flat))

        # 25. Rhythm Regularity (Amapiano = very regular, Maskandi = variable)
        if len(beat_frames) > 1:
            diffs = np.diff(beat_frames)
            if features["beat_interval_mean"] > 0:
                features["rhythm_regularity"] = features["beat_interval_std"] / (features["beat_interval_mean"] + 1e-8)
            else:
                features["rhythm_regularity"] = 0.0
        else:
            features["rhythm_regularity"] = 0.0

        # 26. Pulse Clarity (how strong the beat is)
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

    except Exception as e:
        print(f"Error in extract_features: {e}")
        import traceback
        traceback.print_exc()
        raise


