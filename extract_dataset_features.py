"""Extract the SoundRoot feature table from 30-second WAV clips."""
from pathlib import Path
import pandas as pd
from feature_extraction import extract_features

CLIPS_DIR = Path("data/clips_30sec")
GENRES = ["amapiano", "maskandi", "lekompo", "gospel", "afro_house"]

def main():
    rows = []
    for genre in GENRES:
        genre_dir = CLIPS_DIR / genre
        clips = list(genre_dir.glob("*.wav"))
        print(f"Processing {genre}: {len(clips)} clips")
        for clip_path in clips:
            try:
                features, _, _ = extract_features(clip_path)
                features["genre"] = genre
                features["filename"] = clip_path.name
                rows.append(features)
            except Exception as exc:
                print(f"Error: {clip_path.name}: {exc}")
    df = pd.DataFrame(rows)
    output = Path("data/features.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Features extracted: {len(df)} samples, {len(df.columns)-2} features")
    print(df["genre"].value_counts())

if __name__ == "__main__":
    main()
