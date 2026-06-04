import json
import os
import sys
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TARGET_SIGNS, UNKNOWN_WORDS_POOL, TARGET_WORDS, RAW_DIR, DATASETS_DIR

WLASL_DIR = DATASETS_DIR / "WLASL" / "raw"
WLASL_JSON_PATH = WLASL_DIR / "WLASL_v0.3.json"
WLASL_VIDEOS_DIR = WLASL_DIR / "videos"


def parse_wlasl():
    print(f"Lettura dei metadati WLASL da: {WLASL_JSON_PATH}")

    with WLASL_JSON_PATH.open("r") as f:
        wlasl_data = json.load(f)

    video_copiati = 0
    video_mancanti = 0
    word_counts = {word: 0 for word in TARGET_WORDS}

    for entry in wlasl_data:
        word = entry["gloss"]

        label = None
        is_garbage = False

        if word in TARGET_SIGNS:
            label = word
        elif word in UNKNOWN_WORDS_POOL:
            label = word
            is_garbage = True

        if label:
            for instance in entry["instances"]:
                video_id = instance["video_id"]
                original_filename = f"{video_id}.mp4"
                original_path = WLASL_VIDEOS_DIR / original_filename

                # Creazione differenziata del filename
                if is_garbage:
                    new_filename = f"{label}_garbage_wlasl_{video_id}.mp4"
                    count_key = "unknown"
                else:
                    new_filename = f"{label}_wlasl_{video_id}.mp4"
                    count_key = label

                new_path = RAW_DIR / new_filename

                if not original_path.exists():
                    video_mancanti += 1
                    print(f"❌ Manca: {original_filename} (per classe {label})")
                    continue

                word_counts[count_key] += 1

                if new_path.exists():
                    print(f"⏭️ Salto: {new_filename} poiché è già stato salvato in passato.")
                    continue

                shutil.copy(str(original_path), str(new_path))
                video_copiati += 1
                print(f"✅ Copiato: {new_filename} (Originale: {word})")

    print("\n--- RESOCONTO WLASL ---")
    for word, count in word_counts.items():
        print(f"Classe '{word}': {count} video")

    print(f"\nNuovi video inseriti nell'imbuto: {video_copiati}")
    print(f"Video non trovati nel dataset grezzo: {video_mancanti}")


if __name__ == "__main__":
    parse_wlasl()
