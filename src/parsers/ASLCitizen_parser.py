import os
import csv
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TARGET_SIGNS, UNKNOWN_WORDS_POOL, TARGET_WORDS, RAW_DIR, DATASETS_DIR

# prendiamo i percorsi specifici per il dataset ASL-Citizen
ASLCITIZEN_DIR = DATASETS_DIR / "ASL_Citizen" / "raw"
CSV_FILES = ["train.csv", "val.csv", "test.csv"]
ASLCITIZEN_SPLIT_DIR = ASLCITIZEN_DIR / "ASL_Citizen" / "splits"
ASLCITIZEN_VIDEOS_DIR = ASLCITIZEN_DIR / "ASL_Citizen" / "videos"


def parse_aslcitizen():
    print("Avvio parsing ASL-Citizen...")

    target_mapping = {}

    # 1. Mappiamo SOLO i veri segni target
    for w in TARGET_SIGNS:
        clean_w = w.replace("-", "").upper()
        target_mapping[clean_w] = {"label": w, "is_garbage": False}

    # 2. Mappiamo le parole della pool sconosciuta mantenendo il nome originale
    for w in UNKNOWN_WORDS_POOL:
        clean_w = w.replace("-", "").upper()
        if clean_w not in target_mapping:
            target_mapping[clean_w] = {"label": w, "is_garbage": True}

    video_copiati = 0
    video_mancanti = 0
    word_counts = {word: 0 for word in TARGET_WORDS}

    for csv_filename in CSV_FILES:
        csv_path = ASLCITIZEN_SPLIT_DIR / csv_filename
        if not csv_path.exists():
            print(f"File {csv_filename} non trovato in {ASLCITIZEN_SPLIT_DIR}. Lo salto.")
            continue

        print(f"\nLeggendo metadati da: {csv_filename}...")
        with csv_path.open(mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gloss = row["Gloss"]
                base_gloss = gloss.rstrip("0123456789")

                if base_gloss in target_mapping:

                    info = target_mapping[base_gloss]
                    label = info["label"]
                    is_garbage = info["is_garbage"]

                    participant_id = row["Participant ID"]
                    video_file = row["Video file"]
                    source_path = ASLCITIZEN_VIDEOS_DIR / video_file

                    # Creazione differenziata del filename
                    if is_garbage:
                        new_filename = f"{label}_garbage_aslcitizen_{participant_id}_{video_file}"
                        count_key = "unknown"
                    else:
                        new_filename = f"{label}_aslcitizen_{participant_id}_{video_file}"
                        count_key = label

                    destination_path = RAW_DIR / new_filename

                    if destination_path.exists():
                        word_counts[count_key] += 1
                        continue

                    if source_path.exists():
                        shutil.copy(str(source_path), str(destination_path))
                        video_copiati += 1
                        word_counts[count_key] += 1
                        print(f"    ✅ Copiato: {new_filename} (Gloss: {gloss})")
                    else:
                        video_mancanti += 1
                        print(f"    ❌ Non trovato: {video_file} (Gloss: {gloss})")

    print("\n--- RESOCONTO ASL-CITIZEN ---")
    for word, count in word_counts.items():
        print(f"Classe '{word}': {count} video validi")

    print(f"\nNuovi video copiati nell'imbuto: {video_copiati}")
    print(f"Video non trovati nella cartella originale: {video_mancanti}")


if __name__ == "__main__":
    parse_aslcitizen()
