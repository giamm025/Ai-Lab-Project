import json
import os
import sys
import yt_dlp
from pathlib import Path
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TARGET_SIGNS, UNKNOWN_WORDS_POOL, TARGET_WORDS, RAW_DIR, DATASETS_DIR

MSASL_DIR = DATASETS_DIR / "MS_ASL" / "raw"
JSON_FILES = ["MSASL_train.json", "MSASL_val.json", "MSASL_test.json"]
TEMP_DIR = DATASETS_DIR / "MS_ASL" / "temp"

TEMP_DIR.mkdir(parents=True, exist_ok=True)


def parse_msasl():
    video_creati = 0
    video_persi = 0
    word_counts = {word: 0 for word in TARGET_WORDS}

    for json_filename in JSON_FILES:
        json_path = MSASL_DIR / json_filename
        if not json_path.exists():
            print(f"File {json_filename} non trovato, lo salto.")
            continue

        print(f"\nLettura dei metadati MSASL da: {json_filename}...")
        with json_path.open("r") as f:
            json_data = json.load(f)

        for entry in json_data:
            word = entry.get("clean_text", "")

            label = None
            is_garbage = False

            if word in TARGET_SIGNS:
                label = word
            elif word in UNKNOWN_WORDS_POOL:
                label = word
                is_garbage = True

            if label:
                url = entry.get("url")
                if not url.startswith("http"):
                    url = "https://" + url

                start_time = entry.get("start_time")
                end_time = entry.get("end_time")
                signer_id = entry.get("signer_id")

                # Creazione differenziata del filename
                if is_garbage:
                    new_filename = f"{label}_garbage_msasl_signer{signer_id}_{start_time}.mp4"
                    count_key = "unknown"
                else:
                    new_filename = f"{label}_msasl_signer{signer_id}_{start_time}.mp4"
                    count_key = label

                final_path = RAW_DIR / new_filename
                temp_video_path = TEMP_DIR / f"temp_{signer_id}_{start_time}.mp4"

                if final_path.exists():
                    print(f"⏭️ Salto: {new_filename} esiste già.")
                    word_counts[count_key] += 1
                    continue

                print(f"⏳ Processando [{word}] (da {start_time}s a {end_time}s)...")

                success = download_youtube_video(url, str(temp_video_path))

                if success:
                    try:
                        ffmpeg_extract_subclip(str(temp_video_path), start_time, end_time, targetname=str(final_path))
                        video_creati += 1
                        word_counts[count_key] += 1
                        print(f"    ✅ Salvato: {new_filename}")

                    except Exception as e:
                        print(f"    ❌ Errore nel taglio del video: {e}")

                    finally:
                        if temp_video_path.exists():
                            os.remove(str(temp_video_path))
                else:
                    video_persi += 1

    print("\n--- RESOCONTO MS-ASL ---")
    for word, count in word_counts.items():
        print(f"Classe '{word}': {count} video validi")

    print(f"\nNuovi video ritagliati e inseriti nell'imbuto: {video_creati}")
    print(f"Video non recuperabili (Link morti): {video_persi}")


def download_youtube_video(url, output_path):
    ydl_opts = {"format": "bestvideo[height<=480][ext=mp4]", "outtmpl": output_path, "quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"    ❌ Impossibile scaricare {url} (Forse rimosso da YouTube?)")
        return False


if __name__ == "__main__":
    parse_msasl()
