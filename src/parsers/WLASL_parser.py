import json
import os
import sys 
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)                
sys.path.append(parent_dir)   

from config import TARGET_WORDS, RAW_DIR, DATASETS_DIR

# prendiamo i percorsi specifici per il dataset WALSL
WLASL_DIR = os.path.join(DATASETS_DIR, 'WLASL', 'raw')
WLASL_JSON_PATH = os.path.join(WLASL_DIR, 'WLASL_v0.3.json')
WLASL_VIDEOS_DIR = os.path.join(WLASL_DIR, 'videos')

# funzione per estrarre da WLASL solo i video delle TARGET_WORDS e salvarli in data/raw/ con il formato "parola_dataset_id.mp4"
def parse_wlasl():

    print(f"Lettura dei metadati WLASL da: {WLASL_JSON_PATH}")
    
    # crea la cartella data/raw se non esiste
    os.makedirs(RAW_DIR, exist_ok=True)

    with open(WLASL_JSON_PATH, 'r') as f:
        wlasl_data = json.load(f)
        
    video_copiati = 0
    video_mancanti = 0
    word_counts = {word: 0 for word in TARGET_WORDS}

    # per ogni entry nel dataset
    for entry in wlasl_data:

        # prendiamo la parola associata a quell'entry e vediamo se è una delle nostre parole target
        word = entry['gloss']
        
        # se la parola è una delle nostre TARGET_WORDS
        if word in TARGET_WORDS:

            # prendiamo i video (instances) associati a quella parola
            for instance in entry['instances']:

                # estriamo i campi necessari
                video_id = instance['video_id']
                original_filename = f"{video_id}.mp4"
                original_path = os.path.join(WLASL_VIDEOS_DIR, original_filename)
                
                # creiamo il nuovo filename (parola_dataset_id.mp4)
                new_filename = f"{word}_wlasl_{video_id}.mp4"
                new_path = os.path.join(RAW_DIR, new_filename)
                
                # se il file .mp4 NON esiste sul PC => segnaliamo che è mancante e passiamo al prossimo video
                if not os.path.exists(original_path):
                    video_mancanti += 1
                    print(f"❌ Manca: {original_filename}")
                    continue

                word_counts[word] += 1
                
                # se il file .npy GIA ESISTE => abbiamo gia processato questa parola => saltiamo
                if os.path.exists(new_path):
                    print(f"⏭️ Salto: {new_filename} poiché è già stato salvato in passato.")
                    continue

                # altrimenti => salviamo il video nella cartella data/raw/ con il nuovo nome
                shutil.copy(original_path, new_path)
                video_copiati += 1
                print(f"✅ Copiato: {new_filename}")
                                   
    print("\n--- RESOCONTO WLASL ---")
    for word, count in word_counts.items():
        print(f"Parola '{word}': {count} video")
    
    print(f"")
    print(f"Nuovi video inseriti nell'imbuto: {video_copiati}")
    print(f"Video non trovati nel dataset grezzo: {video_mancanti}")


if __name__ == "__main__":
    parse_wlasl()