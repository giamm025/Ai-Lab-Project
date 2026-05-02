import os
import json

JSON_PATH = '../data/WLASL_v0.3.json'
VIDEO_DIR = '../data/raw/'
PROCESSED_DIR = '../data/processed/'
TARGET_WORDS = ['hello', 'book', 'computer', 'deaf', 'fine']
LABELS_FILEPATH = '../data/labels.json'

# controlla: se labels.json lo carica, altrimenti lo crea (basandosi sugli indici di TARGET_WORDS)
def get_labels():

    # se il file NON esiste => lo crea (basandosi sugli indici di TARGET_WORDS)
    if not os.path.exists(LABELS_FILEPATH):
        return create_label_map()
    
    # altrimenti (gia esiste) => lo apre in sola lettura e lo carica in un dizionario
    with open(LABELS_FILEPATH, 'r') as f:
        label_map = json.load(f)

        # check di consistenza: dobbiamo verificare che ci siano tutte le parole
        for word in TARGET_WORDS:
            if word not in label_map:
                print(f"ATTENZIONE: La parola '{word}' non è nel labels.json!")
        
        return label_map
    

def create_label_map():
    print(f"Creazione del dizionario immutabile in {LABELS_FILEPATH}...")
    label_map = {word: idx for idx, word in enumerate(TARGET_WORDS)}
    with open(LABELS_FILEPATH, 'w') as f:
        json.dump(label_map, f, indent=4)
    return label_map

LABEL_MAP = get_labels()