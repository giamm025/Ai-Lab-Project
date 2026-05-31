import json
from pathlib import Path

''' 
==========================================================================
1. GESTIONE DEI PERCORSI (Pathlib)
==========================================================================
'''

ROOT_DIR        = Path(__file__).resolve().parent.parent
RAW_DIR         = ROOT_DIR / 'data'     / 'raw'
PROCESSED_DIR   = ROOT_DIR / 'data'     / 'processed'
LABELS_FILEPATH = ROOT_DIR / 'data'     / 'labels.json'
DATASETS_DIR    = ROOT_DIR / 'datasets'
MODELS_DIR      = ROOT_DIR / 'models'
RESULTS_DIR     = ROOT_DIR / 'results'

for directory in [RAW_DIR, PROCESSED_DIR, DATASETS_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# le parole che utilizzeremo per costruire il nostro dataset, addestrare e testare il modello (prendendo solo i video relativi a queste parole)
TARGET_WORDS = ["happen", "finally", "late", "not-yet", "misunderstand", "understand"]
SEED = 42

""" 
==========================================================================
GESTIONE ETICHETTE (LABELS)
==========================================================================
"""

# crea il labels.json basandosi sugli indici di TARGET_WORDS
def create_label_map():
    print(f"Creazione del dizionario immutabile in {LABELS_FILEPATH}...")
    label_map = {word: idx for idx, word in enumerate(TARGET_WORDS)}
    with open(LABELS_FILEPATH, "w") as f:
        json.dump(label_map, f, indent=4)
    return label_map


# controlla: se labels.json lo carica, altrimenti lo crea (basandosi sugli indici di TARGET_WORDS)
def get_labels():

    # se il file NON esiste => lo crea (basandosi sugli indici di TARGET_WORDS)
    if not LABELS_FILEPATH.exists():
        return create_label_map()

    # altrimenti (gia esiste) => lo apre in sola lettura e lo carica in un dizionario
    with open(LABELS_FILEPATH, "r") as f:
        label_map = json.load(f)

        # check di consistenza: dobbiamo verificare che ci siano tutte le parole
        for word in TARGET_WORDS:
            if word not in label_map:
                print(f"ATTENZIONE: La parola '{word}' non è nel labels.json!")

        return label_map


LABEL_MAP = get_labels()
