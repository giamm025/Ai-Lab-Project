import os
import json

""" 
==========================================================================
CONFIGURAZIONE GENERALE
==========================================================================
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(
    BASE_DIR, "..", "data", "raw"
)  # data/ è la cartella dove salveremo il NOSTRO dataset, ovvero quello formato dai video presi dai vari dataset
PROCESSED_DIR = os.path.join(
    BASE_DIR, "..", "data", "processed"
)  # la cartella in cui salveremo i file .npy contenenti le coordinate di mani e volto estratti con MediaPipe
LABELS_FILEPATH = os.path.join(
    BASE_DIR, "..", "data", "labels.json"
)  # file immutabile per il mapping Parola -> Numero
DATASETS_DIR = os.path.join(
    BASE_DIR, "..", "datasets"
)  # cartella dove lo script download_datasets.py salva i dataset originali


# le parole che utilizzeremo per costruire il nostro dataset, addestrare e testare il modello (prendendo solo i video relativi a queste parole)
TARGET_WORDS = ["happen", "finally", "late", "not-yet", "misunderstand", "understand"]

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
    if not os.path.exists(LABELS_FILEPATH):
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
