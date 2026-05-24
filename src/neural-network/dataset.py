import os
import torch
import numpy as np
from torch.utils.data import Dataset
from config import PROCESSED_DIR, LABEL_MAP


class SignLanguageDataset(Dataset):

    # costruttore del Dataset: prepara la lista dei file, delle etichette e calcola il padding
    def __init__(self, data_dir):

        self.data_dir = data_dir  # il percorso alla cartella in cui abbiamo salvato i file .npy contenenti le coordinate delle mani di ogni video (nel nostro caso PROCESSED_DIR)
        self.filenames = (
            []
        )  # conterrà la lista dei file .npy contenenti le coordinate delle mani di ogni video
        self.labels = (
            []
        )  # conterrà la lista delle soluzioni. questo array è parallelo a filenames (es. se filenames[0] è "hello_12345.npy" allora labels[0] conterrà "0" che tradotto significa "hello")
        self.max_frames = 0  # conterrà il numero di frame del video più lungo. ci servira per fare il PADDING

        # per ogni file .npy nella cartella data_dire (nel nostro caso PROCESSED_DIR)
        for filename in os.listdir(data_dir):

            # se NON è un file .npy lo saltiamo
            if filename.endswith(".npy"):

                # estriamo la singola parole (poiche ogni filename è tipo "hello_12345.npy" possiamo splittare su '_')
                word = filename.split("_")[0]
                if word in LABEL_MAP:
                    self.filenames.append(
                        filename
                    )  # salviamo il filename (es. "hello_12345.npy") nella lista dei file
                    self.labels.append(
                        LABEL_MAP[word]
                    )  # salviamo la relativa soluzione (es. "0" per "hello")

        # calcoliamo il numero di frame del video più lungo (ci serve per fare il padding)
        self.calculate_max_frames(self.filenames, self.data_dir)

    # funzione per calcolare dinamicamente il numero di frame del video piu lungo (ci serve per il padding)
    def calculate_max_frames(self, filenames, data_dir):
        print("Calcolo la lunghezza massima dei video per il padding...")

        for filename in filenames:
            filepath = os.path.join(data_dir, filename)
            data = np.load(filepath)
            if data.shape[0] > self.max_frames:
                self.max_frames = data.shape[0]

        print(
            f"-> Il video più lungo dura {self.max_frames} frame. Uso questo valore per il padding di tutti i video!"
        )

    # getter per sapere sempre quanti video abbiamo in totale (es. 100 video di "hello", 80 video di "book", ecc.)
    def __len__(self):
        return len(self.filenames)

    # getter per ottenere un video specifico a partire dall'indice
    def __getitem__(self, idx):

        # prende il filename e costruisce il filepath
        filename = self.filenames[idx]
        filepath = os.path.join(self.data_dir, filename)

        # estrae l'etichetta corrispondente a questo video (es. 0 per "hello", 1 per "book", ecc.)
        label = self.labels[idx]
        # estrae i keypoints (1530 coordinate per frame) dal file .npy
        data = np.load(filepath)

        # --------------------------------------------------- PADDING ---------------------------------------------------
        seq_len = data.shape[0]  # estriamo il numero di frame di questo video
        padding = np.zeros(
            (self.max_frames - seq_len, data.shape[1])
        )  # crea matrici di zeri per i frame mancanti
        data = np.vstack(
            (data, padding)
        )  # aggiunge gli zeri alla fine dei dati originali
        # ---------------------------------------------------------------------------------------------------------------

        # converte gli array NumPy (formato Python base) in Tensori (formato PyTorch per la scheda video)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor


# --- TEST DEL CAMERIERE ---
if __name__ == "__main__":
    print("Testo il PyTorch Dataset...")

    # Creiamo un'istanza del nostro Dataset
    my_dataset = SignLanguageDataset(PROCESSED_DIR)

    print(f"\nVideo totali trovati dal Dataset: {len(my_dataset)}")

    # Chiediamo al Dataset di darci il primo video in assoluto (indice 0)
    primo_video, prima_etichetta = my_dataset[0]

    print("\nControllo Qualità sul primo video:")
    print(f"Formato del Tensore: {primo_video.shape}")
    print(f"Etichetta (Numero della parola): {prima_etichetta.item()}")

    # Controllo dinamico sul test
    if primo_video.shape[0] == my_dataset.max_frames and primo_video.shape[1] == 1530:
        print(
            f"\n-> GRANDIOSO! Il padding dinamico funziona. Il video è stato forzato a {my_dataset.max_frames} frame esatti."
        )
        print("Siamo ufficialmente pronti per scrivere la LSTM!")
