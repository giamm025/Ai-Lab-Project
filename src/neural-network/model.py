import torch
import torch.nn as nn

"""
Come abbiamo visto a lezione dovremo implementare due metodi principali:

    - __init__(self): definiamo l'hardware, cioe la struttura del modello, ovvero quali tipi di layer 
                      utilizzeremo e come sono connessi tra loro.

    - forward(self, x): definiamo il software, cioe il flusso dei dati attraverso il modello, come i dati 
                        vengono trasformati dai layer per produrre l'output.
"""


class SignLanguageLSTM(nn.Module):

    # parametri:
    # - input_size:  quanti numeri entrano per ogni frame?
    #                (126 se lavoriamo "Solo Mani", 1530 per "Mani+Volto")
    # - hidden_size: quanto è grande la memoria interna (hidden emmory) della LSTM?
    #                (Es. 64 o 128 neuroni. Piu è grande, più la rete può ricordare, ma più è difficile da addestrare)
    # - num_classes: il numero di canali di uscita... cioe: quali sono le soluzioni possibili?
    #                (nel nostro caso è il numero di parole che vogliamo riconoscere, cioè 5)
    def __init__(self, input_size, hidden_size, num_classes):

        super().__init__()

        # "legge” un intero video, un frame alla volta, e salva nella sua memoria interna (hidden state) un’array di numero che,
        # matematicamente, rappresentano il movimento osservato da inizio video fino a quel frame.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )  # serve a specificare l'ordine dei numeri nel tensore di input.
        # se batch_first=True,  la LSTM si aspetta (batch_size, seq_len, input_size).
        # se batch_first=False, la LSTM si aspetta (seq_len, batch_size, input_size).
        # la LSTM restituirà l'intera lista di ciò che ha pensato ad ogni frame de video

        # creiamo lo stato lineare finale, che prende in input l'ultimo frame prodotto dalla LSTM e restituitsce
        # l'array con le probabilità di ciascuna parola (es. [0.1, 0.7, 0.05, 0.1, 0.05] => 10% "hello", 70% "book", ecc.)
        self.fc = nn.Linear(hidden_size, num_classes)

    # questa è la funzione che viene chiamata quando passiamo i dati alla rete. Definisce il PERCORSO che fanno i dati (da x fino alla predizione).
    def forward(self, x):

        # passiamo i dati alla LSTM.
        frames, (hn, cn) = self.lstm(x)
        # alla fine avremo:
        # frames: contiene ciò che la LSTM ha pensato ad OGNI frame del video                 (forma: [batch_size, seq_len, hidden_size])
        # hn (hidden state): contiene ciò che la LSTM ha pensato all'ULTIMO frame del video   (forma: [num_layers, batch_size, hidden_size])
        # cn (cell state): contiene la "memoria a lungo termine" della LSTM. Viene utilizzata
        #  solo internamente dalla LSTM per decidere cosa scriversi nll'hn
        #  Al mondo esterno passiamo SEMPRE E SOLO gli appunti finali hn      (forma: [num_layers, batch_size, hidden_size])

        # a noi interessa solo la decisione finale, presa dopo aver analizzato TUTTI i frame (cioè a fine video)
        # possiamo quindi "tagliare" frames per prendere solo l'ultimo frame oppure usare direttamente la variabile hn
        out = frames[:, -1, :]

        # passiamo il verdetto finale (dopo aver letto tutti i frame) al livello Lineare per ottenere la predizione
        result = self.fc(out)
        return result


# --- TEST DELL'ARCHITETTURA ---
if __name__ == "__main__":
    from config import TARGET_WORDS

    print("Testo l'architettura della Rete Neurale...\n")

    # Parametri fittizi per il test
    batch_size = 8  # immaginiamo che il DataLoader ci stia dando 8 video alla volta
    seq_len = 108  # la lunghezza massima calcolato prima dentro dataset.py (per oora lo hardcodiamo è solo una prova)
    num_classes = (
        5  # perche ora usiamo solo 5 parole (hello, book, computer, deaf, fine)
    )
    hidden_size = 64  # la grandeza della memoria della LSTM. Per ora mettiamo 64 neuroni di memoria interna (Piu è grande, più la rete può ricordare, ma più è difficile da addestrare)

    # --------------------- SOLO MANI ---------------------

    # creiamo un tensore fittizio (pieno di numeri casuali) tanto per vedere se la rete riesce a processarlo senza errori
    input_size_mani = 126  # (63 mano sx + 63 mano dx)
    dati_finti_mani = torch.randn(batch_size, seq_len, input_size_mani)
    modello_mani = SignLanguageLSTM(
        input_size=input_size_mani, hidden_size=hidden_size, num_classes=num_classes
    )
    predizioni_mani = modello_mani(dati_finti_mani)

    print("--------------------- SOLO MANI ---------------------")
    print(f"Formato Input: {dati_finti_mani.shape}")
    print(
        f"Formato Output: {predizioni_mani.shape} -> (Deve essere: [{batch_size} video, {num_classes} parole])"
    )

    # --------------------- MANI + VOLTO ---------------------
    input_size_volto = (
        1530  # Tutto l'array completo (63 mano sx + 63 mano dx + 1404 volto)
    )
    dati_finti_volto = torch.randn(batch_size, seq_len, input_size_volto)
    modello_volto = SignLanguageLSTM(
        input_size=input_size_volto, hidden_size=hidden_size, num_classes=num_classes
    )
    predizioni_volto = modello_volto(dati_finti_volto)

    print("\n--------------------- MANI + VOLTO ---------------------")
    print(f"Formato Input: {dati_finti_volto.shape}")
    print(
        f"Formato Output: {predizioni_volto.shape} -> (Deve essere: [{batch_size} video, {num_classes} parole])"
    )

    # --------------------- RESOCONTO FINALE ---------------------
    if predizioni_mani.shape == (8, 5) and predizioni_volto.shape == (8, 5):
        print(
            "\n-> MATEMATICAMENTE PERFETTO! La rete riceve i dati e sputa 5 probabilità (una per parola)."
        )
