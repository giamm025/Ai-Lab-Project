import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import torchmetrics
import argparse
import csv
from pathlib import Path

# Fix per i percorsi: diciamo a Python di cercare nella root del progetto
sys.path.append(str(Path(__file__).resolve().parent.parent))

# IMPORTIAMO LE NUOVE FUNZIONI CENTRALIZZATE DAL CONFIG!
from config import TARGET_WORDS, PROCESSED_DIR, SEED, GET_MODEL_PATH, GET_CSV_PATH, EXPERIMENT_SUFFIX
from dataset import SignLanguageDataset
from model import SignLanguageLSTM

# Inizializzo il parser degli argomenti
parser = argparse.ArgumentParser(
    description="Addestra il modello di Riconoscimento LIS."
)

# Aggiungo un parametro opzionale chiamato --modalita
parser.add_argument(
    "--modalita",
    type=str,
    choices=[
        "SOLO_MANI",
        "MANI_VOLTO",
    ],  # Limita le scelte per evitare errori di battitura
    default="MANI_VOLTO",  # Se non scrivi nulla nel terminale, usa questa di default
    help="Scegli se usare i 126 keypoints (SOLO_MANI) o tutti i 1530 keypoints (MANI_VOLTO)",
)

# Leggo cosa ha scritto l'utente nel terminale
args = parser.parse_args()
MODALITA = args.modalita

# Fissa il seed per la riproducibilità (A/B testing) importato dal config
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Stampo un messaggio chiaro per confermare la modalità
print(f"\n⚙️  CONFIGURAZIONE AVVIATA: Modalità {MODALITA} | Esperimento: {EXPERIMENT_SUFFIX}")

"""
1. PREPARARE IL DATASET
Carico i dati estratti con MediaPipe e li divido in Train, Val e Test.
"""
# Creo il dataset completo usando la cartella con i file .npy
full_dataset = SignLanguageDataset(PROCESSED_DIR)

# --- FIX DATA LEAKAGE ---
# Dividiamo il dataset esattamente come fa l'evaluate_model (70/15/15).
# L'addestramento userà il 70% (Train) e si metterà alla prova sul 15% (Val).
# Il restante 15% (Test) NON LO TOCCA MAI, sarà usato solo da evaluate_model.py!
total = len(full_dataset)
train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

training_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])

"""
2. CREARE I DATALOADER
"""
# Batch size piccola (es. 8) perché i tensori video pesano in memoria
batch_size = 8
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# Usiamo i dati di Validazione per testare il modello a ogni epoca!
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

"""
3. SELEZIONARE IL DISPOSITIVO
"""
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")

"""
3.5 CONFIGURAZIONE A/B TEST (Tramite Terminale)
"""

if MODALITA == "SOLO_MANI":
    input_size = 126
else:
    input_size = 1530
"""
4. DEFINIZIONE DEL MODELLO
"""
hidden_size = 64
num_classes = len(TARGET_WORDS)  # 5 parole target

# Istanzio la LSTM e la sposto sulla GPU
model = SignLanguageLSTM(
    input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
).to(device)
print(model)

"""
5. IPERPARAMETRI, LOSS FUNCTION E OPTIMIZER
"""
epochs = 20  # Le LSTM richiedono più epoche rispetto alle reti lineari semplici
learning_rate = 1e-3

loss_fn = torch.nn.CrossEntropyLoss()

"""
Per le LSTM, meglio usare Adam al posto di SGD (che abbiamo visto a lezione). Adam è un ottimizzatore 
più avanzato che combina i vantaggi di due altri ottimizzatori: AdaGrad e RMSProp.

Mentre SGD utilizza un unico learning rate fisso per aggiornare tutti i pesi della rete, 
Adam adatta dinamicamente il learning rate per ogni singolo parametro basandosi sui gradienti passati. 
Questo si rivela fondamentale per le LSTM che elaborano sequenze complesse come i frame di un video, 
in quanto aiuta a mitigare il problema della scomparsa del gradiente (vanishing gradient) e accelera 
notevolmente la convergenza verso la soluzione ottimale. 

Con SGD, specialmente su dati sparsi o tridimensionali come le coordinate spaziali di MediaPipe, 
c'è il forte rischio che l'addestramento risulti eccessivamente lento o si blocchi prima di 
raggiungere una buona accuratezza.
"""

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

"""
6. TRAINING LOOP
"""
def train_loop(train_dataloader, model, loss_fn, optimizer):
    model.train()  # Metto il modello in modalità addestramento
    total_loss = 0 # Accumulatore per il CSV

    # FIX: Ora estraiamo anche 'lengths' dal dataloader
    for batch, (x, y, lengths) in enumerate(train_dataloader):

        # --- TAGLIO DEL TENSORE PER L'A/B TEST ---
        if MODALITA == "SOLO_MANI":
            # x ha forma [batch, seq_len, 402]. Prendo solo tutte le righe e le prime 126 colonne
            x = x[:, :, :126]

        x = x.to(device)
        y = y.to(device)
        lengths = lengths.cpu() # pack_padded_sequence vuole le lunghezze sulla CPU

        # Forward
        pred = model(x, lengths)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        metric(pred, y)

        # Stampo i progressi (uso % 5 dato che ho meno batch rispetto a MNIST)
        if batch % 5 == 0:
            loss_v = loss.item()
            acc = metric(pred, y)
            print(f"Loss: {loss_v:.4f} | Batch: {batch + 1} | Accuratezza batch: {acc:.4f}")

    epoch_acc = metric.compute().item()
    epoch_loss = total_loss / len(train_dataloader)
    print(f"\n---> Accuratezza Finale Epoca (Train): {epoch_acc:.4f}\n")
    metric.reset()
    return epoch_loss, epoch_acc


"""
7. TEST LOOP (Usato come Validation Loop)
"""
def test_loop(test_dataloader, model, loss_fn):
    model.eval()  # Metto il modello in modalità valutazione
    total_loss = 0
    
    with torch.no_grad():  # Spegne i gradienti
        # FIX: Estraiamo anche 'lengths'
        for x, y, lengths in test_dataloader:

            # Applico lo stesso taglio anche in fase di test
            if MODALITA == "SOLO_MANI":
                x = x[:, :, :126]

            x = x.to(device)
            y = y.to(device)
            lengths = lengths.cpu()

            pred = model(x, lengths)
            total_loss += loss_fn(pred, y).item()
            metric(pred, y)

    epoch_acc = metric.compute().item()
    epoch_loss = total_loss / len(test_dataloader)
    print(f"*** ACCURATEZZA DI TEST FINALE: {epoch_acc:.4f} ***\n")
    metric.reset()
    return epoch_loss, epoch_acc


"""
8. ESECUZIONE E SALVATAGGIO
"""
print(f"Inizio addestramento in modalità: {MODALITA}")

# Recuperiamo i path corretti dal config.py
model_save_path = GET_MODEL_PATH(MODALITA)
csv_path = GET_CSV_PATH(MODALITA)

history = []
best_val_acc = 0.0 # Per i vecchi modelli salviamo basandoci sull'accuratezza

for epoch in range(epochs):
    print(f"=============================")
    print(f" EPOCH: {epoch + 1}/{epochs}")
    print(f"=============================")

    train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = test_loop(val_dataloader, model, loss_fn)
    
    # Salviamo le metriche per il CSV
    history.append([epoch + 1, train_loss, train_acc, val_loss, val_acc])
    
    # Salviamo il modello solo se migliora
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 Modello salvato! Nuova migliore Accuratezza: {best_val_acc:.4f}")

# Salvataggio del CSV
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc']) # NB: Modificato in Val_Loss/Val_Acc
    writer.writerows(history)

print("Addestramento completato! 🎉")