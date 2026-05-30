import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import torchmetrics
import argparse
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)

sys.path.append(src_dir)

from config import TARGET_WORDS, PROCESSED_DIR
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

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seme per la riproducibilità (A/B testing)",
)


# Leggo cosa ha scritto l'utente nel terminale
args = parser.parse_args()
MODALITA = args.modalita
SEED = args.seed

# Fissa il seed per la riproducibilità (A/B testing)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Stampo un messaggio chiaro per confermare la modalità
print(f"\n⚙️  CONFIGURAZIONE AVVIATA: Modalità {MODALITA}")

"""
1. PREPARARE IL DATASET
Carico i dati estratti con MediaPipe e li divido in Train e Test.
"""
# Creo il dataset completo usando la cartella con i file .npy
full_dataset = SignLanguageDataset(PROCESSED_DIR)

# Divido il dataset: 80% per l'addestramento, 20% per il test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
training_data, test_data = random_split(full_dataset, [train_size, test_size])

"""
2. CREARE I DATALOADER
"""
# Batch size piccola (es. 8) perché i tensori video pesano in memoria
batch_size = 8
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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

# -----------------------------------------------------------------------
# DIMENSIONI DEL VETTORE DI INPUT (per frame):
#
#   SOLO_MANI  →  126 valori
#                 └─ 21 land. mano sx × 3  =  63
#                 └─ 21 land. mano dx × 3  =  63
#
#   MANI_VOLTO →  402 valori
#                 └─ 21 land. mano sx × 3  =  63
#                 └─ 21 land. mano dx × 3  =  63
#                 └─ 92 land. volto × 3    = 276  (solo labbra + occhi + sopracciglia)
#                    ├─ labbra:               40 land.
#                    ├─ occhio sinistro:       16 land.
#                    ├─ occhio destro:         16 land.
#                    ├─ sopracciglio sx:       10 land.
#                    └─ sopracciglio dx:       10 land.
# -----------------------------------------------------------------------

if MODALITA == "SOLO_MANI":
    input_size = 126  # 63 + 63
else:
    input_size = 402  # 63 + 63 + 276

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
    model.train()   # Metto il modello in modalità addestramento
    total_loss = 0  # Variabile per accumulare la loss totale dell'epoca
    for batch, (x, y) in enumerate(train_dataloader):
        
        # --- TAGLIO DEL TENSORE PER L'A/B TEST ---
        # Il tensore x ha sempre forma [batch, seq_len, 402] (shape salvata da extract_features.py).
        # In SOLO_MANI prendiamo solo le prime 126 colonne (mano sx + mano dx).
        # In MANI_VOLTO usiamo tutte le 402 colonne: nessun taglio necessario.
        if MODALITA == "SOLO_MANI":
            x = x[:, :, :126]  # [batch, seq_len, 402] → [batch, seq_len, 126]

        x = x.to(device)
        y = y.to(device)

        # Forward
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accumulo la loss per il calcolo della media a fine epoca  
        total_loss += loss.item()
        metric(pred, y)

    epoch_acc = metric.compute().item()
    epoch_loss = total_loss / len(train_dataloader)
    
    print(f'---> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
    metric.reset()
    return epoch_loss, epoch_acc

"""
7. TEST LOOP
"""


def test_loop(test_dataloader, model):
    model.eval()  # Metto il modello in modalità valutazione
    with torch.no_grad():  # Spegne i gradienti
        for x, y in test_dataloader:

            # Applico lo stesso taglio anche in fase di test
            if MODALITA == "SOLO_MANI":
                x = x[:, :, :126]  # [batch, seq_len, 402] → [batch, seq_len, 126]

            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            metric(pred, y)

    epoch_acc = metric.compute().item()
    print(f'*** TEST ACCURACY: {epoch_acc:.4f} ***\n')
    metric.reset()
    return epoch_acc

"""
8. ESECUZIONE E SALVATAGGIO
"""
print(f"Inizio addestramento in modalità: {MODALITA}")

# Creiamo le cartelle DIRETTAMENTE NELLA ROOT
MODELS_DIR = os.path.join(root_dir, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
model_save_path = os.path.join(MODELS_DIR, f"best_model_{MODALITA}.pth")

RESULTS_DIR = os.path.join(root_dir, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

history = []
best_test_acc = 0.0
for epoch in range(epochs):
    print(f"=============================")
    print(f" EPOCH: {epoch + 1}/{epochs}")
    print(f"=============================")

    train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_acc = test_loop(test_dataloader, model)
    
    # Salviamo i dati per il grafico
    history.append([epoch + 1, train_loss, train_acc, test_acc])
    
    # Salviamo il modello SOLO se ha superato il suo record personale!
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 Nuovo record! Modello salvato con accuratezza: {best_test_acc:.4f}")

# Salvataggio del CSV
csv_path = os.path.join(RESULTS_DIR, f"training_history_{MODALITA}.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Test_Acc'])
    writer.writerows(history)

print("Addestramento completato! 🎉")
