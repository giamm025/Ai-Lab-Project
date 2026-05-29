import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import torchmetrics
import argparse

# Importo le classi e le costanti
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import PROCESSED_DIR, TARGET_WORDS
from dataset import SignLanguageDataset
from model import SignLanguageLSTM

"""
0. ARGOMENTI DA TERMINALE
"""
parser = argparse.ArgumentParser(
    description="Addestra il modello di Riconoscimento LIS."
)

arg_configs = [
    {
        "name": "--modalita",
        "type": str,
        "choices": ["SOLO_MANI", "MANI_VOLTO"],
        "default": "MANI_VOLTO",
        "help": "Scegli se usare i 126 keypoints (SOLO_MANI) o i 402 keypoints (MANI_VOLTO)",
    },
    {
        "name": "--seed",
        "type": int,
        "default": 42,
        "help": "Seme per la riproducibilità (default: 42)",
    },
    {
        "name": "--epochs",
        "type": int,
        "default": 200,
        "help": "Numero di epoche per l'addestramento (default: 200)",
    },
    {
        "name": "--batch_size",
        "type": int,
        "default": 8,
        "help": "Dimensione del batch per l'addestramento (default: 8)",
    },
    {
        "name": "--patience",
        "type": int,
        "default": 10,
        "help": "Numero di epoche senza miglioramento prima dell'early stopping (default: 10)",
    },
    {
        "name": "--learning_rate",
        "type": float,
        "default": 1e-3,
        "help": "Learning rate per l'ottimizzatore (default: 1e-3)",
    },
    {
        "name": "--hidden_size",
        "type": int,
        "default": 64,
        "help": "Dimensione dell'hidden state della LSTM (default: 64)",
    },
    {
        "name": "--num_layers",
        "type": int,
        "default": 1,
        "help": "Numero di layer della LSTM (default: 1)",
    },
]

for arg in arg_configs:
    name = arg.pop("name")
    parser.add_argument(name, **arg)

args = parser.parse_args()
MODALITA = args.modalita
SEED = args.seed
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
PATIENCE = args.patience
LEARNING_RATE = args.learning_rate
HIDDEN_SIZE = args.hidden_size
NUM_LAYERS = args.num_layers

# Fissa il seed per garantire che ogni run con lo stesso seed produca gli stessi risultati.
# Fondamentale per l'A/B testing: senza seed fisso non si può sapere se un miglioramento
# è dovuto alle modifiche al modello o alla fortuna dell'inizializzazione casuale.
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"\n⚙️  CONFIGURAZIONE AVVIATA: Modalità {MODALITA} | Seed {SEED}")

"""
1. PREPARARE IL DATASET
Carico i dati estratti con MediaPipe e li divido in Train e Test.
"""
full_dataset = SignLanguageDataset(PROCESSED_DIR)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
training_data, test_data = random_split(full_dataset, [train_size, test_size])

"""
2. CREARE I DATALOADER
"""
# Batch size piccola (es. 8) perché i tensori video pesano in memoria
batch_size = BATCH_SIZE
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
3.5 CONFIGURAZIONE A/B TEST
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
hidden_size = HIDDEN_SIZE
num_layers = NUM_LAYERS
num_classes = len(TARGET_WORDS)

model = SignLanguageLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_classes=num_classes,
    num_layers=num_layers,
).to(device)
print(model)

"""
5. IPERPARAMETRI, LOSS FUNCTION E OPTIMIZER
"""
epochs = EPOCHS
learning_rate = LEARNING_RATE

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
    model.train()

    for batch, (x, y, lengths) in enumerate(train_dataloader):

        # --- TAGLIO DEL TENSORE PER L'A/B TEST ---
        # Il tensore x ha sempre forma [batch, seq_len, 402] (shape salvata da extract_features.py).
        # In SOLO_MANI prendiamo solo le prime 126 colonne (mano sx + mano dx).
        # In MANI_VOLTO usiamo tutte le 402 colonne: nessun taglio necessario.
        if MODALITA == "SOLO_MANI":
            x = x[:, :, :126]  # [batch, seq_len, 402] → [batch, seq_len, 126]

        x = x.to(device)
        y = y.to(device)
        # lengths resta sulla CPU: pack_padded_sequence lo richiede esplicitamente

        # Forward
        pred = model(x, lengths)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss_v = loss.item()
            acc = metric(pred, y)
            print(
                f"Loss: {loss_v:.4f} | Batch: {batch + 1} | Accuratezza batch: {acc:.4f}"
            )

    acc = metric.compute()
    print(f"\n---> Accuratezza Finale Epoca (Train): {acc:.4f}\n")
    metric.reset()


"""
7. TEST LOOP
"""


def test_loop(test_dataloader, model, loss_fn):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y, lengths in test_dataloader:

            # stesso taglio applicato anche in fase di test
            if MODALITA == "SOLO_MANI":
                x = x[:, :, :126]  # [batch, seq_len, 402] → [batch, seq_len, 126]

            x = x.to(device)
            y = y.to(device)
            # lengths resta sulla CPU

            pred = model(x, lengths)

            # Accumulo la loss per calcolare la media alla fine
            total_loss += loss_fn(pred, y).item()
            num_batches += 1

            metric(pred, y)

    avg_loss = total_loss / num_batches
    acc = metric.compute()
    print(f"*** TEST → Loss: {avg_loss:.4f} | Accuratezza: {acc:.4f} ***\n")
    metric.reset()

    return avg_loss, acc.item()


"""
8. ESECUZIONE
"""
print(f"Inizio addestramento in modalità: {MODALITA}")

# --- CONFIGURAZIONE EARLY STOPPING ---
# Se la test loss non migliora per `patience` epoche consecutive, l'addestramento
# si ferma automaticamente per evitare overfitting e sprecare tempo di calcolo.
patience = PATIENCE
epochs_no_improve = 0
best_test_loss = float("inf")
model_save_path = os.path.join(current_dir, f"best_model_{MODALITA}.pth")
# -------------------------------------

for epoch in range(epochs):
    print(f"=============================")
    print(f" EPOCH: {epoch + 1}/{epochs}")
    print(f"=============================")

    train_loop(train_dataloader, model, loss_fn, optimizer)
    current_test_loss, current_test_acc = test_loop(test_dataloader, model, loss_fn)

    # --- LOGICA EARLY STOPPING ---
    if current_test_loss < best_test_loss:
        # La loss è migliorata: salviamo il modello e azzeriamo il contatore
        best_test_loss = current_test_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(
            f"💾 Modello salvato! Nuova migliore Loss: {best_test_loss:.4f} (Acc: {current_test_acc:.4f})\n"
        )
    else:
        # La loss NON è migliorata: incrementiamo il contatore
        epochs_no_improve += 1
        print(
            f"⚠️  Nessun miglioramento per {epochs_no_improve}/{patience} epoche consecutive.\n"
        )

        if epochs_no_improve >= patience:
            print("🛑 EARLY STOPPING ATTIVATO: il modello ha smesso di imparare.")
            print(f"   Modello migliore salvato con Loss: {best_test_loss:.4f}")
            break

print("Addestramento completato! 🎉")
