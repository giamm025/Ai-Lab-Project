import sys
import torch
from torch.utils.data import DataLoader, random_split
import torchmetrics
import argparse
import csv
from pathlib import Path

# Forza stdout in UTF-8 nel caso in cui l'output venga reindirizzato su un file
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TARGET_WORDS, PROCESSED_DIR, SEED, GET_MODEL_PATH, GET_EXPERIMENT_DIR, GET_CSV_PATH
from dataset import SignLanguageDataset
from model import SignLanguageLSTM

"""
0. ARGOMENTI DA TERMINALE
"""
parser = argparse.ArgumentParser(description="Addestra il modello di Riconoscimento LIS.")

arg_configs = [
    {"name": "--modalita", "type": str, "choices": ["SOLO_MANI", "MANI_VOLTO"], "default": "MANI_VOLTO", "help": "Scegli se usare i 126 o i 402 keypoints"},
    {"name": "--seed", "type": int, "default": 42, "help": "Seme per la riproducibilità (default: 42)"},
    {"name": "--epochs", "type": int, "default": 200, "help": "Numero di epoche per l'addestramento (default: 200)"},
    {"name": "--batch_size", "type": int, "default": 8, "help": "Dimensione del batch per l'addestramento (default: 8)"},
    {"name": "--patience", "type": int, "default": 10, "help": "Epoche senza miglioramento prima dell'early stopping (default: 10)"},
    {"name": "--learning_rate", "type": float, "default": 1e-3, "help": "Learning rate (default: 1e-3)"},
    {"name": "--hidden_size", "type": int, "default": 64, "help": "Dimensione hidden state LSTM (default: 64)"},
    {"name": "--num_layers", "type": int, "default": 1, "help": "Numero di layer della LSTM (default: 1)"},
    {"name": "--dropout", "type": float, "default": 0.0, "help": "Probabilità di dropout tra LSTM e layer finale (default: 0.0 = disabilitato)"},
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
DROPOUT = args.dropout
if args.seed != SEED:
    SEED = args.seed

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"\n⚙️  CONFIGURAZIONE AVVIATA: Modalità {MODALITA} | Seed {SEED}")

"""
1. PREPARARE IL DATASET
"""
full_dataset = SignLanguageDataset(PROCESSED_DIR)

total = len(full_dataset)
train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

training_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])
print(f"Split dataset: {train_size} train | {val_size} val | {test_size} test")


"""
2. CREARE I DATALOADER
"""
batch_size = BATCH_SIZE
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

"""
3. SELEZIONARE IL DISPOSITIVO
"""
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

"""
3.5 CONFIGURAZIONE A/B TEST
"""
input_size = 126 if MODALITA == "SOLO_MANI" else 402

"""
4. DEFINIZIONE DEL MODELLO
"""
hidden_size = HIDDEN_SIZE
num_layers = NUM_LAYERS
num_classes = len(TARGET_WORDS)

model = SignLanguageLSTM(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_classes=num_classes,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

"""
5. IPERPARAMETRI E OPTIMIZER
"""
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

"""
6. TRAINING LOOP (Fuso: Ritorna i valori per il CSV + usa lengths del compagno)
"""


def train_loop(train_dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0

    for batch, (x, y, lengths) in enumerate(train_dataloader):
        if MODALITA == "SOLO_MANI":
            x = x[:, :, :126]

        x = x.to(device)
        y = y.to(device)
        # lengths resta sulla CPU

        pred = model(x, lengths)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        metric(pred, y)

    epoch_acc = metric.compute().item()
    epoch_loss = total_loss / len(train_dataloader)

    print(f"---> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
    metric.reset()
    return epoch_loss, epoch_acc


def val_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y, lengths in dataloader:
            if MODALITA == "SOLO_MANI":
                x = x[:, :, :126]

            x, y = x.to(device), y.to(device)

            pred = model(x, lengths)
            total_loss += loss_fn(pred, y).item()
            num_batches += 1
            metric(pred, y)

    avg_loss = total_loss / num_batches
    acc = metric.compute().item()
    print(f"--- VAL  → Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    metric.reset()
    return avg_loss, acc


"""
7. TEST LOOP (Fuso: Calcola avg_loss per Early Stopping + usa lengths)
"""


def test_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y, lengths in dataloader:
            if MODALITA == "SOLO_MANI":
                x = x[:, :, :126]

            x, y = x.to(device), y.to(device)

            pred = model(x, lengths)
            total_loss += loss_fn(pred, y).item()
            num_batches += 1
            metric(pred, y)

    avg_loss = total_loss / num_batches
    acc = metric.compute().item()
    metric.reset()
    return avg_loss, acc


"""
8. ESECUZIONE E SALVATAGGIO (Fuso: Early Stopping + Percorsi Root + CSV)
"""
print(f"Inizio addestramento in modalità: {MODALITA}")

model_save_path = GET_MODEL_PATH(MODALITA)
experiment_save_dir = GET_EXPERIMENT_DIR(MODALITA)
csv_path = GET_CSV_PATH(MODALITA)

# Configurazioni Early Stopping e CSV
history = []
patience = PATIENCE
epochs_no_improve = 0
best_val_loss = float("inf")  # Early Stopping usa val_loss, NON test_loss

for epoch in range(EPOCHS):
    print(f"=============================")
    print(f" EPOCH: {epoch + 1}/{EPOCHS}")
    print(f"=============================")

    train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val_loop(val_dataloader, model, loss_fn)

    # Salviamo i dati per il CSV (Aggiunto test_loss!)
    history.append([epoch + 1, train_loss, train_acc, val_loss, val_acc])

    # --- LOGICA EARLY STOPPING E SALVATAGGIO (Dal codice del compagno) ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 Modello salvato! Nuova migliore Val Loss: {best_val_loss:.4f} (Val Acc: {val_acc:.4f})\n")
    else:
        epochs_no_improve += 1
        print(f"⚠️  Nessun miglioramento per {epochs_no_improve}/{patience} epoche consecutive.\n")

        if epochs_no_improve >= patience:
            print("🛑 EARLY STOPPING ATTIVATO: il modello ha smesso di imparare.")
            print(f"   Miglior Val Loss salvata: {best_val_loss:.4f}")
            break

# --- SALVATAGGIO DEL CSV FINALE ---
with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"])
    writer.writerows(history)


# Carichiamo il miglior modello salvato prima di eseguire il test.
print("\n" + "=" * 50)
print("🔬 VALUTAZIONE FINALE SUL TEST SET (una tantum)")
print("=" * 50)
model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
final_test_loss, final_test_acc = test_loop(test_dataloader, model, loss_fn)
print(f"*** ACCURATEZZA DI TEST FINALE: {final_test_acc:.4f} | Loss: {final_test_loss:.4f} ***")
print("=" * 50)

print("\nAddestramento completato! 🎉")
