import sys
import torch
from torch.utils.data import DataLoader, random_split
import torchmetrics
import argparse
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TARGET_WORDS, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, SEED
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

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
training_data, test_data = random_split(full_dataset, [train_size, test_size])

"""
2. CREARE I DATALOADER
"""
batch_size = BATCH_SIZE
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
    hidden_size=hidden_size,
    num_classes=num_classes,
    num_layers=num_layers,
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
    
    print(f'---> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
    metric.reset()
    return epoch_loss, epoch_acc

"""
7. TEST LOOP (Fuso: Calcola avg_loss per Early Stopping + usa lengths)
"""
def test_loop(test_dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y, lengths in test_dataloader:
            if MODALITA == "SOLO_MANI":
                x = x[:, :, :126]  

            x = x.to(device)
            y = y.to(device)

            pred = model(x, lengths)
            
            total_loss += loss_fn(pred, y).item()
            num_batches += 1
            metric(pred, y)

    avg_loss = total_loss / num_batches
    acc = metric.compute().item()
    print(f"*** TEST → Loss: {avg_loss:.4f} | Accuratezza: {acc:.4f} ***\n")
    metric.reset()
    return avg_loss, acc

"""
8. ESECUZIONE E SALVATAGGIO (Fuso: Early Stopping + Percorsi Root + CSV)
"""
print(f"Inizio addestramento in modalità: {MODALITA}")

# prendiamo il path in cui salvare il modello (root/models/best_model_{MODALITA}.pth)
model_save_path = MODELS_DIR / f"best_model_{MODALITA}.pth"

# Configurazioni Early Stopping e CSV
history = []
patience = PATIENCE
epochs_no_improve = 0
best_test_loss = float("inf") # Usiamo la loss per il salvataggio (più preciso dell'accuratezza)

for epoch in range(EPOCHS):
    print(f"=============================")
    print(f" EPOCH: {epoch + 1}/{EPOCHS}")
    print(f"=============================")

    train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
    
    # Salviamo i dati per il CSV (Aggiunto test_loss!)
    history.append([epoch + 1, train_loss, train_acc, test_loss, test_acc])
    
    # --- LOGICA EARLY STOPPING E SALVATAGGIO (Dal codice del compagno) ---
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 Modello salvato in root/models! Nuova migliore Loss: {best_test_loss:.4f} (Acc: {test_acc:.4f})\n")
    else:
        epochs_no_improve += 1
        print(f"⚠️  Nessun miglioramento per {epochs_no_improve}/{patience} epoche consecutive.\n")

        if epochs_no_improve >= patience:
            print("🛑 EARLY STOPPING ATTIVATO: il modello ha smesso di imparare (Overfitting bloccato).")
            print(f"   Modello migliore salvato con Loss: {best_test_loss:.4f}")
            break # Interrompe il ciclo for!

# --- SALVATAGGIO DEL CSV FINALE ---
csv_path = RESULTS_DIR / f"training_history_{MODALITA}.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Test_Loss', 'Test_Acc'])
    writer.writerows(history)

print("Addestramento completato! 🎉")