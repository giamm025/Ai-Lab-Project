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

from config import TARGET_WORDS, RAW_DIR, DATASETS_DIR

from config import PROCESSED_DIR, TARGET_WORDS
from dataset import SignLanguageDataset
from model import SignLanguageLSTM

# Inizializzo il parser degli argomenti
parser = argparse.ArgumentParser(description="Addestra il modello di Riconoscimento LIS.")

# Aggiungo un parametro opzionale chiamato --modalita
parser.add_argument(
    "--modalita", 
    type=str, 
    choices=["SOLO_MANI", "MANI_VOLTO"], # Limita le scelte per evitare errori di battitura
    default="MANI_VOLTO",                # Se non scrivi nulla nel terminale, usa questa di default
    help="Scegli se usare i 126 keypoints (SOLO_MANI) o tutti i 1530 keypoints (MANI_VOLTO)"
)

# Leggo cosa ha scritto l'utente nel terminale
args = parser.parse_args()
MODALITA = args.modalita

# Stampo un messaggio chiaro per confermare la modalità
print(f"\n⚙️  CONFIGURAZIONE AVVIATA: Modalità {MODALITA}")

'''
1. PREPARARE IL DATASET
Carico i dati estratti con MediaPipe e li divido in Train e Test.
'''
# Creo il dataset completo usando la cartella con i file .npy
full_dataset = SignLanguageDataset(PROCESSED_DIR)

# Divido il dataset: 80% per l'addestramento, 20% per il test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
training_data, test_data = random_split(full_dataset, [train_size, test_size])

'''
2. CREARE I DATALOADER
'''
# Batch size piccola (es. 8) perché i tensori video pesano in memoria
batch_size = 8 
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

'''
3. SELEZIONARE IL DISPOSITIVO
'''
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")

'''
3.5 CONFIGURAZIONE A/B TEST (Tramite Terminale)
'''

if MODALITA == "SOLO_MANI":
    input_size = 126  
else:
    input_size = 1530

'''
4. DEFINIZIONE DEL MODELLO
'''
hidden_size = 64
num_classes = len(TARGET_WORDS) # 5 parole target

# Istanzio la LSTM e la sposto sulla GPU
model = SignLanguageLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device) 
print(model)

'''
5. IPERPARAMETRI, LOSS FUNCTION E OPTIMIZER
'''
epochs = 20 # Le LSTM richiedono più epoche rispetto alle reti lineari semplici
learning_rate = 1e-3

loss_fn = torch.nn.CrossEntropyLoss()


'''
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
'''

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)

'''
6. TRAINING LOOP
'''
def train_loop(train_dataloader, model, loss_fn, optimizer):
    model.train() # Metto il modello in modalità addestramento
    
    for batch, (x, y) in enumerate(train_dataloader):
        
        # --- TAGLIO DEL TENSORE PER L'A/B TEST ---
        if MODALITA == "SOLO_MANI":
            # x ha forma [batch, seq_len, 1530]. Prendo solo tutte le righe e le prime 126 colonne
            x = x[:, :, :126] 
            
        x = x.to(device)
        y = y.to(device)

        # Forward
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Stampo i progressi (uso % 5 dato che ho meno batch rispetto a MNIST)
        if batch % 5 == 0:
            loss_v = loss.item()
            acc = metric(pred, y)
            print(f'Loss: {loss_v:.4f} | Batch: {batch + 1} | Accuratezza batch: {acc:.4f}')

    acc = metric.compute()
    print(f'\n---> Accuratezza Finale Epoca (Train): {acc:.4f}\n')
    metric.reset()

'''
7. TEST LOOP
'''
def test_loop(test_dataloader, model):
    model.eval() # Metto il modello in modalità valutazione
    with torch.no_grad(): # Spegne i gradienti
        for x, y in test_dataloader:
            
            # Applico lo stesso taglio anche in fase di test
            if MODALITA == "SOLO_MANI":
                x = x[:, :, :126]
                
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            metric(pred, y)

        acc = metric.compute()
        print(f'*** ACCURATEZZA DI TEST FINALE: {acc:.4f} ***\n')
        metric.reset()

'''
8. ESECUZIONE
'''
print(f"Inizio addestramento in modalità: {MODALITA}")
for epoch in range(epochs):
    print(f'=============================')
    print(f' EPOCH: {epoch + 1}/{epochs}')
    print(f'=============================')
    
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model)
    
print("Addestramento completato! 🎉")