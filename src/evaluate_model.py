import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split
import argparse

from config import MODELS_DIR, RESULTS_DIR, PROCESSED_DIR, TARGET_WORDS, LABEL_MAP, SEED, GET_MODEL_PATH, GET_EXPERIMENT_DIR, GET_CSV_PATH
from neural_network.dataset import SignLanguageDataset
from neural_network.model import SignLanguageLSTM

# =====================================================================
# FUNZIONI DI VISUALIZZAZIONE E REPORTING
# =====================================================================

"""Genera il grafico con le curve di Loss e Accuracy, per un confronto Train vs Test"""
def draw_learning_curves(csv_path, save_dir, modalita):
    
    if not csv_path.exists():
        print("CSV non trovato! Fai prima il train.")
        return
    
    # Prepariamo una "tela" bianca per il grafico grande 10x5 pollici
    plt.figure(figsize=(10, 5))
    
    # Legge il file CSV
    df = pd.read_csv(csv_path)
    
    # Disegniamo la prima linea: L'accuratezza durante lo studio (Train)
    plt.plot(df['Epoch'], df['Train_Acc'], label='Train Accuracy', marker='o')

    # Disegniamo la seconda linea: L'accuratezza durante l'esame (Test)
    plt.plot(df['Epoch'], df['Test_Acc'], label='Test Accuracy', marker='s')
    
    # Aggiungiamo titoli, etichette per gli assi e una griglia di sfondo per renderlo "scientifico"
    plt.title(f'Learning Curves ({modalita})')
    plt.xlabel('Epochs (Epoche)')
    plt.ylabel('Accuracy (Accuratezza)')
    plt.ylim(0, 1.1) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend() 
    
    # Salviamo l'immagine finita come .png nella cartella results/
    save_path = save_dir / f'learning_curve_{modalita}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Grafico Learning Curve salvato in: {save_path}")


"""
Crea il report.txt che contiene:
    - Precision: quante volte il modello indovina la parola specifica?              (es. se dice "happen" 100 volte, quante volte ha ragione davvero?)
    - Recall:    se la parola X è presente 100 volte, quante volte la individua?
    - F1-Score:  è la media armonica tra Precision e Recall. RSe è alto, significa che va tutto bene: il modello è sia 
                 preciso che sensibile. Se è basso, significa che il modello ha problemi in almeno uno dei due aspetti.
    - Support:   quante volte la parola X è presente nel test set

I parametri della funzione sono:
    - solutions:    lista con i valori reali (quello che era davvero)
    - precitions:   lista con i valori predetti (quello che ha capito la rete)
    - target_words: lista con i nomi delle parole (ordinati in base agli indici di LABEL_MAP)
    - save_dir:     cartella dove salvare il report
    - modalita:     SOLO_MANI o MANI_VOLTO (per distinguere i file dei due esperimenti)
"""
def generate_report(solutions, precitions, target_words, save_dir, modalita):
    
    report = classification_report(solutions, precitions, target_names=target_words)
    report_path = save_dir / f'classification_report_{modalita}.txt'
    
    with report_path.open('w') as f:
        f.write(f"--- RISULTATI FINALI {modalita} ---\n\n")
        f.write(report)
        
    print(f"📝 Report Testuale salvato in: {report_path}")


"""
Crea la 'matrice di confusione', in cui:
    - confusion_matrix[x:y]: la risposta corretta era x ma il modello ha detto y

Da cui ne deduciamo che:
    - confusion_matrix[x:x]: quante volte il modello ha indovinato correttamente la parola x (la risposta corretta era x ma il modello ha detto x)
    - confusion_matrix[x:y]: quante volte il modello ha confuso la parola x con la parola y  (la risposta corretta era x ma il modello ha detto y)
"""
def draw_confusion_matrix(solutions, precitions, target_words, save_dir, modalita):
    cm = confusion_matrix(solutions, precitions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_words, yticklabels=target_words)
    
    plt.title(f'Confusion Matrix ({modalita})')
    plt.ylabel('Valore Reale (Quello che era davvero)')
    plt.xlabel('Valore Predetto (Quello che ha capito la rete)')
    
    cm_path = save_dir / f'confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    print(f"📊 Confusion Matrix salvata in: {cm_path}")

# =====================================================================
# LOGICA DI VALUTAZIONE CORE
# =====================================================================
"""Ri-divide il Dataset per ottenere lo stesso Test Set utilizzato alla fine dell'addestramento)"""
def create_test_dataloader(dataset):
    torch.manual_seed(SEED) 
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_data = random_split(dataset, [train_size, test_size]) 
    
    return DataLoader(test_data, batch_size=8, shuffle=False)


"""Carica il modello gia addestrato (.pth) e lo prepara per la valutazione"""
def load_trained_model(model_path, modalita, device):
    input_size = 126 if modalita == "SOLO_MANI" else 402
    num_classes = len(TARGET_WORDS)
    
    model = SignLanguageLSTM(input_size=input_size, hidden_size=64, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

"""
Ri-esegue la fase di test per ottenere la lista di predizioni (risposte del modello) e la lista di soluzioni

NB. Questa esecuzione è diversa da quella di train.py!!!! Li eseguiamo il test AD OGNI EPOCA per ottenere loss/accuracy
    e decretare un "miglior modello" da salvare. QUI, invece, eseguiamo il test SOLO UNA VOLTA, sul MODELLO MIGLIORE per 
    valutare il modello finale e generare i grafici finali. NON avrebbe senso unificare le due logiche.
"""
def run_evaluation(model, dataloader, modalita, device):
    precitions = [] 
    solutions = [] 
    
    print("🤖 Inizio test sul modello salvato...")
    with torch.no_grad(): 
        for x, y, lengths in dataloader:
            if modalita == "SOLO_MANI":
                x = x[:, :, :126] 
            
            x, y, lengths = x.to(device), y.to(device), lengths.cpu()
            
            outputs = model(x, lengths)
            preds = torch.argmax(outputs, dim=1) 
            
            precitions.extend(preds.cpu().numpy())
            solutions.extend(y.cpu().numpy())
            
    return solutions, precitions

# =====================================================================
# MAIN: ESECUZIONE DELLA VALUTAZIONE
# =====================================================================
if __name__ == "__main__":

    # ------------------------------------------ PARSER ------------------------------------------
    parser = argparse.ArgumentParser(description="Valuta il modello e genera grafici.")
    parser.add_argument("--modalita", type=str, choices=["SOLO_MANI", "MANI_VOLTO"], default="MANI_VOLTO")
    args = parser.parse_args()
    MODALITA = args.modalita

    # ------------------------------------------- PATHS ---------------------------------------
    SAVE_PATH = GET_EXPERIMENT_DIR(MODALITA)
    model_file = GET_MODEL_PATH(MODALITA)
    csv_file = GET_CSV_PATH(MODALITA)

    # ----------------------------------- FASE DI TEST E VALUTAZIONE ------------------------------------
    print(f"\n--- AVVIO VALUTAZIONE: {MODALITA} ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not model_file.exists():
        print(f"❌ Errore: Modello non trovato ({model_file}). Addestra prima la rete!")
        sys.exit()
        
    # PREPARAZIONE TEST
    full_dataset = SignLanguageDataset(PROCESSED_DIR)
    test_dataloader = create_test_dataloader(full_dataset)
    model = load_trained_model(model_file, MODALITA, device)
    solutions, precitions = run_evaluation(model, test_dataloader, MODALITA, device)    
    target_words = [word for word, idx in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
    
    # GENERAZIONE GRAFICI E REPORT
    draw_learning_curves(csv_file, SAVE_PATH, MODALITA)
    generate_report(solutions, precitions, target_words, SAVE_PATH, MODALITA)
    draw_confusion_matrix(solutions, precitions, target_words, SAVE_PATH, MODALITA)
    
    print(f"\n✅ Valutazione completata! Controlla la cartella: {SAVE_PATH}")