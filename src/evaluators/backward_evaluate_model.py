import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split
import argparse

from config import MODELS_DIR, RESULTS_DIR, PROCESSED_DIR, TARGET_WORDS, LABEL_MAP, SEED
from neural_network.dataset import SignLanguageDataset
from neural_network.model import SignLanguageLSTM

# =====================================================================
# FUNZIONI DI VISUALIZZAZIONE E REPORTING
# =====================================================================

"""Genera il grafico con le curve di Loss e Accuracy, per un confronto Train vs Val"""
def plot_learning_curves(csv_path, save_dir, modalita):
    
    if not csv_path.exists():
        print(f"⚠️ CSV non trovato in {csv_path}! Salto il grafico Learning Curve.")
        return
    # Prepariamo una "tela" bianca per il grafico grande 10x5 pollici
    plt.figure(figsize=(10, 5))

    # Legge il file CSV
    df = pd.read_csv(csv_path)

    # Disegniamo la prima linea: L'accuratezza durante lo studio (Train)
    plt.plot(df["Epoch"], df["Train_Acc"], label="Train Accuracy", marker="o")

    # Disegniamo la seconda linea: L'accuratezza durante l'esame (Val)
    plt.plot(df["Epoch"], df["Val_Acc"], label="Validation Accuracy", marker="s")

    # Aggiungiamo titoli, etichette per gli assi e una griglia di sfondo per renderlo "scientifico"
    plt.title(f"Learning Curves ({modalita})")
    plt.xlabel("Epochs (Epoche)")
    plt.ylabel("Accuracy (Accuratezza)")
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Salviamo l'immagine finita come .png nella cartella results/
    save_path = save_dir / f"learning_curve_{modalita}.png"
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
    
    report = classification_report(solutions, precitions, labels=list(range(len(target_words))), target_names=target_words, zero_division=0)
    report_path = save_dir / f"classification_report_{modalita}.txt"

    with report_path.open("w", encoding="utf-8") as f:
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_words, yticklabels=target_words)

    plt.title(f"Confusion Matrix ({modalita})")
    plt.ylabel("Valore Reale (Quello che era davvero)")
    plt.xlabel("Valore Predetto (Quello che ha capito la rete)")

    cm_path = save_dir / f"confusion_matrix_{modalita}.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"📊 Confusion Matrix salvata in: {cm_path}")


# =====================================================================
# LOGICA DI VALUTAZIONE CORE
# =====================================================================
# =====================================================================
"""Ri-divide il Dataset per ottenere lo stesso Test Set utilizzato alla fine dell'addestramento)"""
def create_test_dataloader(dataset):
    torch.manual_seed(SEED)
    
    # prendiamo solo gli indici dei video originali (escludendo quelli con '_aug_' nel nome)
    original_indices = [i for i, name in enumerate(dataset.filenames) if "_aug_" not in name]
    
    # creiamo un sub-datset contenente solo video originali, ESCLUDENDO quelli derivanti da data augmentation
    # questo lo useremo in fare di test per evitare di testare il modello su video uguali a quelli di train... ma semplicemente con le mani scambiate
    pure_dataset = torch.utils.data.Subset(dataset, original_indices)
    
    total = len(pure_dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    _, _, test_data = random_split(pure_dataset, [train_size, val_size, test_size])
    return DataLoader(test_data, batch_size=8, shuffle=False)

def load_trained_model(model_path, modalita, version, device):
    # RETROCOMPATIBILITÀ: Se il modello è v1 ed è MANI_VOLTO, si aspetta 1530 ingressi, altrimenti 402
    if modalita == "SOLO_MANI":
        input_size = 126
    elif modalita == "MANI_VOLTO":
        input_size = 1530 if version == "v1" else 402
    else: 
        raise ValueError("Modalità sconosciuta! Scegli SOLO_MANI o MANI_VOLTO.")    
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
    parser.add_argument("--modalita", type=str, choices=["SOLO_MANI", "MANI_VOLTO"], required=True)
    parser.add_argument("--version", type=str, required=True, help="Es. v1, v2")
    parser.add_argument("--desc", type=str, default="", help="Es. S, M, L")
    args = parser.parse_args()
    
    MODALITA = args.modalita
    
    # ------------------------------------------- PATHS ---------------------------------------
    suffix = f"{args.version}_{args.desc}".strip('_')
    SAVE_PATH = RESULTS_DIR / suffix / MODALITA
    model_file = MODELS_DIR / f"model_{suffix}_{MODALITA}.pth"
    csv_file = SAVE_PATH / "training_history.csv"
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # ----------------------------------- FASE DI TEST E VALUTAZIONE ------------------------------------
    print(f"\n--- AVVIO VALUTAZIONE: {MODALITA} | Esperimento: {suffix} ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not model_file.exists():
        print(f"❌ Errore: Modello non trovato ({model_file}). Salto valutazione.")
        sys.exit(1)

    # PREPARAZIONE TEST
    # eseguiamo la fase di test utilizzando IL DATASET SPECIFICO utilizzato durante l'addestramento di quel modello
    # deve per forza essere nel formato "vX_processed_Y" (es. v1_processed_S) 
    dataset_folder_name = f"{args.version}_processed_{args.desc}"
    dynamic_processed_dir = PROCESSED_DIR.parent / dataset_folder_name
    
    # Controllo di sicurezza: se per caso la cartella specifica non esiste, usa quella di default
    if not dynamic_processed_dir.exists():
        print(f"⚠️ Cartella specifica non trovata ({dynamic_processed_dir}). Uso PROCESSED_DIR di default.")
        dynamic_processed_dir = PROCESSED_DIR
    else:
        print(f"📦 Dataset rilevato con successo: {dataset_folder_name}")

    full_dataset = SignLanguageDataset(dynamic_processed_dir)
    test_dataloader = create_test_dataloader(full_dataset)
    model = load_trained_model(model_file, MODALITA, args.version, device)
    solutions, precitions = run_evaluation(model, test_dataloader, MODALITA, device)
    target_words = [word for word, idx in sorted(LABEL_MAP.items(), key=lambda item: item[1])]

    # GENERAZIONE GRAFICI E REPORT
    plot_learning_curves(csv_file, SAVE_PATH, MODALITA)
    generate_report(solutions, precitions, target_words, SAVE_PATH, MODALITA)
    draw_confusion_matrix(solutions, precitions, target_words, SAVE_PATH, MODALITA)

    print(f"\n✅ Valutazione completata! Controlla la cartella: {SAVE_PATH}")