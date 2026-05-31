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
from config import MODELS_DIR, RESULTS_DIR, PROCESSED_DIR, TARGET_WORDS, LABEL_MAP, SEED
from neural_network.dataset import SignLanguageDataset
from neural_network.model import SignLanguageLSTM


# =====================================================================
# FUNZIONE 1: DISEGNARE LE CURVE DI APPRENDIMENTO (Learning Curves)
# =====================================================================
# Questa funzione prende il file CSV creato da train.py e disegna un grafico a linee
# Ci fa vedere visivamente se la rete ha imparato bene o se è andata in Overfitting
def plot_learning_curves(csv_path, save_dir, modalita):

    if not csv_path.exists():
        print("CSV non trovato! Fai prima il train.")
        return

    # Prepariamo una "tela" bianca per il grafico grande 10x5 pollici
    plt.figure(figsize=(10, 5))

    # Legge il file CSV
    df = pd.read_csv(csv_path)

    # Disegniamo la prima linea: L'accuratezza durante lo studio (Train)
    plt.plot(df["Epoch"], df["Train_Acc"], label="Train Accuracy", marker="o")

    # Disegniamo la seconda linea: L'accuratezza durante l'esame (Test)
    plt.plot(df["Epoch"], df["Val_Acc"], label="Validation Accuracy", marker="s")

    # Aggiungiamo titoli, etichette per gli assi e una griglia di sfondo per renderlo "scientifico"
    plt.title(f"Learning Curves ({modalita})")
    plt.xlabel("Epochs (Epoche)")
    plt.ylabel("Accuracy (Accuratezza)")
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Salviamo l'immagine finita come .png nella cartella results/
    plot_path = save_dir / f"learning_curve_{modalita}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"📈 Grafico Learning Curve salvato in: {plot_path}")


# =====================================================================
# FUNZIONE 2: VALUTAZIONE E MATRICE DI CONFUSIONE
# =====================================================================
# esegue di nuovo la fase di TEST, scrive un report dettagliato e disegna la matrice di confusione
def evaluate_and_plot_confusion_matrix(model_path, dataset, modalita, save_dir, device):

    # Ovviamente usiamo lo stesso seed del train.py per eseguire il test sempre sugli stessi video (altrimenti rischiamo che capitino video presi dal training set)
    torch.manual_seed(SEED)
    total = len(dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    _, _, test_data = random_split(dataset, [train_size, val_size, test_size])

    # Creiamo un DataLoader per il test (ci permette di prendere i dati a batch e non tutti in una volta, risparmiando RAM)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)

    # prepariamo il modello
    if modalita == "SOLO_MANI":
        input_size = 126
    elif modalita == "MANI_VOLTO":
        input_size = 402
    else:
        raise ValueError("Modalità sconosciuta! Scegli SOLO_MANI o MANI_VOLTO.")
    num_classes = len(TARGET_WORDS)

    # Creiamo un cervello "vuoto" e poi ci infiliamo dentro i ricordi salvati nel file .pth
    model = SignLanguageLSTM(input_size=input_size, hidden_size=64, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # mettiamo il modello in modalita valutazione (disabilita dropout, batchnorm, ecc)

    all_preds = []  # Qui salveremo tutte le risposte DIL MODELLO (es. "ha detto book")
    all_trues = []  # Qui salveremo tutte le risposte REALI (es. "era davvero book")

    print("🤖 Inizio test sul modello salvato...")
    with torch.no_grad():
        # Modifica qui: aggiungiamo 'lengths' per catturare il terzo valore restituito dal dataset
        for x, y, lengths in test_dataloader:
            if modalita == "SOLO_MANI":
                x = x[:, :, :126]

            x, y, lengths = x.to(device), y.to(device), lengths.cpu()

            # Passiamo x e lengths (già pronte!) al modello
            outputs = model(x, lengths)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(y.cpu().numpy())

    # 3. IL CLASSIFICATION REPORT (La Pagella Dettagliata)
    target_names = [word for word, idx in sorted(LABEL_MAP.items(), key=lambda item: item[1])]

    # LA libreria sklearn calcola in automatico tutte le statistiche fighe (Precision, Recall, F1)
    report = classification_report(all_trues, all_preds, target_names=target_names)

    # Salviamo la pagella in un file di testo (.txt)
    report_path = save_dir / f"classification_report_{modalita}.txt"
    with report_path.open("w") as f:
        f.write(f"--- RISULTATI FINALI {modalita} ---\n\n")
        f.write(report)
    print(f"📝 Report Testuale salvato in: {report_path}")

    # 4. LA CONFUSION MATRIX (La griglia grafica degli errori)
    # Crea una matrice matematica mettendo a confronto risposte Vere vs Predette
    cm = confusion_matrix(all_trues, all_preds)

    # Prepara la tela per disegnare la griglia
    plt.figure(figsize=(8, 6))

    # Seaborn (sns) trasforma la matrice matematica in una bellissima "Heatmap" (Mappa di calore).
    # Più alto è il numero in un quadrato, più il quadrato diventa blu scuro.
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)

    plt.title(f"Confusion Matrix ({modalita})")
    plt.ylabel("Valore Reale (Quello che era davvero)")
    plt.xlabel("Valore Predetto (Quello che ha capito la rete)")

    # Salva il grafico come immagine
    cm_path = save_dir / f"confusion_matrix_{modalita}.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"📊 Confusion Matrix salvata in: {cm_path}")


# =====================================================================
# MAIN: ESECUZIONE DELLA VALUTAZIONE
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuta il modello e genera grafici.")
    parser.add_argument("--modalita", type=str, choices=["SOLO_MANI", "MANI_VOLTO"], default="MANI_VOLTO")
    args = parser.parse_args()

    MODALITA = args.modalita
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # creiamo le sotto cartelle results/SOLO_MANI/ e results/MANI_VOLTO/
    MODALITA_RESULTS_DIR = RESULTS_DIR / MODALITA
    MODALITA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Costruiamo i percorsi usando la nuova cartella dedicata
    model_file = MODELS_DIR / f"best_model_{MODALITA}.pth"
    csv_file = RESULTS_DIR / f"training_history_{MODALITA}.csv"

    print(f"\n--- AVVIO VALUTAZIONE: {MODALITA} ---")

    if not model_file.exists():
        print(f"❌ Errore: Modello non trovato ({model_file}). Addestra prima la rete!")
        sys.exit()

    full_dataset = SignLanguageDataset(PROCESSED_DIR)

    # Disegna le curve
    plot_learning_curves(csv_file, MODALITA_RESULTS_DIR, MODALITA)

    # Esegue il test e disegna la matrice
    evaluate_and_plot_confusion_matrix(model_file, full_dataset, MODALITA, MODALITA_RESULTS_DIR, device)

    print(f"\n✅ Valutazione completata! Controlla la cartella: {MODALITA_RESULTS_DIR}")
