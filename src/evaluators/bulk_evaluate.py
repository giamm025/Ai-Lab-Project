import os
import subprocess
import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Definiamo i percorsi
CURRENT_DIR = Path(__file__).resolve().parent
EVALUATE_SCRIPT = CURRENT_DIR / "backward_evaluate_model.py"
MODELS_DIR = CURRENT_DIR.parent.parent / "models"

# Importiamo RESULTS_DIR dal config (che sta in src/, ovvero CURRENT_DIR.parent)
sys.path.append(str(CURRENT_DIR.parent))
from config import RESULTS_DIR

# Definiamo la matrice dei nostri esperimenti
versions = ["v1", "v2"]
sizes = ["S", "M", "L"]
modalitas = ["SOLO_MANI", "MANI_VOLTO"]

# Struttura per accumulare i dati per i 3 grafici comparativi richiesti
plot_data = {
    ("v1", "SOLO_MANI"): {},
    ("v1", "MANI_VOLTO"): {},
    ("v2", "MANI_VOLTO"): {}
}

# =====================================================================
# FUNZIONI HELPER PER ESTRAZIONE METRICHE E STRUMENTAZIONE GRAFICI
# =====================================================================
def extract_metrics(save_path, modalita):
    """Estrae Test Accuracy, Macro F1 e calcola l'Overfitting Gap adattandosi al formato (Old vs New)."""
    test_acc = 0.0
    macro_f1 = 0.0
    train_acc = 0.0

    # 1. Lettura del report testuale per ottenere Test Accuracy e Macro F1-Score
    report_path = save_path / f"classification_report_{modalita}.txt"
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if "macro" in parts and "avg" in parts:
                    try:
                        macro_f1 = float(parts[4])
                    except:
                        pass
                if "accuracy" in parts:
                    try:
                        test_acc = float(parts[1])
                    except:
                        pass

    # 2. Lettura del CSV adattiva
    # FIX: Cambiato da f"training_history_{modalita}.csv" a "training_history.csv" per allinearsi al config!
    csv_path = save_path / "training_history.csv"
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
            
            if rows:
                is_old_format = 'Val_Acc' in rows[0]
                
                if is_old_format:
                    best_val_acc = -1.0
                    for row in rows:
                        try:
                            val_acc_curr = float(row['Val_Acc'])
                            if val_acc_curr > best_val_acc:
                                best_val_acc = val_acc_curr
                                train_acc = float(row['Train_Acc'])
                        except:
                            pass
                else:
                    best_loss = float('inf')
                    for row in rows:
                        loss_key = 'Test_Loss' if 'Test_Loss' in row else 'Val_Loss'
                        if loss_key in row:
                            try:
                                loss_val = float(row[loss_key])
                                if loss_val < best_loss:
                                    best_loss = loss_val
                                    train_acc = float(row['Train_Acc'])
                            except:
                                pass
    else:
        print(f"⚠️ Attenzione: Impossibile trovare il file CSV in {csv_path}")

    # Calcolo dell'Overfitting Gap reale
    gap = train_acc - test_acc
    return test_acc, macro_f1, gap

def generate_comparison_chart(title, sizes_data, save_path):
    """Genera un grafico a barre raggruppate confrontando le METRICHE sull'asse X per le taglie S, M, L."""
    metrics_labels = ['Test Accuracy', 'F1-Score (Macro)', 'Overfitting Gap']
    
    # Estraiamo le tuple (test_acc, f1, gap) per ogni dimensione di dataset
    s_vals = sizes_data.get('S', (0, 0, 0))
    m_vals = sizes_data.get('M', (0, 0, 0))
    l_vals = sizes_data.get('L', (0, 0, 0))

    # NUOVO LAYOUT RAGGRUPPATO: Mettiamo in fila le metriche per ogni taglia
    s_bars = [s_vals[0], s_vals[1], s_vals[2]]
    m_bars = [m_vals[0], m_vals[1], m_vals[2]]
    l_bars = [l_vals[0], l_vals[1], l_vals[2]]

    x = np.arange(len(metrics_labels))  # 3 gruppi principali (Acc, F1, Gap)
    width = 0.25                        # Spessore delle barre

    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Generiamo le barre raggruppate per colore/dimensione dataset
    rects1 = ax.bar(x - width, s_bars, width, label='Dataset Piccolo (S)', color='#1f77b4')
    rects2 = ax.bar(x, m_bars, width, label='Dataset Medio (M)', color='#2ca02c')
    rects3 = ax.bar(x + width, l_bars, width, label='Dataset Grande (L)', color='#d62728')

    # Abbellimenti grafici accademici
    ax.set_ylabel('Valore Metrica', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=12, fontweight='bold')
    
    # Calcolo dinamico dei limiti Y per evitare che i testi escano fuori
    all_values = s_bars + m_bars + l_bars
    min_val = min(all_values)
    max_val = max(all_values)
    ax.set_ylim(min_val - 0.15 if min_val < 0 else -0.05, max_val + 0.15)
    
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-') # Linea dello zero evidenziata
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='upper right')

    # Funzione per stampare i valori sopra (o sotto se negativi) le barre
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            va_direction = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -12
            
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),  
                        textcoords="offset points",
                        ha='center', va=va_direction, fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# =====================================================================
# LOGICA DI CONTROLLO LOOP (Invariata)
# =====================================================================
print("🚀 AVVIO BULK EVALUATION (Valutazione Massiva dei Modelli)\n")
print("-" * 50)

modelli_valutati = 0
modelli_saltati = 0

for v in versions:
    for size in sizes:
        for mod in modalitas:
            
            suffix = f"{v}_{size}".strip('_')
            expected_model_name = f"model_{suffix}_{mod}.pth"
            expected_model_path = MODELS_DIR / expected_model_name
            
            if expected_model_path.exists():
                print(f"⏳ Valutazione in corso per: {expected_model_name}...")
                
                result = subprocess.run(
                    [sys.executable, str(EVALUATE_SCRIPT), "--modalita", mod, "--version", v, "--desc", size],
                    capture_output=False
                )
                
                if result.returncode == 0:
                    modelli_valutati += 1
                    if (v, mod) in plot_data:
                        save_dir = RESULTS_DIR / suffix / mod
                        metrics = extract_metrics(save_dir, mod)
                        plot_data[(v, mod)][size] = metrics
                else:
                    print(f"❌ Errore durante la valutazione di {expected_model_name}")
            else:
                print(f"⏭️  Saltato: {expected_model_name} non esiste.")
                modelli_saltati += 1
                
print("-" * 50)
print(f"🎉 BULK EVALUATION COMPLETATA!")
print(f"Modelli valutati con successo: {modelli_valutati}")
print(f"Modelli saltati (non trovati): {modelli_saltati}\n")

# =====================================================================
# GENERAZIONE DEI GRAFICI CON IL NUOVO LAYOUT RAGGRUPPATO
# =====================================================================
print("📊 Generazione dei grafici comparativi (Raggruppati per Metrica)...")
for (v, mod), sizes_data in plot_data.items():
    if sizes_data:
        graph_title = f"Confronto Prestazioni Dataset - {v} ({mod})"
        graph_filename = RESULTS_DIR / f"confronto_metriche" / f"confronto_metriche_{v}_{mod}.png"
        graph_filename.parent.mkdir(parents=True, exist_ok=True)
        generate_comparison_chart(graph_title, sizes_data, graph_filename)
        print(f"    ✅ Grafico salvato in: {graph_filename}")