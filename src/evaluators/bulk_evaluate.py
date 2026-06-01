import os
import subprocess
import sys
from pathlib import Path

# Definiamo i percorsi
CURRENT_DIR = Path(__file__).resolve().parent
EVALUATE_SCRIPT = CURRENT_DIR / "backward_evaluate_model.py"
MODELS_DIR = CURRENT_DIR.parent.parent / "models"

# Definiamo la matrice dei nostri esperimenti
versions = ["v1", "v2"]
sizes = ["S", "M", "L"]
modalitas = ["SOLO_MANI", "MANI_VOLTO"]

print("🚀 AVVIO BULK EVALUATION (Valutazione Massiva dei Modelli)\n")
print("-" * 50)

modelli_valutati = 0
modelli_saltati = 0

# Cicliamo su tutte le 12 combinazioni possibili
for v in versions:
    for size in sizes:
        for mod in modalitas:
            
            # Ricostruiamo il nome del file per controllare se esiste
            # Es: model_v1_L_MANI_VOLTO.pth
            suffix = f"{v}_{size}".strip('_')
            expected_model_name = f"model_{suffix}_{mod}.pth"
            expected_model_path = MODELS_DIR / expected_model_name
            
            if expected_model_path.exists():
                print(f"⏳ Valutazione in corso per: {expected_model_name}...")
                
                # Questa riga lancia letteralmente il comando nel terminale in modo invisibile
                result = subprocess.run(
                    [sys.executable, str(EVALUATE_SCRIPT), "--modalita", mod, "--version", v, "--desc", size],
                    capture_output=False # Metti True se non vuoi vedere i log intermedi sul terminale
                )
                
                if result.returncode == 0:
                    modelli_valutati += 1
                else:
                    print(f"❌ Errore durante la valutazione di {expected_model_name}")
            else:
                print(f"⏭️  Saltato: {expected_model_name} non esiste.")
                modelli_saltati += 1
                
print("-" * 50)
print(f"🎉 BULK EVALUATION COMPLETATA!")
print(f"Modelli valutati con successo: {modelli_valutati}")
print(f"Modelli saltati (non trovati): {modelli_saltati}")