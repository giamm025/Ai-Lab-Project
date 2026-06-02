import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = CURRENT_DIR / "train.py"

# Definiamo esattamente la lista dei 9 esperimenti che hai richiesto
esperimenti = [
    {"version": "v1", "desc": "L", "modalita": "SOLO_MANI"},
    {"version": "v1", "desc": "L", "modalita": "MANI_VOLTO"},
    {"version": "v1", "desc": "M", "modalita": "SOLO_MANI"},
    {"version": "v1", "desc": "M", "modalita": "MANI_VOLTO"},
    {"version": "v1", "desc": "S", "modalita": "SOLO_MANI"},
    {"version": "v1", "desc": "S", "modalita": "MANI_VOLTO"},
    
    {"version": "v2", "desc": "L", "modalita": "MANI_VOLTO"},
    {"version": "v2", "desc": "M", "modalita": "MANI_VOLTO"},
    {"version": "v2", "desc": "S", "modalita": "MANI_VOLTO"},
]

print("🚀 AVVIO BULK TRAINING (Addestramento Massivo dei Modelli)\n")
print("-" * 50)

modelli_addestrati = 0
errori = 0

for exp in esperimenti:
    v = exp["version"]
    size = exp["desc"]
    mod = exp["modalita"]
    
    nome_modello = f"model_{v}_{size}_{mod}"
    print(f"\n⏳ Avvio addestramento per: {nome_modello}...")
    print(f"   [Parametri: --version {v} --desc {size} --modalita {mod}]")
    
    # Lancia train.py con i parametri corretti
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--modalita", mod, "--version", v, "--desc", size],
        capture_output=False  # Lascialo False così vedi l'avanzamento delle epoche nel terminale!
    )
    
    if result.returncode == 0:
        print(f"✅ Addestramento completato per {nome_modello}!")
        modelli_addestrati += 1
    else:
        print(f"❌ Errore critico durante l'addestramento di {nome_modello}")
        errori += 1

print("\n" + "=" * 50)
print(f"🎉 BULK TRAINING COMPLETATO!")
print(f"Modelli addestrati con successo: {modelli_addestrati} / {len(esperimenti)}")
if errori > 0:
    print(f"Modelli falliti: {errori}")
print("=" * 50)