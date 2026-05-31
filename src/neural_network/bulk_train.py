import subprocess
import re
import sys
import os
import random
import argparse
from pathlib import Path

# Forza stdout in UTF-8 nel caso in cui l'output venga reindirizzato su un file
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

import config


def main():
    parser = argparse.ArgumentParser(description="Esegui il bulk training del modello.")

    arg_configs = [
        {"name": "--n_runs", "type": int, "default": 10, "help": "Quante volte ripetere l'addestramento (N)"},
        {"name": "--modalita", "nargs": "+", "choices": ["SOLO_MANI", "MANI_VOLTO"], "default": ["SOLO_MANI", "MANI_VOLTO"], "help": "Modalità da testare"},
        {"name": "--epochs", "type": int, "default": 200, "help": "Numero di epoche per l'addestramento (default: 200)"},
        {"name": "--batch_size", "type": int, "default": 8, "help": "Dimensione del batch per l'addestramento (default: 8)"},
        {"name": "--patience", "type": int, "default": 10, "help": "Numero di epoche senza miglioramento prima dell'early stopping (default: 10)"},
        {"name": "--learning_rate", "type": float, "default": 1e-3, "help": "Learning rate per l'ottimizzatore (default: 1e-3)"},
        {"name": "--hidden_size", "type": int, "default": 64, "help": "Dimensione dell'hidden state della LSTM (default: 64)"},
        {"name": "--num_layers", "type": int, "default": 1, "help": "Numero di layer della LSTM (default: 1)"},
    ]

    for arg in arg_configs:
        name = arg.pop("name")
        parser.add_argument(name, **arg)

    args = parser.parse_args()

    N_RUNS = args.n_runs
    MODALITA_LIST = args.modalita

    # Generiamo N_RUNS seed casuali una sola volta
    SEEDS = [random.randint(1, 10000) for _ in range(N_RUNS)]

    print("=====================================================")
    print(f"🚀 INIZIO BULK TRAINING : {N_RUNS} run per ogni modalità")
    print(f"Modalità in test: {', '.join(MODALITA_LIST)}")
    print(f"Seeds scelti per le run: {SEEDS}")
    print("=====================================================\n")

    results = {mod: [] for mod in MODALITA_LIST}

    for mod in MODALITA_LIST:
        print(f"=====================================================")
        print(f"🔍 TESTANDO LA MODALITÀ: {mod}")
        print(f"=====================================================\n")

        for run_idx in range(N_RUNS):
            run = run_idx + 1
            seed = SEEDS[run_idx]
            print(f"🔄 [{mod}] Esecuzione Addestramento {run}/{N_RUNS} (Seed: {seed}) in corso...")

            current_dir = Path(__file__).resolve().parent
            train_script_path = current_dir / "train.py"

            # Assicuriamoci che python trovi config.py impostando il PYTHONPATH alla cartella src/
            env = os.environ.copy()
            env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else str(src_dir)

            # Forza l'output UTF-8 per il processo figlio così scriverà le emoji senza fare crash
            env["PYTHONIOENCODING"] = "utf-8"

            # Lancia train.py usando la sua cartella assoluta
            cmd = [
                sys.executable,
                str(train_script_path),
                "--modalita",
                mod,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--patience",
                str(args.patience),
                "--learning_rate",
                str(args.learning_rate),
                "--hidden_size",
                str(args.hidden_size),
                "--num_layers",
                str(args.num_layers),
            ]

            # Use errors="replace" so we don't crash when reading problematic characters
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            if process.returncode != 0:
                print(f"❌ Errore durante la run {run}. Output di errore:")
                print(process.stderr)
                continue

            output = process.stdout

            # Estraiamo con le espressioni regolari l'ultima accuratezza e loss stampate
            train_accs = re.findall(r"---> Train Loss:\s*[0-9.]+\s*\|\s*Train Acc:\s*([0-9.]+)", output)
            test_accs = re.findall(
                r"\*\*\*\s*ACCURATEZZA DI TEST FINALE:\s*([0-9.]+)\s*\|\s*Loss:\s*[0-9.]+\s*\*\*\*",
                output,
            )
            losses = re.findall(r"\*\*\*\s*ACCURATEZZA DI TEST FINALE:\s*[0-9.]+\s*\|\s*Loss:\s*([0-9.]+)\s*\*\*\*", output)

            if train_accs and test_accs and losses:
                final_train = float(train_accs[-1])
                final_test = float(test_accs[-1])
                final_loss = float(losses[-1])

                results[mod].append(
                    {
                        "run": run,
                        "loss": final_loss,
                        "train_acc": final_train,
                        "test_acc": final_test,
                    }
                )
                print(f"  ✅ [{mod}] Run {run} completata: Loss = {final_loss:.4f} | Train Acc = {final_train:.4f} | Test Acc = {final_test:.4f}\n")
            else:
                print(f"  ⚠️ [{mod}] Run {run} completata, ma non sono riuscito a leggere tutte le metriche dal log.\n")

    # ==========================================
    # RESOCONTO FINALE
    # ==========================================
    print("=====================================================")
    print("📊 RESOCONTO DETTAGLIATO DEGLI ADDESTRAMENTI")
    print("=====================================================")

    for mod in MODALITA_LIST:
        mod_results = results[mod]
        print(f"\n--- 📌 MODALITÀ: {mod} ---")

        if not mod_results:
            print("Nessun risultato registrato.")
            continue

        for res in mod_results:
            print(f"  Run {res['run']:02d} -> Loss: {res['loss']:.4f} | Train: {res['train_acc']:.4f} | Test: {res['test_acc']:.4f}")

        print("  ---------------------------------------------------")
        avg_loss = sum(r["loss"] for r in mod_results) / len(mod_results)
        avg_train = sum(r["train_acc"] for r in mod_results) / len(mod_results)
        avg_test = sum(r["test_acc"] for r in mod_results) / len(mod_results)
        print(f"  🎯 MEDIA LOSS           : {avg_loss:.4f}")
        print(f"  🎯 MEDIA TRAIN ACCURACY : {avg_train:.4f}")
        print(f"  🎯 MEDIA TEST ACCURACY  : {avg_test:.4f}")

    print("\n=====================================================")

    try:
        from playsound import playsound

        sound_path = str(Path(__file__).resolve().parent.parent.parent / "marimba_bloop.mp3")
        if os.path.exists(sound_path):
            playsound(sound_path)
        else:
            print(f"Sound file not found at {sound_path}")
    except ImportError:
        print("\nplaysound module not installed. Run 'pip install playsound==1.2.2' to hear the completion sound.")
    except Exception as e:
        print(f"\nCould not play sound: {e}")


if __name__ == "__main__":
    main()
