import subprocess
import re
import sys

# ==========================================
# COSTANTI DI CONFIGURAZIONE
# ==========================================
N_RUNS = 3  # Quante volte ripetere l'addestramento (N)
M_EPOCHS = 20  # Numero di epoche per ogni addestramento (M)
MODALITA_LIST = ["SOLO_MANI", "MANI_VOLTO"]  # Entrambe le modalità da testare


def main():
    print("=====================================================")
    print(
        f"🚀 INIZIO BULK TRAINING : {N_RUNS} run da {M_EPOCHS} epoche per ogni modalità"
    )
    print(f"Modalità in test: {', '.join(MODALITA_LIST)}")
    print("=====================================================\n")

    results = {mod: [] for mod in MODALITA_LIST}

    for mod in MODALITA_LIST:
        print(f"=====================================================")
        print(f"🔍 TESTANDO LA MODALITÀ: {mod}")
        print(f"=====================================================\n")

        for run in range(1, N_RUNS + 1):
            print(f"🔄 [{mod}] Esecuzione Addestramento {run}/{N_RUNS} in corso...")

            # Lancia train.py come sottoprocesso usando lo stesso eseguibile Python (quello dell'ambiente virtuale)
            cmd = [
                sys.executable,
                "train.py",
                "--modalita",
                mod,
                "--epochs",
                str(M_EPOCHS),
            ]

            # Use errors="replace" so we don't crash when reading problematic characters
            process = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )

            if process.returncode != 0:
                print(f"❌ Errore durante la run {run}. Output di errore:")
                print(process.stderr)
                continue

            output = process.stdout

            # Estraiamo con le espressioni regolari l'ultima accuratezza e loss stampate
            train_accs = re.findall(
                r"---> Accuratezza Finale Epoca \(Train\):\s*([0-9.]+)", output
            )
            test_accs = re.findall(
                r"\*\*\* ACCURATEZZA DI TEST FINALE:\s*([0-9.]+)", output
            )
            losses = re.findall(r"Loss:\s*([0-9.]+)", output)

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
                print(
                    f"  ✅ [{mod}] Run {run} completata: Loss = {final_loss:.4f} | Train Acc = {final_train:.4f} | Test Acc = {final_test:.4f}\n"
                )
            else:
                print(
                    f"  ⚠️ [{mod}] Run {run} completata, ma non sono riuscito a leggere tutte le metriche dal log.\n"
                )

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
            print(
                f"  Run {res['run']:02d} -> Loss: {res['loss']:.4f} | Train: {res['train_acc']:.4f} | Test: {res['test_acc']:.4f}"
            )

        print("  ---------------------------------------------------")
        avg_loss = sum(r["loss"] for r in mod_results) / len(mod_results)
        avg_train = sum(r["train_acc"] for r in mod_results) / len(mod_results)
        avg_test = sum(r["test_acc"] for r in mod_results) / len(mod_results)
        print(f"  🎯 MEDIA LOSS           : {avg_loss:.4f}")
        print(f"  🎯 MEDIA TRAIN ACCURACY : {avg_train:.4f}")
        print(f"  🎯 MEDIA TEST ACCURACY  : {avg_test:.4f}")

    print("\n=====================================================")


if __name__ == "__main__":
    main()
