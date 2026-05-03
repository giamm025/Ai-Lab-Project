import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import PROCESSED_DIR, TARGET_WORDS, LABEL_MAP
from dataset import SignLanguageDataset
from model import SignLanguageLSTM

# ============================================================
#   IPERPARAMETRI  –  modifica qui se necessario
# ============================================================
EPOCHS        = 30
BATCH_SIZE    = 8
LEARNING_RATE = 0.001
HIDDEN_SIZE   = 128
TRAIN_SPLIT   = 0.8        # 80% training, 20% test

INPUT_SIZE    = 126        # SOLO MANI: 63 coord mano sx + 63 coord mano dx
NUM_CLASSES   = len(TARGET_WORDS)

MODELS_DIR    = '../models/'
MODEL_NAME    = 'model_hands_only.pth'
CM_NAME       = 'confusion_matrix_hands_only.png'
# ============================================================


def train():

    # ------------------------------------------------------------------ DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo in uso: {device}")

    # ---------------------------------------------------------------- DATASET
    print("\nCaricamento del Dataset...")
    full_dataset = SignLanguageDataset(PROCESSED_DIR)
    total = len(full_dataset)

    train_size = int(total * TRAIN_SPLIT)
    test_size  = total - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    print(f"Video totali: {total}  |  Train: {train_size}  |  Test: {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ------------------------------------------------------------------ MODELLO
    # Il Dataset restituisce sempre 1530 coordinate (mani + volto).
    # Per il test "Solo Mani" prendiamo solo i primi 126 numeri nel forward pass.
    model = SignLanguageLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ----------------------------------------------------------------- TRAINING
    print(f"\n{'='*55}")
    print(f"  ADDESTRAMENTO  –  SOLO MANI  (input_size={INPUT_SIZE})")
    print(f"{'='*55}")

    train_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for data, labels in train_loader:
            # ---------- tagliamo le prime 126 coordinate (solo mani) ----------
            data   = data[:, :, :INPUT_SIZE].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch:>3}/{EPOCHS}]  Loss: {avg_loss:.4f}")

    # ---------------------------------------------------------------- VALUTAZIONE
    print(f"\n{'='*55}")
    print("  VALUTAZIONE SUL TEST SET")
    print(f"{'='*55}")

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data   = data[:, :, :INPUT_SIZE].to(device)
            labels = labels.to(device)

            outputs    = model(data)
            _, preds   = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    print(f"\nAccuratezza (Solo Mani): {accuracy:.2f}%")

    # ---------------------------------------------------------- CONFUSION MATRIX
    label_names = list(LABEL_MAP.keys())
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix – Solo Mani\nAccuratezza: {accuracy:.2f}%', fontsize=13, pad=15)
    plt.tight_layout()

    os.makedirs(MODELS_DIR, exist_ok=True)
    cm_path = os.path.join(MODELS_DIR, CM_NAME)
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix salvata in: {cm_path}")

    # --------------------------------------------------------------- SALVATAGGIO
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size':       INPUT_SIZE,
        'hidden_size':      HIDDEN_SIZE,
        'num_classes':      NUM_CLASSES,
        'accuracy':         accuracy,
        'epochs':           EPOCHS,
    }, model_path)
    print(f"Modello salvato in: {model_path}")
    print("\n-> Ricordati di salvare labels.json insieme al modello!")


if __name__ == '__main__':
    train()