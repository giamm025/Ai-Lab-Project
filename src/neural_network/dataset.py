import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from config import PROCESSED_DIR, LABEL_MAP, GARBAGE_CLASS, UNKNOWN_TRAIN_VAL_POOL, UNKNOWN_TEST_POOL


class SignLanguageDataset(Dataset):

    def __init__(self, data_dir, exclude_augmented=False):
        self.data_dir = data_dir
        self.exclude_augmented = exclude_augmented
        self.filenames = []
        self.labels = []
        self.max_frames = 0

        for filename in os.listdir(data_dir):
            if filename.endswith(".npy"):
                if self.exclude_augmented and "_aug_" in filename:
                    continue

                word = filename.split("_")[0]

                # Se è un video di calderone, forziamo l'etichetta GARBAGE_CLASS ("unknown")
                if "_garbage_" in filename:
                    self.filenames.append(filename)
                    self.labels.append(LABEL_MAP[GARBAGE_CLASS])

                # Altrimenti comportamento normale per i TARGET_SIGNS
                elif word in LABEL_MAP:
                    self.filenames.append(filename)
                    self.labels.append(LABEL_MAP[word])

        self.calculate_max_frames(self.filenames, self.data_dir)

    def calculate_max_frames(self, filenames, data_dir):
        print("Calcolo la lunghezza massima dei video per il padding...")
        for filename in filenames:
            filepath = os.path.join(data_dir, filename)
            data = np.load(filepath)
            if data.shape[0] > self.max_frames:
                self.max_frames = data.shape[0]
        print(f"-> Il video più lungo dura {self.max_frames} frame. Uso questo valore per il padding!")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.data_dir, filename)
        label = self.labels[idx]
        data = np.load(filepath)
        real_len = data.shape[0]

        # --------------------------------------------------- PADDING ---------------------------------------------------
        seq_len = data.shape[0]
        padding = np.zeros((self.max_frames - seq_len, data.shape[1]))
        data = np.vstack((data, padding))
        # ---------------------------------------------------------------------------------------------------------------

        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        length_tensor = torch.tensor(real_len, dtype=torch.long)

        return data_tensor, label_tensor, length_tensor


"""
Carica i dati e garantisce che i segni dell'UNKNOWN_TEST_POOL non vengano MAI visti durante il training.
"""


def get_stratified_dataset_splits(data_dir, seed):
    base_dataset = SignLanguageDataset(data_dir, exclude_augmented=True)
    split_generator = torch.Generator().manual_seed(seed)

    label_to_indices = {}
    train_val_garbage_idxs = []
    test_garbage_idxs = []

    # 1. Smistamento preliminare: isoliamo i video garbage da quelli normali
    for idx, filename in enumerate(base_dataset.filenames):
        if "_garbage_" in filename:
            word = filename.split("_")[0]
            if word in UNKNOWN_TRAIN_VAL_POOL:
                train_val_garbage_idxs.append(idx)
            elif word in UNKNOWN_TEST_POOL:
                test_garbage_idxs.append(idx)
        else:
            label = base_dataset.labels[idx]
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    # 2. Split stratificato classico per i TARGET_SIGNS
    for label, idxs in label_to_indices.items():
        idxs_tensor = torch.tensor(idxs)
        shuffled_idxs = idxs_tensor[torch.randperm(len(idxs_tensor), generator=split_generator)].tolist()

        n = len(shuffled_idxs)
        n_train = int(0.65 * n)
        n_val = int(0.15 * n)

        train_indices.extend(shuffled_idxs[:n_train])
        val_indices.extend(shuffled_idxs[n_train : n_train + n_val])
        test_indices.extend(shuffled_idxs[n_train + n_val :])

    # 3. Split Zero-Shot per i file GARBAGE
    if train_val_garbage_idxs:
        tvg_tensor = torch.tensor(train_val_garbage_idxs)
        shuffled_tvg = tvg_tensor[torch.randperm(len(tvg_tensor), generator=split_generator)].tolist()

        # Manteniamo la proporzione Train/Val (65% / 15% equivale a ~81% per il train della pool)
        n_tvg = len(shuffled_tvg)
        n_train_garbage = int((0.65 / 0.80) * n_tvg)

        train_indices.extend(shuffled_tvg[:n_train_garbage])
        val_indices.extend(shuffled_tvg[n_train_garbage:])

    if test_garbage_idxs:
        # Questi video finiscono DIRETTAMENTE al 100% nel test set
        test_indices.extend(test_garbage_idxs)

    training_base = Subset(base_dataset, train_indices)
    val_data = Subset(base_dataset, val_indices)
    test_data = Subset(base_dataset, test_indices)

    training_data = AugmentedTrainingWrapper(training_base, data_dir)
    return training_data, val_data, test_data


# =====================================================================
# WRAPPER AUGMENTATION
# =====================================================================
class AugmentedTrainingWrapper(Dataset):

    def __init__(self, base_train_subdataset, data_dir):
        self.base_train = base_train_subdataset
        self.data_dir = data_dir

        self.allowed_originals = set(base_train_subdataset.dataset.filenames[i] for i in base_train_subdataset.indices)

        self.augmented_filenames = []
        self.augmented_labels = []

        for filename in os.listdir(data_dir):
            if filename.endswith(".npy") and "_aug_" in filename:
                parts = filename.split("_aug_")
                original_name = parts[0] + ".npy"

                if original_name in self.allowed_originals:
                    self.augmented_filenames.append(filename)

                    if "_garbage_" in filename:
                        self.augmented_labels.append(LABEL_MAP[GARBAGE_CLASS])
                    else:
                        word = filename.split("_")[0]
                        self.augmented_labels.append(LABEL_MAP[word])

        print(f"   ↳ Trovati {len(self.augmented_filenames)} file aumentati legali per il Training Set.")

    def __len__(self):
        return len(self.base_train) + len(self.augmented_filenames)

    def __getitem__(self, idx):
        if idx < len(self.base_train):
            return self.base_train[idx]

        aug_idx = idx - len(self.base_train)
        filename = self.augmented_filenames[aug_idx]
        label = self.augmented_labels[aug_idx]

        data = np.load(self.data_dir / filename)
        real_len = data.shape[0]

        padding = np.zeros((self.base_train.dataset.max_frames - real_len, data.shape[1]))
        data = np.vstack((data, padding))

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(real_len, dtype=torch.long)


if __name__ == "__main__":
    print("Testo il PyTorch Dataset...")
    my_dataset = SignLanguageDataset(PROCESSED_DIR)
    print(f"\nVideo totali trovati dal Dataset: {len(my_dataset)}")

    primo_video, prima_etichetta, prima_lunghezza = my_dataset[0]
    print("\nControllo Qualità sul primo video:")
    print(f"Formato del Tensore: {primo_video.shape}")
    print(f"Etichetta (Numero della parola): {prima_etichetta.item()}")
    print(f"Lunghezza Reale pre-padding: {prima_lunghezza.item()}")

    if primo_video.shape[0] == my_dataset.max_frames and primo_video.shape[1] == 402:
        print(f"\n-> GRANDIOSO! Il padding dinamico funziona. Il video è stato forzato a {my_dataset.max_frames} frame esatti.")
