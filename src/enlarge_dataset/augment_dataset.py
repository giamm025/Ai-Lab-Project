"""
augment_dataset.py
==================
Data augmentation for MediaPipe keypoint sequences (.npy files).
Balances the dataset so every class reaches TARGET_COUNT samples,
within a strict 10% margin of each other.

Drop this file into your src/ folder and run it from there:
    python augment_dataset.py

Feature layout per frame
------------------------
SOLO_MANI  (126 features):  [0:63]   left hand   |  [63:126]  right hand
MANI_VOLTO (402 features):  [0:63]   left hand   |  [63:126]  right hand
                             [126:402] face NMMs
"""

import sys
import argparse
import random
from pathlib import Path

import numpy as np

# ── locate config.py (one directory up from src/) ───────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, TARGET_WORDS

# ────────────────────────── HYPER-PARAMETERS ────────────────────────────────

TARGET_COUNT   = 300    # desired number of samples per class
MARGIN         = 0.10   # ±10 % balance tolerance  (for the final report)
RANDOM_SEED    = 42

# Spatial jitter — σ of Gaussian noise added to every coordinate
JITTER_STD     = 0.005

# Temporal drop — fraction of frames to remove
TDROP_MIN      = 0.10
TDROP_MAX      = 0.20

# Scaling — uniform scale factor range
SCALE_MIN      = 0.90
SCALE_MAX      = 1.10

# ────────────────────────── FEATURE-LAYOUT CONSTANTS ────────────────────────

LH_START, LH_END = 0,   63    # left-hand  block
RH_START, RH_END = 63, 126    # right-hand block

# ────────────────────────────────────────────────────────────────────────────


def jitter(sequence: np.ndarray, std: float = JITTER_STD) -> np.ndarray:
    """
    Spatial jittering: add zero-mean Gaussian noise to every coordinate.

    Shape: (T, F)  →  (T, F)
    This simulates natural hand-tremor and sensor noise. The standard
    deviation is intentionally tiny (< 1 % of the normalised [0,1] space)
    so the shape of the sign is preserved.
    """

    # siccome nella sequenza possono esserci coordinate assenti (es. le mani non sono nell'inquadratura) applichiamo il jitter 
    # SOLO ai punti che sono presenti (non zero) per evitare di far apparire "rumore" dove in realta non dovrebbe esserci nulla
    seq = sequence.copy()
    T, F = seq.shape
    coords = seq.reshape(-1, 3)
    
    # troviamo i punti che NON sono zeri 
    valid_mask = np.any(coords != 0, axis=1)
    
    # applichiamo il rumore SOLO ai punti validi
    noise = np.random.normal(0.0, std, size=coords.shape).astype(seq.dtype)
    coords[valid_mask] += noise[valid_mask]
    return coords.reshape(T, F)


def temporal_drop(sequence: np.ndarray,
                  drop_min: float = TDROP_MIN,
                  drop_max: float = TDROP_MAX) -> np.ndarray:
    """
    Temporal frame dropping: randomly remove 10–20 % of frames.

    Shape: (T, F)  →  (T', F),  T' ≈ 0.80–0.90 × T
    Simulates a signer who executes the sign slightly faster than the
    original recording.  Frames are removed uniformly at random so the
    result is a realistic sub-sampled trajectory, not a truncation.

    Guard: if only 1 frame remains we return the original to avoid
    degenerate sequences.
    """
    T = sequence.shape[0]
    n_drop = int(round(T * random.uniform(drop_min, drop_max)))
    n_keep = T - n_drop

    # need at least 1 frame after dropping
    n_keep = max(1, n_keep)

    # choose which frame indices to keep (sorted to preserve time order)
    keep_idx = sorted(random.sample(range(T), n_keep))
    return sequence[keep_idx]


def scale(sequence: np.ndarray,
          scale_min: float = SCALE_MIN,
          scale_max: float = SCALE_MAX) -> np.ndarray:
    """
    Uniform spatial scaling: multiply every coordinate by a random factor.

    Shape: (T, F)  →  (T, F)
    Simulates the signer being closer (factor > 1) or further (factor < 1)
    from the camera.  Because MediaPipe outputs normalised coordinates in
    [0, 1], mild over/under-shoot is acceptable.
    """

    # Come per il jitter, applichiamo la scala SOLO ai punti validi (non zero) per evitare 
    # di far diventare "grandi" i punti che in realta non dovrebbero esserci
    seq = sequence.copy()
    T, F = seq.shape
    factor = random.uniform(scale_min, scale_max)
    
    for t in range(T):
        coords = seq[t].reshape(-1, 3)
        valid_mask = np.any(coords != 0, axis=1)
        
        if not np.any(valid_mask):
            continue
            
        # Calcoliamo il centro geometrico (centroid) per x e y
        cx = np.mean(coords[valid_mask, 0])
        cy = np.mean(coords[valid_mask, 1])
        
        # Ingrandiamo mantenendo il centro fisso
        coords[valid_mask, 0] = cx + factor * (coords[valid_mask, 0] - cx)
        coords[valid_mask, 1] = cy + factor * (coords[valid_mask, 1] - cy)
        coords[valid_mask, 2] = coords[valid_mask, 2] * factor # la profondità si scala direttamente
        
        seq[t] = coords.flatten()
    return seq


def mirror(sequence: np.ndarray) -> np.ndarray:
    """
    Horizontal mirroring: invert the X coordinate of every landmark AND
    swap the left-hand block with the right-hand block.

    Shape: (T, F)  →  (T, F)

    Why the swap is mandatory
    -------------------------
    MediaPipe labels landmarks from the SUBJECT's perspective.  After a
    horizontal flip the subject's left hand appears on the right side of
    the image, so what was the left-hand block must become the right-hand
    block — otherwise the anatomy is wrong and the model would learn that
    "left hand on right side" is valid, which it is not.

    X inversion formula:  x' = 1 - x
    Each landmark has three consecutive values (x, y, z).
    We invert only x (index 0, 3, 6, …) within the coordinate triples.
    """
    seq = sequence.copy()
    T, F = seq.shape

    # ── 1. invert X coordinate everywhere ───────────────────────────────────
    # X indices are 0, 3, 6, … for every group of 3 (x, y, z)
    x_indices = np.arange(0, F, 3)
    seq[:, x_indices] = 1.0 - seq[:, x_indices]

    # ── 2. swap left-hand block ↔ right-hand block ──────────────────────────
    lh_block = seq[:, LH_START:LH_END].copy()
    rh_block = seq[:, RH_START:RH_END].copy()
    seq[:, LH_START:LH_END] = rh_block
    seq[:, RH_START:RH_END] = lh_block

    return seq


# ────────────────────────── AUGMENTATION PIPELINE ───────────────────────────

def augment(sequence: np.ndarray,
            jitter_std: float = JITTER_STD,
            drop_min: float = TDROP_MIN,
            drop_max: float = TDROP_MAX,
            scale_min: float = SCALE_MIN,
            scale_max: float = SCALE_MAX) -> np.ndarray:
    """
    Apply a random combination of augmentations to one sequence.

    Strategy
    --------
    * Always apply at least ONE technique.
    * Mirror is chosen independently with 50 % probability so it can
      combine freely with the spatial/temporal transforms.
    * From the remaining three techniques (jitter, temporal_drop, scale)
      we pick 1 or 2 at random — applying all three at once can push the
      sample too far from the original distribution.

    Returns a new array; the input is never modified.
    """
    aug = sequence.copy()
    F = aug.shape[1]        # estraiamo il numero di features per frame (126 o 402) per poter applicare la logica del mirror SOLO alla modalità SOLO_MANI (126)

    def do_jitter(seq): return jitter(seq, jitter_std)
    def do_tdrop(seq): return temporal_drop(seq, drop_min, drop_max)
    def do_scale(seq): return scale(seq, scale_min, scale_max)

    augmentations = [do_jitter, do_tdrop, do_scale]

    # ── spatial / temporal augmentations (pick 1 or 2) ──────────────────────
    n_apply = random.randint(1, 2)
    chosen  = random.sample(augmentations, n_apply)
    for fn in chosen:
        aug = fn(aug)

    # ── mirroring (independent 50 % coin flip) ───────────────────────────────
    if F == 126 and random.random() < 0.5:
        aug = mirror(aug)

    return aug


# ────────────────────────── MAIN LOGIC ───────────────────────────────────────

def collect_existing_files(processed_dir: Path) -> dict[str, list[Path]]:
    """
    Scan processed_dir for .npy files and group them by word.

    Only words in TARGET_WORDS are considered.
    Files generated by a previous run of this script (containing '_aug_')
    are intentionally included in the pool so they can be used as source
    material for further augmentation if needed.
    """
    files_by_word: dict[str, list[Path]] = {w: [] for w in TARGET_WORDS}

    for npy_path in sorted(processed_dir.glob("*.npy")):
        word = npy_path.name.split("_")[0]
        if word in files_by_word:
            files_by_word[word].append(npy_path)

    return files_by_word


def next_aug_index(existing: list[Path], word: str) -> int:
    """
    Determine the next available augmentation index for a given word.

    Scans the filenames that already contain '_aug_' and returns
    max_existing_index + 1, starting at 1 if none exist.
    """
    max_idx = 0
    for p in existing:
        if "_aug_" in p.name:
            try:
                # filename format: word_aug_NNN.npy  (NNN may be padded)
                idx = int(p.stem.split("_aug_")[-1])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return max_idx + 1


def validate_feature_dim(sequence: np.ndarray, path: Path) -> bool:
    """
    Accept only arrays whose feature dimension is 126 (SOLO_MANI) or 402
    (MANI_VOLTO).  Warn and skip anything else so the script never crashes
    on stray files.
    """
    if sequence.ndim != 2:
        print(f"  [SKIP] {path.name}  — unexpected ndim={sequence.ndim}")
        return False
    F = sequence.shape[1]
    if F not in (126, 402, 1530):
        print(f"  [SKIP] {path.name}  — unexpected feature dim={F}")
        return False
    return True


def run_augmentation(
    processed_dir: Path = PROCESSED_DIR,
    target_count:  int  = TARGET_COUNT,
    margin:        float = MARGIN,
    seed:          int  = RANDOM_SEED,
    jitter_std:    float = JITTER_STD,
    drop_min:      float = TDROP_MIN,
    drop_max:      float = TDROP_MAX,
    scale_min:     float = SCALE_MIN,
    scale_max:     float = SCALE_MAX,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    print("=" * 62)
    print("  MediaPipe Keypoint Dataset Augmentation")
    print("=" * 62)
    print(f"  Processed dir : {processed_dir}")
    print(f"  Target count  : {target_count} samples / class")
    print(f"  Balance margin: ±{int(margin * 100)} %")
    print(f"  Random seed   : {seed}")
    print()

    # ── 1. inventory ────────────────────────────────────────────────────────
    files_by_word = collect_existing_files(processed_dir)

    print("Current distribution (before augmentation):")
    for word in TARGET_WORDS:
        n = len(files_by_word[word])
        bar = "█" * n
        print(f"  {word:<20}  {n:>4}  {bar}")
    print()

    # ── 2. augment each under-represented class ──────────────────────────────
    total_generated = 0

    for word in TARGET_WORDS:
        existing = files_by_word[word]
        current  = len(existing)
        needed   = max(0, target_count - current)

        if needed == 0:
            print(f"  {word:<20}  already at {current} — skipping.")
            continue

        print(f"  {word:<20}  need {needed} new samples  "
              f"(current {current} → target {target_count})")

        aug_idx = next_aug_index(existing, word)

        for i in range(needed):
            # pick a random source file (sampling WITH replacement so we can
            # generate more samples than the original pool size)
            src_path = random.choice(existing)
            src_seq  = np.load(str(src_path))

            if not validate_feature_dim(src_seq, src_path):
                # try another file next iteration
                continue

            # apply augmentations
            aug_seq = augment(
                src_seq,
                jitter_std=jitter_std,
                drop_min=drop_min,
                drop_max=drop_max,
                scale_min=scale_min,
                scale_max=scale_max
            )

            # extract the original video name
            parent_filename = src_path.stem.split("_aug_")[0]
            
            # build output filename:  word_aug_NNN.npy
            out_name = f"{parent_filename}_aug_{aug_idx:04d}.npy"
            out_path = processed_dir / out_name
            np.save(str(out_path), aug_seq)

            aug_idx      += 1
            total_generated += 1

            if (i + 1) % 20 == 0 or (i + 1) == needed:
                print(f"    … {i + 1}/{needed} saved", end="\r")

        print(f"    {needed}/{needed} saved ✓                    ")

    # ── 3. final summary ─────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  Final distribution (after augmentation):")
    print("=" * 62)

    final_counts = []
    for word in TARGET_WORDS:
        files_by_word = collect_existing_files(processed_dir)   # re-scan
        n = len(files_by_word[word])
        final_counts.append(n)
        bar = "█" * min(n, 60)
        print(f"  {word:<20}  {n:>4}  {bar}")

    if final_counts:
        min_c, max_c = min(final_counts), max(final_counts)
        mean_c       = sum(final_counts) / len(final_counts)
        spread       = (max_c - min_c) / mean_c   # relative spread

        print()
        print(f"  Min : {min_c}   Max : {max_c}   "
              f"Mean : {mean_c:.1f}   Spread : {spread * 100:.1f} %")

        if spread <= margin:
            print(f"\n  ✅  Dataset is balanced within the ±{int(margin*100)} % margin.")
        else:
            print(f"\n  ⚠️   Spread {spread*100:.1f} % exceeds the "
                  f"±{int(margin*100)} % margin — consider raising target_count "
                  f"or investigating the source files.")

    print(f"\n  Total new files generated: {total_generated}")
    print("=" * 62)


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data augmentation for MediaPipe keypoint sequences.")

    arg_configs = [
        {"name": "--processed-dir", "type": Path, "default": PROCESSED_DIR, "help": "Directory containing processed .npy files"},
        {"name": "--target-count", "type": int, "default": TARGET_COUNT, "help": "Desired number of samples per class"},
        {"name": "--margin", "type": float, "default": MARGIN, "help": "Balance tolerance margin"},
        {"name": "--seed", "type": int, "default": RANDOM_SEED, "help": "Random seed"},
        {"name": "--jitter-std", "type": float, "default": JITTER_STD, "help": "Std dev for spatial jitter"},
        {"name": "--tdrop-min", "type": float, "default": TDROP_MIN, "help": "Min fraction for temporal drop"},
        {"name": "--tdrop-max", "type": float, "default": TDROP_MAX, "help": "Max fraction for temporal drop"},
        {"name": "--scale-min", "type": float, "default": SCALE_MIN, "help": "Min scale factor"},
        {"name": "--scale-max", "type": float, "default": SCALE_MAX, "help": "Max scale factor"},
    ]

    for arg in arg_configs:
        name = arg.pop("name")
        parser.add_argument(name, **arg)

    args = parser.parse_args()

    run_augmentation(
        processed_dir=args.processed_dir,
        target_count=args.target_count,
        margin=args.margin,
        seed=args.seed,
        jitter_std=args.jitter_std,
        drop_min=args.tdrop_min,
        drop_max=args.tdrop_max,
        scale_min=args.scale_min,
        scale_max=args.scale_max
    )