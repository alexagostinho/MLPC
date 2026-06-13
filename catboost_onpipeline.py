"""
CatBoost SED system for MLPC 2026 Task 5 — Standalone Script
=============================================================
Replaces the 15 baseline decision trees with 15 CatBoost classifiers,
keeping the same SED inference pipeline (overlapping segments -> whole-second
predictions -> onset/offset intervals -> evaluated with the official script).

Improvements over the baseline:
  (1) CatBoost (gradient-boosted trees) instead of single decision trees.
  (2) auto_class_weights="Balanced" to handle severe class imbalance.
  (3) Per-class probability thresholds tuned on the validation set
      (segment-based macro F1).
  (4) Temporal context: each segment is augmented with features of its
      +/- WINDOW neighbors, built PER RECORDING (edge-padded by replication).

Usage:
    python catboost_sed.py

Make sure this file lives in the SAME folder as evaluate.py from the challenge,
so the imports work.
"""
#IMPORTANT TO HAVE THIS FILE ON THE SAME FOLDER AS EVALUATE.py
import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score

# Import the evaluation helpers from the challenge's evaluate.py
sys.path.insert(0, os.getcwd())
from evaluate import (
    aggregate_ground_truth_annotations,
    build_segment_frame_from_intervals,
    calculate_f1_score,
)

# ──────────────────────────────────────────────
# CONFIG  (adjust these to your machine)
# ──────────────────────────────────────────────
PATH_TO_DATASET = r"E:\MLPC_new dataset" #your dataset path

WINDOW = 2                       # +/- WINDOW neighbor segments for temporal context
SEED = 42
MAX_TRAINING_FILES = None        # None = all training recordings
MAX_TRAINING_SEGMENTS = 80_000   # subsample for speed; set to None to keep all

OUTPUT_CSV = "predictions_hidden_test_catboost.csv"

CB_PARAMS = dict(
    iterations=1500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    auto_class_weights="Balanced",
    loss_function="Logloss",
    eval_metric="AUC",
    early_stopping_rounds=50,
    task_type="GPU",             # change to "CPU" if no GPU available
    devices="0",
    random_seed=SEED,
    verbose=0,
)

# ──────────────────────────────────────────────
# Constants (must match the baseline)
# ──────────────────────────────────────────────
FEATURE_NAMES = [
    "zcr_mean", "zcr_std", "zcr_min", "zcr_max",
    "melspect_mean", "melspect_std", "melspect_min", "melspect_max",
    "mfcc_mean", "mfcc_std", "mfcc_min", "mfcc_max",
    "mfcc_d_mean", "mfcc_d_std", "mfcc_d_min", "mfcc_d_max",
    "mfcc_d2_mean", "mfcc_d2_std", "mfcc_d2_min", "mfcc_d2_max",
    "flux_mean", "flux_std", "flux_min", "flux_max",
    "flatness_mean", "flatness_std", "flatness_min", "flatness_max",
    "centroid_mean", "centroid_std", "centroid_min", "centroid_max",
    "bandwidth_mean", "bandwidth_std", "bandwidth_min", "bandwidth_max",
    "contrast_mean", "contrast_std", "contrast_min", "contrast_max",
    "rolloff_low_mean", "rolloff_low_std", "rolloff_low_min", "rolloff_low_max",
    "rolloff_high_mean", "rolloff_high_std", "rolloff_high_min", "rolloff_high_max",
    "energy_mean", "energy_std", "energy_min", "energy_max",
    "power_mean", "power_std", "power_min", "power_max",
]

CLASS_NAMES = [
    "bell_ringing", "coffee_machine", "cutlery_dishes", "door_open_close",
    "footsteps", "keyboard_typing", "keychain", "light_switch",
    "microwave", "phone_ringing", "running_water", "toilet_flushing",
    "vacuum_cleaner", "wardrobe_drawer_open_close", "window_open_close",
]

SEGMENT_LENGTH = 1.0
HOP_SIZE = 0.5
N_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PATH_TRAIN = os.path.join(PATH_TO_DATASET, "train")
PATH_VAL = os.path.join(PATH_TO_DATASET, "validation")
PATH_TEST = os.path.join(PATH_TO_DATASET, "test")

for p in [PATH_TRAIN, PATH_VAL, PATH_TEST]:
    assert os.path.isdir(p), f"Directory not found: {p}"

rng = np.random.default_rng(SEED)


# ══════════════════════════════════════════════
# DATA LOADING (same logic as baseline)
# ══════════════════════════════════════════════
def build_feature_matrix(data: dict) -> np.ndarray:
    """Concatenate all named features from one .npz dict into a [T, 960] matrix."""
    arrays = []
    for name in FEATURE_NAMES:
        feat = data[name]
        if feat.ndim == 1:
            feat = feat[:, np.newaxis]
        arrays.append(feat.astype(np.float32))
    return np.concatenate(arrays, axis=1)


def get_segment_labels(data: dict) -> np.ndarray:
    """Majority-vote binary labels of shape [T, C] from annotations [T, C, A]."""
    annotations = data["annotations"]
    binary = (annotations > 0).astype(int)
    votes = binary.sum(axis=2)
    n_annot = binary.shape[2]
    return (votes > (n_annot // 2)).astype(int)


# ══════════════════════════════════════════════
# TEMPORAL CONTEXT (per recording, edge-padded)
# ══════════════════════════════════════════════
def add_temporal_context(X_file: np.ndarray, window: int) -> np.ndarray:
    """[T, D] -> [T, D*(2*window+1)] by stacking shifted, edge-padded copies."""
    if window == 0:
        return X_file
    T = X_file.shape[0]
    shifts = []
    for off in range(-window, window + 1):
        idx = np.clip(np.arange(T) + off, 0, T - 1)
        shifts.append(X_file[idx])
    return np.concatenate(shifts, axis=1)


def build_feature_matrix_with_context(data: dict, window: int) -> np.ndarray:
    X = build_feature_matrix(data)
    return add_temporal_context(X, window)


def load_all_segments_with_context(
    file_list: List[str], window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list, Y_list = [], []
    for fp in file_list:
        data = dict(np.load(fp, allow_pickle=True))
        X_list.append(build_feature_matrix_with_context(data, window))
        Y_list.append(get_segment_labels(data))
    return np.vstack(X_list), np.vstack(Y_list)


# ══════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════
def train_catboost_per_class(
    X_tr: np.ndarray, Y_tr: np.ndarray,
    X_va: np.ndarray, Y_va: np.ndarray,
    class_names: List[str], params: dict,
) -> List[CatBoostClassifier]:
    classifiers = []
    print(f"\n── Training {len(class_names)} CatBoost classifiers ──")
    print(f"{'Class':<28}{'val_AUC':>10}{'iters':>8}{'time':>8}")
    print("-" * 54)
    t_all = time.time()
    for c, cname in enumerate(class_names):
        t0 = time.time()
        clf = CatBoostClassifier(**params)
        clf.fit(X_tr, Y_tr[:, c], eval_set=(X_va, Y_va[:, c]))
        p_va = clf.predict_proba(X_va)[:, 1]
        try:
            auc = roc_auc_score(Y_va[:, c], p_va)
        except ValueError:
            auc = float("nan")
        classifiers.append(clf)
        print(f"{cname:<28}{auc:>10.4f}{clf.tree_count_:>8}{time.time()-t0:>7.1f}s")
    print("-" * 54)
    print(f"Total training time: {time.time()-t_all:.1f}s")
    return classifiers


# ══════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════
def predict_proba_recording(
    filepath: str, classifiers: List[CatBoostClassifier], window: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """One recording -> per-class probabilities at whole-second timestamps."""
    data = dict(np.load(filepath, allow_pickle=True))
    start_times_all = data["start_time"]
    X_all = build_feature_matrix_with_context(data, window)
    proba_all = np.column_stack([
        clf.predict_proba(X_all)[:, 1] for clf in classifiers
    ])
    whole = np.isclose(start_times_all % 1.0, 0.0)
    proba = proba_all[whole]
    times = start_times_all[whole]
    fname = os.path.basename(filepath).replace(".npz", ".wav")
    return proba, times, fname


def proba_to_intervals(
    proba: np.ndarray, thresholds: np.ndarray, times: np.ndarray,
    filename: str, class_names: List[str],
) -> List[Dict]:
    """Apply per-class thresholds, merge consecutive active seconds into intervals."""
    binary = (proba >= thresholds[None, :]).astype(int)
    rows = []
    for c_idx, c_name in enumerate(class_names):
        in_event = False
        onset = None
        for t, p in zip(times, binary[:, c_idx]):
            if p == 1 and not in_event:
                onset = float(t)
                in_event = True
            elif p == 0 and in_event:
                rows.append({"filename": filename, "annotation": c_name,
                             "onset": onset, "offset": float(t)})
                in_event = False
        if in_event:
            rows.append({"filename": filename, "annotation": c_name,
                         "onset": onset,
                         "offset": float(times[-1]) + SEGMENT_LENGTH})
    return rows


def generate_predictions_from_cache(
    cache: List[Tuple[np.ndarray, np.ndarray, str]],
    thresholds: np.ndarray, class_names: List[str],
) -> pd.DataFrame:
    """Build prediction DataFrame from cached per-file probabilities."""
    all_rows = []
    for proba, times, fname in cache:
        all_rows.extend(proba_to_intervals(proba, thresholds, times, fname, class_names))
    if not all_rows:
        return pd.DataFrame(columns=["filename", "annotation", "onset", "offset"])
    return pd.DataFrame(all_rows)


def cache_probabilities(
    file_list: List[str], classifiers: List[CatBoostClassifier], window: int,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    cache = []
    for fp in file_list:
        cache.append(predict_proba_recording(fp, classifiers, window))
    return cache


# ══════════════════════════════════════════════
# EVALUATION using the official script
# ══════════════════════════════════════════════
def evaluate_split(
    pred_df: pd.DataFrame, file_list: List[str], ann_df: pd.DataFrame,
) -> Tuple[float, pd.DataFrame]:
    split_filenames = {os.path.basename(f).replace(".npz", ".wav") for f in file_list}
    ann_split = ann_df[ann_df["filename"].isin(split_filenames)].copy()
    gt = aggregate_ground_truth_annotations(ann_split)
    gt_segments = build_segment_frame_from_intervals(gt, name="ground_truth")
    pred_segments = build_segment_frame_from_intervals(pred_df, name="predictions")
    if len(pred_segments) > 0:
        pred_filenames = pred_segments.index.get_level_values("filename")
        pred_segments = pred_segments[pred_filenames.isin(split_filenames)]
    return calculate_f1_score(gt_segments, pred_segments)


# ══════════════════════════════════════════════
# THRESHOLD TUNING (per class, on validation cache)
# ══════════════════════════════════════════════
def tune_thresholds(
    cache_val: List[Tuple[np.ndarray, np.ndarray, str]],
    val_files: List[str], ann_df: pd.DataFrame, class_names: List[str],
    grid: np.ndarray = np.arange(0.20, 0.81, 0.05),
) -> np.ndarray:
    """Sweep one class's threshold at a time (others held at 0.5)."""
    C = len(class_names)
    best_thr = np.full(C, 0.5)

    print(f"\n── Tuning per-class thresholds (grid {grid.min():.2f}..{grid.max():.2f}) ──")
    print(f"{'Class':<28}{'best_thr':>10}{'F1':>10}")
    print("-" * 48)

    for c_idx, c_name in enumerate(class_names):
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            thr = best_thr.copy()
            thr[c_idx] = t
            pred_df = generate_predictions_from_cache(cache_val, thr, class_names)
            _, results = evaluate_split(pred_df, val_files, ann_df)
            row = results[results["annotation"] == c_name]
            f1_c = 0.0 if row.empty else float(row["f1"].values[0])
            if f1_c > best_f1:
                best_f1 = f1_c
                best_t = float(t)
        best_thr[c_idx] = best_t
        print(f"{c_name:<28}{best_t:>10.2f}{best_f1:>10.4f}")
    return best_thr


# ══════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════
def plot_per_class_comparison(results_nht: pd.DataFrame, macro_nht: float):
    baseline_f1 = {
        "bell_ringing": 0.2882, "coffee_machine": 0.3620, "cutlery_dishes": 0.2873,
        "door_open_close": 0.1773, "footsteps": 0.3326, "keyboard_typing": 0.3590,
        "keychain": 0.3211, "light_switch": 0.2337, "microwave": 0.3896,
        "phone_ringing": 0.4848, "running_water": 0.5776, "toilet_flushing": 0.2904,
        "vacuum_cleaner": 0.4697, "wardrobe_drawer_open_close": 0.1203,
        "window_open_close": 0.0611,
    }
    res = results_nht.copy()
    res["baseline_f1"] = res["annotation"].map(baseline_f1)
    res = res.sort_values("annotation").reset_index(drop=True)

    x = np.arange(len(res))
    width = 0.4
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width/2, res["baseline_f1"], width, label="Baseline DT (macro=0.317)",
           color="steelblue")
    ax.bar(x + width/2, res["f1"], width,
           label=f"CatBoost (macro={macro_nht:.3f})", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(res["annotation"], rotation=45, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-Class F1 — Baseline vs CatBoost (Non-Hidden Test)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("catboost_vs_baseline.png", dpi=150)
    plt.show()
    print("Saved: catboost_vs_baseline.png")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("=" * 72)
    print(f"CatBoost SED  (window=+/-{WINDOW}, GPU={CB_PARAMS['task_type']=='GPU'})")
    print("=" * 72)

    # File lists
    train_files = sorted(glob.glob(os.path.join(PATH_TRAIN, "audio_features", "*.npz")))
    val_all     = sorted(glob.glob(os.path.join(PATH_VAL,   "audio_features", "*.npz")))
    test_files  = sorted(glob.glob(os.path.join(PATH_TEST,  "audio_features", "*.npz")))

    # Local val/test split (same logic as baseline: split validation in half)
    val_arr = np.array(val_all)
    perm = np.random.RandomState(SEED).permutation(len(val_arr))
    half = len(val_arr) // 2
    our_val_files  = val_arr[perm[:half]].tolist()
    our_test_files = val_arr[perm[half:]].tolist()

    print(f"  Training:        {len(train_files)} files")
    print(f"  Local val:       {len(our_val_files)} files")
    print(f"  Non-hidden test: {len(our_test_files)} files")
    print(f"  Hidden test:     {len(test_files)} files")

    # Load annotations for validation (covers both our_val and our_test)
    ann_df = pd.read_csv(os.path.join(PATH_VAL, "annotations.csv"))

    # 1. Training data with temporal context
    train_files_used = train_files
    if MAX_TRAINING_FILES is not None:
        train_files_used = rng.choice(
            train_files, size=min(MAX_TRAINING_FILES, len(train_files)),
            replace=False,
        ).tolist()

    print(f"\nLoading {len(train_files_used)} training recordings (context +/-{WINDOW})...")
    t0 = time.time()
    X_train, Y_train = load_all_segments_with_context(train_files_used, WINDOW)
    print(f"  Loaded {X_train.shape[0]} segments x {X_train.shape[1]} features "
          f"({time.time()-t0:.1f}s)")

    if MAX_TRAINING_SEGMENTS and X_train.shape[0] > MAX_TRAINING_SEGMENTS:
        idx = rng.choice(X_train.shape[0], size=MAX_TRAINING_SEGMENTS, replace=False)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
        print(f"  Subsampled to {X_train.shape[0]} segments.")

    # 2. Local validation features (for early stopping)
    print(f"\nLoading {len(our_val_files)} local validation recordings...")
    X_val, Y_val = load_all_segments_with_context(our_val_files, WINDOW)
    print(f"  X_val={X_val.shape}")

    # 3. Train CatBoost per class
    classifiers = train_catboost_per_class(
        X_train, Y_train, X_val, Y_val, CLASS_NAMES, CB_PARAMS,
    )

    # Free training arrays (they can be huge with context)
    del X_train, Y_train, X_val, Y_val

    # 4. Cache validation probabilities once, then tune thresholds
    print(f"\n── Caching validation probabilities ──")
    cache_val = cache_probabilities(our_val_files, classifiers, WINDOW)
    thresholds = tune_thresholds(cache_val, our_val_files, ann_df, CLASS_NAMES)
    print(f"\nFinal thresholds: {dict(zip(CLASS_NAMES, np.round(thresholds, 2)))}")

    # 5. Evaluate on local validation
    print("\n── Local validation evaluation ──")
    pred_val = generate_predictions_from_cache(cache_val, thresholds, CLASS_NAMES)
    macro_val, results_val = evaluate_split(pred_val, our_val_files, ann_df)
    print(f"=== Local Validation Macro F1: {macro_val:.4f} ===")
    print(results_val.to_string(index=False))

    # 6. Evaluate on non-hidden test (one shot)
    print("\n── Non-hidden test evaluation ──")
    cache_nht = cache_probabilities(our_test_files, classifiers, WINDOW)
    pred_nht = generate_predictions_from_cache(cache_nht, thresholds, CLASS_NAMES)
    macro_nht, results_nht = evaluate_split(pred_nht, our_test_files, ann_df)
    print(f"=== Non-Hidden Test Macro F1: {macro_nht:.4f} ===")
    print(results_nht.to_string(index=False))

    # 7. Hidden test predictions (submission CSV)
    print(f"\n── Generating hidden test predictions ({len(test_files)} files) ──")
    cache_hidden = cache_probabilities(test_files, classifiers, WINDOW)
    pred_hidden = generate_predictions_from_cache(cache_hidden, thresholds, CLASS_NAMES)
    pred_hidden.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(pred_hidden)} event intervals to {OUTPUT_CSV}")

    # 8. Plot comparison
    plot_per_class_comparison(results_nht, macro_nht)

    # 9. Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Baseline (Decision Trees)  Non-hidden test Macro F1: 0.3170")
    print(f"  CatBoost (this run)        Non-hidden test Macro F1: {macro_nht:.4f}")
    print(f"  Improvement:                                         "
          f"{'+' if macro_nht > 0.317 else ''}{macro_nht-0.3170:.4f}")
    return classifiers, thresholds, results_nht, pred_hidden


if __name__ == "__main__":
    main()