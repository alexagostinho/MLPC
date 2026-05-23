"""
Improved CatBoost classifier — adds two things over catboost_classifier.py:
  (1) Temporal context: each segment's feature vector is augmented with the
      features of its +/- WINDOW neighbors, built WITHIN each recording so it
      never mixes segments across files (edges padded by replication).
  (2) Full aggregation set: mean + std + min + max (960 base features instead
      of 480). The min/max capture transient peaks that mean/std smooth away.

Compares against the previous 480-feature mean+std model on the same split.
Run in `qsar_torch` env.
"""
import os
import glob
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(DATA_DIR, "audio_features")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
OUT_DIR = os.path.join(DATA_DIR, "models_catboost_improved")

SEED = 42
OVERLAP_THRESH = 0.5
AGREEMENT_THRESH = 0.5
WINDOW = 2                      # temporal context: +/- WINDOW neighbor segments

BASE_NAMES = ["mfcc", "mfcc_d", "mfcc_d2", "melspect", "zcr", "flux", "flatness",
              "centroid", "bandwidth", "contrast", "rolloff_low", "rolloff_high",
              "energy", "power"]
AGGS = ["mean", "std", "min", "max"]            # (2) full aggregation set
FEATURE_KEYS = [f"{b}_{a}" for b in BASE_NAMES for a in AGGS]

CB_PARAMS = dict(
    iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    auto_class_weights="Balanced", loss_function="Logloss", eval_metric="AUC",
    early_stopping_rounds=50, task_type="GPU", devices="0",
    random_seed=SEED, verbose=0,
)


def aggregate_labels(annotations):
    binary = (annotations >= OVERLAP_THRESH).astype(np.float32)
    return (binary.mean(axis=2) >= AGREEMENT_THRESH).astype(np.int32)


def add_temporal_context(X_file, window):
    """X_file [T, D] -> [T, D*(2*window+1)] by stacking shifted (edge-padded) copies."""
    if window == 0:
        return X_file
    T = X_file.shape[0]
    shifts = []
    for off in range(-window, window + 1):
        idx = np.clip(np.arange(T) + off, 0, T - 1)   # replicate edges
        shifts.append(X_file[idx])
    return np.concatenate(shifts, axis=1)


def load_dataset(window):
    meta = pd.read_csv(META_PATH)
    fname2collector = dict(zip(
        meta["filename"].str.replace(".wav", "", regex=False), meta["collector_id"]))

    npz_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npz")))
    X_parts, y_parts, collectors = [], [], []
    class_names = None
    t0 = time.time()
    for path in npz_files:
        fname = os.path.splitext(os.path.basename(path))[0]
        d = np.load(path, allow_pickle=True)
        if class_names is None:
            class_names = list(d["class_names"])
        feats = [d[k] if d[k].ndim > 1 else d[k][:, None] for k in FEATURE_KEYS]
        X_file = np.concatenate(feats, axis=1).astype(np.float32)
        X_file = add_temporal_context(X_file, window)          # (1) temporal context
        X_parts.append(X_file)
        y_parts.append(aggregate_labels(d["annotations"]))
        collectors.extend([fname2collector.get(fname, "unknown")] * X_file.shape[0])

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    collectors = np.array(collectors)
    print(f"Loaded {len(npz_files)} files in {time.time()-t0:.1f}s: X={X.shape}, y={y.shape} "
          f"(window=+/-{window}, {len(FEATURE_KEYS)} base feats x {2*window+1} frames)")
    return X, y, collectors, class_names


def collector_split(X, collectors, seed=SEED):
    idx = np.arange(len(X))
    tr, tmp = next(GroupShuffleSplit(1, train_size=0.70, random_state=seed).split(idx, groups=collectors))
    vr, ter = next(GroupShuffleSplit(1, train_size=0.50, random_state=seed).split(tmp, groups=collectors[tmp]))
    va, te = tmp[vr], tmp[ter]
    for a, b, na, nb in [(tr, va, "train", "val"), (tr, te, "train", "test"), (va, te, "val", "test")]:
        assert not (set(collectors[a]) & set(collectors[b])), f"leak {na}/{nb}"
    print(f"Split: train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te


def best_threshold(y_true, probs):
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in grid]
    b = int(np.argmax(f1s))
    return grid[b], f1s[b]


def stratified_random_baseline(y_test, y_train, seed=SEED):
    """No-training baseline: predict 1 per class with prob = train class frequency."""
    rng = np.random.RandomState(seed)
    freqs = y_train.mean(axis=0)
    pred = np.zeros_like(y_test)
    for c in range(y_test.shape[1]):
        pred[:, c] = rng.binomial(1, freqs[c], size=y_test.shape[0])
    return pred


def main():
    print("=" * 72)
    print(f"Improved CatBoost — full aggregations + temporal context (+/-{WINDOW})")
    print("=" * 72)
    X, y, collectors, class_names = load_dataset(WINDOW)
    C = len(class_names)
    tr, va, te = collector_split(X, collectors)
    X_tr, y_tr, X_va, y_va, X_te, y_te = X[tr], y[tr], X[va], y[va], X[te], y[te]
    os.makedirs(OUT_DIR, exist_ok=True)

    proba_te = np.zeros((len(X_te), C), dtype=np.float32)
    thr = np.full(C, 0.5)
    print("\n── Training one CatBoost per class (GPU) ──")
    print(f"{'Class':<28}{'val_AUC':>9}{'thr':>7}{'val_F1':>9}{'iters':>7}{'time':>7}")
    print("-" * 67)
    t_all = time.time()
    for c in range(C):
        t0 = time.time()
        clf = CatBoostClassifier(**CB_PARAMS)
        clf.fit(X_tr, y_tr[:, c], eval_set=(X_va, y_va[:, c]))
        p_va = clf.predict_proba(X_va)[:, 1]
        proba_te[:, c] = clf.predict_proba(X_te)[:, 1]
        thr[c], f1_va = best_threshold(y_va[:, c], p_va)
        auc = roc_auc_score(y_va[:, c], p_va) if len(np.unique(y_va[:, c])) > 1 else float("nan")
        clf.save_model(os.path.join(OUT_DIR, f"{class_names[c]}.cbm"))
        print(f"{class_names[c]:<28}{auc:>9.4f}{thr[c]:>7.2f}{f1_va:>9.4f}{clf.tree_count_:>7}{time.time()-t0:>6.1f}s")
    print("-" * 67)
    print(f"Total training time: {time.time()-t_all:.1f}s")

    pred_tuned = (proba_te >= thr[None, :]).astype(int)
    fm = f1_score(y_te, pred_tuned, average="macro", zero_division=0)
    fmi = f1_score(y_te, pred_tuned, average="micro", zero_division=0)

    #no-training baseline on the same test set (requirement 3b)
    pred_base = stratified_random_baseline(y_te, y_tr)
    base_fm = f1_score(y_te, pred_base, average="macro", zero_division=0)
    base_fmi = f1_score(y_te, pred_base, average="micro", zero_division=0)

    print("\n" + "=" * 72)
    print("TEST RESULTS (tuned thresholds)")
    print("=" * 72)
    print(f"{'Model':<32}{'F1_macro':>12}{'F1_micro':>12}")
    print("-" * 56)
    print(f"{'Baseline (random)':<32}{base_fm:>12.4f}{base_fmi:>12.4f}")
    print(f"{'CatBoost improved (tuned-thr)':<32}{fm:>12.4f}{fmi:>12.4f}")
    print(f"  (previous mean+std, no-context model: F1_macro = 0.5384)")

    f1_pc = f1_score(y_te, pred_tuned, average=None, zero_division=0)
    #previous per-class F1 for direct comparison
    prev = {"running_water":0.7795,"keyboard_typing":0.7549,"vacuum_cleaner":0.7259,
            "cutlery_dishes":0.6364,"microwave":0.6282,"keychain":0.6274,"phone_ringing":0.6196,
            "toilet_flushing":0.6103,"footsteps":0.5735,"coffee_machine":0.5668,
            "door_open_close":0.3844,"bell_ringing":0.3825,"window_open_close":0.3255,
            "wardrobe_drawer_open_close":0.2716,"light_switch":0.1893}
    print("\nPer-class F1 (improved vs previous):")
    print(f"{'Class':<28}{'new':>8}{'prev':>8}{'delta':>8}")
    print("-" * 52)
    order = np.argsort(f1_pc)[::-1]
    for i in order:
        p = prev.get(class_names[i], float("nan"))
        print(f"{class_names[i]:<28}{f1_pc[i]:>8.4f}{p:>8.4f}{f1_pc[i]-p:>+8.4f}")

    np.savez(os.path.join(OUT_DIR, "test_predictions.npz"),
             y_test=y_te, proba=proba_te, thresholds=thr, class_names=np.array(class_names))
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
