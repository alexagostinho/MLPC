"""
CatBoost (GPU) multi-label sound-event classifier for the MLPC task.

One binary CatBoost per class (One-vs-Rest), trained on the raw aggregated
features. Trees need no standardization, so this loads the .npz features
directly. Key choices motivated by the EDA:
  - auto_class_weights="Balanced"  → handles the ~25x class imbalance
  - per-class threshold tuning on val → optimizes macro-F1 (not the 0.5 default)
  - collector-level split            → prevents information leakage

Run in the `qsar_torch` conda env (has catboost + CUDA).
"""
import os
import glob
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(DATA_DIR, "audio_features")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
OUT_DIR = os.path.join(DATA_DIR, "models_catboost")

SEED = 42
OVERLAP_THRESH = 0.5
AGREEMENT_THRESH = 0.5

#mean+std feature set (trees ignore scale; melspect block is redundant but
#CatBoost handles that for free, so we keep the full set)
FEATURE_KEYS = [
    "mfcc_mean", "mfcc_std", "mfcc_d_mean", "mfcc_d_std", "mfcc_d2_mean", "mfcc_d2_std",
    "melspect_mean", "melspect_std", "zcr_mean", "zcr_std", "flux_mean", "flux_std",
    "flatness_mean", "flatness_std", "centroid_mean", "centroid_std",
    "bandwidth_mean", "bandwidth_std", "contrast_mean", "contrast_std",
    "rolloff_low_mean", "rolloff_low_std", "rolloff_high_mean", "rolloff_high_std",
    "energy_mean", "energy_std", "power_mean", "power_std",
]

CB_PARAMS = dict(
    iterations=1500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    auto_class_weights="Balanced",
    loss_function="Logloss",
    eval_metric="AUC",            # threshold-free → safe for early stopping
    early_stopping_rounds=50,
    task_type="GPU",
    devices="0",
    random_seed=SEED,
    verbose=0,
)


def aggregate_labels(annotations):
    """Majority vote: [T, C, A] -> binary [T, C]."""
    binary = (annotations >= OVERLAP_THRESH).astype(np.float32)
    return (binary.mean(axis=2) >= AGREEMENT_THRESH).astype(np.int32)


def load_dataset():
    """Load all files into one feature matrix + label matrix + per-segment collector id."""
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
        y_file = aggregate_labels(d["annotations"])
        cid = fname2collector.get(fname, "unknown")

        X_parts.append(X_file)
        y_parts.append(y_file)
        collectors.extend([cid] * X_file.shape[0])

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    collectors = np.array(collectors)
    print(f"Loaded {len(npz_files)} files in {time.time()-t0:.1f}s: "
          f"X={X.shape}, y={y.shape}, {len(np.unique(collectors))} collectors")
    return X, y, collectors, class_names


def collector_split(X, collectors, seed=SEED):
    """70/15/15 split with no collector appearing in two splits."""
    idx = np.arange(len(X))
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=collectors))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_rel, test_rel = next(gss2.split(temp_idx, groups=collectors[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    #leakage check
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        sa = set(collectors[{"train": train_idx, "val": val_idx, "test": test_idx}[a]])
        sb = set(collectors[{"train": train_idx, "val": val_idx, "test": test_idx}[b]])
        assert not (sa & sb), f"Collector leakage between {a} and {b}!"
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
          f"({len(train_idx)/len(X):.0%}/{len(val_idx)/len(X):.0%}/{len(test_idx)/len(X):.0%})")
    return train_idx, val_idx, test_idx


def best_threshold(y_true, probs):
    """Threshold in (0,1) maximizing F1 for one class, searched on a grid."""
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(f1s))
    return grid[best], f1s[best]


def stratified_random_baseline(y_test, y_train, seed=SEED):
    rng = np.random.RandomState(seed)
    freqs = y_train.mean(axis=0)
    pred = np.zeros_like(y_test)
    for c in range(y_test.shape[1]):
        pred[:, c] = rng.binomial(1, freqs[c], size=y_test.shape[0])
    return pred


def main():
    print("=" * 70)
    print("CatBoost (GPU) — multi-label sound event classification")
    print("=" * 70)

    X, y, collectors, class_names = load_dataset()
    C = len(class_names)
    train_idx, val_idx, test_idx = collector_split(X, collectors)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    os.makedirs(OUT_DIR, exist_ok=True)

    proba_va = np.zeros((len(X_va), C), dtype=np.float32)
    proba_te = np.zeros((len(X_te), C), dtype=np.float32)
    thresholds = np.full(C, 0.5)

    print("\n── Training one CatBoost per class (GPU) ──")
    print(f"{'Class':<28}{'pos%':>7}{'val_AUC':>9}{'thr':>7}{'val_F1':>9}{'iters':>7}{'time':>7}")
    print("-" * 74)
    t_all = time.time()
    for c in range(C):
        t0 = time.time()
        clf = CatBoostClassifier(**CB_PARAMS)
        clf.fit(X_tr, y_tr[:, c], eval_set=(X_va, y_va[:, c]))

        p_va = clf.predict_proba(X_va)[:, 1]
        p_te = clf.predict_proba(X_te)[:, 1]
        proba_va[:, c] = p_va
        proba_te[:, c] = p_te

        thr, f1_va = best_threshold(y_va[:, c], p_va)
        thresholds[c] = thr

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_va[:, c], p_va) if len(np.unique(y_va[:, c])) > 1 else float("nan")
        clf.save_model(os.path.join(OUT_DIR, f"{class_names[c]}.cbm"))

        print(f"{class_names[c]:<28}{y_tr[:, c].mean()*100:>6.1f}%{auc:>9.4f}"
              f"{thr:>7.2f}{f1_va:>9.4f}{clf.tree_count_:>7}{time.time()-t0:>6.1f}s")
    print("-" * 74)
    print(f"Total training time: {time.time()-t_all:.1f}s")

    #predictions with tuned vs default thresholds
    pred_te_tuned = (proba_te >= thresholds[None, :]).astype(int)
    pred_te_05 = (proba_te >= 0.5).astype(int)
    pred_te_base = stratified_random_baseline(y_te, y_tr)

    def report(name, pred):
        return (name, f1_score(y_te, pred, average="macro", zero_division=0),
                f1_score(y_te, pred, average="micro", zero_division=0))

    rows = [
        report("Baseline (random)", pred_te_base),
        report("CatBoost @0.5", pred_te_05),
        report("CatBoost tuned-thr", pred_te_tuned),
    ]

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"{'Model':<24}{'F1_macro':>12}{'F1_micro':>12}")
    print("-" * 48)
    for name, fm, fmi in rows:
        print(f"{name:<24}{fm:>12.4f}{fmi:>12.4f}")

    print("\nPer-class F1 (tuned thresholds, test set):")
    f1_pc = f1_score(y_te, pred_te_tuned, average=None, zero_division=0)
    order = np.argsort(f1_pc)[::-1]
    print(f"{'Class':<28}{'F1':>8}{'thr':>7}")
    print("-" * 43)
    for i in order:
        print(f"{class_names[i]:<28}{f1_pc[i]:>8.4f}{thresholds[i]:>7.2f}")

    np.savez(os.path.join(OUT_DIR, "test_predictions.npz"),
             y_test=y_te, proba=proba_te, thresholds=thresholds,
             class_names=np.array(class_names))
    print(f"\nSaved models + predictions to {OUT_DIR}/")


if __name__ == "__main__":
    main()
