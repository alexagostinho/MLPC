"""
CatBoost (GPU) multi-label sound-event classifier for the MLPC task.

One binary CatBoost per class (One-vs-Rest), trained on the raw aggregated
features. Trees need no standardization, so this loads the .npz features
directly. Key choices motivated by the EDA:
  - auto_class_weights="Balanced"  → handles the ~25x class imbalance
  - per-class threshold tuning on val → optimizes macro-F1 (not the 0.5 default)
  - collector-level split            → prevents information leakage

Section 3 (Experiments) support:
  --sweep  systematically varies CatBoost's most important hyperparameters
           (depth, learning_rate, l2_leaf_reg) one at a time around the
           defaults, measures validation macro-F1 for each, saves a plot
           (catboost_hyperparams.png) and a results JSON, then selects the
           best value per hyperparameter for the final model.
  The final model is always compared to a no-training stratified-random
  baseline on the test set (requirement 3b).

Run in the `qsar_torch` conda env (has catboost + CUDA):
  python catboost_classifier.py            # final model with default params
  python catboost_classifier.py --sweep    # HP sweep + plot, then final model
"""
import os
import glob
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # headless: just write the PNG
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(DATA_DIR, "audio_features")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
OUT_DIR = os.path.join(DATA_DIR, "models_catboost")
FIG_DIR = os.path.join(DATA_DIR, "figures")        # report figures

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

#── Hyperparameter sweep configuration ──────────────────────────────────────
#The three knobs that matter most for gradient-boosted trees:
#  depth         — tree complexity / interaction order (under- vs overfitting)
#  learning_rate — shrinkage per boosting round (speed vs generalization)
#  l2_leaf_reg   — L2 regularization on leaf values (overfitting control)
#We vary one at a time around the CB_PARAMS defaults (coordinate search), which
#is cheap and gives one clean "performance vs hyperparameter" curve per knob.
SWEEP_GRID = {
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "l2_leaf_reg": [1, 3, 6, 10],
}
#During the sweep we cap iterations lower than the final model: early stopping
#still applies, but this keeps the (grid x 15-class) fit count tractable.
SWEEP_ITERATIONS = 800
#Optionally subsample training rows during the sweep for speed. None = use all.
#If set (e.g. 40000), state this clearly in the report — it only affects the
#relative HP comparison, the final model is always trained on the full split.
SWEEP_SUBSAMPLE = None


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


def train_ovr(params, X_tr, y_tr, X_va, y_va, X_eval=None,
              save_dir=None, class_names=None, verbose_table=False):
    """
    Train one binary CatBoost per class (One-vs-Rest), tune a per-class
    threshold on the validation set, and report validation macro-F1.

    Returns a dict with the validation probabilities, tuned thresholds, the
    validation macro-F1 (used for model selection), and — if X_eval is given —
    the probabilities on that held-out set (e.g. the test set).
    """
    C = y_tr.shape[1]
    proba_va = np.zeros((len(X_va), C), dtype=np.float32)
    proba_ev = np.zeros((len(X_eval), C), dtype=np.float32) if X_eval is not None else None
    thresholds = np.full(C, 0.5)

    if verbose_table:
        print(f"{'Class':<28}{'pos%':>7}{'val_AUC':>9}{'thr':>7}{'val_F1':>9}{'iters':>7}{'time':>7}")
        print("-" * 74)

    for c in range(C):
        t0 = time.time()
        clf = CatBoostClassifier(**params)
        clf.fit(X_tr, y_tr[:, c], eval_set=(X_va, y_va[:, c]))

        p_va = clf.predict_proba(X_va)[:, 1]
        proba_va[:, c] = p_va
        if X_eval is not None:
            proba_ev[:, c] = clf.predict_proba(X_eval)[:, 1]

        thresholds[c], f1_va = best_threshold(y_va[:, c], p_va)

        if save_dir is not None and class_names is not None:
            clf.save_model(os.path.join(save_dir, f"{class_names[c]}.cbm"))

        if verbose_table:
            auc = roc_auc_score(y_va[:, c], p_va) if len(np.unique(y_va[:, c])) > 1 else float("nan")
            name = class_names[c] if class_names is not None else f"class_{c}"
            print(f"{name:<28}{y_tr[:, c].mean()*100:>6.1f}%{auc:>9.4f}"
                  f"{thresholds[c]:>7.2f}{f1_va:>9.4f}{clf.tree_count_:>7}{time.time()-t0:>6.1f}s")

    pred_va = (proba_va >= thresholds[None, :]).astype(int)
    val_macro_f1 = f1_score(y_va, pred_va, average="macro", zero_division=0)
    return dict(proba_va=proba_va, proba_eval=proba_ev,
                thresholds=thresholds, val_macro_f1=val_macro_f1)


def sweep_hyperparams(X_tr, y_tr, X_va, y_va, grid=SWEEP_GRID):
    """
    Coordinate (one-at-a-time) sweep of the most important hyperparameters.
    For each knob, every other knob is held at its CB_PARAMS default; we record
    the validation macro-F1 for each value. Returns:
      results: {hp: {"values": [...], "val_macro_f1": [...], "best": value}}
    """
    #Optional subsample for speed (relative comparison only)
    if SWEEP_SUBSAMPLE is not None and SWEEP_SUBSAMPLE < len(X_tr):
        rng = np.random.RandomState(SEED)
        sel = rng.choice(len(X_tr), size=SWEEP_SUBSAMPLE, replace=False)
        X_tr_s, y_tr_s = X_tr[sel], y_tr[sel]
        print(f"  (sweep uses a {SWEEP_SUBSAMPLE}-row training subsample for speed)")
    else:
        X_tr_s, y_tr_s = X_tr, y_tr

    sweep_params = {**CB_PARAMS, "iterations": SWEEP_ITERATIONS}

    results = {}
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SWEEP (validation macro-F1, one knob at a time)")
    print("=" * 70)
    for hp, values in grid.items():
        print(f"\n── {hp} (others at default: "
              + ", ".join(f"{k}={CB_PARAMS[k]}" for k in grid if k != hp) + ") ──")
        f1s = []
        for v in values:
            t0 = time.time()
            params = {**sweep_params, hp: v}
            r = train_ovr(params, X_tr_s, y_tr_s, X_va, y_va)
            f1s.append(r["val_macro_f1"])
            print(f"  {hp}={v:<6} val_macro_F1={r['val_macro_f1']:.4f}  ({time.time()-t0:.1f}s)")
        best = values[int(np.argmax(f1s))]
        results[hp] = {"values": list(values), "val_macro_f1": f1s, "best": best}
        print(f"  → best {hp} = {best} (val_macro_F1={max(f1s):.4f})")
    return results


def plot_sweep(results, out_path):
    """One subplot per hyperparameter: val macro-F1 vs value, best value starred."""
    hps = list(results.keys())
    fig, axes = plt.subplots(1, len(hps), figsize=(5 * len(hps), 4.2))
    if len(hps) == 1:
        axes = [axes]
    for ax, hp in zip(axes, hps):
        vals = results[hp]["values"]
        f1s = results[hp]["val_macro_f1"]
        x = np.arange(len(vals))                  # categorical x for even spacing
        ax.plot(x, f1s, "o-", color="steelblue")
        bi = int(np.argmax(f1s))
        ax.plot(x[bi], f1s[bi], "*", ms=18, color="crimson",
                label=f"best = {vals[bi]}")
        ax.set_xticks(x)
        ax.set_xticklabels(vals)
        ax.set_xlabel(hp)
        ax.set_ylabel("validation macro-F1")
        ax.set_title(f"CatBoost: effect of {hp}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("CatBoost hyperparameter sweep (coordinate search, val macro-F1)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved hyperparameter plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true",
                        help="run the hyperparameter sweep + plot before the final model")
    args = parser.parse_args()

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

    #── Section 3a: systematic hyperparameter variation + visualization ──
    final_params = dict(CB_PARAMS)
    if args.sweep:
        sweep_results = sweep_hyperparams(X_tr, y_tr, X_va, y_va)
        os.makedirs(FIG_DIR, exist_ok=True)
        plot_sweep(sweep_results, os.path.join(FIG_DIR, "catboost_hyperparams.png"))

        #select the best value per knob (coordinate selection) for the final model
        for hp, r in sweep_results.items():
            final_params[hp] = r["best"]
        with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
            json.dump({"sweep": sweep_results, "selected": {
                hp: sweep_results[hp]["best"] for hp in sweep_results},
                "fixed": {k: CB_PARAMS[k] for k in ("iterations", "learning_rate",
                          "depth", "l2_leaf_reg")}}, f, indent=2)
        print("\nSelected hyperparameters for final model: "
              + ", ".join(f"{hp}={final_params[hp]}" for hp in sweep_results))

    #── Final model: train on full train split, evaluate on test ──
    print("\n── Training final CatBoost per class (GPU) ──")
    print("Params: " + ", ".join(f"{k}={final_params[k]}"
          for k in ("iterations", "learning_rate", "depth", "l2_leaf_reg")))
    t_all = time.time()
    res = train_ovr(final_params, X_tr, y_tr, X_va, y_va, X_eval=X_te,
                    save_dir=OUT_DIR, class_names=class_names, verbose_table=True)
    print("-" * 74)
    print(f"Total training time: {time.time()-t_all:.1f}s")

    proba_te = res["proba_eval"]
    thresholds = res["thresholds"]

    #predictions with tuned vs default thresholds vs baseline
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
