"""
Linear SVM (One-vs-Rest) multi-label sound-event classifier — the second model
class for the experiments section, deliberately built to be directly comparable
to catboost_classifier.py:

  - same raw aggregated mean+std features (480-d)
  - same collector-level split (GroupShuffleSplit, seed 42) → identical folds
  - same no-training stratified-random baseline
  - same per-class threshold tuning on the validation set to maximize F1

Differences forced by the model:
  - features are z-score standardized (fit on train only); linear SVMs are
    scale-sensitive, trees are not.
  - class_weight="balanced" handles the strong class imbalance.
  - LinearSVC has no probabilities, so thresholds are tuned on the (signed)
    decision-function score instead of a probability.

Section 3a study: sweep the regularization strength C (the single most
important LinearSVC hyperparameter) and visualize validation macro-F1
(svm_hyperparams.png). The best C is used for the final test-set model.

Run in the `qsar_torch` env (sklearn only, CPU):
  python svm_classifier.py
"""
import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

#reuse the exact data/split/baseline logic so the comparison is controlled
from catboost_classifier import (
    load_dataset, collector_split, stratified_random_baseline, DATA_DIR, SEED,
)

OUT_DIR = os.path.join(DATA_DIR, "models_svm")
FIG_DIR = os.path.join(DATA_DIR, "figures")        # report figures

#C = inverse regularization strength: small C → stronger regularization (wider
#margin, more bias); large C → fit training data harder (more variance).
#A first sweep over [0.001, 0.01, 0.1, 1, 10] showed validation macro-F1
#decreasing monotonically with C (0.001->0.473, 0.01->0.466, 0.1->0.464) while
#fit time roughly doubled per step (liblinear primal iterations grow with C), so
#the large-C values are both slower and worse. We therefore focus the grid on
#the strongly-regularized regime and add 0.0001 to confirm the peak.
C_GRID = [0.0001, 0.001, 0.01]


def standardize(X_tr, X_va, X_te):
    """Clean NaN/Inf then z-score standardize, fitting on train only."""
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)


def best_threshold_score(y_true, scores):
    """Threshold on the decision-function score maximizing F1 for one class."""
    lo, hi = np.percentile(scores, 1), np.percentile(scores, 99)
    grid = np.linspace(lo, hi, 91)
    f1s = [f1_score(y_true, (scores >= t).astype(int), zero_division=0) for t in grid]
    b = int(np.argmax(f1s))
    return grid[b], f1s[b]


def train_ovr(C, X_tr, y_tr, X_va, y_va, X_eval=None):
    """
    Train one LinearSVC per class (OvR), tune a per-class threshold on the
    validation decision scores, and return validation macro-F1 (+ optional
    eval-set scores/thresholds for the final model).
    """
    Cn = y_tr.shape[1]
    score_va = np.zeros((len(X_va), Cn), dtype=np.float32)
    score_ev = np.zeros((len(X_eval), Cn), dtype=np.float32) if X_eval is not None else None
    thresholds = np.zeros(Cn, dtype=np.float32)

    for c in range(Cn):
        #dual=False (primal) is far faster when n_samples >> n_features (117k >> 480)
        clf = LinearSVC(C=C, class_weight="balanced", random_state=SEED,
                        dual=False, max_iter=3000)
        clf.fit(X_tr, y_tr[:, c])
        score_va[:, c] = clf.decision_function(X_va)
        if X_eval is not None:
            score_ev[:, c] = clf.decision_function(X_eval)
        thresholds[c], _ = best_threshold_score(y_va[:, c], score_va[:, c])

    pred_va = (score_va >= thresholds[None, :]).astype(int)
    val_macro_f1 = f1_score(y_va, pred_va, average="macro", zero_division=0)
    return dict(score_va=score_va, score_eval=score_ev,
                thresholds=thresholds, val_macro_f1=val_macro_f1)


def sweep_C(X_tr, y_tr, X_va, y_va, grid=C_GRID):
    print("\n" + "=" * 70)
    print("SVM HYPERPARAMETER SWEEP — regularization C (validation macro-F1)")
    print("=" * 70)
    f1s = []
    for C in grid:
        t0 = time.time()
        r = train_ovr(C, X_tr, y_tr, X_va, y_va)
        f1s.append(r["val_macro_f1"])
        print(f"  C={C:<7} val_macro_F1={r['val_macro_f1']:.4f}  ({time.time()-t0:.1f}s)")
    best = grid[int(np.argmax(f1s))]
    print(f"  -> best C = {best} (val_macro_F1={max(f1s):.4f})")
    return {"values": list(grid), "val_macro_f1": f1s, "best": best}


def plot_sweep(res, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(res["values"]))
    ax.plot(x, res["val_macro_f1"], "o-", color="darkgreen")
    bi = int(np.argmax(res["val_macro_f1"]))
    ax.plot(x[bi], res["val_macro_f1"][bi], "*", ms=18, color="crimson",
            label=f"best C = {res['values'][bi]}")
    ax.set_xticks(x)
    ax.set_xticklabels(res["values"])
    ax.set_xlabel("C (inverse regularization strength, log grid)")
    ax.set_ylabel("validation macro-F1")
    ax.set_title("Linear SVM: effect of regularization C")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved hyperparameter plot: {out_path}")


def main():
    print("=" * 70)
    print("Linear SVM (One-vs-Rest) — multi-label sound event classification")
    print("=" * 70)

    X, y, collectors, class_names = load_dataset()
    C = len(class_names)
    tr, va, te = collector_split(X, collectors)
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    X_tr, X_va, X_te = standardize(X_tr, X_va, X_te)
    os.makedirs(OUT_DIR, exist_ok=True)

    #── 3a: sweep C + plot ──
    sweep_res = sweep_C(X_tr, y_tr, X_va, y_va)
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_sweep(sweep_res, os.path.join(FIG_DIR, "svm_hyperparams.png"))
    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
        json.dump(sweep_res, f, indent=2)
    best_C = sweep_res["best"]

    #── final model on the full train split, evaluate on test ──
    print(f"\n── Final LinearSVC (C={best_C}) ──")
    t0 = time.time()
    res = train_ovr(best_C, X_tr, y_tr, X_va, y_va, X_eval=X_te)
    print(f"Trained {C} per-class SVMs in {time.time()-t0:.1f}s")

    score_te = res["score_eval"]
    thr = res["thresholds"]
    pred_tuned = (score_te >= thr[None, :]).astype(int)
    pred_0 = (score_te >= 0.0).astype(int)               # default SVM boundary
    pred_base = stratified_random_baseline(y_te, y_tr)

    def report(name, pred):
        return (name, f1_score(y_te, pred, average="macro", zero_division=0),
                f1_score(y_te, pred, average="micro", zero_division=0))

    rows = [
        report("Baseline (random)", pred_base),
        report("SVM @0 (default)", pred_0),
        report("SVM tuned-thr", pred_tuned),
    ]

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"{'Model':<24}{'F1_macro':>12}{'F1_micro':>12}")
    print("-" * 48)
    for name, fm, fmi in rows:
        print(f"{name:<24}{fm:>12.4f}{fmi:>12.4f}")

    print("\nPer-class F1 (tuned thresholds, test set):")
    f1_pc = f1_score(y_te, pred_tuned, average=None, zero_division=0)
    order = np.argsort(f1_pc)[::-1]
    print(f"{'Class':<28}{'F1':>8}")
    print("-" * 36)
    for i in order:
        print(f"{class_names[i]:<28}{f1_pc[i]:>8.4f}")

    np.savez(os.path.join(OUT_DIR, "test_predictions.npz"),
             y_test=y_te, scores=score_te, thresholds=thr,
             class_names=np.array(class_names))
    print(f"\nSaved predictions to {OUT_DIR}/")


if __name__ == "__main__":
    main()
