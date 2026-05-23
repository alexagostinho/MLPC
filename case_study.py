"""
Section 4 — Case Study and Reflection (CatBoost model).

Uses the per-class CatBoost models trained by catboost_classifier.py
(saved in models_catboost/<class>.cbm + test_predictions.npz for the
tuned thresholds). It:

  (4a) Picks two *test* files (never seen during training, guaranteed by the
       collector-level split) and, for each, draws:
         - the mel-spectrogram over time (the provided melspect_mean feature),
         - an outcome timeline that overlays predicted vs. expected labels,
           colour-coded TP / FP / FN / TN so correct detections and both error
           types are visible at a glance.
       Saved as case_study_<file>.png, plus a printed per-file summary.

  (4b) Reflection: per-class confusion matrices on the whole test set
       (confusion_matrices.png) and a cross-class confusion matrix that shows,
       when the model fires for class i, which class is actually present —
       i.e. which classes get confused (class_confusion.png).

Run in the `qsar_torch` env AFTER catboost_classifier.py has been run:
  python case_study.py
  python case_study.py --files 000123 000456   # pick specific test files
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, multilabel_confusion_matrix

#reuse the exact data/label/split logic from the training script so the
#test set here is identical to the one the model was evaluated on
from catboost_classifier import (
    FEATURE_KEYS, aggregate_labels, collector_split,
    DATA_DIR, FEAT_DIR, META_PATH, OUT_DIR, FIG_DIR, SEED,
)


def load_per_file():
    """Load every file, keeping per-file arrays plus the flat arrays for splitting."""
    meta = pd.read_csv(META_PATH)
    fname2collector = dict(zip(
        meta["filename"].str.replace(".wav", "", regex=False), meta["collector_id"]))

    files = {}                       # fname -> dict(X, y, melspect, t, collector)
    seg_collectors, seg_fnames, seg_counts = [], [], []
    class_names = None
    for path in sorted(glob.glob(os.path.join(FEAT_DIR, "*.npz"))):
        fname = os.path.splitext(os.path.basename(path))[0]
        d = np.load(path, allow_pickle=True)
        if class_names is None:
            class_names = list(d["class_names"])
        feats = [d[k] if d[k].ndim > 1 else d[k][:, None] for k in FEATURE_KEYS]
        X_file = np.concatenate(feats, axis=1).astype(np.float32)
        y_file = aggregate_labels(d["annotations"])
        cid = fname2collector.get(fname, "unknown")
        files[fname] = dict(
            X=X_file, y=y_file,
            melspect=d["melspect_mean"].astype(np.float32),   # [T, 128]
            t=d["start_time"].astype(np.float32),             # [T]
            collector=cid,
        )
        seg_collectors.extend([cid] * X_file.shape[0])
        seg_fnames.extend([fname] * X_file.shape[0])
        seg_counts.append((fname, X_file.shape[0]))
    return files, np.array(seg_collectors), np.array(seg_fnames), class_names


def get_test_files(files, seg_collectors, seg_fnames):
    """Reproduce the training split and return the list of filenames in the test set."""
    n = len(seg_collectors)
    X_dummy = np.zeros((n, 1), dtype=np.float32)        # split only needs length + groups
    _, _, test_idx = collector_split(X_dummy, seg_collectors, seed=SEED)
    test_files = list(dict.fromkeys(seg_fnames[test_idx]))   # preserve order, unique
    return test_files


def load_models(class_names):
    models = []
    for c in class_names:
        clf = CatBoostClassifier()
        clf.load_model(os.path.join(OUT_DIR, f"{c}.cbm"))
        models.append(clf)
    return models


def predict_file(files, fname, models, thresholds):
    """Return probabilities [T, C] and binary predictions [T, C] for one file."""
    X = files[fname]["X"]
    C = len(models)
    proba = np.zeros((X.shape[0], C), dtype=np.float32)
    for c in range(C):
        proba[:, c] = models[c].predict_proba(X)[:, 1]
    pred = (proba >= thresholds[None, :]).astype(int)
    return proba, pred


#── (4a) per-file spectrogram + outcome timeline ─────────────────────────────
def plot_file_case(files, fname, y_pred, class_names, out_path):
    info = files[fname]
    y_true = info["y"]
    t = info["t"]
    T = len(t)

    #outcome code per (class, segment): 0 TN, 1 TP, 2 FP, 3 FN
    outcome = np.zeros_like(y_true)
    outcome[(y_pred == 1) & (y_true == 1)] = 1
    outcome[(y_pred == 1) & (y_true == 0)] = 2
    outcome[(y_pred == 0) & (y_true == 1)] = 3
    cmap = ListedColormap(["#f0f0f0", "#2ca02c", "#d62728", "#1f77b4"])  # TN/TP/FP/FN

    fig, ax = plt.subplots(figsize=(13, 4.8))

    #outcome timeline: predicted vs. expected labels per class over time
    ax.imshow(outcome.T, aspect="auto", origin="lower", interpolation="nearest",
              extent=[t[0], t[-1] if T > 1 else t[0] + 1, -0.5, len(class_names) - 0.5],
              cmap=cmap, vmin=0, vmax=3)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_title(f"{fname} — predicted vs. expected labels")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2ca02c", label="TP (correct)"),
        Patch(color="#d62728", label="FP (false alarm)"),
        Patch(color="#1f77b4", label="FN (missed)"),
        Patch(color="#f0f0f0", label="TN"),
    ], loc="upper right", fontsize=8, ncol=4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def summarize_file(fname, y_true, y_pred, class_names):
    active = np.where(y_true.sum(axis=0) > 0)[0]
    print(f"\n  File {fname}: {y_true.shape[0]} segments, "
          f"{len(active)} classes present.")
    for c in active:
        tp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 1)).sum())
        fp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 0)).sum())
        fn = int(((y_pred[:, c] == 0) & (y_true[:, c] == 1)).sum())
        print(f"    {class_names[c]:<26} present={int(y_true[:,c].sum()):>3}  "
              f"TP={tp:>3} FP={fp:>3} FN={fn:>3}")


#── (4b) confusion matrices on the whole test set ───────────────────────────
def plot_per_class_confusion(y_true, y_pred, class_names, out_path):
    mcm = multilabel_confusion_matrix(y_true, y_pred)   # [C, 2, 2]
    C = len(class_names)
    cols = 5
    rows = int(np.ceil(C / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for c in range(C):
        ax = axes[c]
        m = mcm[c]                                      # [[TN,FP],[FN,TP]]
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        color="black", fontsize=10)
        f1 = f1_score(y_true[:, c], y_pred[:, c], zero_division=0)
        ax.set_title(f"{class_names[c]}\nF1={f1:.2f}", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"], fontsize=7)
    for k in range(C, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Per-class confusion matrices (test set, tuned thresholds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_cross_class_confusion(y_true, y_pred, class_names, out_path):
    """
    Among segments predicted positive for class i (row), which classes are
    actually present (columns)? Row-normalized → reveals systematic confusions
    and label co-occurrence behind false positives.
    """
    C = len(class_names)
    M = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        fired = y_pred[:, i] == 1
        n = fired.sum()
        if n == 0:
            continue
        M[i] = y_true[fired].sum(axis=0) / n            # fraction of those segs where class j present
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(C)); ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(C)); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("actually present class")
    ax.set_ylabel("model predicted class")
    ax.set_title("When the model predicts row-class, which class is truly present?\n"
                 "(row-normalized; diagonal = correct, off-diagonal = confusion)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="fraction of predicted-positive segments")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="*", default=None,
                        help="specific test file ids to visualize (default: auto-pick 2)")
    args = parser.parse_args()

    print("Loading data + reproducing the test split ...")
    files, seg_collectors, seg_fnames, class_names = load_per_file()
    test_files = get_test_files(files, seg_collectors, seg_fnames)
    print(f"Test set: {len(test_files)} files")

    pred_npz = np.load(os.path.join(OUT_DIR, "test_predictions.npz"), allow_pickle=True)
    thresholds = pred_npz["thresholds"]
    models = load_models(class_names)

    #pick two "interesting" test files = most distinct active classes (diverse scenes)
    if args.files:
        chosen = args.files
    else:
        ranked = sorted(test_files,
                        key=lambda f: (files[f]["y"].sum(axis=0) > 0).sum(),
                        reverse=True)
        chosen = ranked[:2]
    print(f"Case-study files: {chosen}")

    os.makedirs(FIG_DIR, exist_ok=True)
    print("\n── (4a) Per-file prediction timelines ──")
    for fname in chosen:
        if fname not in files:
            print(f"  !! {fname} not found, skipping")
            continue
        _, pred = predict_file(files, fname, models, thresholds)
        summarize_file(fname, files[fname]["y"], pred, class_names)
        plot_file_case(files, fname, pred, class_names,
                       os.path.join(FIG_DIR, f"case_study_{fname}.png"))

    print("\n── (4b) Confusion matrices over the whole test set ──")
    #rebuild test predictions per file to get aligned y_true / y_pred
    y_true_all, y_pred_all = [], []
    for fname in test_files:
        _, pred = predict_file(files, fname, models, thresholds)
        y_true_all.append(files[fname]["y"])
        y_pred_all.append(pred)
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    print(f"  test segments: {y_true.shape[0]}  macro-F1={f1_score(y_true,y_pred,average='macro',zero_division=0):.4f}")

    plot_per_class_confusion(y_true, y_pred, class_names,
                             os.path.join(FIG_DIR, "confusion_matrices.png"))
    plot_cross_class_confusion(y_true, y_pred, class_names,
                               os.path.join(FIG_DIR, "class_confusion.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
