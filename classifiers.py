import os
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import pickle
from collections import OrderedDict
from itertools import product as iter_product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")


DATA_DIR = r"E:\MLPC dataset"
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


def load_preprocessed(preprocessed_dir, suffix="_std_mi"):
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(preprocessed_dir, f"{name}{suffix}.npz")
        loaded = np.load(path, allow_pickle=True)
        splits[name] = {
            "X": loaded["X"].astype(np.float32),
            "y": loaded["y"].astype(np.float32),
            "class_names": list(loaded["class_names"]),
        }
    for name, s in splits.items():
        print(f"  {name}: X={s['X'].shape}, y={s['y'].shape}")
    return splits


def evaluate(y_true, y_pred, class_names=None):
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0),
    }


def baseline_stratified_random(y_true, y_train, seed=42):
    rng = np.random.RandomState(seed)
    class_freqs = y_train.mean(axis=0)
    y_pred = np.zeros_like(y_true)
    for c in range(y_true.shape[1]):
        y_pred[:, c] = rng.binomial(1, class_freqs[c], size=y_true.shape[0])
    return y_pred

"""
LinearSVC is much faster than RBF SVM on large datasets (O(n) vs O(n²)).
It can handle the full 118K training set in seconds.

Key hyperparameter:
  - C (regularization): same role as RBF SVM — controls margin vs error trade-off.
"""

SVM_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]


def train_svm_grid(X_train, y_train, X_val, y_val, class_names, c_values):
    print("\n── LinearSVC Hyperparameter Tuning ──")

    results = []
    best_score = -1
    best_model = None
    best_params = None

    for i, C in enumerate(c_values):
        print(f"\n  [{i+1}/{len(c_values)}] C={C}", end=" ... ")
        t0 = time.time()

        clf = OneVsRestClassifier(
            LinearSVC(C=C, max_iter=10000, random_state=RANDOM_SEED),
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        y_pred_tr = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        met_tr = evaluate(y_train, y_pred_tr)
        met_val = evaluate(y_val, y_pred_val)
        elapsed = time.time() - t0

        print(f"train_f1={met_tr['f1_macro']:.4f}, val_f1={met_val['f1_macro']:.4f} ({elapsed:.1f}s)")

        results.append({
            "params": {"C": C},
            "train_f1_macro": met_tr["f1_macro"],
            "val_f1_macro": met_val["f1_macro"],
            "val_f1_per_class": met_val["f1_per_class"],
            "time": elapsed,
        })

        if met_val["f1_macro"] > best_score:
            best_score = met_val["f1_macro"]
            best_model = clf
            best_params = {"C": C}

    print(f"\n  Best LinearSVC: C={best_params['C']} → val F1_macro = {best_score:.4f}")
    return results, best_model, best_params


class SoundEventNet(nn.Module):
    """
    Multi-label feedforward neural network for sound event detection.

    Architecture:
      Input → Linear → BatchNorm → ReLU → Dropout
            → Linear → BatchNorm → ReLU → Dropout
            → Linear → BatchNorm → ReLU → Dropout
            → Linear → Sigmoid (multi-label output)

    BatchNorm stabilizes training, Dropout prevents overfitting.
    """
    def __init__(self, input_dim, n_classes, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_preds = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).cpu().numpy().astype(int)
        all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)


def train_nn(X_train, y_train, X_val, y_val, class_names,
             hidden_dims=(256, 128, 64), dropout=0.3, lr=0.001,
             weight_decay=1e-4, batch_size=512, n_epochs=100, patience=10,
             label=""):
    """Train one NN configuration and return metrics + model."""

    input_dim = X_train.shape[1]
    n_classes = y_train.shape[1]

    #Compute positive class weights for BCE loss (handles imbalance)
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(DEVICE)

    #DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    #Model
    model = SoundEventNet(input_dim, n_classes, hidden_dims, dropout).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                      factor=0.5, patience=5)

    #Training loop with early stopping
    best_val_f1 = -1
    best_epoch = 0
    best_state = None
    history = {"train_loss": [], "val_f1_macro": [], "train_f1_macro": []}

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

        y_pred_val = predict(model, val_loader)
        y_pred_tr = predict(model, train_loader)

        val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
        train_f1 = f1_score(y_train, y_pred_tr, average="macro", zero_division=0)

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_f1_macro"].append(val_f1)
        history["train_f1_macro"].append(train_f1)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:>3}: loss={train_loss:.4f}, "
                  f"train_f1={train_f1:.4f}, val_f1={val_f1:.4f}, "
                  f"lr={current_lr:.6f} ({elapsed:.1f}s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            print(f"    Early stopping at epoch {epoch} (best: epoch {best_epoch})")
            break

    #Restore best model
    model.load_state_dict(best_state)
    model.to(DEVICE)

    #Final evaluation
    y_pred_tr = predict(model, train_loader)
    y_pred_val = predict(model, val_loader)
    met_tr = evaluate(y_train, y_pred_tr)
    met_val = evaluate(y_val, y_pred_val)

    print(f"    Best epoch {best_epoch}: train_f1={met_tr['f1_macro']:.4f}, "
          f"val_f1={met_val['f1_macro']:.4f}")

    return model, met_tr, met_val, history


NN_CONFIGS = [
    {"hidden_dims": (256, 128),       "dropout": 0.3, "lr": 0.001, "weight_decay": 1e-4},
    {"hidden_dims": (256, 128, 64),   "dropout": 0.3, "lr": 0.001, "weight_decay": 1e-4},
    {"hidden_dims": (512, 256, 128),  "dropout": 0.3, "lr": 0.001, "weight_decay": 1e-4},
    {"hidden_dims": (256, 128, 64),   "dropout": 0.2, "lr": 0.001, "weight_decay": 1e-4},
    {"hidden_dims": (256, 128, 64),   "dropout": 0.4, "lr": 0.001, "weight_decay": 1e-4},
    {"hidden_dims": (256, 128, 64),   "dropout": 0.3, "lr": 0.0005, "weight_decay": 1e-4},
    {"hidden_dims": (256, 128, 64),   "dropout": 0.3, "lr": 0.003,  "weight_decay": 1e-4},
    {"hidden_dims": (256, 128, 64),   "dropout": 0.3, "lr": 0.001, "weight_decay": 1e-3},
]


def tune_nn(X_train, y_train, X_val, y_val, class_names, configs):
    print("\n── Neural Network Hyperparameter Tuning ──")

    results = []
    best_score = -1
    best_model = None
    best_config = None
    best_history = None

    for i, cfg in enumerate(configs):
        label = (f"arch={cfg['hidden_dims']}, drop={cfg['dropout']}, "
                 f"lr={cfg['lr']}, wd={cfg['weight_decay']}")
        print(f"\n  [{i+1}/{len(configs)}] {label}")

        model, met_tr, met_val, history = train_nn(
            X_train, y_train, X_val, y_val, class_names,
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            batch_size=512,
            n_epochs=100,
            patience=10,
            label=label,
        )

        results.append({
            "config": cfg,
            "train_f1_macro": met_tr["f1_macro"],
            "val_f1_macro": met_val["f1_macro"],
            "val_f1_per_class": met_val["f1_per_class"],
            "history": history,
        })

        if met_val["f1_macro"] > best_score:
            best_score = met_val["f1_macro"]
            best_model = model
            best_config = cfg
            best_history = history

    print(f"\n  Best NN: {best_config} → val F1_macro = {best_score:.4f}")
    return results, best_model, best_config, best_history

def plot_training_curves(history, title="Best Neural Network"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history["train_loss"], label="Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history["train_f1_macro"], label="Train F1", alpha=0.7)
    ax.plot(history["val_f1_macro"], label="Val F1", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Macro")
    ax.set_title("F1 Score Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig("nn_training_curves.png", dpi=150)
    plt.show()
    print("Saved: nn_training_curves.png")


def plot_svm_hyperparams(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    c_vals = [r["params"]["C"] for r in results]
    train_f1 = [r["train_f1_macro"] for r in results]
    val_f1 = [r["val_f1_macro"] for r in results]

    ax.plot(c_vals, val_f1, "o-", label="Validation", linewidth=2)
    ax.plot(c_vals, train_f1, "o--", alpha=0.5, label="Train")
    ax.set_xscale("log")
    ax.set_xlabel("C (regularization)")
    ax.set_ylabel("F1 Macro")
    ax.set_title("LinearSVC: Effect of C")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("svm_hyperparams.png", dpi=150)
    plt.show()
    print("Saved: svm_hyperparams.png")


def plot_nn_hyperparams(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    #Architecture comparison (filter to same dropout/lr/wd)
    ax = axes[0]
    arch_results = [r for r in results
                    if r["config"]["dropout"] == 0.3
                    and r["config"]["lr"] == 0.001
                    and r["config"]["weight_decay"] == 1e-4]
    archs = [str(r["config"]["hidden_dims"]) for r in arch_results]
    vals = [r["val_f1_macro"] for r in arch_results]
    ax.bar(range(len(archs)), vals, color="#ed7d31", edgecolor="black")
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, rotation=20, fontsize=8)
    ax.set_ylabel("Val F1 Macro")
    ax.set_title("Effect of Architecture")
    ax.grid(True, alpha=0.3, axis="y")

    #Dropout comparison (filter to same arch/lr/wd)
    ax = axes[1]
    drop_results = [r for r in results
                    if r["config"]["hidden_dims"] == (256, 128, 64)
                    and r["config"]["lr"] == 0.001
                    and r["config"]["weight_decay"] == 1e-4]
    drops = [r["config"]["dropout"] for r in drop_results]
    vals = [r["val_f1_macro"] for r in drop_results]
    ax.plot(drops, vals, "o-", color="#ed7d31", linewidth=2)
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("Val F1 Macro")
    ax.set_title("Effect of Dropout")
    ax.grid(True, alpha=0.3)

    #Learning rate comparison
    ax = axes[2]
    lr_results = [r for r in results
                  if r["config"]["hidden_dims"] == (256, 128, 64)
                  and r["config"]["dropout"] == 0.3
                  and r["config"]["weight_decay"] == 1e-4]
    lrs = [r["config"]["lr"] for r in lr_results]
    vals = [r["val_f1_macro"] for r in lr_results]
    ax.plot(lrs, vals, "o-", color="#ed7d31", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Val F1 Macro")
    ax.set_title("Effect of Learning Rate")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Neural Network Hyperparameter Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig("nn_hyperparams.png", dpi=150)
    plt.show()
    print("Saved: nn_hyperparams.png")


def plot_final_comparison(met_svm, met_nn, met_base, class_names):
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, met_base["f1_per_class"], width,
           label="Baseline", color="#aaaaaa", edgecolor="black")
    ax.bar(x, met_svm["f1_per_class"], width,
           label="LinearSVC", color="#5b9bd5", edgecolor="black")
    ax.bar(x + width, met_nn["f1_per_class"], width,
           label="Neural Network", color="#ed7d31", edgecolor="black")

    ax.set_xlabel("Sound Event Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Comparison on Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("final_comparison.png", dpi=150)
    plt.show()
    print("Saved: final_comparison.png")

    #Macro summary
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Baseline", "LinearSVC", "Neural Network"]
    macro_f1 = [met_base["f1_macro"], met_svm["f1_macro"], met_nn["f1_macro"]]
    colors = ["#aaaaaa", "#5b9bd5", "#ed7d31"]
    bars = ax.bar(models, macro_f1, color=colors, edgecolor="black")
    for bar, val in zip(bars, macro_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("F1 Macro")
    ax.set_title("Overall Performance Comparison (Test Set)")
    ax.set_ylim(0, max(macro_f1) * 1.15)
    plt.tight_layout()
    plt.savefig("final_macro_f1.png", dpi=150)
    plt.show()
    print("Saved: final_macro_f1.png")


def plot_train_val_gap(svm_results, nn_results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #SVM
    ax = axes[0]
    tr = [r["train_f1_macro"] for r in svm_results]
    va = [r["val_f1_macro"] for r in svm_results]
    ax.scatter(tr, va, alpha=0.7, s=60, edgecolors="black", c="#5b9bd5")
    lims = [0, max(max(tr), max(va)) + 0.05]
    ax.plot(lims, lims, "r--", alpha=0.5, label="No overfitting")
    ax.set_xlabel("Train F1 Macro")
    ax.set_ylabel("Val F1 Macro")
    ax.set_title("LinearSVC: Train vs Val")
    ax.legend()
    ax.grid(True, alpha=0.3)

    #NN
    ax = axes[1]
    tr = [r["train_f1_macro"] for r in nn_results]
    va = [r["val_f1_macro"] for r in nn_results]
    ax.scatter(tr, va, alpha=0.7, s=60, edgecolors="black", c="#ed7d31")
    lims = [0, max(max(tr), max(va)) + 0.05]
    ax.plot(lims, lims, "r--", alpha=0.5, label="No overfitting")
    ax.set_xlabel("Train F1 Macro")
    ax.set_ylabel("Val F1 Macro")
    ax.set_title("Neural Network: Train vs Val")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("train_val_gap.png", dpi=150)
    plt.show()
    print("Saved: train_val_gap.png")


if __name__ == "__main__":
    #Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    splits = load_preprocessed(PREPROCESSED_DIR, suffix="_std_mi")
    class_names = splits["train"]["class_names"]

    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]
    X_val = splits["val"]["X"]
    y_val = splits["val"]["y"]
    X_test = splits["test"]["X"]
    y_test = splits["test"]["y"]

    #LinearSVC
    print("\n" + "=" * 60)
    print("Classifier 1: LINEAR SVM ")
    print("=" * 60)
    svm_results, best_svm, best_svm_params = train_svm_grid(
        X_train, y_train.astype(int), X_val, y_val.astype(int),
        class_names, SVM_C_VALUES,
    )

    #Neural Network
    print("\n" + "=" * 60)
    print("Classifier 2: PYTORCH NEURAL NETWORK")
    print("=" * 60)
    nn_results, best_nn, best_nn_config, best_history = tune_nn(
        X_train, y_train, X_val, y_val, class_names, NN_CONFIGS,
    )

    #Plots: hyperparameter effects
    print("\n" + "=" * 60)
    print("Hyperparameters visualization:")
    print("=" * 60)
    plot_svm_hyperparams(svm_results)
    plot_nn_hyperparams(nn_results)
    plot_training_curves(best_history)
    plot_train_val_gap(svm_results, nn_results)

    #Final test set evaluation
    print("\n" + "=" * 60)
    print("Final comparison on Test set")
    print("=" * 60)

    #SVM test predictions
    y_pred_svm = best_svm.predict(X_test.astype(np.float64))

    #NN test predictions
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, pin_memory=True)
    y_pred_nn = predict(best_nn, test_loader)

    #Baseline
    y_pred_base = baseline_stratified_random(y_test.astype(int), y_train.astype(int))

    met_svm = evaluate(y_test.astype(int), y_pred_svm)
    met_nn = evaluate(y_test.astype(int), y_pred_nn)
    met_base = evaluate(y_test.astype(int), y_pred_base)

    #Print results
    print(f"\n  {'Model':<20} {'F1 Macro':>10} {'F1 Micro':>10}")
    print("  " + "-" * 42)
    print(f"  {'Baseline':<20} {met_base['f1_macro']:>10.4f} {met_base['f1_micro']:>10.4f}")
    print(f"  {'LinearSVC':<20} {met_svm['f1_macro']:>10.4f} {met_svm['f1_micro']:>10.4f}")
    print(f"  {'Neural Network':<20} {met_nn['f1_macro']:>10.4f} {met_nn['f1_micro']:>10.4f}")

    print(f"\n  Per-class F1 (test set):")
    print(f"  {'Class':<25} {'Baseline':>10} {'SVM':>10} {'NN':>10}")
    print("  " + "-" * 57)
    for i, cname in enumerate(class_names):
        print(f"  {cname:<25} {met_base['f1_per_class'][i]:>10.4f} "
              f"{met_svm['f1_per_class'][i]:>10.4f} "
              f"{met_nn['f1_per_class'][i]:>10.4f}")

    print(f"\n── LinearSVC Classification Report ──")
    print(classification_report(y_test.astype(int), y_pred_svm,
                                target_names=class_names, zero_division=0))
    print(f"\n── Neural Network Classification Report ──")
    print(classification_report(y_test.astype(int), y_pred_nn,
                                target_names=class_names, zero_division=0))

    #Plots: final comparison
    plot_final_comparison(met_svm, met_nn, met_base, class_names)

    #Save everything
    print("\n" + "=" * 60)
    print("SAVING MODELS AND RESULTS")
    print("=" * 60)
    output_dir = os.path.join(DATA_DIR, "models")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "best_svm.pkl"), "wb") as f:
        pickle.dump({"model": best_svm, "params": best_svm_params}, f)
    torch.save({"model_state": best_nn.state_dict(), "config": best_nn_config},
               os.path.join(output_dir, "best_nn.pt"))
    np.savez(os.path.join(output_dir, "test_predictions.npz"),
             y_pred_svm=y_pred_svm, y_pred_nn=y_pred_nn, y_test=y_test)

    print(f"  Saved to {output_dir}/")

    #Summary
    print("\n" + "=" * 60)
    print("Summary for report")
    print("=" * 60)
    print(f"""
    Classifier 1: LinearSVC
      Best C: {best_svm_params['C']}
      Test F1 macro: {met_svm['f1_macro']:.4f}

    Classifier 2: PyTorch Neural Network
      Best config: {best_nn_config}
      Test F1 macro: {met_nn['f1_macro']:.4f}

    Baseline (Stratified Random):
      Test F1 macro: {met_base['f1_macro']:.4f}

    Plots generated:
      svm_hyperparams.png      — C effect on LinearSVC
      nn_hyperparams.png       — architecture/dropout/lr effects
      nn_training_curves.png   — loss and F1 over epochs
      train_val_gap.png        — overfitting analysis
      final_comparison.png     — per-class F1 comparison
      final_macro_f1.png       — overall macro F1 comparison
    """)