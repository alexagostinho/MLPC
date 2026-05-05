import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

data_dir = r"E:\MLPC dataset"
features_dir = r"E:\MLPC dataset\audio_features"
overlap_threshold = 0.5
agreement_threshold = 0.5

def load_all_npz(features_dir):
    """Load all .npz files and return a dict keyed by filename."""
    data = {}
    npz_files = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {features_dir}")

    for path in npz_files:
        fname = os.path.splitext(os.path.basename(path))[0]
        content = dict(np.load(path, allow_pickle=True))
        data[fname] = content

    print(f"Loaded {len(data)} files.")
    return data


def aggregate_majority_vote(annotations, overlap_thresh=0.5, agreement_thresh=0.5):
    """
    Majority voting.
    1) Binarize each annotator's overlap values (>= overlap_thresh → 1).
    2) Average across annotators.
    3) Threshold the average (>= agreement_thresh → positive label).

    Returns: binary labels of shape [T, C].
    """
    binary = (annotations >= overlap_thresh).astype(np.float32)
    avg = binary.mean(axis=2)  # [T, C]
    labels = (avg >= agreement_thresh).astype(np.int32)
    return labels


def aggregate_soft_labels(annotations, overlap_thresh=0.5):
    """
    Soft labels: binarize per-annotator, then average across annotators.
    Returns continuous values in [0, 1] of shape [T, C].
    Useful if your classifier supports soft/probabilistic targets.
    """
    binary = (annotations >= overlap_thresh).astype(np.float32)
    soft = binary.mean(axis=2)  # [T, C]
    return soft


def aggregate_union(annotations, overlap_thresh=0.5):
    """
    Union (any-agreement): label is positive if ANY annotator marked it.
    High recall, potentially noisy.
    Returns: binary labels of shape [T, C].
    """
    binary = (annotations >= overlap_thresh).astype(np.float32)
    labels = (binary.max(axis=2) >= 1.0).astype(np.int32)
    return labels


def aggregate_intersection(annotations, overlap_thresh=0.5):
    """
    Intersection (unanimous): label is positive only if ALL annotators agree.
    High precision, potentially misses many events.
    Returns: binary labels of shape [T, C].
    """
    binary = (annotations >= overlap_thresh).astype(np.float32)
    labels = (binary.min(axis=2) >= 1.0).astype(np.int32)
    return labels

def aggregate_dataset(data, strategy="majority_vote"):
    """
    Apply the chosen aggregation strategy to all files.

    Returns:
        aggregated: dict mapping filename -> labels [T, C]
        class_names: list of class names (consistent across files)
    """
    strategies = {
        "majority_vote": aggregate_majority_vote,
        "soft_labels": aggregate_soft_labels,
        "union": aggregate_union,
        "intersection": aggregate_intersection,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(strategies.keys())}")

    agg_fn = strategies[strategy]
    aggregated = {}
    class_names = None

    for fname, content in data.items():
        annotations = content["annotations"]  # [T, C, A]
        labels = agg_fn(annotations)  # [T, C]
        aggregated[fname] = labels

        #Extract class names (consistent across files)
        if class_names is None:
            class_names = list(content["class_names"])

    return aggregated, class_names


def compute_class_frequencies(aggregated, class_names):
    """Compute how often each class is active across all segments."""
    all_labels = np.concatenate(list(aggregated.values()), axis=0)  # [total_T, C]
    total_segments = all_labels.shape[0]

    #For soft labels, threshold at 0.5 for counting
    if all_labels.dtype == np.float32 and np.any((all_labels > 0) & (all_labels < 1)):
        binary = (all_labels >= 0.5).astype(int)
    else:
        binary = all_labels

    counts = binary.sum(axis=0)
    freqs = counts / total_segments

    print(f"\nTotal segments: {total_segments}")
    print(f"{'Class':<25} {'Count':>8} {'Frequency':>10}")
    print("-" * 45)
    for i, name in enumerate(class_names):
        print(f"{name:<25} {counts[i]:>8} {freqs[i]:>10.4f}")

    return counts, freqs


def compare_strategies(data, class_names_ref=None):
    """
    Compare all aggregation strategies side by side.
    Shows how many positive labels each strategy produces per class.
    """
    strategies = ["majority_vote", "union", "intersection"]
    results = {}

    for strat in strategies:
        agg, class_names = aggregate_dataset(data, strategy=strat)
        all_labels = np.concatenate(list(agg.values()), axis=0)
        counts = all_labels.sum(axis=0)
        results[strat] = counts

    #Also compute soft label mean
    agg_soft, _ = aggregate_dataset(data, strategy="soft_labels")
    all_soft = np.concatenate(list(agg_soft.values()), axis=0)
    soft_means = all_soft.mean(axis=0)

    #Print comparison table
    print(f"\n{'Class':<25}", end="")
    for strat in strategies:
        print(f" {strat:>15}", end="")
    print(f" {'soft_mean':>15}")
    print("-" * (25 + 16 * (len(strategies) + 1)))

    for i, name in enumerate(class_names):
        print(f"{name:<25}", end="")
        for strat in strategies:
            print(f" {int(results[strat][i]):>15}", end="")
        print(f" {soft_means[i]:>15.4f}")

    return results, class_names


def plot_strategy_comparison(data):
    """Bar chart comparing positive label counts across strategies."""
    strategies = ["majority_vote", "union", "intersection"]
    results = {}
    class_names = None

    for strat in strategies:
        agg, class_names = aggregate_dataset(data, strategy=strat)
        all_labels = np.concatenate(list(agg.values()), axis=0)
        results[strat] = all_labels.sum(axis=0)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, strat in enumerate(strategies):
        ax.bar(x + i * width, results[strat], width, label=strat.replace("_", " ").title())

    ax.set_xlabel("Sound Event Class")
    ax.set_ylabel("Number of Positive Segments")
    ax.set_title("Positive Label Counts by Aggregation Strategy")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig("strategy_comparison.png", dpi=150)
    plt.show()
    print("Saved: strategy_comparison.png")


def plot_annotator_agreement(data, class_names=None):
    """
    Visualize inter-annotator agreement per class.
    Shows distribution of agreement levels (fraction of annotators agreeing).
    """
    all_annotations = []
    for fname, content in data.items():
        ann = content["annotations"]  # [T, C, A]
        binary = (ann >= 0.5).astype(float)
        agreement = binary.mean(axis=2)  # [T, C] — fraction of annotators agreeing
        all_annotations.append(agreement)

    all_agreement = np.concatenate(all_annotations, axis=0)  # [total_T, C]

    if class_names is None:
        first_file = next(iter(data.values()))
        class_names = list(first_file["class_names"])

    #Only plot classes that have at least some positive annotations
    active_mask = all_agreement.max(axis=0) > 0
    active_indices = np.where(active_mask)[0]

    n_active = len(active_indices)
    if n_active == 0:
        print("No active classes found.")
        return

    cols = min(4, n_active)
    rows = (n_active + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_active == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for plot_idx, class_idx in enumerate(active_indices):
        ax = axes[plot_idx]
        values = all_agreement[:, class_idx]
        # Only show non-zero values for clarity
        positive_values = values[values > 0]
        if len(positive_values) > 0:
            ax.hist(positive_values, bins=20, edgecolor="black", alpha=0.7)
        ax.set_title(class_names[class_idx], fontsize=10)
        ax.set_xlabel("Annotator Agreement")
        ax.set_ylabel("Count")

    #Hide unused subplots
    for idx in range(n_active, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Inter-Annotator Agreement Distribution (positive segments only)", fontsize=13)
    plt.tight_layout()
    plt.savefig("annotator_agreement.png", dpi=150)
    plt.show()
    print("Saved: annotator_agreement.png")


if __name__ == "__main__":
    #Load data
    data = load_all_npz(features_dir)

    #Inspect one file to understand the shapes
    sample_fname = next(iter(data))
    sample = data[sample_fname]
    print(f"\nSample file: {sample_fname}")
    print(f"  annotations shape: {sample['annotations'].shape}  (T, C, A)")
    print(f"  class_names: {list(sample['class_names'])}")
    print(f"  annotator_ids: {list(sample['annotator_ids'])}")

    #Primary strategy: Majority Vote
    print("\n" + "=" * 60)
    print("PRIMARY STRATEGY: Majority Vote")
    print("=" * 60)
    aggregated, class_names = aggregate_dataset(data, strategy="majority_vote")
    counts, freqs = compute_class_frequencies(aggregated, class_names)

    #Compare all strategies
    print("\n" + "=" * 60)
    print("COMPARISON OF ALL STRATEGIES")
    print("=" * 60)
    compare_strategies(data)

    #Visualizations
    plot_strategy_comparison(data)
    plot_annotator_agreement(data, class_names)

    #Save aggregated labels for downstream use
    print("\n" + "=" * 60)
    print("SAVING AGGREGATED LABELS")
    print("=" * 60)
    output_dir = os.path.join(data_dir, "aggregated_labels")
    os.makedirs(output_dir, exist_ok=True)

    for fname, labels in aggregated.items():
        np.save(os.path.join(output_dir, f"{fname}_labels.npy"), labels)

    print(f"Saved {len(aggregated)} label files to {output_dir}/")
    print(f"Each file has shape [T, {len(class_names)}] with binary labels.")
    print(f"Class order: {class_names}")

    print("\nDone! Use the saved labels for sections 1b, 1c, and beyond.")



