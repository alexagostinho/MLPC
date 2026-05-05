import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, hamming_loss,
    classification_report, multilabel_confusion_matrix,
    average_precision_score, roc_auc_score,
)
from collections import OrderedDict

data_dir = r"E:\MLPC dataset"
preprocessed_dir = os.path.join(data_dir, "preprocessed")
random_seed = 42

def load_preprocessed(preprocessed_dir, suffix="_std_mi"):
    """Load preprocessed splits."""
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(preprocessed_dir, f"{name}{suffix}.npz")
        loaded = np.load(path, allow_pickle=True)
        splits[name] = {
            "X": loaded["X"],
            "y": loaded["y"],
            "class_names": list(loaded["class_names"]),
        }
    
    for name, s in splits.items():
        print(f"  {name}: X={s['X'].shape}, y={s['y'].shape}")
    
    return splits


def analyze_label_characteristics(y, class_names, split_name="train"):
    """
    Analyze label properties that affect metric choice:
    - Class imbalance
    - Multi-label statistics
    - Label cardinality (avg number of active classes per sample)
    """
    n_samples, n_classes = y.shape
    
    print(f"\n── Label Characteristics ({split_name}, {n_samples} samples) ──")
    
    #Class frequencies
    class_counts = y.sum(axis=0)
    class_freqs = class_counts / n_samples
    
    print(f"\n  {'Class':<25} {'Positive':>8} {'Negative':>8} {'Freq':>8} {'Imbalance':>10}")
    print("  " + "-" * 65)
    for i, cname in enumerate(class_names):
        pos = int(class_counts[i])
        neg = n_samples - pos
        freq = class_freqs[i]
        ratio = neg / pos if pos > 0 else float("inf")
        print(f"  {cname:<25} {pos:>8} {neg:>8} {freq:>8.4f} {ratio:>9.1f}:1")
    
    #Overall statistics
    label_cardinality = y.sum(axis=1).mean()
    all_negative = (y.sum(axis=1) == 0).sum()
    
    print(f"\n  Label cardinality (avg active classes per sample): {label_cardinality:.2f}")
    print(f"  Samples with no active class: {all_negative} ({all_negative/n_samples:.1%})")
    print(f"  Overall positive rate: {y.mean():.4f}")
    
    return class_counts, class_freqs

def compute_all_metrics(y_true, y_pred, class_names):
    """
    Compute a comprehensive set of metrics for multi-label classification.
    Returns a dict of metric_name -> value.
    """
    metrics = OrderedDict()
    
    #Sample-averaged metrics
    metrics["accuracy_exact_match"] = accuracy_score(y_true, y_pred)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
    metrics["hamming_score"] = 1 - hamming_loss(y_true, y_pred)
    
    #Macro-averaged (treats all classes equally)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    #Micro-averaged (pools all predictions)
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    
    #Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, cname in enumerate(class_names):
        metrics[f"f1_{cname}"] = per_class_f1[i]
    
    return metrics


def print_metric_comparison(metrics_dict, title=""):
    """Print multiple baselines/models side by side."""
    if title:
        print(f"\n{title}")
        print("=" * (30 + 12 * len(metrics_dict)))
    
    #Collect all metric names
    all_metric_names = list(next(iter(metrics_dict.values())).keys())
    model_names = list(metrics_dict.keys())
    
    #Header
    print(f"  {'Metric':<30}", end="")
    for name in model_names:
        print(f" {name:>12}", end="")
    print()
    print("  " + "-" * (30 + 13 * len(model_names)))
    
    #Only show aggregate metrics (skip per-class for readability)
    aggregate_metrics = [m for m in all_metric_names if not m.startswith("f1_")
                         or m in ["f1_macro", "f1_micro"]]
    
    for metric in aggregate_metrics:
        print(f"  {metric:<30}", end="")
        for name in model_names:
            val = metrics_dict[name].get(metric, float("nan"))
            print(f" {val:>12.4f}", end="")
        print()

def baseline_random_uniform(y_true, seed=42):
    """
    Random baseline: predict 0 or 1 with equal probability (50/50)
    for each class independently.
    """
    rng = np.random.RandomState(seed)
    y_pred = rng.randint(0, 2, size=y_true.shape)
    return y_pred


def baseline_random_stratified(y_true, y_train, seed=42):
    """
    Stratified random baseline: predict 1 with probability equal to
    the class frequency in the training set.
    
    This is the baseline suggested in the task description — it generates
    predictions based on empirical class frequencies.
    """
    rng = np.random.RandomState(seed)
    class_freqs = y_train.mean(axis=0)  # [C]
    
    y_pred = np.zeros_like(y_true)
    for c in range(y_true.shape[1]):
        y_pred[:, c] = rng.binomial(1, class_freqs[c], size=y_true.shape[0])
    
    return y_pred


def baseline_all_zeros(y_true):
    """
    Predict all zeros (no events). This is often a strong baseline
    when classes are rare (high imbalance).
    """
    return np.zeros_like(y_true)


def baseline_all_ones(y_true):
    """
    Predict all ones (all events active). Usually performs poorly
    but useful as a reference.
    """
    return np.ones_like(y_true)


def baseline_majority_class(y_true, y_train):
    """
    Per-class majority: predict the most frequent label (0 or 1)
    for each class based on training frequencies.
    """
    class_freqs = y_train.mean(axis=0)
    majority = (class_freqs >= 0.5).astype(int)
    y_pred = np.tile(majority, (y_true.shape[0], 1))
    return y_pred


def evaluate_baselines(y_val, y_train, class_names):
    """Run all baselines and compare metrics."""
    baselines = OrderedDict()
    
    #Generate predictions
    preds = {
        "Random 50/50": baseline_random_uniform(y_val),
        "Stratified Random": baseline_random_stratified(y_val, y_train),
        "All Zeros": baseline_all_zeros(y_val),
        "All Ones": baseline_all_ones(y_val),
        "Majority Class": baseline_majority_class(y_val, y_train),
    }
    
    #Compute metrics for each
    for name, y_pred in preds.items():
        baselines[name] = compute_all_metrics(y_val, y_pred, class_names)
    
    return baselines, preds


def per_class_baseline_analysis(y_val, y_train, class_names):
    """
    Show per-class F1 for the stratified random baseline.
    Reveals which classes are hard even for a frequency-aware predictor.
    """
    y_pred = baseline_random_stratified(y_val, y_train)
    per_class_f1 = f1_score(y_val, y_pred, average=None, zero_division=0)
    class_freqs = y_train.mean(axis=0)
    
    print(f"\n── Per-Class Stratified Random Baseline ──")
    print(f"  {'Class':<25} {'Train Freq':>10} {'F1':>8}")
    print("  " + "-" * 45)
    for i, cname in enumerate(class_names):
        print(f"  {cname:<25} {class_freqs[i]:>10.4f} {per_class_f1[i]:>8.4f}")


def analyze_upper_bound(data_dir, class_names):
    """
    Discuss whether perfect performance is achievable.
    
    Limiting factors:
    1. Inter-annotator disagreement → ambiguous ground truth
    2. Overlap thresholding → boundary segments are noisy
    3. Feature resolution (1s windows, 0.5s hop) → temporal smearing
    4. Feature expressiveness → some events may not be distinguishable
       from these acoustic features alone
    """
    print("\n── Upper Bound Discussion ──")
    print("""
    Perfect performance (F1 = 1.0) is likely NOT achievable because:
    
    1. ANNOTATOR DISAGREEMENT: Multiple annotators often disagree on 
       whether a sound event is present, especially for ambiguous or 
       quiet events. The aggregated labels inherit this noise — they 
       are not ground truth but a consensus estimate.
    
    2. TEMPORAL BOUNDARIES: Annotations have continuous onset/offset 
       times, but features are aggregated into 1-second windows with 
       0.5s hop. Segments at event boundaries receive partial overlap 
       values that get binarized, introducing label noise.
    
    3. FEATURE LIMITATIONS: The provided acoustic features (MFCCs, 
       spectral descriptors) are summary statistics over 1-second 
       windows. They may not capture fine-grained temporal structure 
       needed to distinguish similar-sounding events.
    
    4. CLASS AMBIGUITY: Some sound classes may be acoustically similar 
       (e.g., different types of impacts or speech vs. TV audio), 
       making them inherently hard to separate.
    
    A realistic upper bound is likely in the range of F1 = 0.7–0.9 
    depending on the class, with well-defined events (e.g., vacuum 
    cleaner) being easier than transient or ambiguous ones.
    """)

def plot_baseline_comparison(baselines, primary_metric="f1_macro"):
    """Bar chart comparing baselines on the primary metric."""
    names = list(baselines.keys())
    values = [baselines[n][primary_metric] for n in names]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#aaaaaa", "#5b9bd5", "#aaaaaa", "#aaaaaa", "#aaaaaa"]
    bars = ax.bar(names, values, color=colors, edgecolor="black", alpha=0.8)
    
    # Highlight the stratified random baseline
    ax.set_ylabel(primary_metric.replace("_", " ").title())
    ax.set_title(f"Baseline Comparison — {primary_metric.replace('_', ' ').title()}")
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("baseline_comparison.png", dpi=150)
    plt.show()
    print("Saved: baseline_comparison.png")


def plot_metric_sensitivity(y_val, y_train, class_names):
    """
    Show how different metrics react to different baselines.
    Helps justify why macro F1 is better than accuracy for this task.
    """
    preds = {
        "Random 50/50": baseline_random_uniform(y_val),
        "Stratified Random": baseline_random_stratified(y_val, y_train),
        "All Zeros": baseline_all_zeros(y_val),
        "Majority Class": baseline_majority_class(y_val, y_train),
    }
    
    metrics_to_show = ["f1_macro", "f1_micro", "accuracy_exact_match", "hamming_score"]
    
    fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(18, 5))
    
    for ax, metric in zip(axes, metrics_to_show):
        values = []
        for name, y_pred in preds.items():
            m = compute_all_metrics(y_val, y_pred, class_names)
            values.append(m[metric])
        
        ax.bar(list(preds.keys()), values, color="steelblue", alpha=0.8, edgecolor="black")
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        
        for j, val in enumerate(values):
            ax.text(j, val + 0.02, f"{val:.3f}", ha="center", fontsize=8)
    
    plt.suptitle("How Different Metrics React to Baselines", fontsize=13)
    plt.tight_layout()
    plt.savefig("metric_sensitivity.png", dpi=150)
    plt.show()
    print("Saved: metric_sensitivity.png")

if __name__ == "__main__":
    #Load data
    print("=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)
    splits = load_preprocessed(preprocessed_dir, suffix="_std_mi")
    class_names = splits["train"]["class_names"]
    y_train = splits["train"]["y"]
    y_val = splits["val"]["y"]
    
    #Analyze label characteristics → justifies metric choice
    print("\n" + "=" * 60)
    print("LABEL CHARACTERISTICS (motivating metric choice)")
    print("=" * 60)
    analyze_label_characteristics(y_train, class_names, "train")
    analyze_label_characteristics(y_val, class_names, "val")
    
    #Metric justification
    print("\n" + "=" * 60)
    print("METRIC CHOICE: Macro-averaged F1 Score")
    print("=" * 60)
    print("""
    We choose MACRO-AVERAGED F1 as the primary evaluation metric.
    
    Why F1 over accuracy?
    - The dataset is highly imbalanced: most segments have no active 
      sound event for most classes. A classifier predicting all zeros 
      achieves high accuracy but is useless.
    - F1 is the harmonic mean of precision and recall, rewarding 
      classifiers that both detect events (recall) and avoid false 
      alarms (precision).
    
    Why macro over micro?
    - Macro-averaging computes F1 per class, then averages. This gives 
      equal weight to every class, regardless of frequency.
    - Micro-averaging would be dominated by the most frequent classes, 
      hiding poor performance on rare but important events.
    - Since KIAL wants robust detection across ALL sound event classes, 
      macro F1 ensures rare classes are not ignored.
    
    Complementary metrics reported:
    - Per-class F1: identifies which specific classes are easy/hard
    - Hamming loss: fraction of incorrectly predicted labels
    - Micro F1: overall prediction quality weighted by frequency
    """)
    
    #Evaluate baselines
    print("=" * 60)
    print("Baseline evaluation")
    print("=" * 60)
    baselines, baseline_preds = evaluate_baselines(y_val, y_train, class_names)
    print_metric_comparison(baselines, title="Baseline Comparison on Validation Set")
    
    #Per-class breakdown
    per_class_baseline_analysis(y_val, y_train, class_names)
    
    #Upper bound discussion
    print("\n" + "=" * 60)
    print("Upper bound on performance")
    print("=" * 60)
    analyze_upper_bound(data_dir, class_names)
    
    #Visualizations
    print("\n" + "=" * 60)
    print("Visualizations")
    print("=" * 60)
    plot_baseline_comparison(baselines, primary_metric="f1_macro")
    plot_metric_sensitivity(y_val, y_train, class_names)
    
    #Summary for report
    print("\n" + "=" * 60)
    print("Summary for report")
    print("=" * 60)
    strat_f1 = baselines["Stratified Random"]["f1_macro"]
    zeros_f1 = baselines["All Zeros"]["f1_macro"]
    majority_f1 = baselines["Majority Class"]["f1_macro"]
    print(f"""
    Primary metric: Macro-averaged F1 Score
    
    Baseline results (validation set):
      Stratified Random:  F1_macro = {strat_f1:.4f}  ← primary baseline
      All Zeros:          F1_macro = {zeros_f1:.4f}
      Majority Class:     F1_macro = {majority_f1:.4f}
    
    The stratified random baseline is the fairest comparison because it 
    uses class frequency information without learning from features. 
    Any trained classifier should substantially outperform this baseline.
    
    Perfect performance is unlikely due to annotator disagreement, 
    temporal boundary noise, and acoustic ambiguity between classes.
    
    Plots generated:
      baseline_comparison.png  — bar chart of baselines
      metric_sensitivity.png   — why macro F1 > accuracy for this task
    """)
