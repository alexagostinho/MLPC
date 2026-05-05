import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
import warnings

warnings.filterwarnings("ignore")

data_dir = r"E:\MLPC dataset"
splits_dir = os.path.join(data_dir, "splits")
output_dir = os.path.join(data_dir, "preprocessed")

random_seed = 42

def load_splits(splits_dir):
    """Load train/val/test splits produced by data_split.py."""
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(splits_dir, f"{name}.npz")
        loaded = np.load(path, allow_pickle=True)
        splits[name] = {
            "X": loaded["X"].astype(np.float32),
            "y": loaded["y"].astype(np.int32),
            "filenames": loaded["filenames"],
            "class_names": list(loaded["class_names"]),
            "feature_keys": list(loaded["feature_keys"]),
        }

    print(f"Loaded splits:")
    for name, s in splits.items():
        print(f"  {name}: X={s['X'].shape}, y={s['y'].shape}")

    return splits

def clean_invalid_values(splits):
    """
    Replace NaN and Inf values with 0.
    Some acoustic features can produce NaN (e.g., spectral flatness
    on silent segments). Must be handled before normalization.
    """
    total_fixed = 0
    for name, s in splits.items():
        X = s["X"]
        mask = ~np.isfinite(X)
        n_invalid = mask.sum()
        if n_invalid > 0:
            X[mask] = 0.0
            total_fixed += n_invalid
            print(f"  {name}: replaced {n_invalid} NaN/Inf values "
                  f"({n_invalid / X.size:.4%} of all values)")
        s["X"] = X

    if total_fixed == 0:
        print("  No NaN/Inf values found.")

    return splits

def analyze_features(X_train, feature_keys):
    """
    Analyze feature statistics to motivate preprocessing choices.
    Shows scale differences across feature groups.
    """
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    mins = np.min(X_train, axis=0)
    maxs = np.max(X_train, axis=0)

    print(f"\nFeature statistics (train set, {X_train.shape[1]} features):")
    print(f"  Mean  range: [{means.min():.4f}, {means.max():.4f}]")
    print(f"  Std   range: [{stds.min():.6f}, {stds.max():.4f}]")
    print(f"  Min   range: [{mins.min():.4f}, {mins.max():.4f}]")
    print(f"  Max   range: [{maxs.min():.4f}, {maxs.max():.4f}]")

    # Identify features with zero or near-zero variance
    low_var = np.where(stds < 1e-8)[0]
    if len(low_var) > 0:
        print(f"  ⚠ {len(low_var)} features have near-zero variance (will be removed)")

    return means, stds, mins, maxs


def plot_feature_scales(X_train, feature_keys):
    """Visualize the scale differences across features — motivates normalization."""
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    axes[0].bar(range(len(means)), np.abs(means), color="steelblue", alpha=0.7)
    axes[0].set_title("Absolute Feature Means (before normalization)")
    axes[0].set_xlabel("Feature index")
    axes[0].set_ylabel("|Mean|")
    axes[0].set_yscale("log")

    axes[1].bar(range(len(stds)), stds, color="coral", alpha=0.7)
    axes[1].set_title("Feature Standard Deviations (before normalization)")
    axes[1].set_xlabel("Feature index")
    axes[1].set_ylabel("Std")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("feature_scales_before.png", dpi=150)
    plt.show()
    print("Saved: feature_scales_before.png")


def remove_low_variance(X_train, X_val, X_test, feature_names, threshold=1e-8):
    """
    Remove features with variance below threshold.
    Fit on train only — apply same mask to val/test.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)

    kept_mask = selector.get_support()
    kept_names = [f for f, keep in zip(feature_names, kept_mask) if keep]
    removed = sum(~kept_mask)

    print(f"\nVariance threshold ({threshold}):")
    print(f"  Removed {removed} features, kept {len(kept_names)}")

    return X_train_sel, X_val_sel, X_test_sel, kept_names

def standardize(X_train, X_val, X_test):
    """
    Z-score standardization: zero mean, unit variance.

    Why: Many classifiers (SVM, kNN, neural nets) are sensitive to
    feature scales. Acoustic features span vastly different ranges
    (e.g., energy vs. MFCC coefficients), so without normalization,
    high-magnitude features dominate distance-based methods.

    Critical: Fit scaler on TRAIN ONLY to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nStandardization (z-score):")
    print(f"  Train mean range after: [{X_train_scaled.mean(axis=0).min():.6f}, "
          f"{X_train_scaled.mean(axis=0).max():.6f}]")
    print(f"  Train std  range after: [{X_train_scaled.std(axis=0).min():.6f}, "
          f"{X_train_scaled.std(axis=0).max():.6f}]")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def select_by_mutual_information(X_train, y_train, class_names,
                                 feature_names, top_k=None, top_frac=0.5):
    """
    Rank features by mutual information with each class label.
    Aggregate MI across classes (mean) and keep the top features.

    Why: Reduces dimensionality, removes irrelevant features, speeds
    up training, and can improve generalization.

    Fit on train only.
    """
    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]

    if top_k is None:
        top_k = max(10, int(n_features * top_frac))

    print(f"\nMutual Information feature selection:")
    print(f"  Computing MI for {n_features} features × {n_classes} classes...")

    #Compute MI for each class separately
    mi_scores = np.zeros((n_features, n_classes))
    for c in range(n_classes):
        mi_scores[:, c] = mutual_info_classif(
            X_train, y_train[:, c],
            discrete_features=False,
            random_state=random_seed,
        )

    #Aggregate: mean MI across classes
    mi_mean = mi_scores.mean(axis=1)

    #Select top_k features
    top_indices = np.argsort(mi_mean)[::-1][:top_k]
    top_indices = np.sort(top_indices)  #restore original order

    selected_names = [feature_names[i] for i in top_indices]

    print(f"  Selected {top_k} / {n_features} features")
    print(f"  Top 10 features by MI:")
    ranking = np.argsort(mi_mean)[::-1]
    for rank in range(min(10, len(ranking))):
        idx = ranking[rank]
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"    {rank + 1:>3}. {name:<30} MI={mi_mean[idx]:.4f}")

    return top_indices, mi_scores, mi_mean


def plot_mutual_information(mi_mean, feature_names, top_k=30):
    """Visualize feature importance by MI — useful for the report."""
    ranking = np.argsort(mi_mean)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, 8))
    names = [feature_names[i] if i < len(feature_names) else f"feat_{i}"
             for i in ranking]
    scores = mi_mean[ranking]

    ax.barh(range(len(names)), scores, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Mutual Information")
    ax.set_title(f"Top {top_k} Features by Mutual Information")
    plt.tight_layout()
    plt.savefig("feature_importance_mi.png", dpi=150)
    plt.show()
    print("Saved: feature_importance_mi.png")

def apply_pca(X_train, X_val, X_test, variance_retained=0.95):
    """
    PCA dimensionality reduction, keeping enough components to
    explain `variance_retained` fraction of total variance.

    Why: Can speed up training (fewer features) and reduce noise.
    Some classifiers (kNN, SVM) benefit from fewer dimensions.

    Note: Apply AFTER standardization. Fit on train only.
    """
    pca = PCA(n_components=variance_retained, random_state=random_seed)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    print(f"\nPCA ({variance_retained:.0%} variance retained):")
    print(f"  Components: {pca.n_components_} / {X_train.shape[1]}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    return X_train_pca, X_val_pca, X_test_pca, pca


def plot_pca_variance(pca):
    """Cumulative explained variance plot — shows how many components you need."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", markersize=3)
    ax.axhline(y=0.95, color="red", linestyle="--", label="95% threshold")
    ax.axhline(y=0.99, color="orange", linestyle="--", label="99% threshold")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA: Cumulative Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pca_variance.png", dpi=150)
    plt.show()
    print("Saved: pca_variance.png")

def plot_correlation_matrix(X_train, feature_names, max_features=50):
    """
    Correlation heatmap of features. Helps justify removing
    redundant features or applying PCA.
    """
    n = min(max_features, X_train.shape[1])
    corr = np.corrcoef(X_train[:, :n].T)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(np.abs(corr), cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(f"Feature Correlation Matrix (first {n} features)")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    plt.colorbar(im, label="|Correlation|")
    plt.tight_layout()
    plt.savefig("feature_correlation.png", dpi=150)
    plt.show()
    print("Saved: feature_correlation.png")

    #Count highly correlated pairs
    high_corr = np.sum(np.abs(corr) > 0.9) - n  # subtract diagonal
    high_corr //= 2  # each pair counted twice
    print(f"  Highly correlated feature pairs (|r| > 0.9): {high_corr}")

def build_feature_names(feature_keys, data_sample):
    """
    Create a name for every column in the concatenated feature matrix.
    E.g., 'mfcc_mean_0', 'mfcc_mean_1', ..., 'zcr_mean_0', ...
    """
    names = []
    for key in feature_keys:
        if key in data_sample:
            dim = data_sample[key].shape[1] if data_sample[key].ndim > 1 else 1
        else:
            dim = 1
        if dim == 1:
            names.append(key)
        else:
            for d in range(dim):
                names.append(f"{key}_{d}")
    return names


def save_preprocessed(splits, output_dir, suffix=""):
    """Save preprocessed splits."""
    os.makedirs(output_dir, exist_ok=True)

    for name in ["train", "val", "test"]:
        s = splits[name]
        fname = f"{name}{suffix}.npz"
        np.savez_compressed(
            os.path.join(output_dir, fname),
            X=s["X"],
            y=s["y"],
            filenames=s.get("filenames", np.array([])),
            class_names=np.array(s["class_names"]),
        )

    print(f"\nSaved preprocessed data to {output_dir}/")


if __name__ == "__main__":
    #Load splits
    print("=" * 60)
    print("LOADING SPLIT DATA")
    print("=" * 60)
    splits = load_splits(splits_dir)
    class_names = splits["train"]["class_names"]
    feature_keys = splits["train"]["feature_keys"]

    #Build detailed feature names
    #We need a sample NPZ to know per-key dimensions
    import glob

    sample_path = sorted(glob.glob(os.path.join(
        data_dir, "audio_features", "*.npz")))[0]
    sample_data = dict(np.load(sample_path, allow_pickle=True))
    feature_names = build_feature_names(feature_keys, sample_data)
    print(f"  Total named features: {len(feature_names)}")

    #Clean invalid values
    print("\n" + "=" * 60)
    print("Cleaning invalid values")
    print("=" * 60)
    splits = clean_invalid_values(splits)

    #Analyze raw feature scales
    print("\n" + "=" * 60)
    print("Feature analysis (before preprocessing)")
    print("=" * 60)
    analyze_features(splits["train"]["X"], feature_keys)
    plot_feature_scales(splits["train"]["X"], feature_keys)

    #Correlation analysis
    print("\n" + "=" * 60)
    print("Correlation analysis")
    print("=" * 60)
    plot_correlation_matrix(splits["train"]["X"], feature_names)

    #Remove zero-variance features
    print("\n" + "=" * 60)
    print("Removing zero variance features")
    print("=" * 60)
    X_tr, X_va, X_te, kept_names = remove_low_variance(
        splits["train"]["X"], splits["val"]["X"], splits["test"]["X"],
        feature_names,
    )
    splits["train"]["X"] = X_tr
    splits["val"]["X"] = X_va
    splits["test"]["X"] = X_te
    feature_names = kept_names

    #Standardization
    print("\n" + "=" * 60)
    print("Standardization")
    print("=" * 60)
    X_tr, X_va, X_te, scaler = standardize(
        splits["train"]["X"], splits["val"]["X"], splits["test"]["X"],
    )
    splits["train"]["X"] = X_tr
    splits["val"]["X"] = X_va
    splits["test"]["X"] = X_te

    #Mutual information feature selection
    print("\n" + "=" * 60)
    print("Mutual information feature selection")
    print("=" * 60)
    mi_indices, mi_scores, mi_mean = select_by_mutual_information(
        splits["train"]["X"], splits["train"]["y"],
        class_names, feature_names,
        top_frac=0.5,  #keep top 50% of features; adjust as needed
    )
    plot_mutual_information(mi_mean, feature_names)

    #Apply MI selection
    for name in ["train", "val", "test"]:
        splits[name]["X"] = splits[name]["X"][:, mi_indices]
    selected_names = [feature_names[i] for i in mi_indices]
    print(f"  Final feature count after MI selection: {len(selected_names)}")

    #Save version with MI selection + standardization (no PCA)
    print("\n" + "=" * 60)
    print("SAVING PREPROCESSED DATA (standardized + MI selected)")
    print("=" * 60)
    save_preprocessed(splits, output_dir, suffix="_std_mi")

    #Optional: also produce a PCA version
    print("\n" + "=" * 60)
    print("PCA DIMENSIONALITY REDUCTION (optional alternative)")
    print("=" * 60)
    X_tr_pca, X_va_pca, X_te_pca, pca = apply_pca(
        splits["train"]["X"], splits["val"]["X"], splits["test"]["X"],
        variance_retained=0.95,
    )
    plot_pca_variance(pca)

    #Save PCA version
    splits_pca = {}
    for name, X_pca in [("train", X_tr_pca), ("val", X_va_pca), ("test", X_te_pca)]:
        splits_pca[name] = {
            "X": X_pca,
            "y": splits[name]["y"],
            "filenames": splits[name].get("filenames", np.array([])),
            "class_names": class_names,
        }
    save_preprocessed(splits_pca, output_dir, suffix="_std_mi_pca")

    #Summary
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"  Original features:          {len(feature_keys)} keys")
    print(f"  After expansion:            {len(build_feature_names(feature_keys, sample_data))} columns")
    print(f"  After low-variance removal: {len(kept_names)} columns")
    print(f"  After MI selection:         {len(selected_names)} columns")
    print(f"  After PCA (95% var):        {pca.n_components_} components")
    print()
    print("  Output files:")
    print(f"    {output_dir}/*_std_mi.npz     — standardized + MI selected")
    print(f"    {output_dir}/*_std_mi_pca.npz — standardized + MI + PCA")
    print()
    print("  Plots generated:")
    print("    feature_scales_before.png   — motivates normalization")
    print("    feature_correlation.png     — motivates PCA / feature removal")
    print("    feature_importance_mi.png   — motivates feature selection")
    print("    pca_variance.png            — shows PCA component count choice")
    print()
    print("  Use the _std_mi version as your primary preprocessed data.")
    print("  Try the _pca version if training is too slow or you want to compare.")
    print("\nReady for Section 2 (Evaluation) and Section 3 (Experiments).")