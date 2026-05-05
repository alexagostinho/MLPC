
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from itertools import combinations

data_dir = r"E:\MLPC dataset"
feature_dir = os.path.join(data_dir, "audio_features")
metadata_dir = os.path.join(data_dir, "metadata.csv")
overlap_threshold = 0.5
agreement_threshold = 0.5

#Split ratios (approximate, since we split by collector)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random_seed = 42

def load_all_data(features_dir, metadata_path):
    """Load all NPZ files and metadata."""
    #Load metadata
    metadata = pd.read_csv(metadata_path)

    #Load features
    data = {}
    npz_files = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
    for path in npz_files:
        fname = os.path.splitext(os.path.basename(path))[0]
        content = dict(np.load(path, allow_pickle=True))
        data[fname] = content

    print(f"Loaded {len(data)} NPZ files and metadata with {len(metadata)} entries.")
    return data, metadata


def aggregate_labels(annotations, overlap_thresh=0.5, agreement_thresh=0.5):
    """Majority vote aggregation: [T, C, A] -> [T, C]."""
    binary = (annotations >= overlap_thresh).astype(np.float32)
    avg = binary.mean(axis=2)
    labels = (avg >= agreement_thresh).astype(np.int32)
    return labels


def build_file_info(data, metadata):
    """
    Build a DataFrame with one row per file, containing:
    - filename
    - collector_id (grouping key to prevent leakage)
    - num_segments
    - per-class positive counts (for stratification)
    - class_profile string (for stratified grouping)
    """
    rows = []
    class_names = None

    for fname, content in data.items():
        annotations = content["annotations"]
        labels = aggregate_labels(annotations)  # [T, C]

        if class_names is None:
            class_names = list(content["class_names"])

        #Get collector_id from metadata
        #filename column might have extension or not — handle both
        meta_row = metadata[metadata["filename"].str.replace(".wav", "", regex=False) == fname]
        if meta_row.empty:
            meta_row = metadata[metadata["filename"] == fname]

        collector_id = meta_row["collector_id"].values[0] if not meta_row.empty else "unknown"

        #Per-class presence: does this file contain at least one positive segment?
        class_presence = (labels.sum(axis=0) > 0).astype(int)

        row = {
            "filename": fname,
            "collector_id": collector_id,
            "num_segments": labels.shape[0],
            "labels": labels,  # store for later
        }
        for i, cname in enumerate(class_names):
            row[f"has_{cname}"] = class_presence[i]
            row[f"count_{cname}"] = int(labels[:, i].sum())

        rows.append(row)

    df = pd.DataFrame(rows)

    #Create a class profile string for stratification heuristic
    #e.g., "1_0_1_1_0" indicating which classes are present in the file
    presence_cols = [f"has_{c}" for c in class_names]
    df["class_profile"] = df[presence_cols].astype(str).agg("_".join, axis=1)

    print(f"\nFile-level info:")
    print(f"  Total files: {len(df)}")
    print(f"  Unique collectors: {df['collector_id'].nunique()}")
    print(f"  Total segments: {df['num_segments'].sum()}")

    return df, class_names

def split_by_collector(file_info, class_names, train_ratio=0.7, val_ratio=0.15,
                       test_ratio=0.15, seed=42):
    """
    Split at the COLLECTOR level to prevent information leakage.

    Why collector-level?
    - Segments from the same recording share background noise, acoustics,
      microphone characteristics → file-level split is the minimum.
    - Recordings from the same collector often share the same room, device,
      and ambient conditions → collector-level split is stronger.

    Strategy:
    1. Group files by collector.
    2. Compute per-collector class distributions.
    3. Greedily assign collectors to splits trying to balance class distributions.
    """
    rng = np.random.RandomState(seed)

    #Aggregate class counts per collector
    count_cols = [f"count_{c}" for c in class_names]
    collector_info = file_info.groupby("collector_id").agg(
        num_files=("filename", "count"),
        num_segments=("num_segments", "sum"),
        **{col: (col, "sum") for col in count_cols}
    ).reset_index()

    total_segments = collector_info["num_segments"].sum()
    target_train = int(total_segments * train_ratio)
    target_val = int(total_segments * val_ratio)

    #Shuffle collectors
    collectors = collector_info.sample(frac=1, random_state=rng).reset_index(drop=True)

    #Greedy assignment: assign collectors one by one, picking the split
    #that is furthest below its target segment count
    train_ids, val_ids, test_ids = [], [], []
    train_segs, val_segs, test_segs = 0, 0, 0

    for _, row in collectors.iterrows():
        cid = row["collector_id"]
        n = row["num_segments"]

        #How far each split is from its target
        train_gap = target_train - train_segs
        val_gap = target_val - val_segs
        test_gap = (total_segments - target_train - target_val) - test_segs

        #Assign to the split with the largest gap
        gaps = {"train": train_gap, "val": val_gap, "test": test_gap}
        best = max(gaps, key=gaps.get)

        if best == "train":
            train_ids.append(cid)
            train_segs += n
        elif best == "val":
            val_ids.append(cid)
            val_segs += n
        else:
            test_ids.append(cid)
            test_segs += n

    #Map back to files
    split_map = {}
    for cid in train_ids:
        fnames = file_info[file_info["collector_id"] == cid]["filename"].tolist()
        for f in fnames:
            split_map[f] = "train"
    for cid in val_ids:
        fnames = file_info[file_info["collector_id"] == cid]["filename"].tolist()
        for f in fnames:
            split_map[f] = "val"
    for cid in test_ids:
        fnames = file_info[file_info["collector_id"] == cid]["filename"].tolist()
        for f in fnames:
            split_map[f] = "test"

    file_info["split"] = file_info["filename"].map(split_map)

    print(f"\nSplit results (by collector):")
    print(f"  Train collectors: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"  Train segments: {train_segs} ({train_segs / total_segments:.1%})")
    print(f"  Val segments:   {val_segs} ({val_segs / total_segments:.1%})")
    print(f"  Test segments:  {test_segs} ({test_segs / total_segments:.1%})")

    return file_info, split_map


def verify_no_leakage(file_info):
    """Verify that no collector appears in multiple splits."""
    for split_name in ["train", "val", "test"]:
        collectors = set(file_info[file_info["split"] == split_name]["collector_id"])
        for other in ["train", "val", "test"]:
            if other == split_name:
                continue
            other_collectors = set(file_info[file_info["split"] == other]["collector_id"])
            overlap = collectors & other_collectors
            if overlap:
                print(f"  WARNING: Collectors {overlap} appear in both {split_name} and {other}!")
            else:
                print(f"  ✓ No collector overlap between {split_name} and {other}")


def analyze_class_distributions(file_info, data, class_names):
    """
    Compute and display segment-level class frequencies per split.
    Returns a DataFrame suitable for the report.
    """
    split_stats = {}

    for split_name in ["train", "val", "test"]:
        split_files = file_info[file_info["split"] == split_name]["filename"].tolist()

        all_labels = []
        for fname in split_files:
            annotations = data[fname]["annotations"]
            labels = aggregate_labels(annotations)
            all_labels.append(labels)

        if all_labels:
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_labels = np.zeros((0, len(class_names)))

        total = all_labels.shape[0]
        counts = all_labels.sum(axis=0)
        freqs = counts / total if total > 0 else counts

        split_stats[split_name] = {
            "total_segments": total,
            "counts": counts,
            "frequencies": freqs,
        }

    #Print table
    print(f"\n{'Class':<25}", end="")
    for split_name in ["train", "val", "test"]:
        print(f" {split_name + '_freq':>12} {split_name + '_cnt':>10}", end="")
    print()
    print("-" * 95)

    for i, cname in enumerate(class_names):
        print(f"{cname:<25}", end="")
        for split_name in ["train", "val", "test"]:
            freq = split_stats[split_name]["frequencies"][i]
            cnt = int(split_stats[split_name]["counts"][i])
            print(f" {freq:>12.4f} {cnt:>10}", end="")
        print()

    print(f"\n{'TOTAL SEGMENTS':<25}", end="")
    for split_name in ["train", "val", "test"]:
        total = split_stats[split_name]["total_segments"]
        print(f" {'':>12} {total:>10}", end="")
    print()

    return split_stats


def plot_class_distributions(split_stats, class_names):
    """Bar chart of class frequencies per split — for the report."""
    splits = ["train", "val", "test"]
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, split_name in enumerate(splits):
        freqs = split_stats[split_name]["frequencies"]
        ax.bar(x + i * width, freqs, width, label=split_name.capitalize())

    ax.set_xlabel("Sound Event Class")
    ax.set_ylabel("Frequency (fraction of positive segments)")
    ax.set_title("Class Frequency Distribution Across Splits")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig("class_distribution_splits.png", dpi=150)
    plt.show()
    print("Saved: class_distribution_splits.png")

def build_feature_matrices(data, file_info, class_names,
                           feature_keys=None):
    """
    Concatenate features and labels across files for each split.

    Returns:
        splits_data: dict with keys 'train', 'val', 'test', each containing:
            - 'X': feature matrix [N, D]
            - 'y': label matrix [N, C]
            - 'filenames': list of source filenames per segment
    """
    if feature_keys is None:
        #Default: use a broad set of features
        feature_keys = [
            "mfcc_mean", "mfcc_std",
            "mfcc_d_mean", "mfcc_d_std",
            "mfcc_d2_mean", "mfcc_d2_std",
            "melspect_mean", "melspect_std",
            "zcr_mean", "zcr_std",
            "flux_mean", "flux_std",
            "flatness_mean", "flatness_std",
            "centroid_mean", "centroid_std",
            "bandwidth_mean", "bandwidth_std",
            "contrast_mean", "contrast_std",
            "rolloff_low_mean", "rolloff_low_std",
            "rolloff_high_mean", "rolloff_high_std",
            "energy_mean", "energy_std",
            "power_mean", "power_std",
        ]

    #Verify which feature keys actually exist in the data
    sample = next(iter(data.values()))
    available_keys = [k for k in feature_keys if k in sample]
    missing_keys = [k for k in feature_keys if k not in sample]
    if missing_keys:
        print(f"  Note: These feature keys not found, skipping: {missing_keys}")
    feature_keys = available_keys

    splits_data = {}

    for split_name in ["train", "val", "test"]:
        split_files = file_info[file_info["split"] == split_name]["filename"].tolist()

        X_parts = []
        y_parts = []
        fname_parts = []

        for fname in split_files:
            content = data[fname]

            #Concatenate features
            feats = [content[k] for k in feature_keys]
            X_file = np.concatenate(feats, axis=1)  # [T, D]

            #Aggregate labels
            labels = aggregate_labels(content["annotations"])  # [T, C]

            X_parts.append(X_file)
            y_parts.append(labels)
            fname_parts.extend([fname] * X_file.shape[0])

        X = np.concatenate(X_parts, axis=0) if X_parts else np.empty((0, 0))
        y = np.concatenate(y_parts, axis=0) if y_parts else np.empty((0, len(class_names)))

        splits_data[split_name] = {
            "X": X,
            "y": y,
            "filenames": fname_parts,
        }

        print(f"  {split_name}: X={X.shape}, y={y.shape}")

    return splits_data, feature_keys


def save_splits(splits_data, class_names, feature_keys, output_dir):
    """Save split data as .npz files for downstream use."""
    os.makedirs(output_dir, exist_ok=True)

    for split_name, sdata in splits_data.items():
        np.savez_compressed(
            os.path.join(output_dir, f"{split_name}.npz"),
            X=sdata["X"],
            y=sdata["y"],
            filenames=np.array(sdata["filenames"]),
            class_names=np.array(class_names),
            feature_keys=np.array(feature_keys),
        )

    print(f"\nSaved split data to {output_dir}/")
    print(f"  Files: train.npz, val.npz, test.npz")
    print(f"  Load with: data = np.load('train.npz', allow_pickle=True)")


if __name__ == "__main__":
    #Load everything
    data, metadata = load_all_data(feature_dir, metadata_dir)

    #Build file-level info
    file_info, class_names = build_file_info(data, metadata)

    #Split by collector
    print("\n" + "=" * 60)
    print("Splitting by collector (preventing information leakage)")
    print("=" * 60)
    file_info, split_map = split_by_collector(
        file_info, class_names,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        seed=random_seed,
    )

    #Verify no leakage
    print("\n── Leakage Check ──")
    verify_no_leakage(file_info)

    #Analyze class distributions
    print("\n" + "=" * 60)
    print("Class distribuition across splits")
    print("=" * 60)
    split_stats = analyze_class_distributions(file_info, data, class_names)
    plot_class_distributions(split_stats, class_names)

    #Build feature matrices
    print("\n" + "=" * 60)
    print("Building feature matrices")
    print("=" * 60)
    splits_data, feature_keys = build_feature_matrices(data, file_info, class_names)

    #Save
    output_dir = os.path.join(data_dir, "splits")
    save_splits(splits_data, class_names, feature_keys, output_dir)

    #Save the split mapping for reference
    split_df = file_info[["filename", "collector_id", "split", "num_segments"]].copy()
    split_df.to_csv(os.path.join(output_dir, "split_mapping.csv"), index=False)
    print(f"Saved split mapping to {output_dir}/split_mapping.csv")

    print("\n" + "=" * 60)
    print("DONE — Ready for Section 1c (Preprocessing) and beyond.")
    print("=" * 60)