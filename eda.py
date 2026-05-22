"""Quick real EDA on the local MLPC dataset. Read-only; prints summary stats."""
import os, glob
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(DATA_DIR, "audio_features")
OVERLAP, AGREE = 0.5, 0.5

meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
ann_csv = pd.read_csv(os.path.join(DATA_DIR, "annotations.csv"))
npz_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npz")))

print(f"=== FILES ===")
print(f"npz files            : {len(npz_files)}")
print(f"metadata rows        : {len(meta)}")
print(f"annotations.csv rows : {len(ann_csv)}")
print(f"metadata columns     : {list(meta.columns)}")
if "collector_id" in meta:
    print(f"unique collectors    : {meta['collector_id'].nunique()}")
    print(f"files per collector  : min={meta.groupby('collector_id').size().min()}, "
          f"median={int(meta.groupby('collector_id').size().median())}, "
          f"max={meta.groupby('collector_id').size().max()}")

# --- inspect one file ---
s = dict(np.load(npz_files[0], allow_pickle=True))
class_names = list(s["class_names"])
C = len(class_names)
print(f"\n=== SHAPES (sample {os.path.basename(npz_files[0])}) ===")
print(f"keys: {sorted(s.keys())}")
print(f"annotations shape    : {s['annotations'].shape}  (T, C, A)")
print(f"num classes C        : {C}")
print(f"num annotators (this file): {s['annotations'].shape[2]}")

# feature dims for mean+std set used by the code
feat_keys = ["mfcc_mean","mfcc_std","mfcc_d_mean","mfcc_d_std","mfcc_d2_mean","mfcc_d2_std",
             "melspect_mean","melspect_std","zcr_mean","zcr_std","flux_mean","flux_std",
             "flatness_mean","flatness_std","centroid_mean","centroid_std","bandwidth_mean",
             "bandwidth_std","contrast_mean","contrast_std","rolloff_low_mean","rolloff_low_std",
             "rolloff_high_mean","rolloff_high_std","energy_mean","energy_std","power_mean","power_std"]
avail = [k for k in feat_keys if k in s]
D = sum(s[k].shape[1] if s[k].ndim > 1 else 1 for k in avail)
print(f"mean+std feature dim D: {D}  ({len(avail)} keys present)")
allkeys = [k for k in s if k not in ("annotations","class_names","annotator_ids",
            "is_own_recording","start_time","end_time")]
print(f"all feature keys available: {len(allkeys)}")

# --- aggregate labels across all files (majority vote) + stats ---
def agg(a):
    b = (a >= OVERLAP).astype(np.float32)
    return (b.mean(axis=2) >= AGREE).astype(np.int32)

total_segs = 0
class_pos = np.zeros(C, dtype=np.int64)
n_annotators = []
labels_per_seg = []   # number of positive classes per segment
nan_count = 0
val_count = 0
T_per_file = []
for p in npz_files:
    d = np.load(p, allow_pickle=True)
    a = d["annotations"]
    n_annotators.append(a.shape[2])
    lab = agg(a)
    total_segs += lab.shape[0]
    T_per_file.append(lab.shape[0])
    class_pos += lab.sum(axis=0)
    labels_per_seg.append(lab.sum(axis=1))
    X = np.concatenate([d[k] if d[k].ndim > 1 else d[k][:, None] for k in avail], axis=1)
    nan_count += (~np.isfinite(X)).sum()
    val_count += X.size

labels_per_seg = np.concatenate(labels_per_seg)
print(f"\n=== SCALE ===")
print(f"total segments       : {total_segs:,}")
print(f"segments/file        : min={min(T_per_file)}, median={int(np.median(T_per_file))}, max={max(T_per_file)}")
print(f"annotators/file      : min={min(n_annotators)}, median={int(np.median(n_annotators))}, max={max(n_annotators)}")
print(f"NaN/Inf in features  : {nan_count:,} / {val_count:,} ({nan_count/val_count:.4%})")

print(f"\n=== MULTI-LABEL DENSITY ===")
print(f"avg positive classes / segment: {labels_per_seg.mean():.3f}")
print(f"segments with 0 labels        : {(labels_per_seg==0).mean():.1%}")
print(f"segments with 1 label         : {(labels_per_seg==1).mean():.1%}")
print(f"segments with >=2 labels      : {(labels_per_seg>=2).mean():.1%}")
print(f"max simultaneous labels       : {labels_per_seg.max()}")

print(f"\n=== CLASS FREQUENCY (majority-vote, segment-level) ===")
order = np.argsort(class_pos)[::-1]
print(f"{'Class':<28}{'pos_count':>12}{'freq':>10}")
print("-"*50)
for i in order:
    print(f"{class_names[i]:<28}{class_pos[i]:>12,}{class_pos[i]/total_segs:>10.4f}")
print(f"\nimbalance ratio (max freq / min nonzero freq): "
      f"{class_pos.max()/max(class_pos[class_pos>0].min(),1):.0f}x")
print(f"classes with zero positives: {(class_pos==0).sum()}")
