"""Data pipeline: load per-recording feature sequences, split without leakage,
standardize, and wrap in DataLoaders.

Each recording is a sequence of 1-second frames; every frame is a 960-d vector
of aggregated (mean/std/min/max) features. Labels are per-frame, multi-label
over 15 classes. The collector-level split keeps all recordings from one person
in a single fold so the model can't cheat via recording-style leakage.
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Feature layout (order matters: it fixes the 960-d concatenation).
BASE_NAMES = ["mfcc", "mfcc_d", "mfcc_d2", "melspect", "zcr", "flux", "flatness",
              "centroid", "bandwidth", "contrast", "rolloff_low", "rolloff_high",
              "energy", "power"]
AGGS = ["mean", "std", "min", "max"]
FEATURE_KEYS = [f"{b}_{a}" for b in BASE_NAMES for a in AGGS]


def aggregate_labels(annotations, overlap_thresh, agreement_thresh):
    """[T, C, n_annot] soft annotations -> [T, C] binary frame labels."""
    binary = (annotations >= overlap_thresh).astype(np.float32)
    return (binary.mean(axis=2) >= agreement_thresh).astype(np.float32)


def load_sequences(cfg):
    """Return per-file (X[T,D], y[T,C]) lists, the collector of each file, and
    the class names."""
    meta = pd.read_csv(cfg.meta_path)
    fname2collector = dict(zip(
        meta["filename"].str.replace(".wav", "", regex=False), meta["collector_id"]))
    npz_files = sorted(glob.glob(os.path.join(cfg.feat_dir, "*.npz")))

    Xs, ys, collectors, class_names = [], [], [], None
    t0 = time.time()
    for path in npz_files:
        fname = os.path.splitext(os.path.basename(path))[0]
        d = np.load(path, allow_pickle=True)
        if class_names is None:
            class_names = list(d["class_names"])
        feats = [d[k] if d[k].ndim > 1 else d[k][:, None] for k in FEATURE_KEYS]
        Xs.append(np.concatenate(feats, axis=1).astype(np.float32))
        ys.append(aggregate_labels(d["annotations"], cfg.overlap_thresh,
                                   cfg.agreement_thresh))
        collectors.append(fname2collector.get(fname, "unknown"))
    print(f"Loaded {len(Xs)} sequences in {time.time() - t0:.1f}s, "
          f"D={Xs[0].shape[1]}, C={len(class_names)}")
    return Xs, ys, np.array(collectors), class_names


def split_by_collector(collectors, seed):
    """Assign whole collectors to train/val/test (70/15/15) -> no leakage."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(collectors)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    sets = {"train": set(uniq[:n_tr]), "val": set(uniq[n_tr:n_tr + n_va]),
            "test": set(uniq[n_tr + n_va:])}
    idx = {s: np.array([i for i, c in enumerate(collectors) if c in cs])
           for s, cs in sets.items()}
    print(f"Files: train={len(idx['train'])}, val={len(idx['val'])}, "
          f"test={len(idx['test'])}")
    return idx


def standardize_stats(Xs, train_idx):
    """Per-feature mean/std computed on train segments only."""
    train_cat = np.concatenate([Xs[i] for i in train_idx], axis=0)
    mean = train_cat.mean(axis=0)
    std = train_cat.std(axis=0) + 1e-6
    return mean, std


def pos_weight_from(ys, train_idx):
    """BCE pos_weight = #neg / #pos per class, from train segment frequencies."""
    y_train = np.concatenate([ys[i] for i in train_idx], axis=0)
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    return (neg / (pos + 1e-6)).astype(np.float32)


class SeqDataset(Dataset):
    """Standardized feature sequences for a set of file indices."""

    def __init__(self, Xs, ys, indices, mean, std):
        self.X = [(Xs[i] - mean) / std for i in indices]
        self.y = [ys[i] for i in indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])


def collate(batch):
    """Pad a batch of variable-length sequences; return padded X, y, lengths."""
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    xp = pad_sequence(xs, batch_first=True)          # [B, Tmax, D]
    yp = pad_sequence(ys, batch_first=True)          # [B, Tmax, C]
    return xp, yp, lengths


def make_loaders(cfg, Xs, ys, idx, mean, std):
    """Build train/val/test DataLoaders (only the train loader is shuffled)."""
    return {s: DataLoader(SeqDataset(Xs, ys, idx[s], mean, std),
                          batch_size=cfg.batch_size, shuffle=(s == "train"),
                          collate_fn=collate, num_workers=cfg.num_workers)
            for s in ["train", "val", "test"]}
