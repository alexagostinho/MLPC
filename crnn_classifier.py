"""
Frame-wise CRNN for multi-label sound event detection.

Each recording is a sequence of per-second feature vectors. The model learns
temporal context end-to-end (instead of the hand-built +/-2 window used for the
trees):

    [T, 960] per-segment features
      -> Conv1d stack (local temporal patterns: onsets/offsets)
      -> BiGRU       (longer-range context across the recording)
      -> Linear -> 15 sigmoid, per timestep (multi-label)

A single network predicts all 15 classes at once, so rare classes share the
representation learned from common ones. Collector-level split + pos_weight +
per-class threshold tuning, to stay comparable with the CatBoost results (0.584).

Run in `qsar_torch` env (PyTorch + CUDA).
"""
import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(DATA_DIR, "audio_features")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
OUT_DIR = os.path.join(DATA_DIR, "models_crnn")

SEED = 42
OVERLAP_THRESH = 0.5
AGREEMENT_THRESH = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_NAMES = ["mfcc", "mfcc_d", "mfcc_d2", "melspect", "zcr", "flux", "flatness",
              "centroid", "bandwidth", "contrast", "rolloff_low", "rolloff_high",
              "energy", "power"]
AGGS = ["mean", "std", "min", "max"]
FEATURE_KEYS = [f"{b}_{a}" for b in BASE_NAMES for a in AGGS]

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def aggregate_labels(annotations):
    binary = (annotations >= OVERLAP_THRESH).astype(np.float32)
    return (binary.mean(axis=2) >= AGREEMENT_THRESH).astype(np.float32)


def load_sequences():
    """Return per-file (X[T,D], y[T,C]) lists plus the collector of each file."""
    meta = pd.read_csv(META_PATH)
    fname2collector = dict(zip(
        meta["filename"].str.replace(".wav", "", regex=False), meta["collector_id"]))
    npz_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.npz")))

    Xs, ys, collectors, class_names = [], [], [], None
    t0 = time.time()
    for path in npz_files:
        fname = os.path.splitext(os.path.basename(path))[0]
        d = np.load(path, allow_pickle=True)
        if class_names is None:
            class_names = list(d["class_names"])
        feats = [d[k] if d[k].ndim > 1 else d[k][:, None] for k in FEATURE_KEYS]
        Xs.append(np.concatenate(feats, axis=1).astype(np.float32))
        ys.append(aggregate_labels(d["annotations"]))
        collectors.append(fname2collector.get(fname, "unknown"))
    print(f"Loaded {len(Xs)} sequences in {time.time()-t0:.1f}s, "
          f"D={Xs[0].shape[1]}, C={len(class_names)}")
    return Xs, ys, np.array(collectors), class_names


def split_by_collector(collectors, seed=SEED):
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
    print(f"Files: train={len(idx['train'])}, val={len(idx['val'])}, test={len(idx['test'])}")
    return idx


class SeqDataset(Dataset):
    def __init__(self, Xs, ys, indices, mean, std):
        self.X = [(Xs[i] - mean) / std for i in indices]
        self.y = [ys[i] for i in indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])


def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    xp = pad_sequence(xs, batch_first=True)          # [B, Tmax, D]
    yp = pad_sequence(ys, batch_first=True)          # [B, Tmax, C]
    return xp, yp, lengths


class CRNN(nn.Module):
    def __init__(self, in_dim, n_classes, conv_dim=128, gru_dim=128,
                 conv_layers=2, dropout=0.3):
        super().__init__()
        convs = []
        prev = in_dim
        for _ in range(conv_layers):
            convs += [nn.Conv1d(prev, conv_dim, kernel_size=3, padding=1),
                      nn.BatchNorm1d(conv_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = conv_dim
        self.conv = nn.Sequential(*convs)
        self.gru = nn.GRU(conv_dim, gru_dim, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.head = nn.Linear(2 * gru_dim, n_classes)

    def forward(self, x, lengths):
        # x: [B, T, D]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)          # [B, T, conv_dim]
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)        # [B, T, 2*gru]
        return self.head(out)                                      # [B, T, C] logits


def gather_preds(model, loader):
    """Return concatenated per-segment (unpadded) probs and labels."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xp, yp, lengths in loader:
            xp = xp.to(DEVICE)
            logits = model(xp, lengths)
            p = torch.sigmoid(logits).cpu().numpy()
            yp = yp.numpy()
            for b, L in enumerate(lengths.numpy()):
                probs.append(p[b, :L])
                labels.append(yp[b, :L])
    return np.concatenate(probs), np.concatenate(labels)


def best_thresholds(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    thr = np.zeros(y.shape[1])
    for c in range(y.shape[1]):
        f1s = [f1_score(y[:, c], (p[:, c] >= t).astype(int), zero_division=0) for t in grid]
        thr[c] = grid[int(np.argmax(f1s))]
    return thr


def main():
    print("=" * 72)
    print(f"CRNN (Conv1d -> BiGRU) multi-label SED — device={DEVICE}")
    print("=" * 72)
    Xs, ys, collectors, class_names = load_sequences()
    C = len(class_names)
    idx = split_by_collector(collectors)

    # standardization stats from train segments only
    train_cat = np.concatenate([Xs[i] for i in idx["train"]], axis=0)
    mean = train_cat.mean(axis=0); std = train_cat.std(axis=0) + 1e-6
    del train_cat

    loaders = {s: DataLoader(SeqDataset(Xs, ys, idx[s], mean, std),
                             batch_size=32, shuffle=(s == "train"),
                             collate_fn=collate, num_workers=0)
               for s in ["train", "val", "test"]}

    # pos_weight from train segment frequencies
    y_train_all = np.concatenate([ys[i] for i in idx["train"]], axis=0)
    pos = y_train_all.sum(axis=0); neg = len(y_train_all) - pos
    pos_weight = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32, device=DEVICE)

    model = CRNN(Xs[0].shape[1], C).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=3)

    os.makedirs(OUT_DIR, exist_ok=True)
    best_f1, best_state, best_epoch = -1, None, 0
    n_epochs, patience = 60, 8

    print("\n── Training ──")
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for xp, yp, lengths in loaders["train"]:
            xp, yp = xp.to(DEVICE), yp.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xp, lengths)
            # mask padded timesteps
            mask = (torch.arange(xp.size(1), device=DEVICE)[None, :]
                    < lengths.to(DEVICE)[:, None]).float()[:, :, None]
            loss = (criterion(logits, yp) * mask).sum() / (mask.sum() * yp.size(-1))
            loss.backward()
            optimizer.step()
            total += loss.item() * xp.size(0)

        p_va, y_va = gather_preds(model, loaders["val"])
        val_f1 = f1_score(y_va, (p_va >= 0.5).astype(int), average="macro", zero_division=0)
        scheduler.step(val_f1)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  epoch {epoch:>2}: loss={total/len(loaders['train'].dataset):.4f} "
                  f"val_F1@0.5={val_f1:.4f} ({time.time()-t0:.1f}s)")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    torch.save({"state": best_state, "mean": mean, "std": std,
                "class_names": class_names}, os.path.join(OUT_DIR, "crnn.pt"))

    # tune thresholds on val, evaluate on test
    p_va, y_va = gather_preds(model, loaders["val"])
    thr = best_thresholds(y_va, p_va)
    p_te, y_te = gather_preds(model, loaders["test"])
    pred_tuned = (p_te >= thr[None, :]).astype(int)
    pred_05 = (p_te >= 0.5).astype(int)

    fm = f1_score(y_te, pred_tuned, average="macro", zero_division=0)
    fmi = f1_score(y_te, pred_tuned, average="micro", zero_division=0)
    fm05 = f1_score(y_te, pred_05, average="macro", zero_division=0)

    print("\n" + "=" * 72)
    print("TEST RESULTS")
    print("=" * 72)
    print(f"  CRNN @0.5        : F1_macro = {fm05:.4f}")
    print(f"  CRNN tuned-thr   : F1_macro = {fm:.4f}   F1_micro = {fmi:.4f}")
    print(f"  (CatBoost improved reference: F1_macro = 0.5842)")

    f1_pc = f1_score(y_te, pred_tuned, average=None, zero_division=0)
    cb_prev = {"keyboard_typing":0.8225,"running_water":0.7934,"vacuum_cleaner":0.7729,
               "cutlery_dishes":0.6861,"keychain":0.6842,"toilet_flushing":0.6739,
               "footsteps":0.6477,"microwave":0.6419,"coffee_machine":0.6416,
               "phone_ringing":0.6319,"door_open_close":0.4701,"bell_ringing":0.4171,
               "window_open_close":0.3654,"wardrobe_drawer_open_close":0.3134,
               "light_switch":0.2013}
    print("\nPer-class F1 (CRNN vs CatBoost-improved):")
    print(f"{'Class':<28}{'CRNN':>8}{'CatB':>8}{'delta':>8}")
    print("-" * 52)
    for i in np.argsort(f1_pc)[::-1]:
        cb = cb_prev.get(class_names[i], float("nan"))
        print(f"{class_names[i]:<28}{f1_pc[i]:>8.4f}{cb:>8.4f}{f1_pc[i]-cb:>+8.4f}")

    np.savez(os.path.join(OUT_DIR, "test_predictions.npz"),
             y_test=y_te, proba=p_te, thresholds=thr, class_names=np.array(class_names))
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
