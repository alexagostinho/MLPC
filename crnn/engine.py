"""Training and inference engine: pure compute, no argument parsing or file I/O.

The trainer selects the best epoch by validation macro-F1 @0.5, drives the LR
scheduler on the same signal, and early-stops on patience.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import macro_f1


def gather_preds(model, loader, device):
    """Return concatenated per-segment (unpadded) probs and labels over a loader."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xp, yp, lengths in loader:
            xp = xp.to(device)
            logits = model(xp, lengths)
            p = torch.sigmoid(logits).cpu().numpy()
            yp = yp.numpy()
            for b, L in enumerate(lengths.numpy()):
                probs.append(p[b, :L])
                labels.append(yp[b, :L])
    return np.concatenate(probs), np.concatenate(labels)


def _masked_bce(criterion, logits, yp, lengths, device):
    """BCE averaged over real (non-padded) timesteps and classes."""
    mask = (torch.arange(logits.size(1), device=device)[None, :]
            < lengths.to(device)[:, None]).float()[:, :, None]
    return (criterion(logits, yp) * mask).sum() / (mask.sum() * yp.size(-1))


def train_model(cfg, model, loaders, pos_weight, device):
    """Train the model; return (best_state, best_val_f1, history).

    ``history`` is a list of per-epoch dicts (epoch, train_loss, val_f1) suitable
    for dumping to JSON.
    """
    weight = pos_weight.to(device) if (cfg.use_pos_weight and pos_weight is not None) else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.sched_factor, patience=cfg.sched_patience)

    best_f1, best_state, best_epoch = -1.0, None, 0
    history = []

    print("\n-- Training --")
    for epoch in range(1, cfg.n_epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch:>3}", leave=False)
        for xp, yp, lengths in pbar:
            xp, yp = xp.to(device), yp.to(device)
            optimizer.zero_grad()
            logits = model(xp, lengths)
            loss = _masked_bce(criterion, logits, yp, lengths, device)
            loss.backward()
            optimizer.step()
            total += loss.item() * xp.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total / len(loaders["train"].dataset)
        p_va, y_va = gather_preds(model, loaders["val"], device)
        val_f1 = macro_f1(y_va, (p_va >= 0.5).astype(int))
        scheduler.step(val_f1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1})

        if epoch % 2 == 0 or epoch == 1:
            print(f"  epoch {epoch:>2}: loss={train_loss:.4f} "
                  f"val_F1@0.5={val_f1:.4f} ({time.time() - t0:.1f}s)")

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= cfg.patience:
            print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    print(f"Best val F1@0.5 = {best_f1:.4f} at epoch {best_epoch}")
    return best_state, best_f1, history
