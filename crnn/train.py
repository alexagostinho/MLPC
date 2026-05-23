"""Entry point: train one CRNN run end-to-end and log it.

    python -m crnn.train                       # defaults from config.py
    python -m crnn.train --lr 5e-4 --gru-dim 256 --dropout 0.4
    python -m crnn.train --run-name big_gru    # name the run folder

Outputs land in ``runs/<run_name>/``:
    config.json          exact hyperparameters used
    history.json         per-epoch train loss / val F1
    metrics.json         final test metrics + per-class F1
    crnn.pt              best checkpoint (+ standardization stats, class names)
    test_predictions.npz y_test, probabilities, tuned thresholds
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

# Allow running this file directly (e.g. IDE "Run" button: `python crnn/train.py`)
# in addition to `python -m crnn.train`. When run as a script there is no parent
# package, so put the repo root on sys.path and declare the package ourselves.
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "crnn"

import numpy as np
import torch

from . import data as D
from .config import Config, add_config_args, config_from_args
from .engine import gather_preds, train_model
from .metrics import (CATBOOST_MACRO, best_thresholds, macro_f1, micro_f1,
                      report_per_class)
from .model import CRNN
from .utils import get_device, set_seed


def run(cfg: Config) -> dict:
    device = get_device(cfg.device)
    set_seed(cfg.seed)

    run_name = cfg.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 72)
    print(f"CRNN (Conv1d -> BiGRU) multi-label SED - device={device} - run={run_name}")
    print("=" * 72)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # ---- data ----
    Xs, ys, collectors, class_names = D.load_sequences(cfg)
    C = len(class_names)
    idx = D.split_by_collector(collectors, cfg.seed)
    mean, std = D.standardize_stats(Xs, idx["train"])
    pos_weight = torch.tensor(D.pos_weight_from(ys, idx["train"]), device=device)
    loaders = D.make_loaders(cfg, Xs, ys, idx, mean, std)

    # ---- model ----
    model = CRNN.from_config(cfg, Xs[0].shape[1], C).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ---- train ----
    best_state, best_val_f1, history = train_model(cfg, model, loaders, pos_weight, device)
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "mean": mean, "std": std,
                "class_names": class_names, "config": cfg.to_dict()},
               os.path.join(run_dir, "crnn.pt"))
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ---- tune thresholds on val, evaluate on test ----
    p_va, y_va = gather_preds(model, loaders["val"], device)
    thr = best_thresholds(y_va, p_va, cfg.thr_min, cfg.thr_max, cfg.thr_steps)
    p_te, y_te = gather_preds(model, loaders["test"], device)
    pred_tuned = (p_te >= thr[None, :]).astype(int)
    pred_05 = (p_te >= 0.5).astype(int)

    fm = macro_f1(y_te, pred_tuned)
    fmi = micro_f1(y_te, pred_tuned)
    fm05 = macro_f1(y_te, pred_05)

    print("\n" + "=" * 72)
    print("TEST RESULTS")
    print("=" * 72)
    print(f"  CRNN @0.5        : F1_macro = {fm05:.4f}")
    print(f"  CRNN tuned-thr   : F1_macro = {fm:.4f}   F1_micro = {fmi:.4f}")
    print(f"  (CatBoost improved reference: F1_macro = {CATBOOST_MACRO:.4f})")

    f1_pc = report_per_class(y_te, pred_tuned, class_names)

    np.savez(os.path.join(run_dir, "test_predictions.npz"),
             y_test=y_te, proba=p_te, thresholds=thr,
             class_names=np.array(class_names))
    metrics = {"val_f1_best": float(best_val_f1),
               "test_f1_macro_05": float(fm05),
               "test_f1_macro_tuned": float(fm), "test_f1_micro_tuned": float(fmi),
               "test_f1_per_class": {class_names[i]: float(f1_pc[i]) for i in range(C)},
               "n_params": int(n_params)}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved run to {run_dir}/")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train a CRNN for multi-label sound event detection.")
    add_config_args(parser)
    cfg = config_from_args(parser.parse_args())
    run(cfg)


if __name__ == "__main__":
    main()
