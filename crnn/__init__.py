"""Frame-wise CRNN for multi-label sound event detection.

Conv1d (local temporal patterns) -> BiGRU (long-range context) -> per-timestep
sigmoid over 15 classes. Trained with a collector-level split, class-frequency
weighting, and per-class threshold tuning.

Layout:
    config.py   all hyperparameters (one dataclass) + CLI overrides
    data.py     loading, collector split, standardization, DataLoaders
    model.py    the CRNN module
    engine.py   training loop + inference
    metrics.py  thresholds, F1, per-class report
    train.py    entry point (`python -m crnn.train`)

Run inside the `qsar_torch` conda env (PyTorch + CUDA).
"""
from .config import Config
from .model import CRNN

__all__ = ["Config", "CRNN"]
