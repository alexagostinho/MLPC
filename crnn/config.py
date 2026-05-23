"""Central configuration for the CRNN sound-event-detection model.

Every hyperparameter the experiment depends on lives in the single ``Config``
dataclass below. That is the whole point of this layout: to tune by hand you
either edit a default here, or override it on the command line without touching
any other file::

    python -m crnn.train --lr 5e-4 --gru-dim 256 --conv-layers 3 --dropout 0.4

Each run is written to ``runs/<run_name>/`` together with the exact config used
(``config.json``), so manual experiments stay reproducible and comparable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Repo root = parent of the `crnn/` package. Data lives next to the package.
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---- data ----
    feat_dir: str = str(REPO_ROOT / "audio_features")
    meta_path: str = str(REPO_ROOT / "metadata.csv")
    overlap_thresh: float = 0.5       # annotator overlap -> binary positive
    agreement_thresh: float = 0.5     # fraction of annotators that must agree
    batch_size: int = 32
    num_workers: int = 0

    # ---- model architecture ----
    conv_dim: int = 192
    conv_layers: int = 2
    kernel_size: int = 3
    gru_dim: int = 256
    gru_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.35

    # ---- optimization ----
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 100
    patience: int = 12                # early-stopping patience (epochs)
    sched_factor: float = 0.5         # ReduceLROnPlateau LR multiplier
    sched_patience: int = 3
    use_pos_weight: bool = True       # class-frequency weighting in BCE

    # ---- threshold tuning / evaluation ----
    thr_min: float = 0.05
    thr_max: float = 0.95
    thr_steps: int = 91               # grid points in [thr_min, thr_max]

    # ---- bookkeeping ----
    seed: int = 42
    device: str = "auto"              # "auto" | "cuda" | "cpu"
    runs_dir: str = str(REPO_ROOT / "runs")
    run_name: str = ""                # empty -> timestamp (set in train.py)

    def to_dict(self) -> dict:
        return asdict(self)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add one ``--flag`` per Config field so any default can be overridden.

    Defaults are left as ``None`` so we can tell "user passed it" from "use the
    dataclass default" in :func:`config_from_args`.
    """
    group = parser.add_argument_group("config overrides")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            group.add_argument(flag, dest=f.name,
                               action=argparse.BooleanOptionalAction, default=None)
        else:
            group.add_argument(flag, dest=f.name, type=type(f.default),
                               default=None, metavar=f"{f.default!r}")


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a Config from parsed args, applying only the flags the user set."""
    overrides = {f.name: getattr(args, f.name) for f in fields(Config)
                 if getattr(args, f.name, None) is not None}
    return Config(**overrides)
