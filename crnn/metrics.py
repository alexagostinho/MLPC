"""Evaluation helpers: per-class threshold tuning, F1 scores, and reporting."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

# Reference numbers from the improved CatBoost model, for side-by-side reporting.
CATBOOST_REF = {
    "keyboard_typing": 0.8225, "running_water": 0.7934, "vacuum_cleaner": 0.7729,
    "cutlery_dishes": 0.6861, "keychain": 0.6842, "toilet_flushing": 0.6739,
    "footsteps": 0.6477, "microwave": 0.6419, "coffee_machine": 0.6416,
    "phone_ringing": 0.6319, "door_open_close": 0.4701, "bell_ringing": 0.4171,
    "window_open_close": 0.3654, "wardrobe_drawer_open_close": 0.3134,
    "light_switch": 0.2013,
}
CATBOOST_MACRO = 0.5842


def best_thresholds(y, p, thr_min=0.05, thr_max=0.95, thr_steps=91):
    """Per-class decision threshold that maximizes F1 on the given probs."""
    grid = np.linspace(thr_min, thr_max, thr_steps)
    thr = np.zeros(y.shape[1])
    for c in range(y.shape[1]):
        f1s = [f1_score(y[:, c], (p[:, c] >= t).astype(int), zero_division=0)
               for t in grid]
        thr[c] = grid[int(np.argmax(f1s))]
    return thr


def macro_f1(y, pred):
    return f1_score(y, pred, average="macro", zero_division=0)


def micro_f1(y, pred):
    return f1_score(y, pred, average="micro", zero_division=0)


def per_class_f1(y, pred):
    return f1_score(y, pred, average=None, zero_division=0)


def report_per_class(y, pred, class_names):
    """Print a per-class F1 table against the CatBoost reference. Returns the
    per-class F1 array."""
    f1_pc = per_class_f1(y, pred)
    print("\nPer-class F1 (CRNN vs CatBoost-improved):")
    print(f"{'Class':<28}{'CRNN':>8}{'CatB':>8}{'delta':>8}")
    print("-" * 52)
    for i in np.argsort(f1_pc)[::-1]:
        cb = CATBOOST_REF.get(class_names[i], float("nan"))
        print(f"{class_names[i]:<28}{f1_pc[i]:>8.4f}{cb:>8.4f}{f1_pc[i] - cb:>+8.4f}")
    return f1_pc
