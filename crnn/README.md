# CRNN — frame-wise multi-label sound event detection

Conv1d → BiGRU → per-timestep sigmoid over 15 classes. Refactor of the old
`crnn_classifier.py` into a small package so it's easy to read and tune by hand.

## Run

```bash
conda activate qsar_torch
python -m crnn.train                       # train with defaults from config.py
```

## Tune hyperparameters by hand

Every knob lives in the `Config` dataclass in `config.py`. Two ways to change it:

1. **Edit the default** in `config.py` (good for a setting you've settled on).
2. **Override on the CLI** (good for sweeping one run at a time) — every field is
   a flag, `_` → `-`:

```bash
python -m crnn.train --lr 5e-4 --gru-dim 256 --conv-layers 3 --dropout 0.4
python -m crnn.train --no-bidirectional --gru-layers 1        # boolean flags
python -m crnn.train --run-name big_gru                       # name the output folder
```

`python -m crnn.train --help` lists every tunable with its default.

## What each run produces

Each run writes to `runs/<run_name>/` (default `run_name` = timestamp):

| file | contents |
|------|----------|
| `config.json` | exact hyperparameters used — the record of *what you ran* |
| `history.json` | per-epoch train loss / val F1 |
| `metrics.json` | final test macro/micro F1 + per-class F1 |
| `crnn.pt` | best checkpoint + standardization stats + class names |
| `test_predictions.npz` | labels, probabilities, tuned thresholds |

Because the config is saved next to the metrics, you can compare manual runs
later without guessing which settings produced which number.

## Module map

| file | responsibility |
|------|----------------|
| `config.py` | all hyperparameters + CLI parsing |
| `data.py` | load sequences, collector split, standardization, DataLoaders |
| `model.py` | the `CRNN` module only |
| `engine.py` | training loop + inference (no I/O) |
| `metrics.py` | threshold tuning, F1, per-class report, CatBoost reference |
| `train.py` | wires it together and logs the run |
```
