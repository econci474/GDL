"""
results_tracker.py — Append one row per completed split to sweep_results.csv.

Uses filelock for safe concurrent writes from parallel training runs.
Falls back to a simple append if filelock is not installed.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

SWEEP_RESULTS_PATH = Path(cfg.results_dir) / "sweep_results.csv"

COLUMNS = [
    "dataset", "model", "loss_type",
    "K", "seed", "split",
    "lr", "weight_decay", "patience", "max_epochs", "hidden_dim",
    "beta", "lambda_r", "entropy_floor", "per_class_r",
    "band_lower", "band_upper",
    "best_epoch", "best_val_loss", "best_val_acc",
    "timestamp",
]


def append_result(row: dict) -> None:
    """Append one result row to sweep_results.csv (thread/process safe)."""
    row.setdefault("timestamp", datetime.now().isoformat())

    # Ensure all columns present (fill missing with None)
    full_row = {col: row.get(col, None) for col in COLUMNS}

    SWEEP_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SWEEP_RESULTS_PATH.exists()

    def _write():
        with open(SWEEP_RESULTS_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(full_row)

    try:
        from filelock import FileLock
        lock_path = SWEEP_RESULTS_PATH.with_suffix(".lock")
        with FileLock(str(lock_path), timeout=30):
            _write()
    except ImportError:
        # filelock not installed — safe for sequential runs
        _write()
