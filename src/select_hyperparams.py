"""
select_hyperparams.py — Select best hyperparameters from sweep_results.csv.

For each (dataset, model, loss_type), finds the hyperparameter tuple that
minimises the sum of best_val_loss across all K depths (and splits).

Usage:
    python src/select_hyperparams.py
    python src/select_hyperparams.py --hetero-split-mode first
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

SWEEP_RESULTS_PATH = Path(cfg.results_dir) / "sweep_results.csv"
BEST_HYPERPARAMS_PATH = Path(cfg.results_dir) / "best_hyperparams.csv"

# Columns that define a hyperparameter configuration
HYPERPARAM_COLS = [
    "lr", "weight_decay", "patience", "max_epochs", "hidden_dim",
    "beta", "lambda_r", "entropy_floor", "per_class_r",
    "band_lower", "band_upper",
]


def select_hyperparams(hetero_split_mode: str = "all") -> pd.DataFrame:
    """
    For each (dataset, model, loss_type):
      - Optionally restrict heterophilous datasets to split=0
      - Group by hyperparameter tuple
      - Sum best_val_loss across all K (and splits)
      - Return the tuple with the lowest total val loss

    Args:
        hetero_split_mode: 'first' to use only split 0 for hetero datasets,
                           'all' to use all splits.
    Returns:
        DataFrame with one row per (dataset, model, loss_type).
    """
    if not SWEEP_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Sweep results not found: {SWEEP_RESULTS_PATH}")

    df = pd.read_csv(SWEEP_RESULTS_PATH)
    print(f"Loaded {len(df)} rows from {SWEEP_RESULTS_PATH}")

    # Optionally restrict hetero datasets to split 0
    if hetero_split_mode == "first":
        hetero_mask = df["dataset"].isin(cfg.heterophilous_datasets)
        df = df[~hetero_mask | (df["split"] == 0)].copy()
        print(f"  Restricted hetero datasets to split=0: {len(df)} rows remaining")

    results = []
    group_cols = ["dataset", "model", "loss_type"]

    for group_keys, group_df in df.groupby(group_cols):
        dataset, model, loss_type = group_keys

        # For each hyperparameter combination, sum val loss across all K and splits
        agg = (
            group_df
            .groupby(HYPERPARAM_COLS, dropna=False)["best_val_loss"]
            .sum()
            .reset_index()
            .rename(columns={"best_val_loss": "total_val_loss"})
        )

        # Pick the combination with the lowest total val loss
        best_idx = agg["total_val_loss"].idxmin()
        best_row = agg.loc[best_idx].to_dict()

        result = {
            "dataset": dataset,
            "model": model,
            "loss_type": loss_type,
            "total_val_loss": best_row["total_val_loss"],
            "n_runs_aggregated": len(group_df),
        }
        result.update({col: best_row[col] for col in HYPERPARAM_COLS})
        results.append(result)

        print(f"  {dataset} / {model} / {loss_type}: "
              f"best total_val_loss={best_row['total_val_loss']:.4f} "
              f"lr={best_row['lr']}, wd={best_row['weight_decay']}, "
              f"patience={best_row['patience']}")

    best_df = pd.DataFrame(results)
    BEST_HYPERPARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(BEST_HYPERPARAMS_PATH, index=False)
    print(f"\nSaved best hyperparameters to: {BEST_HYPERPARAMS_PATH}")
    return best_df


def main():
    parser = argparse.ArgumentParser(description="Select best hyperparameters from sweep results")
    parser.add_argument(
        "--hetero-split-mode", type=str, default="all",
        choices=["all", "first"],
        help="'first': use only split 0 for heterophilous datasets. "
             "'all': aggregate over all splits (default)."
    )
    args = parser.parse_args()

    best_df = select_hyperparams(hetero_split_mode=args.hetero_split_mode)
    print("\nBest hyperparameters:")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
