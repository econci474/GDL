"""
select_hyperparams.py — Select best hyperparameters from sweep_results.csv.

For each (dataset, model, loss_type), finds the hyperparameter tuple that
minimises the sum of best_val_loss across all K depths (and splits).

Also generates:
  - Per-layer best_hyperparams CSVs (best val_acc per K value, with "_per_layer" suffix)
  - Per-band best_hyperparams CSVs for ce_plus_R so that both band configurations
    can be evaluated systematically.

Usage:
    python src/select_hyperparams.py
    python src/select_hyperparams.py --hetero-split-mode first
    python src/select_hyperparams.py --model GCN --sweep-csv sweep_results_GCN.csv
    python src/select_hyperparams.py --results-dir /content/drive/MyDrive/GDL/sweep_results --model GCN
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

# Columns that define a hyperparameter configuration
HYPERPARAM_COLS = [
    "lr", "weight_decay", "patience", "max_epochs", "hidden_dim",
    "beta", "lambda_r", "entropy_floor", "per_class_r",
    "band_lower", "band_upper",
]

# Band configurations to generate separate best_hyperparams files for.
# Each entry: (band_lower, band_upper, filename_suffix)
CE_PLUS_R_BANDS = [
    (-1.0,  0.0,   "band-1.0to0.0"),
    (-1.5,  0.25,  "band-1.5to0.25"),
]


def _best_per_group(df: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset, model, loss_type), pick the lowest-total-val-loss config."""
    results = []
    group_cols = ["dataset", "model", "loss_type"]
    for group_keys, group_df in df.groupby(group_cols):
        dataset, model, loss_type = group_keys
        agg = (
            group_df
            .groupby(HYPERPARAM_COLS, dropna=False)["best_val_loss"]
            .agg(total_val_loss="sum", n_runs_aggregated="count")
            .reset_index()
        )
        best_row = agg.loc[agg["total_val_loss"].idxmin()].to_dict()
        result = {
            "dataset": dataset, "model": model, "loss_type": loss_type,
            "total_val_loss": best_row["total_val_loss"],
            "n_runs_aggregated": best_row["n_runs_aggregated"],
        }
        result.update({col: best_row[col] for col in HYPERPARAM_COLS})
        results.append(result)
        print(f"  {dataset}/{model}/{loss_type}: "
              f"best total_val_loss={best_row['total_val_loss']:.4f} "
              f"lr={best_row['lr']}, wd={best_row['weight_decay']}, "
              f"patience={best_row['patience']}")
    return pd.DataFrame(results)


def _best_per_group_per_layer(df: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset, model, loss_type, K), pick the highest-mean-val-acc config."""
    results = []
    group_cols = ["dataset", "model", "loss_type", "K"]
    for group_keys, group_df in df.groupby(group_cols):
        dataset, model, loss_type, K = group_keys
        agg = (
            group_df
            .groupby(HYPERPARAM_COLS, dropna=False)["best_val_acc"]
            .agg(mean_val_acc="mean", n_runs_aggregated="count")
            .reset_index()
        )
        best_row = agg.loc[agg["mean_val_acc"].idxmax()].to_dict()
        result = {
            "dataset": dataset, "model": model, "loss_type": loss_type, "K": K,
            "mean_val_acc": best_row["mean_val_acc"],
            "n_runs_aggregated": best_row["n_runs_aggregated"],
        }
        result.update({col: best_row[col] for col in HYPERPARAM_COLS})
        results.append(result)
        print(f"  {dataset}/{model}/{loss_type}/K={K}: "
              f"best mean_val_acc={best_row['mean_val_acc']:.4f} "
              f"lr={best_row['lr']}, wd={best_row['weight_decay']}, "
              f"patience={best_row['patience']}")
    return pd.DataFrame(results)


def select_hyperparams(
    sweep_csv: Path,
    results_dir: Path,
    model_suffix: str,
    hetero_split_mode: str = "all",
):
    """
    Select best hyperparameters from a sweep CSV.

    Outputs:
      <results_dir>/best_hyperparams<suffix>.csv                    — best overall (across all K)
      <results_dir>/best_hyperparams<suffix>_per_layer.csv          — best per K layer
      <results_dir>/best_hyperparams<suffix>_band-*.csv             — best for each ce_plus_R band
      <results_dir>/best_hyperparams<suffix>_band-*_per_layer.csv   — per-band, per-K
    """
    if not sweep_csv.exists():
        raise FileNotFoundError(f"Sweep results not found: {sweep_csv}")

    df = pd.read_csv(sweep_csv)
    print(f"Loaded {len(df)} rows from {sweep_csv}")

    # Optionally restrict hetero datasets to split 0
    if hetero_split_mode == "first":
        hetero_mask = df["dataset"].isin(cfg.heterophilous_datasets)
        df = df[~hetero_mask | (df["split"] == 0)].copy()
        print(f"  Restricted hetero datasets to split=0: {len(df)} rows remaining")

    results_dir.mkdir(parents=True, exist_ok=True)
    suf = f"_{model_suffix}" if model_suffix else ""

    # -- 1. Overall best (aggregated across all K) ----------------------------
    print("\n-- Overall best hyperparameters (across all K) --")
    best_overall = _best_per_group(df)
    out = results_dir / f"best_hyperparams{suf}.csv"
    best_overall.to_csv(out, index=False)
    print(f"Saved -> {out}")

    # -- 2. Per-layer best (best val_acc for each K) ---------------------------
    print("\n-- Best hyperparameters per layer (per K) --")
    best_per_layer = _best_per_group_per_layer(df)
    out = results_dir / f"best_hyperparams{suf}_per_layer.csv"
    best_per_layer.to_csv(out, index=False)
    print(f"Saved -> {out}")

    # -- 3. Per-band best for ce_plus_R ----------------------------------------
    ce_r_df = df[df["loss_type"] == "ce_plus_R"].copy()
    for band_lower, band_upper, band_label in CE_PLUS_R_BANDS:
        band_df = ce_r_df[
            (ce_r_df["band_lower"] == band_lower) &
            (ce_r_df["band_upper"] == band_upper)
        ].copy()
        if band_df.empty:
            print(f"\n  [SKIP] No data for ce_plus_R band ({band_lower}, {band_upper})")
            continue

        print(f"\n-- Best hyperparameters: ce_plus_R band ({band_lower}, {band_upper}) --")
        best_band = _best_per_group(band_df)
        out = results_dir / f"best_hyperparams{suf}_{band_label}.csv"
        best_band.to_csv(out, index=False)
        print(f"Saved -> {out}")

        print(f"\n-- Best hyperparameters per layer: ce_plus_R band ({band_lower}, {band_upper}) --")
        best_band_per_layer = _best_per_group_per_layer(band_df)
        out = results_dir / f"best_hyperparams{suf}_{band_label}_per_layer.csv"
        best_band_per_layer.to_csv(out, index=False)
        print(f"Saved -> {out}")

    return best_overall


def main():
    parser = argparse.ArgumentParser(
        description="Select best hyperparameters from sweep results"
    )
    parser.add_argument(
        "--hetero-split-mode", type=str, default="all",
        choices=["all", "first"],
        help="'first': use only split 0 for heterophilous datasets (default: all).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name suffix for output files, e.g. 'GCN' -> best_hyperparams_GCN.csv",
    )
    parser.add_argument(
        "--sweep-csv", type=str, default=None,
        help="Path to sweep_results CSV. Defaults to <results-dir>/sweep_results[_MODEL].csv",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Override results directory (e.g. /content/drive/MyDrive/GDL/sweep_results). "
             "Makes CWD irrelevant on Colab.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(cfg.results_dir)
    model_suffix = args.model or ""

    if args.sweep_csv:
        sweep_csv = Path(args.sweep_csv)
    else:
        suf = f"_{model_suffix}" if model_suffix else ""
        # Local layout: results/tables/sweep_results_GCN.csv
        # Colab layout: <results_dir>/sweep_results_GCN.csv (flat)
        local_candidate = Path(cfg.tables_dir) / f"sweep_results{suf}.csv"
        colab_candidate = results_dir / f"sweep_results{suf}.csv"
        sweep_csv = local_candidate if local_candidate.exists() else colab_candidate

    select_hyperparams(
        sweep_csv=sweep_csv,
        results_dir=results_dir,
        model_suffix=model_suffix,
        hetero_split_mode=args.hetero_split_mode,
    )


if __name__ == "__main__":
    main()
