"""
run_sweep.py — Hyperparameter sweep runner.

Iterates over all combinations of (dataset, model, loss_type, K, seed, [split])
and calls the appropriate training script as a subprocess.

Usage:
    # Full sweep — everything from config
    python scripts/run_sweep.py \\
        --datasets all --models all --loss-types all \\
        --K-values all --seeds all --split-mode first

    # Filtered sweep
    python scripts/run_sweep.py \\
        --datasets Cora PubMed \\
        --models GCN GAT \\
        --loss-types ce_only weighted_ce \\
        --K-values 2 4 6 8 \\
        --seeds 0 1 \\
        --split-mode first
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg

# Try to import pandas for skip-existing (optional)
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

ALL_LOSS_TYPES = ["ce_only", "weighted_ce", "ce_plus_R", "weighted_ce_plus_R", "R_only"]
# Loss types that need entropy-specific hyperparams
ENTROPY_LOSS_TYPES = {"ce_plus_R", "weighted_ce_plus_R", "R_only"}
# Loss types that use lambda_R
LAMBDA_R_LOSS_TYPES = {"ce_plus_R", "weighted_ce_plus_R", "R_only"}


def expand_all(arg_list, all_values, cast=None):
    """If arg_list == ['all'], return all_values. Otherwise return cast(x) for each x."""
    if arg_list == ["all"]:
        return all_values
    return [cast(x) for x in arg_list] if cast else arg_list


def build_gnn_grid(dataset: str) -> list[dict]:
    """Build list of hyperparam dicts for standard GNN (train_gnn.py)."""
    if dataset in cfg.homophilous_datasets:
        grid = cfg.sweep_homophilous
    else:
        grid = cfg.sweep_heterophilous

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


def build_entropy_grid(dataset: str, loss_type: str) -> list[dict]:
    """Build list of hyperparam dicts for classifier head (train_gnn_entropy.py)."""
    if dataset in cfg.homophilous_datasets:
        base_grid = cfg.sweep_homophilous
    else:
        base_grid = cfg.sweep_heterophilous

    base_keys = list(base_grid.keys())
    base_combos = list(itertools.product(*[base_grid[k] for k in base_keys]))

    eg = cfg.sweep_entropy

    if loss_type in LAMBDA_R_LOSS_TYPES:
        # Full entropy grid: beta + lambda_R + entropy_floor + per_class_R + band
        entropy_keys = ["beta", "lambda_R", "entropy_floor", "per_class_R", "band"]
        entropy_combos = list(itertools.product(
            eg["beta"], eg["lambda_R"], eg["entropy_floor"],
            eg["per_class_R"], eg["band"]
        ))
    else:
        # ce_only / weighted_ce: only sweep beta; no regulariser params
        entropy_keys = ["beta"]
        entropy_combos = [(b,) for b in eg["beta"]]

    all_combos = []
    for base_combo in base_combos:
        base_dict = dict(zip(base_keys, base_combo))
        for entropy_combo in entropy_combos:
            entropy_dict = dict(zip(entropy_keys, entropy_combo))
            # Unpack band tuple into band_lower, band_upper (only present for regulariser types)
            if "band" in entropy_dict:
                band = entropy_dict.pop("band")
                entropy_dict["band_lower"] = band[0]
                entropy_dict["band_upper"] = band[1]
            all_combos.append({**base_dict, **entropy_dict})

    return all_combos


def load_completed_runs(results_csv: Path) -> set:
    """Load set of completed run signatures from sweep_results.csv."""
    if not _PANDAS_AVAILABLE or not results_csv.exists():
        return set()
    try:
        df = pd.read_csv(results_csv)
        completed = set()
        key_cols = ['dataset', 'model', 'loss_type', 'K', 'seed', 'split',
                    'lr', 'weight_decay', 'hidden_dim']
        # Only use columns that exist
        key_cols = [c for c in key_cols if c in df.columns]
        for _, row in df.iterrows():
            sig = tuple(str(row[c]) for c in key_cols)
            completed.add(sig)
        print(f'[resume] Loaded {len(completed)} completed runs from {results_csv}')
        return completed
    except Exception as e:
        print(f'[resume] Warning: could not load {results_csv}: {e}')
        return set()


def make_run_signature(cmd: list, key_args: list) -> tuple:
    """Extract key argument values from a command list to form a run signature."""
    vals = {}
    for i, token in enumerate(cmd):
        if token in key_args and i + 1 < len(cmd):
            vals[token] = cmd[i + 1]
    return tuple(vals.get(k, '') for k in key_args)


def run_training(cmd: list[str], dry_run: bool = False) -> int:
    """Run a training command as subprocess. Returns returncode."""
    print(f"\n>>> {' '.join(cmd)}")
    if dry_run:
        print("    [DRY RUN — skipped]")
        return 0
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"    [WARNING] Command exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep runner")
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        help="Datasets to sweep, or 'all'")
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Models to sweep, or 'all'")
    parser.add_argument("--loss-types", nargs="+", default=["all"],
                        help="Loss types to sweep, or 'all'")
    parser.add_argument("--K-values", nargs="+", default=["all"],
                        help="K values to sweep, or 'all'")
    parser.add_argument("--seeds", nargs="+", default=["all"],
                        help="Seeds to sweep, or 'all'")
    parser.add_argument("--split-mode", type=str, default="first",
                        choices=["auto", "first", "all"],
                        help="Split mode for heterophilous datasets (default: first)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs already present in sweep_results.csv (for resuming)")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python executable to use")
    args = parser.parse_args()

    # Expand 'all' shorthands
    datasets   = expand_all(args.datasets,    cfg.datasets)
    models     = expand_all(args.models,      cfg.models)
    loss_types = expand_all(args.loss_types,  ALL_LOSS_TYPES)
    K_values   = expand_all(args.K_values,    list(range(1, cfg.K_max + 1)), cast=int)
    seeds      = expand_all(args.seeds,       cfg.seeds, cast=int)

    print(f"\n{'='*60}")
    print(f"Sweep configuration:")
    print(f"  Datasets:   {datasets}")
    print(f"  Models:     {models}")
    print(f"  Loss types: {loss_types}")
    print(f"  K values:   {K_values}")
    print(f"  Seeds:      {seeds}")
    print(f"  Split mode: {args.split_mode}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"{'='*60}\n")

    # Load completed runs for resume support
    sweep_csv = ROOT / 'results' / 'sweep_results.csv'
    completed_runs = load_completed_runs(sweep_csv) if args.skip_existing else set()
    skipped = 0

    total_runs = 0
    failed_runs = []

    for dataset in datasets:
        for model in models:
            for loss_type in loss_types:
                # Determine which training script and hyperparam grid to use
                if loss_type == "ce_only" and False:
                    # NOTE: ce_only can be run with either script.
                    # We use train_gnn_entropy.py for all loss types for consistency.
                    pass

                is_entropy = loss_type in ENTROPY_LOSS_TYPES or loss_type in ("ce_only", "weighted_ce")
                # Always use train_gnn_entropy.py for classifier heads
                # Use train_gnn.py only for pure baseline (no classifier head)
                # For now: all loss types go through train_gnn_entropy.py
                use_entropy_script = True

                if use_entropy_script:
                    hyperparam_grid = build_entropy_grid(dataset, loss_type)
                    script = "src/train_gnn_entropy.py"
                else:
                    hyperparam_grid = build_gnn_grid(dataset)
                    script = "src/train_gnn.py"

                for K in K_values:
                    for seed in seeds:
                        for hp in hyperparam_grid:
                            cmd = [
                                args.python, script,
                                "--dataset", dataset,
                                "--model", model,
                                "--K", str(K),
                                "--seed", str(seed),
                                "--split-mode", args.split_mode,
                                "--loss-type", loss_type,
                                "--lr", str(hp["lr"]),
                                "--weight-decay", str(hp["weight_decay"]),
                                "--patience", str(hp["patience"]),
                                "--max-epochs", str(hp["max_epochs"]),
                                "--hidden-dim", str(hp["hidden_dim"]),
                            ]

                            # Entropy-specific flags
                            if "beta" in hp:
                                cmd += ["--beta", str(hp["beta"])]
                            if "lambda_R" in hp and loss_type in LAMBDA_R_LOSS_TYPES:
                                cmd += ["--lambda-r", str(hp["lambda_R"])]
                            if "entropy_floor" in hp and hp["entropy_floor"] is not None:
                                cmd += ["--entropy-floor", str(hp["entropy_floor"])]
                            if hp.get("per_class_R"):
                                cmd += ["--per-class-r"]
                            if "band_lower" in hp:
                                cmd += ["--band-lower", str(hp["band_lower"])]
                            if "band_upper" in hp:
                                cmd += ["--band-upper", str(hp["band_upper"])]

                            # Check if already completed (resume support)
                            if args.skip_existing and completed_runs:
                                sig_keys = ['--dataset', '--model', '--loss-type',
                                            '--K', '--seed', '--lr', '--weight-decay', '--hidden-dim']
                                sig = make_run_signature(cmd, sig_keys)
                                if sig in completed_runs:
                                    skipped += 1
                                    continue

                            rc = run_training(cmd, dry_run=args.dry_run)
                            total_runs += 1
                            if rc != 0:
                                failed_runs.append(cmd)

    print(f"\n{'='*60}")
    print(f"Sweep complete. Total runs: {total_runs}")
    if failed_runs:
        print(f"Failed runs ({len(failed_runs)}):")
        for cmd in failed_runs:
            print(f"  {' '.join(cmd)}")
    else:
        print("All runs completed successfully.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
