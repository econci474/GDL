"""
Evaluate the best saved model checkpoint on the test set.

This script should ONLY be run ONCE after training is complete to
get the final test accuracy. This ensures proper separation between
training/validation and test sets.

Usage:
    # Single run
    python src/evaluate_final.py --dataset Cora --model GCN --K 3 --seed 0
    python src/evaluate_final.py --dataset Cora --model GCN --K 3 --seed 0 \\
        --loss-type weighted_ce_plus_R

    # Batch evaluation from best_hyperparams.csv (per-group: same hyperparams for all K)
    python src/evaluate_final.py --from-best-hyperparams
    python src/evaluate_final.py --from-best-hyperparams --seeds all --K-values all

    # Batch evaluation from best_hyperparams_per_layer.csv (per-layer: K-specific hyperparams)
    python src/evaluate_final.py --from-best-hyperparams \\
        --best-hyperparams-path results/best_hyperparams_GCN_per_layer.csv \\
        --split-mode all --output-path results/comparison_tables/final_results_GCN_per_layer.csv
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from src.datasets import load_dataset
from src.models import GCNNet, GATNet, GraphSAGENet
from src.utils import to_device, get_device

FINAL_RESULTS_PATH = Path(cfg.tables_dir) / "final_results.csv"

FINAL_RESULTS_COLUMNS = [
    "dataset", "model", "method", "loss_type",
    "K", "seed", "split",
    "hidden_dim", "lr", "weight_decay", "max_epochs", "patience",
    "beta", "lambda_r", "entropy_floor", "per_class_r", "band_lower", "band_upper",
    "best_epoch", "best_val_loss", "best_val_acc",
    "test_acc", "test_loss",
    "evaluated_at",
]


def build_model(model_name: str, data, num_classes: int, K: int, config: dict):
    """Factory for models."""
    dropout_input  = config.get("dropout_input")
    dropout_middle = config.get("dropout_middle")
    if model_name == "GCN":
        return GCNNet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            dropout_input=dropout_input,
            dropout_middle=dropout_middle,
            normalize=True,
        )
    elif model_name == "GAT":
        return GATNet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            heads=config.get("gat_heads", 8),
            dropout_input=dropout_input,
            dropout_middle=dropout_middle,
        )
    elif model_name == "GraphSAGE":
        return GraphSAGENet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            aggr=config.get("sage_aggr", "mean"),
            dropout_input=dropout_input,
            dropout_middle=dropout_middle,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use GCN, GAT, GraphSAGE.")


@torch.no_grad()
def evaluate_test_set(model, data, device, use_classifier_head=False):
    """Evaluate model on test set."""
    model.eval()
    data = to_device(data, device)

    if use_classifier_head:
        layer_logits, _ = model.forward_with_classifier_head(data)
        logits = layer_logits[-1]
    else:
        logits = model(data)

    test_loss = F.cross_entropy(logits[data.test_mask], data.y[data.test_mask]).item()
    test_pred = logits[data.test_mask].argmax(dim=1)
    test_acc = (test_pred == data.y[data.test_mask]).sum() / data.test_mask.sum()

    return float(test_loss), float(test_acc.item())


def _build_loss_dir_candidates(loss_type: str, config: dict) -> list:
    """Generate all plausible directory name variants for a given loss/config.

    Returns a list of candidate names, most-preferred first.  This handles
    historical inconsistencies where the training script's default band or
    float format changed across runs:
      - band (-1.0, 0.0): sometimes saved with explicit suffix, sometimes no suffix
      - band (-1.5, 0.25): sometimes saved as 0.25 (:.2f), sometimes as 0.2 (:.1f)
    """
    DECORATED_LOSS_TYPES = {"ce_plus_R", "weighted_ce_plus_R", "R_only"}
    if loss_type not in DECORATED_LOSS_TYPES:
        return [loss_type or "ce_only"]

    def _base_parts(config):
        parts = []
        parts.append(f"R{config.get('lambda_R', 1.0):.1f}")
        parts.append(config.get('R_mode', 'smooth'))
        if config.get('entropy_floor') is not None:
            parts.append(f"floor{config.get('entropy_floor'):.2f}")
        if config.get('per_class_R', False):
            parts.append("perclass")
        return parts

    base = _base_parts(config)
    band_lower = config.get('band_lower', -1.0)
    band_upper = config.get('band_upper', 0.0)

    # Generate band suffix variants: both .1f and .2f precision
    band_1f = f"band{band_lower:.1f}to{band_upper:.1f}"
    band_2f = f"band{band_lower:.2f}to{band_upper:.2f}"
    band_suffixes_with = list(dict.fromkeys([band_1f, band_2f]))  # deduped, ordered

    candidates = []
    is_default_band = (band_lower == -1.0 and band_upper == 0.0)

    if is_default_band:
        # Primary: no suffix (current convention); fallback: explicit suffix
        candidates.append(f"{loss_type}_{'_'.join(base)}")
        for sfx in band_suffixes_with:
            candidates.append(f"{loss_type}_{'_'.join(base + [sfx])}")
    else:
        # Primary: with suffix (.1f); fallbacks: .2f, then no suffix
        for sfx in band_suffixes_with:
            candidates.append(f"{loss_type}_{'_'.join(base + [sfx])}")
        candidates.append(f"{loss_type}_{'_'.join(base)}")

    return candidates


def build_loss_dir(loss_type: str, config: dict) -> str:
    """Return the primary (most-likely) loss directory name for a given config.

    Regulariser loss types ('ce_plus_R', 'weighted_ce_plus_R', 'R_only') get a
    decorated name encoding key hyperparameters.
    Plain CE types ('ce_only', 'weighted_ce') use the bare loss_type string.
    """
    return _build_loss_dir_candidates(loss_type, config)[0]


def resolve_checkpoint_path(dataset, model_name, K, seed, split_id, loss_type, config):
    """Resolve the path to best.pt, trying multiple directory name variants.

    Tries all plausible loss_dir names (accounting for historical format
    differences in the band suffix) and returns the first that exists on disk.
    If none exist, returns the primary (most-preferred) path so the caller
    can emit a meaningful 'not found' message.
    """
    candidates = _build_loss_dir_candidates(loss_type or "ce_only", config)
    primary_path = None

    for i, loss_dir in enumerate(candidates):
        base_dir = (
            Path(cfg.classifier_heads_dir)
            / loss_dir / dataset / model_name
            / f"seed_{seed}" / f"K_{K}"
        )
        if split_id is not None and split_id >= 0:
            base_dir = base_dir / f"split_{split_id}"
        path = base_dir / "best.pt"
        if i == 0:
            primary_path = path  # saved for error messages
        if path.exists():
            return path

    return primary_path  # not found — caller will print skip message


def append_final_result(row: dict) -> None:
    """Append one row to final_results.csv."""
    full_row = {col: row.get(col, None) for col in FINAL_RESULTS_COLUMNS}
    FINAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FINAL_RESULTS_PATH.exists()
    df = pd.DataFrame([full_row])
    df.to_csv(FINAL_RESULTS_PATH, mode="a", header=write_header, index=False)


def evaluate_single(
    dataset, model_name, K, seed, split_id, loss_type,
    config, args, device
):
    """Evaluate one (dataset, model, K, seed, split, loss_type) combination."""
    # Load dataset
    data, num_classes, dataset_kind = load_dataset(
        dataset,
        root_dir=args.root_dir,
        planetoid_normalize=args.normalize_planetoid,
        planetoid_split=args.planetoid_split,
    )

    # Select split masks
    if split_id is not None and split_id >= 0 and data.train_mask.dim() > 1:
        data = data.clone()
        data.train_mask = data.train_mask[:, split_id]
        data.val_mask   = data.val_mask[:, split_id]
        data.test_mask  = data.test_mask[:, split_id]

    # Resolve checkpoint
    checkpoint_path = resolve_checkpoint_path(
        dataset, model_name, K, seed, split_id, loss_type, config
    )

    if not checkpoint_path.exists():
        print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
        return None

    # Build model — use hidden_dim from checkpoint if available
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hp = checkpoint.get("hyperparams", {})
    eval_config = {**config, **hp}

    model = build_model(model_name, data, num_classes, K, eval_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    use_classifier_head = True  # sweep always uses train_gnn_entropy.py (classifier heads for all loss types)
    test_loss, test_acc = evaluate_test_set(model, data, device, use_classifier_head)

    method = "Classifier Heads" if use_classifier_head else "Baseline GNN"

    result = dict(
        dataset=dataset,
        model=model_name,
        method=method,
        loss_type=loss_type or "ce_only",
        K=K,
        seed=seed,
        split=split_id if split_id is not None else -1,
        # Hyperparams from checkpoint
        hidden_dim=hp.get("hidden_dim", eval_config.get("hidden_dim")),
        lr=hp.get("lr", eval_config.get("lr")),
        weight_decay=hp.get("weight_decay", eval_config.get("weight_decay")),
        max_epochs=hp.get("max_epochs", eval_config.get("max_epochs")),
        patience=hp.get("patience", eval_config.get("patience")),
        beta=hp.get("beta"),
        lambda_r=hp.get("lambda_R"),
        entropy_floor=hp.get("entropy_floor"),
        per_class_r=hp.get("per_class_R"),
        band_lower=hp.get("band_lower"),
        band_upper=hp.get("band_upper"),
        # Checkpoint val metrics
        best_epoch=checkpoint.get("epoch"),
        best_val_loss=checkpoint.get("val_loss"),
        best_val_acc=checkpoint.get("val_acc"),
        # Test metrics
        test_acc=test_acc,
        test_loss=test_loss,
        evaluated_at=datetime.now().isoformat(),
    )

    print(
        f"  {dataset}/{model_name}/K={K}/seed={seed}/split={split_id} "
        f"[{loss_type}] -> test_acc={test_acc:.4f}"
    )
    return result


def _apply_hp_row_to_config(hp_row, base_config):
    """Override base_config with hyperparameter values from a CSV row."""
    run_config = dict(base_config)
    for col in ["lr", "weight_decay", "patience", "max_epochs", "hidden_dim",
                "beta", "lambda_r", "entropy_floor", "per_class_r",
                "band_lower", "band_upper"]:
        val = hp_row.get(col)
        if pd.notna(val):
            cfg_key = "lambda_R" if col == "lambda_r" else \
                      "per_class_R" if col == "per_class_r" else col
            run_config[cfg_key] = val
    return run_config


def _get_split_ids(dataset, args):
    """Return list of split IDs to evaluate for a dataset."""
    if dataset in cfg.heterophilous_datasets:
        if args.split_mode == "first":
            return [0]
        else:
            data, _, _ = load_dataset(dataset, root_dir=args.root_dir,
                                      planetoid_normalize=args.normalize_planetoid,
                                      planetoid_split=args.planetoid_split)
            n_splits = data.train_mask.size(1) if data.train_mask.dim() > 1 else 1
            return list(range(n_splits))
    return [None]  # homophilous: no split dimension


def run_from_best_hyperparams(args):
    """Batch evaluation using best_hyperparams.csv.

    Supports two CSV formats:
      - Per-group (no K column): one row per (dataset, model, loss_type).
        The same hyperparameters are used for all K values.
      - Per-layer (K column present): one row per (dataset, model, loss_type, K).
        Each K gets its own optimal hyperparameters.
    """
    if hasattr(args, 'best_hyperparams_path') and args.best_hyperparams_path:
        best_hp_path = Path(args.best_hyperparams_path)
    else:
        best_hp_path = Path(cfg.results_dir) / "best_hyperparams.csv"
    if not best_hp_path.exists():
        raise FileNotFoundError(
            f"best_hyperparams file not found at {best_hp_path}. "
            "Run src/select_hyperparams.py first, or pass --best-hyperparams-path."
        )

    best_df = pd.read_csv(best_hp_path)
    print(f"Loaded {len(best_df)} best hyperparam configs from {best_hp_path}")

    per_layer_mode = "K" in best_df.columns
    print(f"Mode: {'per-layer (K-specific hyperparams)' if per_layer_mode else 'per-group (same hyperparams for all K)'}")

    # Expand seeds
    if args.seeds == ["all"]:
        seeds = cfg.seeds
    else:
        seeds = [int(s) for s in args.seeds]

    device = get_device()
    config = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    all_results = []

    if per_layer_mode:
        # -- Per-layer mode: one row per (dataset, model, loss_type, K) ----------
        # Iterate rows directly — each row already specifies the K to evaluate.
        for _, hp_row in best_df.iterrows():
            dataset    = hp_row["dataset"]
            model_name = hp_row["model"]
            loss_type  = hp_row["loss_type"]
            K          = int(hp_row["K"])

            base = {**config, **(cfg.defaults_homophilous if dataset in cfg.homophilous_datasets
                                  else cfg.defaults_heterophilous)}
            run_config = _apply_hp_row_to_config(hp_row, base)
            split_ids  = _get_split_ids(dataset, args)

            for seed in seeds:
                for split_id in split_ids:
                    result = evaluate_single(
                        dataset, model_name, K, seed, split_id, loss_type,
                        run_config, args, device
                    )
                    if result is not None:
                        all_results.append(result)
                        append_final_result(result)

    else:
        # -- Per-group mode: one row per (dataset, model, loss_type) --------------
        # Expand K values from CLI args.
        if args.K_values == ["all"]:
            K_values = list(range(1, cfg.K_max + 1))
        else:
            K_values = [int(k) for k in args.K_values]

        for _, hp_row in best_df.iterrows():
            dataset    = hp_row["dataset"]
            model_name = hp_row["model"]
            loss_type  = hp_row["loss_type"]

            base = {**config, **(cfg.defaults_homophilous if dataset in cfg.homophilous_datasets
                                  else cfg.defaults_heterophilous)}
            run_config = _apply_hp_row_to_config(hp_row, base)
            split_ids  = _get_split_ids(dataset, args)

            for K in K_values:
                for seed in seeds:
                    for split_id in split_ids:
                        result = evaluate_single(
                            dataset, model_name, K, seed, split_id, loss_type,
                            run_config, args, device
                        )
                        if result is not None:
                            all_results.append(result)
                            append_final_result(result)

    print(f"\nEvaluation complete. {len(all_results)} runs saved to {FINAL_RESULTS_PATH}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate best model on test set")

    # Batch mode
    parser.add_argument("--from-best-hyperparams", action="store_true",
                        help="Batch evaluate all configs in best_hyperparams.csv")
    parser.add_argument("--best-hyperparams-path", type=str, default=None,
                        help="Path to best_hyperparams CSV (e.g. results/best_hyperparams_GCN.csv). "
                             "Defaults to <results-dir>/best_hyperparams.csv")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override for results directory (e.g. /content/drive/MyDrive/GDL/sweep_results). "
                             "Sets all cfg path variables relative to this root, making CWD irrelevant.")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Explicit path for output CSV (e.g. final_results_GCN_band-1.0to0.0.csv). "
                             "Overrides the default final_results.csv.")
    parser.add_argument("--seeds", nargs="+", default=["all"],
                        help="Seeds to evaluate (used with --from-best-hyperparams), or 'all'")
    parser.add_argument("--K-values", nargs="+", default=["all"],
                        help="K values to evaluate (used with --from-best-hyperparams), or 'all'")
    parser.add_argument("--split-mode", type=str, default="first",
                        choices=["first", "all"],
                        help="For hetero datasets: 'first' (split 0) or 'all' splits")

    # Single-run mode
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-id", type=int, default=None)
    parser.add_argument("--loss-type", type=str, default=None)
    parser.add_argument("--use-classifier-head", action="store_true")
    parser.add_argument("--classifier-heads-dir", type=str, default=None,
                        help="Override cfg.classifier_heads_dir (e.g. D:/GCN_eval/classifier_heads). "
                             "Only overrides the checkpoint lookup path, not the output CSV.")

    # Dataset options
    parser.add_argument("--root-dir", type=str, default="data")
    parser.add_argument("--normalize-planetoid", action="store_true", default=True)
    parser.add_argument("--planetoid-split", type=str, default="public")

    args = parser.parse_args()

    global FINAL_RESULTS_PATH

    # Override cfg paths if --results-dir is given (makes CWD irrelevant on Colab)
    if args.results_dir:
        r = Path(args.results_dir)
        cfg.results_dir          = str(r)
        cfg.runs_dir             = str(r / "runs")
        cfg.tables_dir           = str(r / "tables")
        cfg.figures_dir          = str(r / "figures")
        cfg.classifier_heads_dir = str(r / "classifier_heads")
        FINAL_RESULTS_PATH = r / "tables" / "final_results.csv"

    # Override output path if explicitly given
    if args.output_path:
        FINAL_RESULTS_PATH = Path(args.output_path)

    # Override classifier_heads_dir if explicitly given
    if args.classifier_heads_dir:
        cfg.classifier_heads_dir = args.classifier_heads_dir

    if args.from_best_hyperparams:
        run_from_best_hyperparams(args)
        return

    # ── Single-run mode ──────────────────────────────────────────────
    if not all([args.dataset, args.model, args.K is not None, args.seed is not None]):
        parser.error("Single-run mode requires --dataset, --model, --K, --seed")

    config = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    if args.dataset in cfg.homophilous_datasets:
        config.update(cfg.defaults_homophilous)
    elif args.dataset in cfg.heterophilous_datasets:
        config.update(cfg.defaults_heterophilous)

    device = get_device()

    # Resolve checkpoint
    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir) / "best.pt"
    else:
        checkpoint_path = resolve_checkpoint_path(
            args.dataset, args.model, args.K, args.seed,
            args.split_id, args.loss_type, config
        )

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"\n{'='*70}")
    print(f"Final Test Set Evaluation")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset} | Model: {args.model} | K={args.K} | Seed={args.seed}")
    if args.split_id is not None:
        print(f"Split: {args.split_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")

    # Load dataset
    data, num_classes, _ = load_dataset(
        args.dataset,
        root_dir=args.root_dir,
        planetoid_normalize=args.normalize_planetoid,
        planetoid_split=args.planetoid_split,
    )
    if args.split_id is not None and data.train_mask.dim() > 1:
        data = data.clone()
        data.train_mask = data.train_mask[:, args.split_id]
        data.val_mask   = data.val_mask[:, args.split_id]
        data.test_mask  = data.test_mask[:, args.split_id]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    hp = checkpoint.get("hyperparams", {})
    eval_config = {**config, **hp}

    model = build_model(args.model, data, num_classes, args.K, eval_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "val_acc" in checkpoint:
        print(f"Validation Performance (from checkpoint):")
        print(f"   Val Acc:    {checkpoint['val_acc']:.4f}")
        print(f"   Val Loss:   {checkpoint['val_loss']:.4f}")
        print(f"   Best Epoch: {checkpoint['epoch']}\n")

    use_head = args.use_classifier_head or (
        args.loss_type and args.loss_type not in ("ce_only", "standard")
    )
    test_loss, test_acc = evaluate_test_set(model, data, device, use_classifier_head=use_head)

    print(f"🎯 Final Test Set Performance:")
    print(f"   Test Acc:  {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}\n")

    result = dict(
        dataset=args.dataset, model=args.model,
        method="Classifier Heads" if use_head else "Baseline GNN",
        loss_type=args.loss_type or "ce_only",
        K=args.K, seed=args.seed,
        split=args.split_id if args.split_id is not None else -1,
        hidden_dim=hp.get("hidden_dim", eval_config.get("hidden_dim")),
        lr=hp.get("lr"), weight_decay=hp.get("weight_decay"),
        max_epochs=hp.get("max_epochs"), patience=hp.get("patience"),
        beta=hp.get("beta"), lambda_r=hp.get("lambda_R"),
        entropy_floor=hp.get("entropy_floor"), per_class_r=hp.get("per_class_R"),
        band_lower=hp.get("band_lower"), band_upper=hp.get("band_upper"),
        best_epoch=checkpoint.get("epoch"),
        best_val_loss=checkpoint.get("val_loss"),
        best_val_acc=checkpoint.get("val_acc"),
        test_acc=test_acc, test_loss=test_loss,
        evaluated_at=datetime.now().isoformat(),
    )
    append_final_result(result)
    print(f"💾 Results saved to: {FINAL_RESULTS_PATH}\n")
    print(f"{'='*70}")
    print(f"Evaluation complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
