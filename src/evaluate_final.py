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

    # Batch evaluation from best_hyperparams.csv
    python src/evaluate_final.py --from-best-hyperparams
    python src/evaluate_final.py --from-best-hyperparams --seeds all --K-values all
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


def resolve_checkpoint_path(dataset, model_name, K, seed, split_id, loss_type, config):
    """Resolve the path to best.pt given run parameters."""
    if loss_type and loss_type != "ce_only":
        # Classifier heads directory
        base_dir = (
            Path(cfg.classifier_heads_dir)
            / loss_type / dataset / model_name
            / f"seed_{seed}" / f"K_{K}"
        )
    else:
        # Standard GNN runs directory
        base_dir = (
            Path(config["runs_dir"])
            / dataset / model_name
            / f"seed_{seed}" / f"K_{K}"
        )

    if split_id is not None and split_id >= 0:
        base_dir = base_dir / f"split_{split_id}"

    return base_dir / "best.pt"


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

    use_classifier_head = loss_type not in (None, "ce_only", "standard")
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
        f"[{loss_type}] → test_acc={test_acc:.4f}"
    )
    return result


def run_from_best_hyperparams(args):
    """Batch evaluation using best_hyperparams.csv."""
    best_hp_path = Path(cfg.results_dir) / "best_hyperparams.csv"
    if not best_hp_path.exists():
        raise FileNotFoundError(
            f"best_hyperparams.csv not found at {best_hp_path}. "
            "Run src/select_hyperparams.py first."
        )

    best_df = pd.read_csv(best_hp_path)
    print(f"Loaded {len(best_df)} best hyperparam configs from {best_hp_path}")

    # Expand seeds and K values
    if args.seeds == ["all"]:
        seeds = cfg.seeds
    else:
        seeds = [int(s) for s in args.seeds]

    if args.K_values == ["all"]:
        K_values = list(range(1, cfg.K_max + 1))
    else:
        K_values = [int(k) for k in args.K_values]

    device = get_device()
    config = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}

    all_results = []

    for _, hp_row in best_df.iterrows():
        dataset    = hp_row["dataset"]
        model_name = hp_row["model"]
        loss_type  = hp_row["loss_type"]

        # Apply dataset-type defaults then best hyperparams
        if dataset in cfg.homophilous_datasets:
            run_config = {**config, **cfg.defaults_homophilous}
        else:
            run_config = {**config, **cfg.defaults_heterophilous}

        # Override with best hyperparams
        for col in ["lr", "weight_decay", "patience", "max_epochs", "hidden_dim",
                    "beta", "lambda_r", "entropy_floor", "per_class_r",
                    "band_lower", "band_upper"]:
            val = hp_row.get(col)
            if pd.notna(val):
                # Map lambda_r → lambda_R for config key
                cfg_key = "lambda_R" if col == "lambda_r" else \
                          "per_class_R" if col == "per_class_r" else col
                run_config[cfg_key] = val

        # Determine splits to evaluate
        if dataset in cfg.heterophilous_datasets:
            if args.split_mode == "first":
                split_ids = [0]
            else:
                # Load dataset to find number of splits
                data, _, _ = load_dataset(dataset, root_dir=args.root_dir,
                                          planetoid_normalize=args.normalize_planetoid,
                                          planetoid_split=args.planetoid_split)
                n_splits = data.train_mask.size(1) if data.train_mask.dim() > 1 else 1
                split_ids = list(range(n_splits))
        else:
            split_ids = [None]  # homophilous: no split dimension

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
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    # Dataset options
    parser.add_argument("--root-dir", type=str, default="data")
    parser.add_argument("--normalize-planetoid", action="store_true", default=True)
    parser.add_argument("--planetoid-split", type=str, default="public")

    args = parser.parse_args()

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
