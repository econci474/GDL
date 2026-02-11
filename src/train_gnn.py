"""Training script for GNN models with early stopping."""

import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from src.datasets import load_dataset
from src.models import GCNNet, GATNet, GraphSAGENet
from src.utils import set_seed, to_device, get_device
import numpy as np

def train_epoch(model, data, optimizer, device):
    """Train for one epoch."""
    model.train()
    data = to_device(data, device)
    
    optimizer.zero_grad()
    logits = model(data)
    
    # Cross-entropy loss on training nodes
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    # Compute training accuracy
    pred = logits[data.train_mask].argmax(dim=1)
    correct = (pred == data.y[data.train_mask]).sum()
    acc = correct / data.train_mask.sum()
    
    return loss.item(), acc.item()


@torch.no_grad()
def evaluate(model, data, device):
    """Evaluate on validation and test sets."""
    model.eval()
    data = to_device(data, device)
    
    logits = model(data)
    
    # Validation metrics
    val_loss = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask]).item()
    val_pred = logits[data.val_mask].argmax(dim=1)
    val_acc = (val_pred == data.y[data.val_mask]).sum() / data.val_mask.sum()
    
    # Test metrics
    test_pred = logits[data.test_mask].argmax(dim=1)
    test_acc = (test_pred == data.y[data.test_mask]).sum() / data.test_mask.sum()
    
    return float(val_loss), float(val_acc.item()), float(test_acc.item())

def get_num_splits(data):
    return data.train_mask.size(1) if data.train_mask.dim() > 1 else 1

def select_split_masks(data, split_id: int):
    # Make a copy so we don't overwrite masks for later splits
    data_s = data.clone()
    if data_s.train_mask.dim() > 1:
        data_s.train_mask = data_s.train_mask[:, split_id]
        data_s.val_mask   = data_s.val_mask[:, split_id]
        data_s.test_mask  = data_s.test_mask[:, split_id]
    return data_s


def build_model(model_name: str, data, num_classes: int, K: int, config: dict):
    """Factory for models."""
    if model_name == "GCN":
        return GCNNet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            dropout=None,
            normalize=True,
        )
    elif model_name == "GAT":
        return GATNet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            heads=config.get("gat_heads", 8),
            dropout=None,
        )
    elif model_name == "GraphSAGE":
        return GraphSAGENet(
            num_features=data.num_features,
            hidden_dim=config["hidden_dim"],
            num_classes=num_classes,
            K=K,
            aggr=config.get("sage_aggr", "mean"),
            dropout=None,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use GCN, GAT, GraphSAGE.")


def run_one_split(
    *,
    dataset_name: str,
    model_name: str,
    K: int,
    seed: int,
    config: dict,
    data_split,
    num_classes: int,
    output_dir: Path,
):
    """Train + early stop on ONE split. Saves best checkpoint + training log."""
    device = get_device()
    set_seed(seed)

    model = build_model(model_name, data_split, num_classes, K, config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_log = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss, train_acc = train_epoch(model, data_split, optimizer, device)
        val_loss, val_acc, test_acc = evaluate(model, data_split, device)

        scheduler.step(val_loss)

        train_log.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                test_acc=test_acc,
            )
        )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

        # Early stopping on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                dict(
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    val_loss=val_loss,
                    val_acc=val_acc,
                    test_acc=test_acc,
                ),
                output_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(train_log).to_csv(output_dir / "train_log.csv", index=False)

    print(f"\n✓ Split complete: {output_dir}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val acc:  {best_val_acc:.4f}")
    print(f"  Best test acc: {best_test_acc:.4f}")

    return dict(best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_acc=best_val_acc, best_test_acc=best_test_acc)


def train_gnn(dataset_name: str, model_name: str, K: int, seed: int, args, config: dict):
    # Load dataset (keep 2D masks if present)
    data, num_classes, dataset_kind = load_dataset(
    dataset_name,
    root_dir=args.root_dir,
    planetoid_normalize=args.normalize_planetoid,
    planetoid_split=args.planetoid_split,
    )

    # Decide split behavior
    num_splits = get_num_splits(data)

    if args.split_mode == "auto":
        # Heterophilous datasets with multiple splits -> all splits by default; otherwise one split
        split_ids = list(range(num_splits)) if num_splits > 1 else [0]
    elif args.split_mode == "all":
        split_ids = list(range(num_splits))
    elif args.split_mode == "first":
        split_ids = [0]
    elif args.split_mode == "index":
        split_ids = [args.split_id]
    else:
        raise ValueError(f"Unknown split_mode: {args.split_mode}")

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Kind: {dataset_kind} | Planetoid split: {args.planetoid_split}")
    print(f"Model: {model_name} | K={K} | seed={seed}")
    print(f"Mask splits available: {num_splits} | Using split_ids: {split_ids}")
    print(f"Normalize planetoid: {args.normalize_planetoid} | Hetero split: {args.split_mode}")
    print(f"{'='*60}\n")

    base_dir = Path(config["runs_dir"]) / dataset_name / model_name / f"seed_{seed}" / f"K_{K}"

    results = []
    for split_id in split_ids:
        data_split = select_split_masks(data, split_id)

        # Only add split folder if there are multiple splits or you explicitly requested a split index
        if num_splits > 1:
            split_tag = f"split_{split_id}"
            output_dir = base_dir / split_tag
        else:
            output_dir = base_dir  # single split datasets

        r = run_one_split(
            dataset_name=dataset_name,
            model_name=model_name,
            K=K,
            seed=seed,
            config=config,
            data_split=data_split,
            num_classes=num_classes,
            output_dir=output_dir,
        )
        r["split_id"] = split_id
        results.append(r)

    # If multiple splits were run, summarize mean±std
    if len(results) > 1:
        test_accs = [r["best_test_acc"] for r in results]
        mean = float(np.mean(test_accs))
        std = float(np.std(test_accs, ddof=1))
        print(f"\nSUMMARY for {args.dataset} ({args.model}, K={args.K}, seed={seed}):")
        print(f"  Test Acc: {mean:.4f} ± {std:.4f} over {len(results)} splits")

        # Save summary at the seed level
        base_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(base_dir / "summary.csv", index=False)
        print(f"  Saved: {base_dir / 'summary.csv'}")


def main():
    import sys
    from pathlib import Path
    # Add project root to path to import config
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg
    
    parser = argparse.ArgumentParser(description='Train GNN model')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (Cora, PubMed, Roman-empire, Minesweeper)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (GCN, GAT, GraphSAGE)')
    parser.add_argument('--K', type=int, default=8,
                       help='Number of layers')
    parser.add_argument('--seed', type=str, default='0',
                       help='Random seed or "all" to run all seeds from config')
    
    #Root directory for datasets loader
    parser.add_argument("--root-dir", type=str, default="data",
                        help="Root directory for dataset downloads/cache")

    # Choose Planetoid split protocol
    parser.add_argument("--planetoid-split", type=str, default="public",
                        choices=["public", "full", "random"],
                        help="Planetoid split protocol for Cora/PubMed")

    # Normalization toggles
    parser.add_argument("--normalize-planetoid", action="store_true", default=True,
                        help="Apply NormalizeFeatures() to Cora/PubMed (default: True)")
    parser.add_argument("--no-normalize-planetoid", dest="normalize_planetoid", action="store_false",
                        help="Disable NormalizeFeatures() for Cora/PubMed")

    # Split handling for heterophilous datasets
    parser.add_argument("--split-mode", type=str, default="auto",
                        choices=["auto", "all", "first", "index"],
                        help=(
                            "How to handle multi-split masks: "
                            "auto (all splits if available), all, first (split 0), index (use --split-id)"
                        ))
    parser.add_argument("--split-id", type=int, default=0,
                        help="Which split column to use if --split-mode index")

    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Handle seed argument
    if args.seed.lower() == 'all':
        seeds_to_run = config['seeds']
        print(f"\n🔄 Running all seeds: {seeds_to_run}\n")
    else:
        seeds_to_run = [int(args.seed)]
    
    # Train model for each seed
    for seed in seeds_to_run:
        train_gnn(args.dataset, args.model, args.K, seed, args, config)
        print()  # Add spacing between seeds


if __name__ == '__main__':
    main()
