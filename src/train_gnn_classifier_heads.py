"""Training script for GNN models with layer-wise classifier heads and depth-decayed auxiliary losses."""

import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
import sys
import math
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from src.datasets import load_dataset
from src.models import GCNNet, GATNet, GraphSAGENet
from src.utils import set_seed, to_device, get_device
import numpy as np


def compute_class_weights(data):
    """
    Compute class weights for class-weighted cross-entropy.
    Returns tensor of shape (C,) where weights[c] = 1/n_c
    """
    num_classes = int(data.y.max()) + 1
    class_counts = torch.bincount(data.y[data.train_mask], minlength=num_classes).float()
    # Avoid division by zero
    class_weights = torch.where(class_counts > 0, 1.0 / class_counts, torch.zeros_like(class_counts))
    return class_weights


def train_epoch_multi_layer(model, data, optimizer, device, loss_type, beta, K, class_weights=None):
    """
    Train for one epoch with multi-layer loss.
    
    Args:
        loss_type: 'exponential' or 'class-weighted'
        beta: Decay parameter for exponential weighting
        K: Number of GNN layers
        class_weights: For class-weighted loss
    """
    model.train()
    data = to_device(data, device)
    
    optimizer.zero_grad()
    
    # Get predictions from all layers
    layer_logits, layer_probs = model.forward_with_classifier_head(data)
    
    # Compute per-layer losses
    layer_losses = []
    for k, logits_k in enumerate(layer_logits):
        if loss_type == 'class-weighted':
            # Class-weighted cross-entropy (from image formula)
            # L_wCE = -(1/N) Σ (N/(C·n_y_v)) log(e^{z_v,y_v} / Σ_k e^{z_v,k})
            # Simplified: use F.cross_entropy with weight parameter
            loss_k = F.cross_entropy(
                logits_k[data.train_mask],
                data.y[data.train_mask],
                weight=class_weights.to(device) if class_weights is not None else None
            )
        else:  # exponential
            # Standard cross-entropy
            loss_k = F.cross_entropy(
                logits_k[data.train_mask],
                data.y[data.train_mask]
            )
        layer_losses.append(loss_k)
    
    # Compute depth-decayed combined loss
    # L_total = L_final + Σ_{k=0}^{K-1} α_k · L_k
    # where α_k = exp(-β(K-k))
    total_loss = layer_losses[-1]  # Final layer with weight 1.0
    
    for k in range(K):  # k = 0 to K-1 (intermediate layers)
        alpha_k = math.exp(-beta * (K - k))
        total_loss = total_loss + alpha_k * layer_losses[k]
    
    total_loss.backward()
    optimizer.step()
    
    # Compute training accuracy using final layer
    pred = layer_logits[-1][data.train_mask].argmax(dim=1)
    correct = (pred == data.y[data.train_mask]).sum()
    acc = correct / data.train_mask.sum()
    
    # Return total loss and per-layer losses for logging
    layer_loss_values = [l.item() for l in layer_losses]
    return total_loss.item(), acc.item(), layer_loss_values


@torch.no_grad()
def evaluate_multi_layer(model, data, device, loss_type, class_weights=None):
    """Evaluate on validation and test sets using final layer."""
    model.eval()
    data = to_device(data, device)
    
    layer_logits, _ = model.forward_with_classifier_head(data)
    final_logits = layer_logits[-1]
    
    # Validation metrics
    if loss_type == 'class-weighted':
        val_loss = F.cross_entropy(
            final_logits[data.val_mask],
            data.y[data.val_mask],
            weight=class_weights.to(device) if class_weights is not None else None
        ).item()
    else:
        val_loss = F.cross_entropy(final_logits[data.val_mask], data.y[data.val_mask]).item()
    
    val_pred = final_logits[data.val_mask].argmax(dim=1)
    val_acc = (val_pred == data.y[data.val_mask]).sum() / data.val_mask.sum()
    
    return float(val_loss), float(val_acc.item())


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
    loss_type: str,
    beta: float,
):
    """Train + early stop on ONE split with multi-layer loss."""
    device = get_device()
    set_seed(seed)

    model = build_model(model_name, data_split, num_classes, K, config).to(device)
    
    # Compute class weights if needed
    class_weights = None
    if loss_type == 'class-weighted':
        class_weights = compute_class_weights(data_split)
        print(f"Using class weights: {class_weights}")

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
    best_epoch = 0
    patience_counter = 0
    train_log = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss, train_acc, layer_losses = train_epoch_multi_layer(
            model, data_split, optimizer, device, loss_type, beta, K, class_weights
        )
        val_loss, val_acc = evaluate_multi_layer(
            model, data_split, device, loss_type=loss_type, class_weights=class_weights
        )

        scheduler.step(val_loss)

        # Log per-layer losses and hyperparameters
        log_entry = dict(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            # Hyperparameter metadata
            lr=config["lr"],
            patience=config["patience"],
            max_epochs=config["max_epochs"],
            beta=beta,
            loss_type=loss_type,
            K=K,
        )
        # Add per-layer train losses
        for k, loss_k in enumerate(layer_losses):
            log_entry[f'train_loss_layer_{k}'] = loss_k
        
        train_log.append(log_entry)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

        # Early stopping on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                dict(
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    val_loss=val_loss,
                    val_acc=val_acc,
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

    return dict(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc
    )


def train_gnn_classifier_heads(
    dataset_name: str,
    model_name: str,
    K: int,
    seed: int,
    args,
    config: dict
):
    # Load dataset
    data, num_classes, dataset_kind = load_dataset(
        dataset_name,
        root_dir=args.root_dir,
        planetoid_normalize=args.normalize_planetoid,
        planetoid_split=args.planetoid_split,
    )

    # Decide split behavior
    num_splits = get_num_splits(data)

    if args.split_mode == "auto":
        split_ids = list(range(num_splits)) if num_splits > 1 else [0]
    elif args.split_mode == "all":
        split_ids = list(range(num_splits))
    elif args.split_mode == "first":
        split_ids = [0]
    else:
        raise ValueError(f"Unknown split_mode: {args.split_mode}")

    # Base directory with loss_type
    base_dir = Path(cfg.classifier_heads_dir) / args.loss_type / dataset_name / model_name / f"seed_{seed}" / f"K_{K}"

    # Print header
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Features: {data.num_features}")
    print(f"  Classes: {num_classes}")
    print(f"  Train nodes: {data.train_mask[:, 0].sum() if data.train_mask.dim() > 1 else data.train_mask.sum()}")
    if num_splits > 1:
        print(f"  Val nodes:   {data.val_mask[:, 0].sum()}")
        print(f"  Test nodes:  {data.test_mask[:, 0].sum()}")
    else:
        print(f"  Val nodes:   {data.val_mask.sum()}")
        print(f"  Test nodes:  {data.test_mask.sum()}")
    print(f"\nModel: {model_name} | K={K} | seed={seed}")
    print(f"Loss type: {args.loss_type} | Beta: {args.beta}")
    print(f"Dataset kind: {dataset_kind} | Planetoid split: {args.planetoid_split}")
    print(f"Splits to train: {len(split_ids)}")
    print(f"{'='*60}\n")

    all_results = []

    for split_id in split_ids:
        data_split = select_split_masks(data, split_id)

        if num_splits > 1:
            output_dir = base_dir / f"split_{split_id}"
        else:
            output_dir = base_dir

        result = run_one_split(
            dataset_name=dataset_name,
            model_name=model_name,
            K=K,
            seed=seed,
            config=config,
            data_split=data_split,
            num_classes=num_classes,
            output_dir=output_dir,
            loss_type=args.loss_type,
            beta=args.beta,
        )
        all_results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Training complete for {len(split_ids)} split(s)")
    if len(all_results) > 1:
        avg_val_acc = sum(r['best_val_acc'] for r in all_results) / len(all_results)
        print(f"Average validation accuracy: {avg_val_acc:.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train GNN with layer-wise classifier heads")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="Model name (GCN, GAT, GraphSAGE)")
    parser.add_argument("--K", type=int, required=True, help="Number of GNN layers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--loss-type", type=str, default="exponential",
                        choices=["exponential", "class-weighted"],
                        help="Loss type: exponential (depth-decay) or class-weighted")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Decay parameter for exponential weighting (default: 0.5)")
    parser.add_argument("--root-dir", type=str, default="data",
                        help="Root directory for datasets")
    parser.add_argument("--normalize-planetoid", action="store_true",
                        help="Apply normalization for Planetoid datasets")
    parser.add_argument("--planetoid-split", type=str, default="public",
                        choices=["public", "full", "random"],
                        help="Which split to use for Planetoid datasets")
    parser.add_argument("--split-mode", type=str, default="auto",
                        choices=["auto", "all", "first"],
                        help="Split mode: auto (default), all, or first")

    args = parser.parse_args()

    # Load config
    config = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "hidden_dim": cfg.hidden_dim,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "gat_heads": getattr(cfg, "gat_heads", 8),
        "sage_aggr": getattr(cfg, "sage_aggr", "mean"),
    }

    train_gnn_classifier_heads(
        dataset_name=args.dataset,
        model_name=args.model,
        K=args.K,
        seed=args.seed,
        args=args,
        config=config
    )


if __name__ == "__main__":
    main()
