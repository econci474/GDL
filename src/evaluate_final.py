"""
Evaluate the best saved model checkpoint on the test set.

This script should ONLY be run ONCE after training is complete to 
get the final test accuracy. This ensures proper separation between 
training/validation and test sets.

Usage:
    python src/evaluate_final.py --dataset Cora --model GCN --K 3 --seed 0
    python src/evaluate_final.py --dataset Cora --model GCN --K 3 --seed 0 --loss-type ce_plus_R_R1.0_hard
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


@torch.no_grad()
def evaluate_test_set(model, data, device, use_classifier_head=False):
    """Evaluate model on test set."""
    model.eval()
    data = to_device(data, device)
    
    if use_classifier_head:
        # For models with classifier heads, use final layer
        layer_logits, _ = model.forward_with_classifier_head(data)
        logits = layer_logits[-1]
    else:
        logits = model(data)
    
    # Test metrics
    test_loss = F.cross_entropy(logits[data.test_mask], data.y[data.test_mask]).item()
    test_pred = logits[data.test_mask].argmax(dim=1)
    test_acc = (test_pred == data.y[data.test_mask]).sum() / data.test_mask.sum()
    
    return float(test_loss), float(test_acc.item())


def main():
    parser = argparse.ArgumentParser(description='Evaluate best model on test set')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (Cora, PubMed, Roman-empire, Minesweeper)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (GCN, GAT, GraphSAGE)')
    parser.add_argument('--K', type=int, required=True,
                       help='Number of layers')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed')
    parser.add_argument('--split-id', type=int, default=None,
                       help='Split ID for heterophilous datasets (optional)')
    
    # Checkpoint location
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Custom checkpoint directory (optional). If not specified, uses runs_dir or classifier_heads_dir.')
    parser.add_argument('--loss-type', type=str, default=None,
                       help='For classifier heads: exponential, class-weighted, or custom like ce_plus_R_R1.0_hard')
    parser.add_argument('--use-classifier-head', action='store_true',
                       help='Use forward_with_classifier_head for models with auxiliary heads')
    
    # Dataset options
    parser.add_argument('--root-dir', type=str, default='data')
    parser.add_argument('--normalize-planetoid', action='store_true', default=True)
    parser.add_argument('--planetoid-split', type=str, default='public')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Load dataset
    data, num_classes, dataset_kind = load_dataset(
        args.dataset,
        root_dir=args.root_dir,
        planetoid_normalize=args.normalize_planetoid,
        planetoid_split=args.planetoid_split,
    )
    
    # Handle split for heterophilous datasets
    if args.split_id is not None and data.train_mask.dim() > 1:
        data = data.clone()
        data.train_mask = data.train_mask[:, args.split_id]
        data.val_mask = data.val_mask[:, args.split_id]
        data.test_mask = data.test_mask[:, args.split_id]
    
    # Determine checkpoint path
    if args.checkpoint_dir:
        checkpoint_path = Path(args.checkpoint_dir) / "best.pt"
    elif args.loss_type:
        # Classifier heads directory
        base_dir = Path(cfg.classifier_heads_dir) / args.loss_type / args.dataset / args.model / f"seed_{args.seed}" / f"K_{args.K}"
        if args.split_id is not None:
            base_dir = base_dir / f"split_{args.split_id}"
        checkpoint_path = base_dir / "best.pt"
    else:
        # Default runs directory
        base_dir = Path(config['runs_dir']) / args.dataset / args.model / f"seed_{args.seed}" / f"K_{args.K}"
        if args.split_id is not None:
            base_dir = base_dir / f"split_{args.split_id}"
        checkpoint_path = base_dir / "best.pt"
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("\nTip: Use --checkpoint-dir, --loss-type, or --split-id to specify the correct path.")
        return
    
    print(f"\n{'='*70}")
    print(f"Final Test Set Evaluation")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} (K={args.K})")
    print(f"Seed: {args.seed}")
    if args.split_id is not None:
        print(f"Split: {args.split_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    # Build model
    device = get_device()
    model = build_model(args.model, data, num_classes, args.K, config).to(device)
    
    # Load best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Show validation performance from checkpoint
    if 'val_acc' in checkpoint:
        print(f"Validation Performance (from checkpoint):")
        print(f"   Val Acc: {checkpoint['val_acc']:.4f}")
        print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"   Best Epoch: {checkpoint['epoch']}\n")
    
    # Evaluate on test set (ONCE!)
    test_loss, test_acc = evaluate_test_set(model, data, device, use_classifier_head=args.use_classifier_head)
    
    print(f"🎯 Final Test Set Performance:")
    print(f"   Test Acc: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}\n")
    
    # Save test results to CSV for permanent record
    test_results = {
        'dataset': args.dataset,
        'model': args.model,
        'K': args.K,
        'seed': args.seed,
        'split_id': args.split_id if args.split_id is not None else -1,
        'loss_type': args.loss_type if args.loss_type else 'standard',
        'test_acc': test_acc,
        'test_loss': test_loss,
        'val_acc': checkpoint.get('val_acc'),
        'val_loss': checkpoint.get('val_loss'),
        'best_epoch': checkpoint.get('epoch'),
        'evaluated_at': datetime.now().isoformat(),
    }
    
    results_file = checkpoint_path.parent / 'test_log.csv'
    df = pd.DataFrame([test_results])
    
    # Append if file exists, otherwise create new
    if results_file.exists():
        df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        df.to_csv(results_file, index=False)
    
    print(f"💾 Results saved to: {results_file}\n")
    
    print(f"{'='*70}")
    print(f"Evaluation complete")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
