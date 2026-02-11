"""Extract layer-wise predictions from trained classifier head models."""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from src.datasets import load_dataset
from src.models import GCNNet, GATNet, GraphSAGENet
from src.utils import set_seed, to_device, get_device


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


def extract_classifier_outputs(dataset_name, model_name, K, seed, config, loss_type='exponential'):
    """
    Extract layer-wise predictions from trained classifier head model.
    
    Args:
        dataset_name: Name of dataset
        model_name: 'GCN', 'GAT', or 'GraphSAGE'
        K: Number of layers
        seed: Random seed
        config: Configuration dictionary
        loss_type: 'exponential' or 'class-weighted'
    """
    set_seed(seed)
    device = get_device()
    
    print(f"\n{'='*60}")
    print(f"Extracting classifier outputs: {model_name} on {dataset_name}")
    print(f"  K={K}, seed={seed}, loss_type={loss_type}")
    print(f"{'='*60}")
    
    # Load dataset
    data, num_classes, dataset_kind = load_dataset(
        dataset_name,
        root_dir=config.get('root_dir', 'data'),
        planetoid_normalize=config.get('planetoid_normalize', False),
        planetoid_split=config.get('planetoid_split', 'public'),
    )
    
    # Create model (same architecture as training)
    model = build_model(model_name, data, num_classes, K, config)
    
    # Load trained checkpoint
    checkpoint_path = Path(cfg.classifier_heads_dir) / loss_type / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / 'best.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val acc: {checkpoint['val_acc']:.4f}")
    print(f"  Test acc: {checkpoint['test_acc']:.4f}")
    
    # Extract layer-wise predictions
    data = to_device(data, device)
    
    with torch.no_grad():
        layer_logits, layer_probs = model.forward_with_classifier_head(data)
    
    # Move to CPU and convert to numpy
    layer_logits_cpu = [logits.cpu().numpy() for logits in layer_logits]
    layer_probs_cpu = [probs.cpu().numpy() for probs in layer_probs]
    
    # Prepare output dictionary for layer_logits.npz
    logits_dict = {}
    for k, logits_k in enumerate(layer_logits_cpu):
        # Split by mask
        train_mask = data.train_mask.cpu().numpy()
        val_mask = data.val_mask.cpu().numpy()
        test_mask = data.test_mask.cpu().numpy()
        
        logits_dict[f'train_logits_{k}'] = logits_k[train_mask]
        logits_dict[f'val_logits_{k}'] = logits_k[val_mask]
        logits_dict[f'test_logits_{k}'] = logits_k[test_mask]
    
    # Prepare output dictionary for layer_probs.npz
    probs_dict = {}
    for k, probs_k in enumerate(layer_probs_cpu):
        probs_dict[f'train_probs_{k}'] = probs_k[train_mask]
        probs_dict[f'val_probs_{k}'] = probs_k[val_mask]
        probs_dict[f'test_probs_{k}'] = probs_k[test_mask]
    
    # Add metadata
    logits_dict['K'] = K
    logits_dict['num_classes'] = num_classes
    probs_dict['K'] = K
    probs_dict['num_classes'] = num_classes
    
    # Save to files
    logits_path = checkpoint_path.parent / 'layer_logits.npz'
    probs_path = checkpoint_path.parent / 'layer_probs.npz'
    
    np.savez(logits_path, **logits_dict)
    np.savez(probs_path, **probs_dict)
    
    print(f"\n✓ Classifier outputs extracted!")
    print(f"  Total layers: {len(layer_logits_cpu)} (k=0..{K})")
    for k in range(len(layer_logits_cpu)):
        print(f"  k={k}: logits shape {layer_logits_cpu[k].shape}, probs shape {layer_probs_cpu[k].shape}")
    print(f"\nSaved to:")
    print(f"  {logits_path}")
    print(f"  {probs_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract predictions from trained classifier heads')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model', type=str, required=True, help='Model name (GCN, GAT, GraphSAGE)')
    parser.add_argument('--K', type=int, default=8, help='Number of layers')
    parser.add_argument('--seed', type=str, default='0',
                       help='Random seed or "all" to run all seeds from config')
    parser.add_argument('--loss-type', type=str, default='exponential',
                        choices=['exponential', 'class-weighted'],
                        help='Loss type used during training')
    parser.add_argument('--root-dir', type=str, default='data',
                        help='Root directory for datasets')
    parser.add_argument('--normalize-planetoid', action='store_true',
                        help='Apply normalization for Planetoid datasets')
    parser.add_argument('--planetoid-split', type=str, default='public',
                        choices=['public', 'full', 'random'],
                        help='Which split to use for Planetoid datasets')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    config['root_dir'] = args.root_dir
    config['planetoid_normalize'] = args.normalize_planetoid
    config['planetoid_split'] = args.planetoid_split
    
    # Handle seed argument
    if args.seed.lower() == 'all':
        seeds_to_run = config['seeds']
        print(f"\n🔄 Running all seeds: {seeds_to_run}\n")
    else:
        seeds_to_run = [int(args.seed)]
    
    # Extract outputs for each seed
    for seed in seeds_to_run:
        extract_classifier_outputs(
            args.dataset,
            args.model,
            args.K,
            seed,
            config,
            loss_type=args.loss_type
        )
        print()  # Add spacing between seeds


if __name__ == '__main__':
    main()
