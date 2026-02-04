"""Extract layer-wise embeddings from trained GNN models."""

import argparse
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from src.datasets import load_dataset
from src.models import GCNNet, GATNet
from src.utils import set_seed, to_device, get_device


def extract_embeddings(dataset_name, model_name, K, seed, config):
    """
    Extract layer-wise embeddings from a trained model.
    
    Args:
        dataset_name: Name of dataset
        model_name: 'GCN' or 'GAT'
        K: Number of layers
        seed: Random seed
        config: Configuration dictionary
    """
    set_seed(seed)
    device = get_device()
    
    print(f"\n{'='*60}")
    print(f"Extracting embeddings: {model_name} on {dataset_name} (K={K}, seed={seed})")
    print(f"{'='*60}")
    
    # Load dataset
    data = load_dataset(dataset_name)
    
    # Create model (same architecture as training)
    if model_name == 'GCN':
        model = GCNNet(
            num_features=data.num_features,
            hidden_dim=config['hidden_dim'],
            num_classes=int(data.y.max().item()) + 1,
            K=K,
            dropout=None  # No dropout during inference
        )
    elif model_name == 'GAT':
        model = GATNet(
            num_features=data.num_features,
            hidden_dim=config['hidden_dim'],
            num_classes=int(data.y.max().item()) + 1,
            K=K,
            heads=config['gat_heads'],
            dropout=None
        )
    elif model_name == 'GraphSAGE':
        model = GraphSAGENet(
            num_features=data.num_features,
            hidden_dim=config['hidden_dim'],
            num_classes=int(data.y.max().item()) + 1,
            K=K,
            dropout=None
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load trained checkpoint
    checkpoint_path = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / 'best.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val acc: {checkpoint['val_acc']:.4f}")
    print(f"  Test acc: {checkpoint['test_acc']:.4f}")
    
    # Extract embeddings
    data = to_device(data, device)
    
    with torch.no_grad():
        embeddings, logits = model.forward_with_embeddings(data)
    
    # Move to CPU and convert to dictionary
    embeddings_dict = {k: emb.cpu() for k, emb in enumerate(embeddings)}
    
    # Save embeddings with metadata
    output = {
        'embeddings': embeddings_dict,  # {0: h_0, 1: h_1, ..., K: h_K}
        'labels': data.y.cpu(),
        'train_mask': data.train_mask.cpu(),
        'val_mask': data.val_mask.cpu(),
        'test_mask': data.test_mask.cpu(),
        'K': K,
        'num_nodes': data.num_nodes,
    }
    
    # Save to file
    output_path = checkpoint_path.parent / 'embeddings.pt'
    torch.save(output, output_path)
    
    print(f"\n✓ Embeddings extracted!")
    print(f"  Total depths: {len(embeddings_dict)} (k=0..{K})")
    for k, emb in embeddings_dict.items():
        print(f"  k={k}: shape {emb.shape}")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained GNN')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=str, default='0',
                       help='Random seed or "all" to run all seeds from config')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Handle seed argument
    if args.seed.lower() == 'all':
        seeds_to_run = config['seeds']
        print(f"\n🔄 Running all seeds: {seeds_to_run}\n")
    else:
        seeds_to_run = [int(args.seed)]
    
    # Extract embeddings for each seed
    for seed in seeds_to_run:
        extract_embeddings(args.dataset, args.model, args.K, seed, config)
        print()  # Add spacing between seeds


if __name__ == '__main__':
    main()
