"""
Extract embeddings from trained model checkpoints.

The train_gnn.py script only saves model checkpoints (best.pt), not embeddings.
This script loads those checkpoints, runs a forward pass, and saves the embeddings.
"""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import load_dataset
from src.models import GCNNet, GATNet, GraphSAGENet
from src.utils import set_seed, get_device
import config as cfg


def build_model(model_name: str, data, num_classes: int, K: int):
    """Build model architecture (same as train_gnn.py)"""
    num_features = data.x.shape[1]
    
    if model_name == 'GCN':
        model = GCNNet(
            num_features=num_features,
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes,
            K=K,
            dropout=cfg.dropout, 
            normalize=True
        )
    elif model_name == 'GAT':
        model = GATNet(
            num_features=num_features,
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes,
            K=K,
            heads=cfg.gat_heads,
            dropout=cfg.dropout
        )
    elif model_name == 'GraphSAGE':
        model = GraphSAGENet(
            num_features=num_features,
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes,
            K=K,
            dropout=cfg.dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def extract_embeddings(dataset_name, model_name, K, seed, checkpoint_path, output_path):
    """
    Extract embeddings from a trained checkpoint.
    
    Args:
        dataset_name: Dataset name
        model_name: Model name (GCN, GAT, GraphSAGE)
        K: Number of GNN layers
        seed: Random seed
        checkpoint_path: Path to best.pt checkpoint
        output_path: Where to save embeddings.pt
    """
    device = get_device()
    set_seed(seed)
    
    # Load dataset
    data, num_classes, _ = load_dataset(dataset_name)
    data = data.to(device)
    
    # Build model
    model = build_model(model_name, data, num_classes, K).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract embeddings
    with torch.no_grad():
        embeddings, logits = model.forward_with_embeddings(data)
    
    # Convert to CPU and create dict structure
    embeddings_dict = {}
    for k, emb in enumerate(embeddings):
        embeddings_dict[k] = emb.cpu()
    
    # Save in the same structure our other code expects
    output_data = {
        'embeddings': embeddings_dict,
        'labels': data.y.cpu(),
        'train_mask': data.train_mask.cpu(),
        'val_mask': data.val_mask.cpu(),
        'test_mask': data.test_mask.cpu(),
        'K': K,
        'num_nodes': data.num_nodes
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output_path)
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from trained checkpoints')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--results-dir', type=str, default='results/runs')
    
    args = parser.parse_args()
    
    # Path to checkpoint directory
    checkpoint_dir = Path(args.results_dir) / args.dataset / args.model / f'seed_{args.seed}' / f'K_{args.K}'
    
    # Check if this is homophilous (no splits) or heterophilous (with splits)
    homophilous_checkpoint = checkpoint_dir / 'best.pt'
    
    if homophilous_checkpoint.exists():
        # Homophilous dataset - single checkpoint
        output_path = checkpoint_dir / 'embeddings.pt'
        
        # Skip if already exists
        if output_path.exists():
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"\nSKIPPED (already exists): {output_path}")
            print(f"   Size: {size_mb:.1f} MB\n")
            return
        
        print(f"\n{'='*60}")
        print(f"EXTRACTING EMBEDDINGS (Homophilous)")
        print(f"{'='*60}")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"K: {args.K}")
        print(f"Seed: {args.seed}")
        print(f"Checkpoint: {homophilous_checkpoint}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")
        
        # Extract
        embeddings_data = extract_embeddings(
            args.dataset, args.model, args.K, args.seed,
            homophilous_checkpoint, output_path
        )
        
        # Summary
        num_layers = len(embeddings_data['embeddings'])
        size_mb = output_path.stat().st_size / 1024 / 1024
        
        print(f"\n✓ SUCCESS!")
        print(f"  Extracted {num_layers} layers (k=0..{num_layers-1})")
        print(f"  File size: {size_mb:.1f} MB")
        print(f"  Saved to: {output_path}")
        print(f"{'='*60}\n")
        
    else:
        # Heterophilous dataset - check for split folders
        split_dirs = sorted([d for d in checkpoint_dir.glob('split_*') if d.is_dir()])
        
        if not split_dirs:
            print(f"No checkpoints found in {checkpoint_dir}")
            print(f"   Looked for: best.pt or split_*/best.pt")
            return
        
        print(f"\n{'='*60}")
        print(f"EXTRACTING EMBEDDINGS (Heterophilous - {len(split_dirs)} splits)")
        print(f"{'='*60}")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"K: {args.K}")
        print(f"Seed: {args.seed}")
        print(f"Splits: {[d.name for d in split_dirs]}")
        print(f"{'='*60}\n")
        
        extracted_count = 0
        skipped_count = 0
        
        for split_dir in split_dirs:
            split_name = split_dir.name
            checkpoint_path = split_dir / 'best.pt'
            output_path = split_dir / 'embeddings.pt'
            
            if not checkpoint_path.exists():
                print(f"{split_name}: No checkpoint found")
                continue
            
            # Skip if already exists
            if output_path.exists():
                size_mb = output_path.stat().st_size / 1024 / 1024
                print(f"{split_name}: SKIPPED (embeddings exist, {size_mb:.1f} MB)")
                skipped_count += 1
                continue
            
            # Extract for this split
            print(f"\nExtracting {split_name}...")
            embeddings_data = extract_embeddings(
                args.dataset, args.model, args.K, args.seed,
                checkpoint_path, output_path
            )
            
            num_layers = len(embeddings_data['embeddings'])
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"SUCCESS: {num_layers} layers, {size_mb:.1f} MB")
            extracted_count += 1
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"  Extracted: {extracted_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Total splits: {len(split_dirs)}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

