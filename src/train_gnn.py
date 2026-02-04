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
from src.models import GCNNet, GATNet
from src.utils import set_seed, to_device, get_device


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
    
    return val_loss, val_acc.item(), test_acc.item()


def train_gnn(dataset_name, model_name, K, seed, config):
    """
    Train a GNN model with early stopping.
    
    Args:
        dataset_name: Name of dataset ('Cora', etc.)
        model_name: 'GCN' or 'GAT'
        K: Number of layers
        seed: Random seed
        config: Configuration dictionary
    """
    # Set seed for reproducibility
    set_seed(seed)
    device = get_device()
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {dataset_name} (K={K}, seed={seed})")
    print(f"{'='*60}")
    
    data = load_dataset(dataset_name)
    
    from src.models import GCNNet, GATNet, GraphSAGENet
    
    if model_name == 'GCN':
        model = GCNNet(num_features=data.num_features, 
                       hidden_dim=config['hidden_dim'], 
                       num_classes=int(data.y.max().item()) + 1, # Assuming data.num_classes is not available directly
                       K=K,
                       dropout=None)
    elif model_name == 'GAT':
        model = GATNet(num_features=data.num_features, 
                       hidden_dim=config['hidden_dim'], 
                       num_classes=int(data.y.max().item()) + 1, # Assuming data.num_classes is not available directly
                       K=K,
                       heads=config.get('gat_heads', 8),
                       dropout=None)
    elif model_name == 'GraphSAGE':
        model = GraphSAGENet(num_features=data.num_features,
                            hidden_dim=config['hidden_dim'],
                            num_classes=int(data.y.max().item()) + 1, # Assuming data.num_classes is not available directly
                            K=K,
                            aggr=config.get('sage_aggr', 'mean'),
                            dropout=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    train_log = []
    
    output_dir = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining for up to {config['max_epochs']} epochs...")
    print(f"Early stopping patience: {config['patience']}")
    
    for epoch in range(1, config['max_epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, device)
        
        # Evaluate
        val_loss, val_acc, test_acc = evaluate(model, data, device)
        
        # Log
        train_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_acc': test_acc
        })
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'test_acc': test_acc,
            }, output_dir / 'best.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Save training log
    log_df = pd.DataFrame(train_log)
    log_df.to_csv(output_dir / 'train_log.csv', index=False)
    
    print(f"\n✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_dir / 'best.pt'}")
    print(f"  Log saved to: {output_dir / 'train_log.csv'}")
    

def main():
    import sys
    from pathlib import Path
    # Add project root to path to import config
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import config as cfg
    
    parser = argparse.ArgumentParser(description='Train GNN model')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (Cora, PubMed, etc.)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (GCN, GAT)')
    parser.add_argument('--K', type=int, default=8,
                       help='Number of layers')
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
    
    # Train model for each seed
    for seed in seeds_to_run:
        train_gnn(args.dataset, args.model, args.K, seed, config)
        print()  # Add spacing between seeds


if __name__ == '__main__':
    main()
