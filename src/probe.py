"""Linear probing at each depth using PyTorch."""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from src.metrics import compute_nll, compute_accuracy, entropy_from_probs, compute_entropy_stats
from src.utils import set_seed, get_device


def train_linear_probe(X_train, y_train, X_val, y_val, num_classes, weight_decay, seed, device, max_epochs=500):
    """
    Train a linear probe (logistic regression) using PyTorch.
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        num_classes: Number of classes
        weight_decay: L2 regularization strength
        seed: Random seed
        device: Device to use
        max_epochs: Maximum training epochs
        
    Returns:
        Trained model
    """
    set_seed(seed)
    
    # Create linear probe
    input_dim = X_train.shape[1]
    probe = nn.Linear(input_dim, num_classes).to(device)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01, weight_decay=weight_decay)
    
    # Training loop
    best_val_loss = float('inf')
    best_state = None
    patience = 50
    patience_counter = 0
    
    for epoch in range(max_epochs):
        probe.train()
        optimizer.zero_grad()
        
        logits = probe(X_train)
        loss = F.cross_entropy(logits, y_train)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val)
            val_loss = F.cross_entropy(val_logits, y_val).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = probe.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    probe.load_state_dict(best_state)
    probe.eval()
    
    return probe


def probe_at_depth(embeddings, labels, train_mask, val_mask, test_mask, weight_decay_values, seed, device):
    """
    Train and evaluate a linear probe at a single depth with grid search over weight_decay.
    
    Args:
        embeddings: [N, D] embeddings (torch tensor)
        labels: [N] labels (torch tensor)
        train_mask, val_mask, test_mask: [N] boolean masks (torch tensor)
        weight_decay_values: List of weight decay values to try
        seed: Random seed
        device: Device to use
        
    Returns:
        dict with metrics: val_nll, val_acc, test_acc, entropies, etc.
    """
    set_seed(seed)
    
    # Move to device
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    # Extract sets
    X_train = embeddings[train_mask]
    y_train = labels[train_mask]
    X_val = embeddings[val_mask]
    y_val = labels[val_mask]
    X_test = embeddings[test_mask]
    y_test = labels[test_mask]
    
    num_classes = int(labels.max().item()) + 1
    
    # Grid search over weight decay
    best_val_nll = float('inf')
    best_probe = None
    best_wd = None
    
    for wd in weight_decay_values:
        probe = train_linear_probe(X_train, y_train, X_val, y_val, num_classes, wd, seed, device)
        
        # Evaluate on validation set
        with torch.no_grad():
            val_logits = probe(X_val)
            val_probs = F.softmax(val_logits, dim=1)
            val_nll_wd = F.cross_entropy(val_logits, y_val).item()
        
        if val_nll_wd < best_val_nll:
            best_val_nll = val_nll_wd
            best_probe = probe
            best_wd = wd
    
    # Evaluate best probe on val and test
    with torch.no_grad():
        val_logits = best_probe(X_val)
        test_logits = best_probe(X_test)
        
        p_val = F.softmax(val_logits, dim=1).cpu().numpy()
        p_test = F.softmax(test_logits, dim=1).cpu().numpy()
    
    y_val_np = y_val.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    # Compute metrics
    val_nll = compute_nll(p_val, y_val_np)
    val_acc = compute_accuracy(np.argmax(p_val, axis=1), y_val_np)
    test_acc = compute_accuracy(np.argmax(p_test, axis=1), y_test_np)
    
    # Compute entropy statistics
    val_entropy_stats = compute_entropy_stats(p_val, y_val_np)
    test_entropy_stats = compute_entropy_stats(p_test, y_test_np)
    
    results = {
        'best_weight_decay': best_wd,
        'val_nll': val_nll,
        'val_acc': val_acc,
        'val_entropy_mean': val_entropy_stats['mean'],
        'val_entropy_std': val_entropy_stats['std'],
        'test_acc': test_acc,
        'test_entropy_mean': test_entropy_stats['mean'],
        'test_entropy_std': test_entropy_stats['std'],
        'correct_entropy_mean': test_entropy_stats['correct_mean'],
        'incorrect_entropy_mean': test_entropy_stats['incorrect_mean'],
    }
    
    return results


def probe_all_depths(dataset_name, model_name, K, seed, config):
    """
    Probe at all depths k=0..K and save results.
    
    Args:
        dataset_name: Dataset name
        model_name: Model name
        K: Maximum depth
        seed: Random seed
        config: Configuration dict
    """
    device = get_device()
    
    print(f"\n{'='*60}")
    print(f"Probing: {model_name} on {dataset_name} (K={K}, seed={seed})")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Load embeddings with K in path
    embeddings_path = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / f'K_{K}' / 'embeddings.pt'
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    
    data = torch.load(embeddings_path)
    embeddings_dict = data['embeddings']
    labels = data['labels']
    train_mask = data['train_mask']
    val_mask = data['val_mask']
    test_mask = data['test_mask']
    
    print(f"Loaded embeddings from {embeddings_path}")
    print(f"Found {len(embeddings_dict)} depths (k=0..{K})")
    
    # Convert weight decay grid (analogous to C values in sklearn)
    # C in sklearn is 1/weight_decay, so we invert the probe_C_values
    weight_decay_values = [1.0/c if c > 0 else 0.0 for c in config['probe_C_values']]
    weight_decay_values = sorted(weight_decay_values)
    
    print(f"Weight decay grid: {weight_decay_values}")
    
    # Probe at each depth
    results_list = []
    
    for k in range(K + 1):
        print(f"\nProbing at depth k={k}...")
        
        emb = embeddings_dict[k]
        results = probe_at_depth(
            emb, labels, train_mask, val_mask, test_mask,
            weight_decay_values, seed, device
        )
        
        # Add depth to results
        results['k'] = k
        results_list.append(results)
        
        print(f"  Val NLL: {results['val_nll']:.4f}, Val Acc: {results['val_acc']:.4f}")
        print(f"  Test Acc: {results['test_acc']:.4f}, Test Entropy: {results['test_entropy_mean']:.4f}")
        print(f"  Best weight_decay: {results['best_weight_decay']:.6f}")
    
    # Create DataFrame and save
    df = pd.DataFrame(results_list)
    
    # Reorder columns
    cols = ['k', 'val_nll', 'val_acc', 'val_entropy_mean', 'val_entropy_std',
            'test_acc', 'test_entropy_mean', 'test_entropy_std',
            'correct_entropy_mean', 'incorrect_entropy_mean', 'best_weight_decay']
    df = df[cols]
    
    # Save to tables directory
    output_dir = Path(config['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'{dataset_name}_{model_name}_K{K}_seed{seed}_probe.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Probing complete!")
    print(f"  Results saved to: {output_path}")
    print(f"\nSummary:")
    print(df[['k', 'val_nll', 'val_acc', 'test_acc', 'test_entropy_mean']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Linear probing at each depth (PyTorch)')
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
    
    # Probe for each seed
    for seed in seeds_to_run:
        probe_all_depths(args.dataset, args.model, args.K, seed, config)
        print()  # Add spacing between seeds


if __name__ == '__main__':
    main()
