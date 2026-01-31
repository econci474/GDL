"""Linear probing at each depth to compute metrics."""

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from metrics import compute_nll, compute_accuracy, entropy_from_probs, compute_entropy_stats
from utils import set_seed


def probe_at_depth(embeddings, labels, train_mask, val_mask, test_mask, C_values, seed):
    """
    Train and evaluate a linear probe at a single depth.
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] labels
        train_mask, val_mask, test_mask: [N] boolean masks
        C_values: List of regularization values to try
        seed: Random seed
        
    Returns:
        dict with metrics: val_nll, val_acc, test_acc, entropies, etc.
    """
    set_seed(seed)
    
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(train_mask, torch.Tensor):
        train_mask = train_mask.numpy()
    if isinstance(val_mask, torch.Tensor):
        val_mask = val_mask.numpy()
    if isinstance(test_mask, torch.Tensor):
        test_mask = test_mask.numpy()
    
    # Extract training, validation, and test sets
    X_train = embeddings[train_mask]
    y_train = labels[train_mask]
    X_val = embeddings[val_mask]
    y_val = labels[val_mask]
    X_test = embeddings[test_mask]
    y_test = labels[test_mask]
    
    # Grid search for best C on validation set
    param_grid = {'C': C_values}
    probe = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=seed
    )
    
    grid_search = GridSearchCV(
        probe,
        param_grid,
        scoring='accuracy',
        cv=[(list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_val))))],  # Single split: train vs val
        refit=True
    )
    
    # Fit on train+val combined for grid search
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    grid_search.fit(X_train_val, y_train_val)
    
    best_probe = grid_search.best_estimator_
    best_C = grid_search.best_params_['C']
    
    # Get probabilities for val and test sets
    p_val = best_probe.predict_proba(X_val)
    p_test = best_probe.predict_proba(X_test)
    
    # Compute metrics
    val_nll = compute_nll(p_val, y_val)
    val_acc = compute_accuracy(np.argmax(p_val, axis=1), y_val)
    test_acc = compute_accuracy(np.argmax(p_test, axis=1), y_test)
    
    # Compute entropy statistics
    val_entropy_stats = compute_entropy_stats(p_val, y_val)
    test_entropy_stats = compute_entropy_stats(p_test, y_test)
    
    results = {
        'best_C': best_C,
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
    print(f"\n{'='*60}")
    print(f"Probing: {model_name} on {dataset_name} (K={K}, seed={seed})")
    print(f"{'='*60}")
    
    # Load embeddings
    embeddings_path = Path(config['runs_dir']) / dataset_name / model_name / f'seed_{seed}' / 'embeddings.pt'
    
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
    
    # Probe at each depth
    results_list = []
    
    for k in range(K + 1):
        print(f"\nProbing at depth k={k}...")
        
        emb = embeddings_dict[k]
        results = probe_at_depth(
            emb, labels, train_mask, val_mask, test_mask,
            config['probe_C_values'], seed
        )
        
        # Add depth to results
        results['k'] = k
        results_list.append(results)
        
        print(f"  Val NLL: {results['val_nll']:.4f}, Val Acc: {results['val_acc']:.4f}")
        print(f"  Test Acc: {results['test_acc']:.4f}, Test Entropy: {results['test_entropy_mean']:.4f}")
        print(f"  Best C: {results['best_C']}")
    
    # Create DataFrame and save
    df = pd.DataFrame(results_list)
    
    # Reorder columns
    cols = ['k', 'val_nll', 'val_acc', 'val_entropy_mean', 'val_entropy_std',
            'test_acc', 'test_entropy_mean', 'test_entropy_std',
            'correct_entropy_mean', 'incorrect_entropy_mean', 'best_C']
    df = df[cols]
    
    # Save to tables directory
    output_dir = Path(config['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'{dataset_name}_{model_name}_seed{seed}_probe.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Probing complete!")
    print(f"  Results saved to: {output_path}")
    print(f"\nSummary:")
    print(df[['k', 'val_nll', 'val_acc', 'test_acc', 'test_entropy_mean']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Linear probing at each depth')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    probe_all_depths(args.dataset, args.model, args.K, args.seed, config)


if __name__ == '__main__':
    main()
