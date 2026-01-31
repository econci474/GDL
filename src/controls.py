"""Negative control experiments: random labels and layer-0 baseline."""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.datasets import load_dataset
from src.metrics import entropy_from_probs
from src.utils import set_seed


def layer0_baseline(dataset_name, config, seed=0):
    """
    Probe on layer 0 (raw features) as baseline control.
    This is already computed in the main probe.py script (k=0).
    
    This function just extracts and reports those results.
    """
    probe_file = Path(config['tables_dir']) / f'{dataset_name}_GCN_seed{seed}_probe.csv'
    
    if not probe_file.exists():
        print(f"⚠ Probe results not found: {probe_file}")
        return None
    
    probe_df = pd.read_csv(probe_file)
    layer0_results = probe_df[probe_df['k'] == 0].iloc[0]
    
    print(f"\n📊 Layer-0 Baseline (raw features):")
    print(f"  Val Accuracy:  {layer0_results['val_acc']:.4f}")
    print(f"  Test Accuracy: {layer0_results['test_acc']:.4f}")
    print(f"  Val Entropy:   {layer0_results['val_entropy_mean']:.4f}")
    print(f"  Test Entropy:  {layer0_results['test_entropy_mean']:.4f}")
    
    return layer0_results


def random_label_control(dataset_name, K, seed, config):
    """
    Negative control: Train probes with randomly permuted training labels.
    
    Expect:
    - Accuracy ≈ 1/C (random chance)
    - Entropy ≈ log(C) (uniform distribution)
    """
    set_seed(seed)
    
    # Load data
    data = load_dataset(dataset_name)
    
    # Permute training labels only
    permuted_y = data.y.clone()
    train_indices = torch.where(data.train_mask)[0]
    permuted_train_labels = permuted_y[train_indices][torch.randperm(len(train_indices))]
    permuted_y[train_indices] = permuted_train_labels
    
    # Load embeddings from the trained model
    embeddings_dir = Path(config['runs_dir']) / dataset_name / 'GCN' / f'seed_{seed}'
    embeddings_file = embeddings_dir / 'embeddings.pt'
    
    if not embeddings_file.exists():
        print(f"⚠ Embeddings not found: {embeddings_file}")
        return None
    
    embeddings_data = torch.load(embeddings_file)
    embeddings_list = embeddings_data['embeddings']
    
    num_classes = int(data.y.max().item()) + 1
    expected_entropy = np.log(num_classes)  # Expected entropy for uniform distribution
    
    results = []
    
    print(f"\n🎲 Random Label Control ({dataset_name}, K={K}, seed={seed})")
    print(f"  Expected accuracy: {1/num_classes:.4f}")
    print(f"  Expected entropy: {expected_entropy:.4f}")
    print(f"\n  Probing with permuted labels...")
    
    for k in range(len(embeddings_list)):
        h_k = embeddings_list[k].numpy()
        
        # Train probe with permuted labels
        X_train = h_k[data.train_mask.numpy()]
        y_train = permuted_y[data.train_mask].numpy()
        
        X_val = h_k[data.val_mask.numpy()]
        y_val = data.y[data.val_mask].numpy()  # Use TRUE labels for evaluation
        
        X_test = h_k[data.test_mask.numpy()]
        y_test = data.y[data.test_mask].numpy()  # Use TRUE labels for evaluation
        
        # Grid search over regularization
        param_grid = {'C': config['probe_C_values']}
        clf = LogisticRegression(
            solver='lbfgs', 
            max_iter=1000,
            random_state=seed
        )
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='neg_log_loss', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_clf = grid_search.best_estimator_
        
        # Predict probabilities
        p_val = best_clf.predict_proba(X_val)
        p_test = best_clf.predict_proba(X_test)
        
        # Compute metrics
        val_acc = best_clf.score(X_val, y_val)
        test_acc = best_clf.score(X_test, y_test)
        
        # Compute NLL
        val_nll = -np.mean(np.log(p_val[np.arange(len(y_val)), y_val] + 1e-10))
        test_nll = -np.mean(np.log(p_test[np.arange(len(y_test)), y_test] + 1e-10))
        
        # Compute entropy
        val_entropy = entropy_from_probs(p_val)
        test_entropy = entropy_from_probs(p_test)
        
        results.append({
            'k': k,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'val_nll': val_nll,
            'test_nll': test_nll,
            'val_entropy_mean': val_entropy.mean(),
            'val_entropy_std': val_entropy.std(),
            'test_entropy_mean': test_entropy.mean(),
            'test_entropy_std': test_entropy.std(),
            'best_C': grid_search.best_params_['C']
        })
        
        if k % 2 == 0:
            print(f"    k={k}: test_acc={test_acc:.4f}, test_entropy={test_entropy.mean():.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = (
        Path(config['tables_dir']) / 
        f'{dataset_name}_GCN_seed{seed}_control_random_labels.csv'
    )
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Random label control complete")
    print(f"  Mean test accuracy: {results_df['test_acc'].mean():.4f} (expected: {1/num_classes:.4f})")
    print(f"  Mean test entropy: {results_df['test_entropy_mean'].mean():.4f} (expected: {expected_entropy:.4f})")
    print(f"  Saved to: {output_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Negative control experiments')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='GCN',
                       help='Model name (currently only GCN supported)')
    parser.add_argument('--K', type=int, default=8,
                       help='Number of layers')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--control', type=str, required=True,
                       choices=['layer0', 'random_labels', 'all'],
                       help='Which control to run')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Create output directory
    Path(config['tables_dir']).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Negative Controls: {args.dataset}")
    print(f"{'='*60}")
    
    if args.control in ['layer0', 'all']:
        layer0_baseline(args.dataset, config, args.seed)
    
    if args.control in ['random_labels', 'all']:
        random_label_control(args.dataset, args.K, args.seed, config)
    
    print(f"\n{'='*60}")
    print("✓ Control experiments complete")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
