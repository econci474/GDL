"""
Visualize per-node entropy vs correct-class probability across depths.

Creates a multi-panel scatter plot where:
- Each dot is a validation node
- X-axis: Predictive entropy
- Y-axis: Probability assigned to correct class
- Color: True class label
- Panels: One per depth k
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def entropy_from_probs(probs, eps=1e-10):
    """Compute entropy from probability distributions."""
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def plot_entropy_vs_prob(dataset, model, K, seed, config, split='val'):
    """
    Create scatter plot of entropy vs correct-class probability.
    
    Args:
        dataset: Dataset name
        model: Model name
        K: Maximum depth
        seed: Random seed
        config: Config dict
        split: 'val' or 'test'
    """
    # Load per-node arrays
    arrays_path = Path(config['results_dir']) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}_pernode.npz'
    
    if not arrays_path.exists():
        print(f"Error: Per-node arrays not found at {arrays_path}")
        return
    
    data = np.load(arrays_path)
    k_list = data['k_list']
    
    # Load dataset to get true labels
    from src.datasets import load_dataset
    graph_data, _, _ = load_dataset(dataset)  # Returns (data, num_classes, dataset_kind)
    labels = graph_data.y.numpy()
    num_classes = len(np.unique(labels))
    
    # Determine which nodes to plot
    if split == 'val':
        mask = graph_data.val_mask.numpy()
    else:
        mask = graph_data.test_mask.numpy()
    
    plot_indices = np.where(mask)[0]
    plot_labels = labels[plot_indices]
    
    # Create figure with subplots
    num_depths = len(k_list)
    ncols = min(3, num_depths)
    nrows = int(np.ceil(num_depths / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if num_depths == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Color map for classes
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    # Plot each depth
    for idx, k in enumerate(k_list):
        ax = axes[idx]
        
        # Load probabilities for this depth
        p_key = f'p_{split}_{k}'
        if p_key not in data:
            print(f"Warning: {p_key} not found, skipping k={k}")
            continue
        
        probs = data[p_key]  # [N_split, num_classes] - already filtered to split!
        
        # Compute entropy
        H = entropy_from_probs(probs)
        
        # Get probability of correct class for each node in this split
        # probs is already filtered to the split, so use plot_labels
        assert len(probs) == plot_labels.shape[0], f"Shape mismatch: probs {len(probs)} vs labels {plot_labels.shape[0]}"
        p_correct = probs[np.arange(len(probs)), plot_labels]
        
        # Scatter plot, one class at a time for legend
        for c in range(num_classes):
            class_mask = plot_labels == c
            if class_mask.sum() > 0:
                # Add count to label only for k=0
                if k == 0:
                    label = f'Class {c} (n={class_mask.sum()})'
                else:
                    label = f'Class {c}'
                
                ax.scatter(H[class_mask], p_correct[class_mask],
                          c=[colors[c]], label=label, 
                          alpha=0.6, s=20, edgecolors='none')
        
        ax.set_xlabel('Predictive Entropy')
        ax.set_ylabel('P(Correct Class)')
        ax.set_title(f'Depth k={k}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    # Hide unused subplots
    for idx in range(num_depths, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{dataset}/{model} (K={K}, seed={seed}, {split} set):\n'
                 f'Per-Node Entropy vs Correct-Class Probability',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = figures_dir / f'{dataset}_{model}_k{K}_seed{seed}_{split}_entropy_vs_prob.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_entropy_vs_prob_aggregated(dataset, model, K, seeds, config, split='val', seed_mode='aggregated'):
    """
    Create aggregated scatter plot with mean probabilities and entropies across seeds.
    
    Args:
        dataset: Dataset name
        model: Model name
        K: Maximum depth
        seeds: List of seeds to aggregate
        config: Config dict
        split: 'val' or 'test'
    """
    # Load dataset to get true labels
    from src.datasets import load_dataset
    graph_data, _, _ = load_dataset(dataset)
    labels = graph_data.y.numpy()
    num_classes = len(np.unique(labels))
    
    # Determine which nodes to plot
    if split == 'val':
        mask = graph_data.val_mask.numpy()
    else:
        mask = graph_data.test_mask.numpy()
    
    plot_indices = np.where(mask)[0]
    plot_labels = labels[plot_indices]
    
    # Load data from all seeds
    all_probs = {}  # k -> list of probs arrays
    k_list = None
    
    for seed in seeds:
        arrays_path = Path(config['results_dir']) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}_pernode.npz'
        
        if not arrays_path.exists():
            print(f"Warning: Per-node arrays not found for seed {seed}, skipping")
            continue
        
        data = np.load(arrays_path)
        if k_list is None:
            k_list = data['k_list']
        
        # Collect probabilities for each depth
        for k in k_list:
            p_key = f'p_{split}_{k}'
            if p_key in data:
                if k not in all_probs:
                    all_probs[k] = []
                all_probs[k].append(data[p_key])
    
    if not all_probs or k_list is None:
        print("Error: No valid data found across seeds")
        return
    
    # Calculate class mean trajectories for summary panel
    class_mean_entropy = {}  # k -> array of shape [num_classes]
    class_mean_p_correct = {}  # k -> array of shape [num_classes]
    
    for k in k_list:
        if k not in all_probs or len(all_probs[k]) == 0:
            continue
        
        mean_probs = np.mean(all_probs[k], axis=0)
        H = entropy_from_probs(mean_probs)
        p_correct = mean_probs[np.arange(len(mean_probs)), plot_labels]
        
        # Compute mean per class
        class_H = np.zeros(num_classes)
        class_P = np.zeros(num_classes)
        for c in range(num_classes):
            class_mask = plot_labels == c
            if class_mask.sum() > 0:
                class_H[c] = H[class_mask].mean()
                class_P[c] = p_correct[class_mask].mean()
        
        class_mean_entropy[k] = class_H
        class_mean_p_correct[k] = class_P
    
    # Create figure with an extra row for trajectory summary
    num_depths = len(k_list)
    ncols = min(3, num_depths)
    nrows = int(np.ceil(num_depths / ncols)) + 1  # +1 for trajectory panel
    
    fig = plt.figure(figsize=(5*ncols, 4*nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.4, wspace=0.3)
    
    # Color map for classes
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    # Plot individual depth scatters
    for idx, k in enumerate(k_list):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        
        if k not in all_probs or len(all_probs[k]) == 0:
            print(f"Warning: No data for k={k}, skipping")
            continue
        
        # Average probabilities across seeds
        mean_probs = np.mean(all_probs[k], axis=0)  # [N_split, num_classes]
        
        # Compute entropy from mean probabilities
        H = entropy_from_probs(mean_probs)
        
        # Get probability of correct class
        assert len(mean_probs) == plot_labels.shape[0], f"Shape mismatch: probs {len(mean_probs)} vs labels {plot_labels.shape[0]}"
        p_correct = mean_probs[np.arange(len(mean_probs)), plot_labels]
        
        # Scatter plot, one class at a time for legend
        for c in range(num_classes):
            class_mask = plot_labels == c
            if class_mask.sum() > 0:
                # Add count to label only for k=0
                if k == 0:
                    label = f'Class {c} (n={class_mask.sum()})'
                else:
                    label = f'Class {c}'
                
                ax.scatter(H[class_mask], p_correct[class_mask],
                          c=[colors[c]], label=label, 
                          alpha=0.6, s=20, edgecolors='none')
        
        ax.set_xlabel('Mean Entropy', fontsize=10)
        ax.set_ylabel('Mean P(Correct)', fontsize=10)
        ax.set_title(f'Depth k={k}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    # Add trajectory summary panel spanning bottom row
    ax_traj = fig.add_subplot(gs[-1, :])
    
    # Plot trajectories for each class
    for c in range(num_classes):
        entropy_vals = [class_mean_entropy[k][c] for k in k_list if k in class_mean_entropy]
        p_correct_vals = [class_mean_p_correct[k][c] for k in k_list if k in class_mean_p_correct]
        
        if len(entropy_vals) > 0:
            # Draw connecting line
            ax_traj.plot(entropy_vals, p_correct_vals, '-', color=colors[c], 
                        alpha=0.5, linewidth=2, zorder=1)
            
            # Draw points for each depth
            for i, k in enumerate(k_list):
                if k in class_mean_entropy:
                    ax_traj.scatter(entropy_vals[i], p_correct_vals[i], 
                                  c=[colors[c]], s=100, edgecolors='black', 
                                  linewidth=1.5, zorder=2, alpha=0.8)
                    # Annotate with k value
                    if i == len(entropy_vals) - 1:  # Only label last point
                        ax_traj.annotate(f'C{c}', (entropy_vals[i], p_correct_vals[i]),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=9, fontweight='bold', color=colors[c])
    
    ax_traj.set_xlabel('Mean Entropy', fontsize=12, fontweight='bold')
    ax_traj.set_ylabel('Mean P(Correct Class)', fontsize=12, fontweight='bold')
    ax_traj.set_title('Class Trajectory Summary (k=0 ? K)', fontsize=13, fontweight='bold')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlim(left=0)
    ax_traj.set_ylim(0, 1)
    
    # Add arrow showing direction
    ax_traj.annotate('', xy=(0.95, 0.05), xytext=(0.85, 0.05),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax_traj.text(0.90, 0.08, 'Increasing\nDepth', transform=ax_traj.transAxes,
                ha='center', fontsize=10, color='gray', style='italic')
    
    plt.suptitle(f'{dataset}/{model} (K={K}, seeds={seeds}, {split} set):\\n'
                 f'Per-Node Mean Entropy vs Mean Correct-Class Probability',
                 fontsize=14, y=0.995)
    
    # Save figure
    figures_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    if seed_mode == 'custom':
        # Create not_seed_2 subfolder for custom seed lists
        figures_dir = figures_dir / 'not_seed_2'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    seed_str = '_'.join(map(str, seeds)) if seed_mode == 'custom' else 'all'
    output_path = figures_dir / f'{dataset}_{model}_k{K}_seed_{seed_str}_{split}_entropy_vs_prob_with_trajectories.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_entropy_vs_correctness(dataset, model, K, seed, config, split='val'):
    """
    Create scatter plot of entropy vs binary correctness (correct/incorrect).
    
    Args:
        dataset: Dataset name
        model: Model name
        K: Maximum depth
        seed: Random seed
        config: Config dict
        split: 'val' or 'test'
    """
    # Load per-node arrays
    arrays_path = Path(config['results_dir']) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}_pernode.npz'
    
    if not arrays_path.exists():
        print(f"Error: Per-node arrays not found at {arrays_path}")
        return
    
    data = np.load(arrays_path)
    k_list = data['k_list']
    
    # Load dataset to get true labels
    from src.datasets import load_dataset
    graph_data, _, _ = load_dataset(dataset)
    labels = graph_data.y.numpy()
    
    # Determine which nodes to plot
    if split == 'val':
        mask = graph_data.val_mask.numpy()
    else:
        mask = graph_data.test_mask.numpy()
    
    plot_indices = np.where(mask)[0]
    plot_labels = labels[plot_indices]
    
    # Create figure with subplots
    num_depths = len(k_list)
    ncols = min(3, num_depths)
    nrows = int(np.ceil(num_depths / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if num_depths == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Colors for correct/incorrect
    color_correct = 'green'
    color_incorrect = 'red'
    
    # Plot each depth
    for idx, k in enumerate(k_list):
        ax = axes[idx]
        
        # Load probabilities for this depth
        p_key = f'p_{split}_{k}'
        if p_key not in data:
            print(f"Warning: {p_key} not found, skipping k={k}")
            continue
        
        probs = data[p_key]  # [N_split, num_classes] - already filtered to split!
        
        # Compute entropy
        H = entropy_from_probs(probs)
        
        # Get predicted labels and determine correctness
        assert len(probs) == plot_labels.shape[0], f"Shape mismatch: probs {len(probs)} vs labels {plot_labels.shape[0]}"
        pred_labels = np.argmax(probs, axis=1)
        is_correct = (pred_labels == plot_labels)
        
        # Get probability of predicted class (max probability)
        p_pred = np.max(probs, axis=1)
        
        # Count correct/incorrect
        n_correct = is_correct.sum()
        n_incorrect = (~is_correct).sum()
        
        # Scatter plot for incorrect predictions
        if n_incorrect > 0:
            label_incorrect = f'Incorrect (n={n_incorrect})' if k == 0 else 'Incorrect'
            ax.scatter(H[~is_correct], p_pred[~is_correct],
                      c=color_incorrect, label=label_incorrect,
                      alpha=0.6, s=20, edgecolors='none')
        
        # Scatter plot for correct predictions
        if n_correct > 0:
            label_correct = f'Correct (n={n_correct})' if k == 0 else 'Correct'
            ax.scatter(H[is_correct], p_pred[is_correct],
                      c=color_correct, label=label_correct,
                      alpha=0.6, s=20, edgecolors='none')
        
        ax.set_xlabel('Predictive Entropy')
        ax.set_ylabel('P(Predicted Class)')
        ax.set_title(f'Depth k={k}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    # Hide unused subplots
    for idx in range(num_depths, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{dataset}/{model} (K={K}, seed={seed}, {split} set):\n'
                 f'Per-Node Entropy vs Prediction Correctness',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = figures_dir / f'{dataset}_{model}_k{K}_seed{seed}_{split}_entropy_vs_correctness.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_entropy_vs_correctness_aggregated(dataset, model, K, seeds, config, split='val', seed_mode='aggregated'):
    """
    Create aggregated correctness plot: average probs across seeds, then classify by argmax.
    
    Args:
        dataset: Dataset name
        model: Model name
        K: Maximum depth
        seeds: List of seeds to aggregate
        config: Config dict
        split: 'val' or 'test'
    """
    # Load dataset to get true labels
    from src.datasets import load_dataset
    graph_data, _, _ = load_dataset(dataset)
    labels = graph_data.y.numpy()
    
    # Determine which nodes to plot
    if split == 'val':
        mask = graph_data.val_mask.numpy()
    else:
        mask = graph_data.test_mask.numpy()
    
    plot_indices = np.where(mask)[0]
    plot_labels = labels[plot_indices]
    
    # Load data from all seeds
    all_probs = {}  # k -> list of probs arrays
    k_list = None
    
    for seed in seeds:
        arrays_path = Path(config['results_dir']) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}_pernode.npz'
        
        if not arrays_path.exists():
            print(f"Warning: Per-node arrays not found for seed {seed}, skipping")
            continue
        
        data = np.load(arrays_path)
        if k_list is None:
            k_list = data['k_list']
        
        # Collect probabilities for each depth
        for k in k_list:
            p_key = f'p_{split}_{k}'
            if p_key in data:
                if k not in all_probs:
                    all_probs[k] = []
                all_probs[k].append(data[p_key])
    
    if not all_probs or k_list is None:
        print("Error: No valid data found across seeds")
        return
    
    # Create figure with subplots
    num_depths = len(k_list)
    ncols = min(3, num_depths)
    nrows = int(np.ceil(num_depths / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if num_depths == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Colors for correct/incorrect
    color_correct = 'green'
    color_incorrect = 'red'
    
    # Plot each depth
    for idx, k in enumerate(k_list):
        ax = axes[idx]
        
        if k not in all_probs or len(all_probs[k]) == 0:
            print(f"Warning: No data for k={k}, skipping")
            continue
        
        # Average probabilities across seeds
        mean_probs = np.mean(all_probs[k], axis=0)  # [N_split, num_classes]
        
        # Compute entropy from mean probabilities
        H = entropy_from_probs(mean_probs)
        
        # Get predicted labels from mean probabilities
        assert len(mean_probs) == plot_labels.shape[0], f"Shape mismatch: probs {len(mean_probs)} vs labels {plot_labels.shape[0]}"
        pred_labels = np.argmax(mean_probs, axis=1)
        is_correct = (pred_labels == plot_labels)
        
        # Get probability of predicted class (max probability)
        p_pred = np.max(mean_probs, axis=1)
        
        # Count correct/incorrect
        n_correct = is_correct.sum()
        n_incorrect = (~is_correct).sum()
        
        # Scatter plot for incorrect predictions
        if n_incorrect > 0:
            label_incorrect = f'Incorrect (n={n_incorrect})' if k == 0 else 'Incorrect'
            ax.scatter(H[~is_correct], p_pred[~is_correct],
                      c=color_incorrect, label=label_incorrect,
                      alpha=0.6, s=20, edgecolors='none')
        
        # Scatter plot for correct predictions
        if n_correct > 0:
            label_correct = f'Correct (n={n_correct})' if k == 0 else 'Correct'
            ax.scatter(H[is_correct], p_pred[is_correct],
                      c=color_correct, label=label_correct,
                      alpha=0.6, s=20, edgecolors='none')
        
        ax.set_xlabel('Mean Predictive Entropy')
        ax.set_ylabel('Mean P(Predicted Class)')
        ax.set_title(f'Depth k={k}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    # Hide unused subplots
    for idx in range(num_depths, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{dataset}/{model} (K={K}, seeds={seeds}, {split} set):\n'
                 f'Per-Node Mean Entropy vs Prediction Correctness',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    if seed_mode == 'custom':
        # Create not_seed_2 subfolder for custom seed lists
        figures_dir = figures_dir / 'not_seed_2'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    seed_str = '_'.join(map(str, seeds)) if seed_mode == 'custom' else 'all'
    output_path = figures_dir / f'{dataset}_{model}_k{K}_seed_{seed_str}_{split}_entropy_vs_correctness.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot entropy vs correct-class probability or correctness')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=str, default='0', help='Seed, "all", or comma-separated list like "0,1,3"')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--plot_type', type=str, default='probability', 
                       choices=['probability', 'correctness'],
                       help='Plot type: probability (per-class) or correctness (binary)')
    
    args = parser.parse_args()
    
    # Convert config to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Parse seeds
    if args.seed.lower() == 'all':
        seeds = config['seeds']
        seed_mode = 'aggregated'
    elif ',' in args.seed:
        # Custom seed list like "0,1,3"
        seeds = [int(s.strip()) for s in args.seed.split(',')]
        seed_mode = 'custom'
    else:
        # Single seed
        seed = int(args.seed)
        seeds = None
        seed_mode = 'single'
    
    if args.plot_type == 'correctness':
        # Correctness plot (binary: correct/incorrect)
        if seed_mode == 'single':
            print(f"Creating entropy vs correctness plot for {args.dataset}/{args.model}")
            print(f"K={args.K}, seed={seed}, split={args.split}")
            plot_entropy_vs_correctness(args.dataset, args.model, args.K, seed, config, args.split)
        else:
            # Aggregated across seeds
            print(f"Creating aggregated entropy vs correctness plot for {args.dataset}/{args.model}")
            print(f"K={args.K}, seeds={seeds}, split={args.split}")
            plot_entropy_vs_correctness_aggregated(args.dataset, args.model, args.K, seeds, config, args.split, seed_mode)
        
    else:
        # Probability plot (per-class)
        if seed_mode == 'single':
            print(f"Creating entropy vs probability plot for {args.dataset}/{args.model}")
            print(f"K={args.K}, seed={seed}, split={args.split}")
            plot_entropy_vs_prob(args.dataset, args.model, args.K, seed, config, args.split)
        else:
            # Aggregated plot across seeds
            print(f"Creating aggregated entropy vs probability plot for {args.dataset}/{args.model}")
            print(f"K={args.K}, seeds={seeds}, split={args.split}")
            plot_entropy_vs_prob_aggregated(args.dataset, args.model, args.K, seeds, config, args.split, seed_mode)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

