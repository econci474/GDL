"""
Comprehensive comparison plots showing all metrics for all 3 training methods.
Integrates 4 visualization types:
1. Comprehensive metrics (accuracy, entropy, confidence) 
2. Per-class confidence heatmaps
3. Scatter plots (entropy vs correct-class probability)
4. Per-class entropy heatmaps
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def load_all_methods_data(dataset, model, K, seed, loss_types, split='val'):
    """Load classifier head data for specified loss types.
    
    Args:
        dataset: Dataset name
        model: Model name
        K: Number of layers
        seed: Random seed
        loss_types: List of loss type directory names (e.g., ['ce_plus_R_R1.0_hard', 'ce_plus_R_R1.0_smooth'])
        split: 'val' or 'test'
    """
    
    # Load the dataset to get labels and masks
    from src.datasets import load_dataset as load_ds
    data_obj, num_classes, _ = load_ds(
        dataset,
        root_dir='data',
        planetoid_normalize=False,
        planetoid_split='public'
    )
    
    # Get labels and split mask
    labels = data_obj.y.numpy()
    if split == 'val':
        split_mask = data_obj.val_mask.numpy()
    else:
        split_mask = data_obj.test_mask.numpy()
    
    # Load classifier outputs for each loss type
    data_dict = {}
    for loss_type in loss_types:
        if loss_type == 'probe':
            # Special handling for probe data from arrays
            probe_path = Path(cfg.results_dir) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}_pernode.npz'
            if not probe_path.exists():
                print(f"Warning: Probe data not found: {probe_path}")
                continue
            probe_data = np.load(probe_path)
            
            # Extract probabilities for each layer
            probs_list = []
            for k in range(K + 1):
                probe_key = f'p_{split}_{k}'
                probs_list.append(probe_data[probe_key])
            
            data_dict['probe'] = probs_list
        else:
            # Load classifier head data
            loss_path = Path(cfg.classifier_heads_dir) / loss_type / dataset / model / f'seed_{seed}' / f'K_{K}' / 'layer_probs.npz'
            if not loss_path.exists():
                print(f"Warning: Data not found for {loss_type}: {loss_path}")
                continue
            loss_data = np.load(loss_path)
            
            # Extract probabilities for each layer
            probs_list = []
            for k in range(K + 1):
                key = f'{split}_probs_{k}'
                probs_list.append(loss_data[key])
            
            data_dict[loss_type] = probs_list
    
    # Filter labels to split nodes only
    labels_split = labels[split_mask]
    
    data_dict['labels'] = labels_split
    return data_dict



def calc_accuracy(probs_list, labels):
    """Calculate accuracy at each layer."""
    accs = []
    for k in range(len(probs_list)):
        preds = probs_list[k].argmax(axis=1)
        acc = (preds == labels).mean()
        accs.append(acc)
    return accs


def calc_entropy(probs):
    """Calculate entropy per node."""
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)


def plot_comprehensive_3methods(data, dataset, model, K, seed, loss_types, output_dir):
    """
    Create comprehensive comparison plot showing all metrics for specified loss types.
    2x3 grid:
    - Row 1: Test accuracy, mean entropy, mean max probability
    - Row 2: Confidence distributions at k=0, k=K/2, k=K
    """
    labels = data['labels']
    
    # Build methods_data from provided loss_types
    methods_data = {}
    for loss_type in loss_types:
        if loss_type in data:
            methods_data[loss_type] = data[loss_type]
    
    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Training Method Comparison: {dataset} | {model} | K={K} | seed={seed}', 
                 fontsize=15, fontweight='bold')
    
    layers = list(range(K + 1))
    layer_labels = [f'k={k}' for k in layers]
    colors = ['C0', 'C1', 'C2']
    markers = ['o', 's', '^']
    
    # Plot 1: Test Accuracy
    ax = axes[0, 0]
    for idx, (name, probs_list) in enumerate(methods_data.items()):
        accs = calc_accuracy(probs_list, labels)
        ax.plot(layers, accs, marker=markers[idx], label=name, linewidth=2.5,  
                markersize=9, color=colors[idx])
    
    ax.set_xlabel('Layer Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Test Accuracy by Layer', fontsize=13, fontweight='bold')
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    # Plot 2: Mean Entropy
    ax = axes[0, 1]
    for idx, (name, probs_list) in enumerate(methods_data.items()):
        entropies = []
        for k in range(K + 1):
            ent = calc_entropy(probs_list[k]).mean()
            entropies.append(ent)
        ax.plot(layers, entropies, marker=markers[idx], label=name, linewidth=2.5,
                markersize=9, color=colors[idx])
    
    ax.set_xlabel('Layer Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Entropy', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Entropy by Layer', fontsize=13, fontweight='bold')
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    # Plot 3: Mean Max Probability
    ax = axes[0, 2]
    for idx, (name, probs_list) in enumerate(methods_data.items()):
        max_probs = []
        for k in range(K + 1):
            max_prob = probs_list[k].max(axis=1).mean()
            max_probs.append(max_prob)
        ax.plot(layers, max_probs, marker=markers[idx], label=name, linewidth=2.5,
                markersize=9, color=colors[idx])
    
    ax.set_xlabel('Layer Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Max Probability', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence by Layer', fontsize=13, fontweight='bold')
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    # Plots 4-6: Confidence distributions at selected layers
    selected_layers = [0, K//2, K]
    for idx_subplot, k in enumerate(selected_layers):
        ax = axes[1, idx_subplot]
        
        for idx, (name, probs_list) in enumerate(methods_data.items()):
            max_probs = probs_list[k].max(axis=1)
            ax.hist(max_probs, bins=25, alpha=0.4, label=name, density=True,
                    color=colors[idx], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Max Probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'Layer k={k}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
    
    plt.tight_layout()
    
    # Save with loss types in filename
    loss_types_str = '_vs_'.join(loss_types)
    output_path = output_dir / f'{dataset}_GCN_k{K}_seed{seed}_comprehensive_{loss_types_str}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Comprehensive plot saved to: {output_path}')
    plt.close()


def plot_per_class_confidence(data, dataset, model, K, seed, loss_types, output_dir):
    """
    Create per-class confidence heatmaps for specified loss types.
    Includes heatmaps and bar charts for selected classes.
    """
    labels = data['labels']
    
    # Get first loss type data to determine dimensions
    first_loss_type = loss_types[0]
    n_classes = data[first_loss_type][0].shape[1]
    
    
    # Calculate per-class mean max probabilities for each loss type
    class_conf_data = {}
    for loss_type in loss_types:
        if loss_type not in data:
            continue
        probs_list = data[loss_type]
        conf = np.zeros((K + 1, n_classes))
        for k in range(K + 1):
            for c in range(n_classes):
                class_mask = (labels == c)
                if class_mask.sum() > 0:
                    conf[k, c] = probs_list[k][class_mask].max(axis=1).mean()
        class_conf_data[loss_type] = conf
    
    # Create visualization
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Probe vs Classifier Head: Confidence Analysis\\n{dataset} | {model} | K={K} | seed={seed}',  
                 fontsize=14, fontweight='bold')
    
    layers = list(range(K + 1))
    layer_labels = [f'k={k}' for k in layers]
    
    # Row 1: Heatmaps
    cmaps = ['YlOrRd', 'YlGnBu', 'PuRd', 'Greens', 'Oranges', 'Purples']
    methods_heatmap = [(loss_type, class_conf_data[loss_type], cmaps[idx % len(cmaps)]) 
                       for idx, loss_type in enumerate(loss_types) if loss_type in class_conf_data]
    
    for idx, (name, conf_data, cmap) in enumerate(methods_heatmap):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(conf_data.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        ax.set_xlabel('Layer Depth', fontsize=11, fontweight='bold')
        ax.set_ylabel('Class', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}: Mean Max Probability', fontsize=12, fontweight='bold')
        ax.set_xticks(layers)
        ax.set_xticklabels(layer_labels)
        ax.set_yticks(range(n_classes))
        
        # Add values in cells
        for k in range(K + 1):
            for c in range(n_classes):
                text = ax.text(k, c, f'{conf_data[k, c]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Mean Max Probability')
    
    # Row 2: Class-wise comparison for selected classes
    selected_classes = [0, min(n_classes//2, n_classes-1), n_classes-1]
    for idx, c in enumerate(selected_classes[:3]):  # Max 3 classes
        ax = fig.add_subplot(gs[1, idx])
        
        x = np.arange(len(layers))
        n_methods = len(methods_heatmap)
        width = 0.8 / n_methods
        offsets = np.linspace(-0.4, 0.4, n_methods)
        
        for method_idx, (loss_type, conf_data, _) in enumerate(methods_heatmap):
            ax.bar(x + offsets[method_idx], conf_data[:, c], width, label=loss_type, alpha=0.8, color=f'C{method_idx}')
        
        ax.set_xlabel('Layer Depth', fontsize=10)
        ax.set_ylabel('Mean Max Probability', fontsize=10)
        ax.set_title(f'Class {c}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save with loss types in filename
    loss_types_str = '_vs_'.join(loss_types)
    output_path = output_dir / f'{dataset}_GCN_k{K}_seed{seed}_per_class_conf_{loss_types_str}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Per-class confidence plot saved to: {output_path}')
    plt.close()


def plot_scatter_entropy_vs_prob(data, dataset, model, K, seed, loss_types, output_dir):
    """
    Create scatter plots: Entropy vs Correct-Class Probability
    N rows (one per loss type), K+1 columns (one per layer)
    """
    labels = data['labels']
    
    # Get n_classes from first available loss type
    first_loss_type = loss_types[0]
    n_classes = data[first_loss_type][0].shape[1]
    
    # Calculate metrics for each method
    def calc_metrics(probs_list, labels):
        metrics = []
        for k in range(K + 1):
            probs_k = probs_list[k]
            entropy = calc_entropy(probs_k)
            correct_class_prob = probs_k[np.arange(len(probs_k)), labels]
            metrics.append((entropy, correct_class_prob))
        return metrics
    
    # Calculate metrics for each loss type
    all_metrics = {}
    for loss_type in loss_types:
        if loss_type in data:
            all_metrics[loss_type] = calc_metrics(data[loss_type], labels)
    
    # Create figure
    n_methods = len(all_metrics)
    fig, axes = plt.subplots(n_methods, K + 1, figsize=(15, 4 * n_methods))
    if K == 0:
        axes = axes.reshape(n_methods, 1)
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Entropy vs Correct-Class Probability\\n{dataset} | {model} | K={K} | seed={seed}',
                 fontsize=14, fontweight='bold')
    
    methods = [(loss_type, all_metrics[loss_type]) for loss_type in loss_types if loss_type in all_metrics]
    
    # Color palette for classes
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # Count nodes per class
    class_counts = np.bincount(labels, minlength=n_classes)
    
    for row_idx, (method_name, metrics) in enumerate(methods):
        for k in range(K + 1):
            ax = axes[row_idx, k]
            
            entropy, correct_prob = metrics[k]
            
            # Scatter plot with class colors
            for c in range(n_classes):
                class_mask = labels == c
                ax.scatter(entropy[class_mask], correct_prob[class_mask],
                          c=[colors[c]], label=f'Class {c} (n={class_counts[c]})',
                          alpha=0.6, s=25, edgecolors='black', linewidth=0.3)
            
            # Formatting
            if k == 0:
                ax.set_ylabel('Correct-Class\\nProbability', fontsize=10, fontweight='bold')
            
            if row_idx == 2:  # Bottom row
                ax.set_xlabel('Predictive Entropy', fontsize=10, fontweight='bold')
            
            # Title
            if row_idx == 0:
                ax.set_title(f'Depth k={k}', fontsize=11, fontweight='bold')
            
            # Row label
            if k == 0:
                ax.text(-0.35, 0.5, method_name, transform=ax.transAxes,
                       fontsize=11, fontweight='bold', rotation=90,
                       verticalalignment='center')
            
            # Legend only on rightmost plot
            if k == K:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                         fontsize=7, framealpha=0.9)
            
            ax.set_xlim([0, max(2, entropy.max() * 1.1)])
            ax.set_ylim([0, 1.0])
            ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    # Save with loss types in filename
    loss_types_str = '_vs_'.join(loss_types)
    output_path = output_dir / f'{dataset}_GCN_k{K}_seed{seed}_scatter_{loss_types_str}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Scatter plot saved to: {output_path}')
    plt.close()


def plot_per_class_entropy(data, dataset, model, K, seed, loss_types, output_dir):
    """
    Create per-class ENTROPY heatmaps for specified loss types.
    """
    labels = data['labels']
    
    # Get n_classes
    first_loss_type = loss_types[0]
    n_classes = data[first_loss_type][0].shape[1]
    
    # Calculate per-class mean entropy for each loss type
    entropy_data_dict = {}
    for loss_type in loss_types:
        if loss_type not in data:
            continue
        probs_list = data[loss_type]
        class_entropy = np.zeros((K + 1, n_classes))
        
        for k in range(K + 1):
            ent = calc_entropy(probs_list[k])
            for c in range(n_classes):
                class_mask = (labels == c)
                if class_mask.sum() > 0:
                    class_entropy[k, c] = ent[class_mask].mean()
        
        entropy_data_dict[loss_type] = class_entropy
    
    # Create N-row visualization
    from matplotlib.gridspec import GridSpec
    n_methods = len(entropy_data_dict)
    fig = plt.figure(figsize=(16, 4 * n_methods))
    gs = GridSpec(n_methods, 1, height_ratios=[1] * n_methods, hspace=0.35)
    
    fig.suptitle(f'Per-Class Mean Entropy Comparison\\n{dataset} | {model} | K={K} | seed={seed}', 
                 fontsize=15, fontweight='bold')
    
    layers = list(range(K + 1))
    layer_labels = [f'k={k}' for k in layers]
    
    class_counts = np.bincount(labels, minlength=n_classes)
    class_labels_with_counts = [f'Class {c}\\n(n={count})' for c, count in enumerate(class_counts)]
    
    cmaps = ['YlOrRd', 'YlGnBu', 'PuRd', 'Greens', 'Oranges', 'Purples']
    methods = [(loss_type, entropy_data_dict[loss_type], cmaps[idx % len(cmaps)]) 
               for idx, loss_type in enumerate(loss_types) if loss_type in entropy_data_dict]
    
    for row_idx, (method_name, entropy_data, cmap) in enumerate(methods):
        ax = fig.add_subplot(gs[row_idx, 0])
        
        # Plot heatmap
        im = ax.imshow(entropy_data.T, aspect='auto', cmap=cmap, vmin=0, vmax=max(2, entropy_data.max()))
        ax.set_xlabel('Layer Depth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Class', fontsize=12, fontweight='bold')
        ax.set_title(method_name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(layers)
        ax.set_xticklabels(layer_labels, fontsize=11)
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(class_labels_with_counts, fontsize=10)
        
        # Add values in cells
        for k in range(K + 1):
            for c in range(n_classes):
                val = entropy_data[k, c]
                color = 'white' if val > 1.0 else 'black'
                ax.text(k, c, f'{val:.2f}', ha="center", va="center",  
                       color=color, fontsize=9, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mean Entropy', fontsize=11)
    
    plt.tight_layout()
    
    # Save with loss types in filename
    loss_types_str = '_vs_'.join(loss_types)
    output_path = output_dir / f'{dataset}_GCN_k{K}_seed{seed}_entropy_{loss_types_str}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'✓ Entropy heatmap saved to: {output_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive comparison plots')
    parser.add_argument('--dataset', type=str, required=True, choices=['Cora', 'PubMed'])
    parser.add_argument('--model', type=str, required=True, choices=['GCN'])
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--loss-types', type=str, nargs='+', required=True,
                       help='Loss type directory names to compare (e.g., ce_plus_R_R1.0_hard ce_plus_R_R1.0_smooth)')
    
    args = parser.parse_args()
    
    print(f'\\n{"="*60}')
    print(f'Generating comprehensive comparison plots:')
    print(f'  Dataset: {args.dataset}, Model: {args.model}')
    print(f'  K={args.K}, seed={args.seed}, split={args.split}')
    print(f'  Loss types: {", ".join(args.loss_types)}')
    print(f'{"="*60}\\n')
    
    # Load data
    print('Loading data...')
    data = load_all_methods_data(args.dataset, args.model, args.K, args.seed, args.loss_types, args.split)
    print('✓ Data loaded successfully')
    
    # Create output directory - all plots go in same directory
    output_dir = Path(cfg.figures_dir) / args.dataset / args.model / f'K_{args.K}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single directory for all plot types
    dirs = {
        'comprehensive': output_dir,
        'per_class_conf': output_dir,
        'scatter': output_dir,
        'entropy': output_dir
    }
    
    # Generate all plots
    print('\\nGenerating plots...')
    plot_comprehensive_3methods(data, args.dataset, args.model, args.K, args.seed, args.loss_types, dirs['comprehensive'])
    plot_per_class_confidence(data, args.dataset, args.model, args.K, args.seed, args.loss_types, dirs['per_class_conf'])
    plot_scatter_entropy_vs_prob(data, args.dataset, args.model, args.K, args.seed, args.loss_types, dirs['scatter'])
    plot_per_class_entropy(data, args.dataset, args.model, args.K, args.seed, args.loss_types, dirs['entropy'])
    
    print(f'\\n✓ All plots generated successfully!')
    print(f'  Output directories:')
    for name, path in dirs.items():
        print(f'    - {name}: {path}')


if __name__ == '__main__':
    main()
