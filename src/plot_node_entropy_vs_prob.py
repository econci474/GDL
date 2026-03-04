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


def _lt(loss_type: str) -> str:
    """Return a filename suffix for the given loss_type.

    Always includes the loss type so classifier-head plots are never
    confused with linear probe plots (e.g. '_ce_only', '_ce_plus_R_...').
    """
    return '_' + (loss_type or 'ce_only')


def _pretty_lt(loss_type: str) -> str:
    """Return a human-readable label for a loss_type (for plot titles)."""
    lt = loss_type or 'ce_only'
    lt = lt.replace('ce_plus_R_', '').replace('ce_only', 'CE only')
    lt = lt.replace('_smooth_', ' ').replace('_smooth', ' band-1.0to0.0')
    lt = lt.replace('band-1.5to0.2', 'band-1.5to0.25').replace('_', ' ')
    return lt


def _fig_lt(loss_type: str) -> str:
    """Return a normalised filename suffix for OUTPUT figures.

    Unlike _lt() (for pernode lookups), this:
      - Adds explicit band-1.0to0.0 when no band suffix exists
      - Normalises known precision truncations (band-1.5to0.2 → band-1.5to0.25)

    Examples:
      ce_only                               → _ce_only
      ce_plus_R_R10.0_smooth               → _ce_plus_R_R10.0_smooth_band-1.0to0.0
      ce_plus_R_R10.0_smooth_band-1.5to0.2 → _ce_plus_R_R10.0_smooth_band-1.5to0.25
    """
    import re
    lt = loss_type or 'ce_only'
    m = re.match(r'(ce_plus_R_R[\d.]+_smooth)(_band.+)?$', lt)
    if m:
        base = m.group(1)
        band = m.group(2) or '_band-1.0to0.0'        # add default band if missing
        band = band.replace('band-1.5to0.2', 'band-1.5to0.25')  # fix truncation
        return '_' + base + band
    return '_' + lt


def entropy_from_probs(probs, eps=1e-10):
    """Compute entropy from probability distributions."""
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def plot_entropy_vs_prob(dataset, model, K, seed, config, split='val', loss_type='ce_only'):
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
    arrays_path = Path(config['results_dir']) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}{_lt(loss_type)}_pernode.npz'
    
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

    # Sort classes by ascending node count (matches other plot functions)
    class_order = sorted(range(num_classes),
                         key=lambda c: int((plot_labels == c).sum()))

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
        
        # Scatter plot, one class at a time for legend (ascending count order)
        for c in class_order:
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
    
    plt.suptitle(f'{dataset}/{model} (K={K}, seed={seed}, {split} set)  [{_pretty_lt(loss_type)}]:\n'
                 f'Per-Node Entropy vs Correct-Class Probability',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = figures_dir / f'{dataset}_{model}_k{K}_seed{seed}{_fig_lt(loss_type)}_{split}_entropy_vs_prob.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_entropy_vs_prob_aggregated(dataset, model, K, seeds, config, split='val', seed_mode='aggregated', loss_type='ce_only'):
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
        arrays_path = Path(config['results_dir']) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}{_lt(loss_type)}_pernode.npz'
        
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
    
    num_depths = len(k_list)
    ncols = min(3, num_depths)
    nrows_scatter = int(np.ceil(num_depths / ncols))
    nrows = nrows_scatter + 1  # +1 for trajectory panel
    height_ratios = [1] * nrows_scatter + [2]  # trajectory row is 2x taller

    fig = plt.figure(figsize=(5*ncols, 4*nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.2, wspace=0.2, height_ratios=height_ratios)
    
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
    
    # --- Per-class trajectory panels -----------------------------------------
    # Layout: num_classes panels arranged in (traj_rows x traj_ncols) below scatter
    # The whole combined figure uses traj_ncols columns so scatter and trajectory
    # panels share the same grid — avoids the IndexError when traj_ncols > ncols.
    traj_ncols    = max(3, ncols)  # always at least 3 columns
    traj_nrows    = int(np.ceil(num_classes / traj_ncols))
    nrows_scatter = int(np.ceil(num_depths / traj_ncols))  # use traj_ncols here too
    nrows         = nrows_scatter + traj_nrows
    height_ratios = [1] * nrows_scatter + [1.1] * traj_nrows

    # Rebuild figure with updated layout (traj_ncols columns throughout)
    plt.close(fig)
    fig = plt.figure(figsize=(5 * traj_ncols, 4.5 * nrows))
    gs  = fig.add_gridspec(nrows, traj_ncols, hspace=0.35, wspace=0.2,
                           height_ratios=height_ratios)

    # Re-draw the per-depth scatter panels
    for idx, k in enumerate(k_list):
        row = idx // traj_ncols
        col = idx % traj_ncols
        ax  = fig.add_subplot(gs[row, col])
        if k not in all_probs or len(all_probs[k]) == 0:
            continue
        mean_probs = np.mean(all_probs[k], axis=0)
        H          = entropy_from_probs(mean_probs)
        p_correct  = mean_probs[np.arange(len(mean_probs)), plot_labels]
        for c in range(num_classes):
            cm = plot_labels == c
            if cm.sum() > 0:
                lbl = f'Class {c} (n={cm.sum()})' if k == 0 else f'Class {c}'
                ax.scatter(H[cm], p_correct[cm], c=[colors[c]], label=lbl,
                           alpha=0.6, s=20, edgecolors='none')
        ax.set_xlabel('Mean Entropy', fontsize=10)
        ax.set_ylabel('Mean P(Correct)', fontsize=10)
        ax.set_title(f'Depth k={k}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(0, 1)
        if idx == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)

    # Hide unused scatter slots (when num_depths < traj_ncols)
    for slot in range(num_depths, nrows_scatter * traj_ncols):
        fig.add_subplot(gs[slot // traj_ncols, slot % traj_ncols]).axis('off')

    # Depth colormap (k=0 → blue, k=K → red)
    depth_cmap = plt.cm.coolwarm
    K_max_val  = max(k_list)
    depth_norm = plt.Normalize(vmin=0, vmax=K_max_val)

    # Sort classes by ascending n for trajectory panels
    class_sizes   = sorted(range(num_classes), key=lambda c: int((plot_labels == c).sum()))
    ax_last = None

    for panel_idx, c in enumerate(class_sizes):
        tr = nrows_scatter + panel_idx // traj_ncols
        tc = panel_idx % traj_ncols
        ax_c = fig.add_subplot(gs[tr, tc])

        entropy_vals   = [class_mean_entropy[k][c]   for k in k_list if k in class_mean_entropy]
        p_correct_vals = [class_mean_p_correct[k][c] for k in k_list if k in class_mean_p_correct]
        k_vals         = [k for k in k_list if k in class_mean_entropy]

        if len(entropy_vals) == 0:
            ax_c.axis('off'); continue

        # Individual node scatter for this class, colored by depth
        for k in k_list:
            if k not in all_probs or len(all_probs[k]) == 0:
                continue
            mean_probs = np.mean(all_probs[k], axis=0)
            H          = entropy_from_probs(mean_probs)
            p_correct  = mean_probs[np.arange(len(mean_probs)), plot_labels]
            cm         = plot_labels == c
            if cm.sum() > 0:
                ax_c.scatter(H[cm], p_correct[cm],
                             color=depth_cmap(depth_norm(k)),
                             alpha=0.35, s=12, edgecolors='none', zorder=1)

        # Mean trajectory line with arrows between consecutive depths
        ax_c.plot(entropy_vals, p_correct_vals, '-',
                  color=colors[c], alpha=0.6, linewidth=2, zorder=2)
        for i in range(len(k_vals) - 1):
            dx = entropy_vals[i+1] - entropy_vals[i]
            dy = p_correct_vals[i+1] - p_correct_vals[i]
            ax_c.annotate('', xy=(entropy_vals[i+1], p_correct_vals[i+1]),
                          xytext=(entropy_vals[i], p_correct_vals[i]),
                          arrowprops=dict(arrowstyle='->', color=colors[c],
                                         lw=1.5, mutation_scale=12),
                          zorder=3)

        # Mark k=0 start with hollow circle, k=K end with filled circle
        ax_c.scatter(entropy_vals[0], p_correct_vals[0],
                     marker='o', s=55, facecolors='none',
                     edgecolors=colors[c], linewidth=1.8, zorder=6,
                     label='k=0 (start)')
        ax_c.scatter(entropy_vals[-1], p_correct_vals[-1],
                     marker='o', s=55, color=colors[c],
                     edgecolors='black', linewidth=0.8, zorder=6,
                     label=f'k={K_max_val} (end)')

        n_class = int((plot_labels == c).sum())
        ax_c.set_title(f'Class {c}  (n={n_class})', fontweight='bold', fontsize=11)
        ax_c.set_xlabel('Mean Entropy', fontsize=9)
        ax_c.set_ylabel('Mean P(Correct)', fontsize=9)
        ax_c.grid(True, alpha=0.3)
        ax_c.set_xlim(left=0); ax_c.set_ylim(0, 1)
        # No fixed right xlim — auto-scale to match upper scatter panels
        ax_c.legend(fontsize=7, loc='upper right', framealpha=0.85)
        ax_last = ax_c

    # Hide unused trajectory panels
    for c in range(num_classes, traj_nrows * traj_ncols):
        tr = nrows_scatter + c // traj_ncols
        tc = c % traj_ncols
        if tr < nrows:
            fig.add_subplot(gs[tr, tc]).axis('off')

    seeds_str = ', '.join(map(str, seeds))
    fig.suptitle(
        f'{model}/{dataset}, K={K}, seeds=[{seeds_str}], {split} set  [{_pretty_lt(loss_type)}]\n'
        f'Per-Node Mean Entropy vs Mean Correct-Class Probability',
        fontsize=14, fontweight='bold', y=0.985
    )
    # Use subplots_adjust instead of tight_layout — avoids the colorbar incompatibility warning
    # and directly controls all spacing
    fig.subplots_adjust(top=0.94, bottom=0.04, left=0.07, right=0.96,
                        hspace=0.3, wspace=0.2)

    # Colorbar next to last trajectory panel only, full height (added after tight_layout)
    if ax_last is not None:
        sm = plt.cm.ScalarMappable(cmap=depth_cmap, norm=depth_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_last, shrink=1.0, pad=0.04, aspect=15,
                            location='right')
        cbar.set_label('Depth k', fontsize=11)
        cbar.set_ticks(k_list)


    # ------------------------------------------------------------------ #
    # Save combined figure                                                 #
    # ------------------------------------------------------------------ #
    figures_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Always use 'seed_all' for any multi-seed aggregated plot
    stem = f'{dataset}_{model}_k{K}_seed_all{_fig_lt(loss_type)}_{split}'

    output_path = figures_dir / f'{stem}_entropy_vs_prob_with_trajectories.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # ------------------------------------------------------------------ #
    # Layers-only figure (depth scatter panels)                           #
    # ------------------------------------------------------------------ #
    ncols = min(3, num_depths)  # re-derive ncols for layers-only figure
    fig_l = plt.figure(figsize=(5 * ncols, 4 * nrows_scatter))
    gs_l  = fig_l.add_gridspec(nrows_scatter, ncols, hspace=0.35, wspace=0.2)
    for idx, k in enumerate(k_list):
        ax = fig_l.add_subplot(gs_l[idx // ncols, idx % ncols])
        if k not in all_probs or len(all_probs[k]) == 0:
            continue
        mean_probs = np.mean(all_probs[k], axis=0)
        H          = entropy_from_probs(mean_probs)
        p_correct  = mean_probs[np.arange(len(mean_probs)), plot_labels]
        for c in range(num_classes):
            cm = plot_labels == c
            if cm.sum() > 0:
                lbl = f'Class {c} (n={cm.sum()})' if k == 0 else f'Class {c}'
                ax.scatter(H[cm], p_correct[cm], c=[colors[c]], label=lbl,
                           alpha=0.6, s=20, edgecolors='none')
        ax.set_xlabel('Mean Entropy', fontsize=10)
        ax.set_ylabel('Mean P(Correct)', fontsize=10)
        ax.set_title(f'Depth k={k}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(0, 1)
        if idx == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
    top_l = min(0.90, 0.78 + 0.04 * max(nrows_scatter - 1, 0))
    fig_l.subplots_adjust(top=top_l, bottom=0.06, left=0.07, right=0.97,
                          hspace=0.35, wspace=0.2)
    fig_l.suptitle(
        f'{model}/{dataset}, K={K}, seeds=[{seeds_str}], {split} set\n'
        f'Per-Node Mean Entropy vs Mean Correct-Class Probability, by Layer',
        fontsize=13, fontweight='bold', x=0.5, y=min(0.98, top_l + 0.10)
    )
    layers_path = figures_dir / f'{stem}_layers.png'
    fig_l.savefig(layers_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {layers_path}")
    plt.close(fig_l)

    # ------------------------------------------------------------------ #
    # Classes-only figure (trajectory panels)                             #
    # ------------------------------------------------------------------ #
    fig_c = plt.figure(figsize=(5 * traj_ncols, 4.5 * traj_nrows + 1.0))
    gs_c  = fig_c.add_gridspec(traj_nrows, traj_ncols, hspace=0.35, wspace=0.2)
    ax_last_c = None
    for panel_idx, c in enumerate(class_sizes):
        tr = panel_idx // traj_ncols
        tc = panel_idx % traj_ncols
        ax_c = fig_c.add_subplot(gs_c[tr, tc])

        entropy_vals   = [class_mean_entropy[k][c]   for k in k_list if k in class_mean_entropy]
        p_correct_vals = [class_mean_p_correct[k][c] for k in k_list if k in class_mean_p_correct]
        k_vals         = [k for k in k_list if k in class_mean_entropy]

        if len(entropy_vals) == 0:
            ax_c.axis('off'); continue

        for k in k_list:
            if k not in all_probs or len(all_probs[k]) == 0:
                continue
            mean_probs = np.mean(all_probs[k], axis=0)
            H          = entropy_from_probs(mean_probs)
            p_correct  = mean_probs[np.arange(len(mean_probs)), plot_labels]
            cm         = plot_labels == c
            if cm.sum() > 0:
                ax_c.scatter(H[cm], p_correct[cm],
                             color=depth_cmap(depth_norm(k)),
                             alpha=0.35, s=12, edgecolors='none', zorder=1)

        ax_c.plot(entropy_vals, p_correct_vals, '-',
                  color=colors[c], alpha=0.6, linewidth=2, zorder=2)
        for i in range(len(k_vals) - 1):
            dx = entropy_vals[i+1] - entropy_vals[i]
            dy = p_correct_vals[i+1] - p_correct_vals[i]
            ax_c.annotate('', xy=(entropy_vals[i+1], p_correct_vals[i+1]),
                          xytext=(entropy_vals[i], p_correct_vals[i]),
                          arrowprops=dict(arrowstyle='->', color=colors[c],
                                         lw=1.5, mutation_scale=12),
                          zorder=3)
        ax_c.scatter(entropy_vals[0], p_correct_vals[0],
                     marker='o', s=55, facecolors='none',
                     edgecolors=colors[c], linewidth=1.8, zorder=6,
                     label='k=0 (start)')
        ax_c.scatter(entropy_vals[-1], p_correct_vals[-1],
                     marker='o', s=55, color=colors[c],
                     edgecolors='black', linewidth=0.8, zorder=6,
                     label=f'k={K_max_val} (end)')

        n_class = int((plot_labels == c).sum())
        ax_c.set_title(f'Class {c}  (n={n_class})', fontweight='bold', fontsize=11)
        ax_c.set_xlabel('Mean Entropy', fontsize=9)
        ax_c.set_ylabel('Mean P(Correct)', fontsize=9)
        ax_c.grid(True, alpha=0.3)
        ax_c.set_xlim(left=0); ax_c.set_ylim(0, 1)
        ax_c.legend(fontsize=7, loc='upper right', framealpha=0.85)
        ax_last_c = ax_c

    # Hide unused panels
    for extra in range(num_classes, traj_nrows * traj_ncols):
        tr = extra // traj_ncols
        tc = extra % traj_ncols
        if tr < traj_nrows:
            fig_c.add_subplot(gs_c[tr, tc]).axis('off')

    # Colorbar on the classes figure
    if ax_last_c is not None:
        sm = plt.cm.ScalarMappable(cmap=depth_cmap, norm=depth_norm)
        sm.set_array([])
        cbar_c = fig_c.colorbar(sm, ax=ax_last_c, shrink=1.0, pad=0.04, aspect=15,
                                location='right')
        cbar_c.set_label('Depth k', fontsize=11)
        cbar_c.set_ticks(k_list)

    # top_c / y_title: leave enough room for 2-line suptitle above panel titles
    # Panel titles render ABOVE the axes frame, so we need clear space above top_c.
    _fig_h = 4.5 * traj_nrows + 1.0  # approximate figure height in inches
    top_c   = min(0.94, 0.97 - 0.90 / _fig_h)
    y_title = min(0.98, top_c + 0.06 + 0.30 / _fig_h)
    fig_c.subplots_adjust(top=top_c, bottom=0.08, left=0.07, right=0.96,
                          hspace=0.3, wspace=0.2)
    fig_c.suptitle(
        f'{model}/{dataset}, K={K}, seeds=[{seeds_str}], {split} set\n'
        f'Per-Node Mean Entropy vs Mean Correct-Class Probability, by Class',
        fontsize=13, fontweight='bold', x=0.5, y=y_title
    )
    classes_path = figures_dir / f'{stem}_classes.png'
    fig_c.savefig(classes_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {classes_path}")
    plt.close(fig_c)


def plot_entropy_vs_correctness(dataset, model, K, seed, config, split='val', split_idx=None, loss_type='ce_only'):
    """
    Create scatter plot of entropy vs binary correctness (correct/incorrect).
    
    Args:
        dataset: Dataset name
        model: Model name
        K: Maximum depth
        seed: Random seed
        config: Config dict
        split: 'val' or 'test'
        split_idx: If set, use _split{N}_ pernode filename (heterophilous datasets).
    """
    # Build path with optional split suffix
    split_sfx   = f'_split{split_idx}' if split_idx is not None else ''
    arrays_path = Path(config['results_dir']) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}{split_sfx}{_lt(loss_type)}_pernode.npz'
    
    if not arrays_path.exists():
        print(f"Error: Per-node arrays not found at {arrays_path}")
        return
    
    data   = np.load(arrays_path)
    k_list = data['k_list']
    
    # Load dataset to get true labels
    from src.datasets import load_dataset
    graph_data, _, _ = load_dataset(dataset)
    labels = graph_data.y.numpy()
    
    # Determine which nodes to plot
    # For multi-split datasets val_mask / test_mask are (N, num_splits)
    if split_idx is not None:
        val_mask  = graph_data.val_mask.numpy()[:, split_idx]
        test_mask = graph_data.test_mask.numpy()[:, split_idx]
    else:
        val_mask  = graph_data.val_mask.numpy()
        test_mask = graph_data.test_mask.numpy()
    mask = val_mask if split == 'val' else test_mask
    
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
    
    plt.suptitle(f'{dataset}/{model} (K={K}, seed={seed}, {split} set)  [{_pretty_lt(loss_type)}]:\n'
                 f'Per-Node Entropy vs Prediction Correctness',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = figures_dir / f'{dataset}_{model}_k{K}_seed{seed}{_fig_lt(loss_type)}_{split}_entropy_vs_correctness.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_entropy_vs_correctness_aggregated(dataset, model, K, seeds, config, split='val', seed_mode='aggregated', split_idx=None, loss_type='ce_only'):
    """
    Create aggregated correctness plot: average probs across seeds, then classify by argmax.
    
    Args:
        dataset: Dataset name
        model: Model name
        K: Maximum depth
        seeds: List of seeds to aggregate
        config: Config dict
        split: 'val' or 'test'
        split_idx: If set, load per-node arrays with split suffix (e.g. _split0_pernode.npz).
                   Required for heterophilous datasets (Roman-empire, Squirrel).
    """
    # Load dataset to get true labels
    from src.datasets import load_dataset
    graph_data, _, _ = load_dataset(dataset)
    labels = graph_data.y.numpy()
    
    # Determine which nodes to plot
    # For multi-split datasets, val_mask/test_mask are (N, num_splits)
    if split_idx is not None:
        val_mask  = graph_data.val_mask.numpy()[:, split_idx]
        test_mask = graph_data.test_mask.numpy()[:, split_idx]
    else:
        val_mask  = graph_data.val_mask.numpy()
        test_mask = graph_data.test_mask.numpy()
    if split == 'val':
        mask = val_mask
    else:
        mask = test_mask
    
    plot_indices = np.where(mask)[0]
    plot_labels = labels[plot_indices]
    
    # Load data from all seeds
    all_probs = {}  # k -> list of probs arrays
    k_list = None
    
    # Build filename suffix for heterophilous datasets
    split_sfx = f'_split{split_idx}' if split_idx is not None else ''

    for seed in seeds:
        arrays_path = Path(config['results_dir']) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}{split_sfx}{_lt(loss_type)}_pernode.npz'
        
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
    
    plt.suptitle(f'{dataset}/{model} (K={K}, seeds={seeds}, {split} set)  [{_pretty_lt(loss_type)}]:\n'
                 f'Per-Node Mean Entropy vs Prediction Correctness',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Always use 'seed_all' for any multi-seed aggregated plot
    output_path = figures_dir / f'{dataset}_{model}_k{K}_seed_all{_fig_lt(loss_type)}_{split}_entropy_vs_correctness.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_entropy_vs_prob_allsplits(dataset, model, K, seeds, config,
                                   split_indices=None, split_sides=None,
                                   loss_type='ce_only'):
    """
    Variant of plot_entropy_vs_prob_aggregated for datasets with multiple train/val/test splits
    (e.g. Roman-Empire with split0–split9).  Pools all val+test observations across all splits
    and all seeds, then plots them together.

    File naming: {dataset}_{model}_K{K}_seed{seed}_split{split_idx}_pernode.npz
    Keys inside: p_val_{k}, p_test_{k}  (and k_list)
    """
    from src.datasets import load_dataset
    graph_data, _, _ = load_dataset(dataset)
    labels = graph_data.y.numpy()

    if split_indices is None:
        split_indices = list(range(10))   # default: all 10 splits
    if split_sides is None:
        split_sides = ['val', 'test']     # default: pool both sides

    # val_mask / test_mask are (N, num_splits) for multi-split datasets
    val_mask_all  = graph_data.val_mask.numpy()   # (N, num_splits)
    test_mask_all = graph_data.test_mask.numpy()  # (N, num_splits)
    num_classes   = int(labels.max()) + 1

    # ------------------------------------------------------------------ #
    # Load data: average across seeds per (split_idx, side), then pool    #
    # ------------------------------------------------------------------ #
    per_key   = {}   # k -> {(split_idx, vt) -> list of seed probs}
    label_key = {}   # (split_idx, vt) -> label array
    k_list    = None

    for seed in seeds:
        for split_idx in split_indices:
            arrays_path = (Path(config['results_dir']) / 'arrays' /
                           f'{dataset}_{model}_K{K}_seed{seed}_split{split_idx}{_lt(loss_type)}_pernode.npz')
            if not arrays_path.exists():
                print(f"  Warning: {arrays_path.name} not found, skipping")
                continue
            data = np.load(arrays_path)
            if k_list is None:
                k_list = data['k_list']

            val_idx  = np.where(val_mask_all[:, split_idx])[0]
            test_idx = np.where(test_mask_all[:, split_idx])[0]
            side_map = {'val': val_idx, 'test': test_idx}

            for k in k_list:
                if k not in per_key:
                    per_key[k] = {}
                for vt in split_sides:
                    p_key  = f'p_{vt}_{k}'
                    idx    = side_map[vt]
                    if p_key in data:
                        key_id = (split_idx, vt)
                        per_key[k].setdefault(key_id, []).append(data[p_key])
                        if key_id not in label_key:
                            label_key[key_id] = labels[idx]

    if not per_key or k_list is None:
        print("Error: No valid data found")
        return

    pooled_probs  = {}
    pooled_labels = {}
    for k in k_list:
        probs_list, labels_list = [], []
        for key_id, seed_probs in per_key.get(k, {}).items():
            avg = np.mean(seed_probs, axis=0)    # (n_subset, C) averaged over seeds
            probs_list.append(avg)
            labels_list.append(label_key[key_id])
        if probs_list:
            pooled_probs[k]  = np.concatenate(probs_list, axis=0)
            pooled_labels[k] = np.concatenate(labels_list, axis=0)

    if not pooled_probs:
        print("Error: pooled probs empty")
        return

    # ------------------------------------------------------------------ #
    # Pre-compute per-class mean entropy and p_correct across all splits  #
    # ------------------------------------------------------------------ #
    class_mean_entropy   = {}
    class_mean_p_correct = {}
    for k in k_list:
        if k not in pooled_probs:
            continue
        H         = entropy_from_probs(pooled_probs[k])
        lab       = pooled_labels[k]
        p_correct = pooled_probs[k][np.arange(len(lab)), lab]
        class_H, class_P = np.zeros(num_classes), np.zeros(num_classes)
        for c in range(num_classes):
            cm = lab == c
            if cm.sum() > 0:
                class_H[c] = H[cm].mean()
                class_P[c] = p_correct[cm].mean()
        class_mean_entropy[k]   = class_H
        class_mean_p_correct[k] = class_P

    num_depths    = len(k_list)
    ncols         = min(3, num_depths)
    traj_ncols    = max(3, ncols)  # always at least 3 columns
    traj_nrows    = int(np.ceil(num_classes / traj_ncols))
    nrows_scatter = int(np.ceil(num_depths / traj_ncols))  # use traj_ncols to match gridspec
    nrows         = nrows_scatter + traj_nrows
    height_ratios = [1] * nrows_scatter + [1.1] * traj_nrows

    colors     = plt.cm.tab20(np.linspace(0, 1, num_classes))  # tab20 for 18 classes
    depth_cmap = plt.cm.coolwarm
    K_max_val  = max(k_list)
    depth_norm = plt.Normalize(vmin=0, vmax=K_max_val)

    fig = plt.figure(figsize=(5 * traj_ncols, 4.5 * nrows))
    gs  = fig.add_gridspec(nrows, traj_ncols, hspace=0.35, wspace=0.2,
                           height_ratios=height_ratios)

    # ── Scatter panels (one per depth) ───────────────────────────────── #
    for idx, k in enumerate(k_list):
        ax = fig.add_subplot(gs[idx // traj_ncols, idx % traj_ncols])
        if k not in pooled_probs:
            continue
        H         = entropy_from_probs(pooled_probs[k])
        lab       = pooled_labels[k]
        p_correct = pooled_probs[k][np.arange(len(lab)), lab]
        for c in range(num_classes):
            cm = lab == c
            if cm.sum() > 0:
                lbl = f'Class {c} (n={cm.sum()})' if k == 0 else f'Class {c}'
                ax.scatter(H[cm], p_correct[cm], c=[colors[c]], label=lbl,
                           alpha=0.4, s=10, edgecolors='none')
        ax.set_xlabel('Mean Entropy', fontsize=10)
        ax.set_ylabel('Mean P(Correct)', fontsize=10)
        ax.set_title(f'Depth k={k}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(0, 1)
        if idx == 0:
            ax.legend(loc='best', fontsize=6, framealpha=0.9, ncol=2)

    # Hide unused scatter slots (when num_depths < traj_ncols)
    for slot in range(num_depths, nrows_scatter * traj_ncols):
        fig.add_subplot(gs[slot // traj_ncols, slot % traj_ncols]).axis('off')

    # ── Trajectory panels (one per class) ────────────────────────────── #
    class_sizes = sorted(range(num_classes), key=lambda c: int((pooled_labels[k_list[0]] == c).sum())
                         if k_list[0] in pooled_labels else 0)
    ax_last = None

    for panel_idx, c in enumerate(class_sizes):
        tr   = nrows_scatter + panel_idx // traj_ncols
        tc   = panel_idx % traj_ncols
        ax_c = fig.add_subplot(gs[tr, tc])

        entropy_vals   = [class_mean_entropy[k][c]   for k in k_list if k in class_mean_entropy]
        p_correct_vals = [class_mean_p_correct[k][c] for k in k_list if k in class_mean_p_correct]
        k_vals         = [k for k in k_list if k in class_mean_entropy]

        if not entropy_vals:
            ax_c.axis('off'); continue

        for k in k_list:
            if k not in pooled_probs:
                continue
            H         = entropy_from_probs(pooled_probs[k])
            lab       = pooled_labels[k]
            p_correct = pooled_probs[k][np.arange(len(lab)), lab]
            cm        = lab == c
            if cm.sum() > 0:
                ax_c.scatter(H[cm], p_correct[cm],
                             color=depth_cmap(depth_norm(k)),
                             alpha=0.25, s=8, edgecolors='none', zorder=1)

        ax_c.plot(entropy_vals, p_correct_vals, '-',
                  color=colors[c], alpha=0.7, linewidth=2, zorder=2)
        for i in range(len(k_vals) - 1):
            ax_c.annotate('', xy=(entropy_vals[i+1], p_correct_vals[i+1]),
                          xytext=(entropy_vals[i], p_correct_vals[i]),
                          arrowprops=dict(arrowstyle='->', color=colors[c],
                                         lw=1.5, mutation_scale=12), zorder=3)
        ax_c.scatter(entropy_vals[0],  p_correct_vals[0],
                     marker='o', s=55, facecolors='none',
                     edgecolors=colors[c], linewidth=1.8, zorder=6, label='k=0 (start)')
        ax_c.scatter(entropy_vals[-1], p_correct_vals[-1],
                     marker='o', s=55, color=colors[c],
                     edgecolors='black', linewidth=0.8, zorder=6,
                     label=f'k={K_max_val} (end)')

        n_class = int((pooled_labels[k_list[0]] == c).sum()) if k_list[0] in pooled_labels else '?'
        ax_c.set_title(f'Class {c}  (n≈{n_class})', fontweight='bold', fontsize=10)
        ax_c.set_xlabel('Mean Entropy', fontsize=8)
        ax_c.set_ylabel('Mean P(Correct)', fontsize=8)
        ax_c.grid(True, alpha=0.3)
        ax_c.set_xlim(left=0); ax_c.set_ylim(0, 1)
        ax_c.legend(fontsize=6, loc='upper right', framealpha=0.85)
        ax_last = ax_c

    # Hide unused trailing panels
    for extra in range(num_classes, traj_nrows * traj_ncols):
        tr = nrows_scatter + extra // traj_ncols
        tc = extra % traj_ncols
        if tr < nrows:
            fig.add_subplot(gs[tr, tc]).axis('off')

    seeds_str  = ', '.join(map(str, seeds))
    splits_str = ('all_splits' if len(split_indices) > 1
                  else f'split{split_indices[0]}')
    sides_str  = '+'.join(split_sides)
    fig.suptitle(
        f'{model}/{dataset}, K={K}, seeds=[{seeds_str}], {splits_str} ({sides_str})  [{_pretty_lt(loss_type)}]\n'
        f'Per-Node Mean Entropy vs Mean Correct-Class Probability',
        fontsize=14, fontweight='bold', y=0.985
    )
    fig.subplots_adjust(top=0.94, bottom=0.04, left=0.07, right=0.96,
                        hspace=0.3, wspace=0.2)

    if ax_last is not None:
        sm = plt.cm.ScalarMappable(cmap=depth_cmap, norm=depth_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_last, shrink=1.0, pad=0.04, aspect=15,
                            location='right')
        cbar.set_label('Depth k', fontsize=11)
        cbar.set_ticks(k_list)

    # ── Save combined figure ──────────────────────────────────────────── #
    figures_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = 'all' if set(seeds) == set(config.get('seeds', seeds)) else '_'.join(map(str, seeds))
    stem = f'{dataset}_{model}_k{K}_seed_{seed_tag}_{splits_str}_{sides_str}{_fig_lt(loss_type)}'

    combined_path = figures_dir / f'{stem}_entropy_vs_prob_with_trajectories.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {combined_path}")
    plt.close()

    # ── Layers-only figure ────────────────────────────────────────────── #
    fig_l = plt.figure(figsize=(5 * ncols, 4 * nrows_scatter))
    gs_l  = fig_l.add_gridspec(nrows_scatter, ncols, hspace=0.35, wspace=0.2)
    for idx, k in enumerate(k_list):
        ax = fig_l.add_subplot(gs_l[idx // ncols, idx % ncols])
        if k not in pooled_probs:
            continue
        H         = entropy_from_probs(pooled_probs[k])
        lab       = pooled_labels[k]
        p_correct = pooled_probs[k][np.arange(len(lab)), lab]
        for c in range(num_classes):
            cm = lab == c
            if cm.sum() > 0:
                lbl = f'Class {c} (n={cm.sum()})' if k == 0 else f'Class {c}'
                ax.scatter(H[cm], p_correct[cm], c=[colors[c]], label=lbl,
                           alpha=0.4, s=10, edgecolors='none')
        ax.set_xlabel('Mean Entropy', fontsize=10)
        ax.set_ylabel('Mean P(Correct)', fontsize=10)
        ax.set_title(f'Depth k={k}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(0, 1)
        if idx == 0:
            ax.legend(loc='best', fontsize=6, framealpha=0.9, ncol=2)
    fig_l.suptitle(
        f'{model}/{dataset}, K={K}, seeds=[{seeds_str}], {splits_str} ({sides_str})\n'
        f'Per-Node Mean Entropy vs Mean Correct-Class Probability, by Layer',
        fontsize=13, fontweight='bold'
    )
    top_l = min(0.90, 0.78 + 0.04 * max(nrows_scatter - 1, 0))
    fig_l.subplots_adjust(top=top_l, bottom=0.06, left=0.07, right=0.97,
                          hspace=0.35, wspace=0.2)
    layers_path = figures_dir / f'{stem}_layers.png'
    fig_l.savefig(layers_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {layers_path}")
    plt.close(fig_l)

    # ── Classes-only figure ───────────────────────────────────────────── #
    fig_c = plt.figure(figsize=(5 * traj_ncols, 4.5 * traj_nrows + 1.0))
    gs_c  = fig_c.add_gridspec(traj_nrows, traj_ncols, hspace=0.35, wspace=0.2)
    ax_last_c = None
    for panel_idx, c in enumerate(class_sizes):
        tr, tc = panel_idx // traj_ncols, panel_idx % traj_ncols
        ax_c = fig_c.add_subplot(gs_c[tr, tc])

        entropy_vals   = [class_mean_entropy[k][c]   for k in k_list if k in class_mean_entropy]
        p_correct_vals = [class_mean_p_correct[k][c] for k in k_list if k in class_mean_p_correct]
        k_vals         = [k for k in k_list if k in class_mean_entropy]

        if not entropy_vals:
            ax_c.axis('off'); continue

        for k in k_list:
            if k not in pooled_probs:
                continue
            H         = entropy_from_probs(pooled_probs[k])
            lab       = pooled_labels[k]
            p_correct = pooled_probs[k][np.arange(len(lab)), lab]
            cm        = lab == c
            if cm.sum() > 0:
                ax_c.scatter(H[cm], p_correct[cm],
                             color=depth_cmap(depth_norm(k)),
                             alpha=0.25, s=8, edgecolors='none', zorder=1)

        ax_c.plot(entropy_vals, p_correct_vals, '-',
                  color=colors[c], alpha=0.7, linewidth=2, zorder=2)
        for i in range(len(k_vals) - 1):
            ax_c.annotate('', xy=(entropy_vals[i+1], p_correct_vals[i+1]),
                          xytext=(entropy_vals[i], p_correct_vals[i]),
                          arrowprops=dict(arrowstyle='->', color=colors[c],
                                         lw=1.5, mutation_scale=12), zorder=3)
        ax_c.scatter(entropy_vals[0],  p_correct_vals[0],
                     marker='o', s=55, facecolors='none',
                     edgecolors=colors[c], linewidth=1.8, zorder=6, label='k=0 (start)')
        ax_c.scatter(entropy_vals[-1], p_correct_vals[-1],
                     marker='o', s=55, color=colors[c],
                     edgecolors='black', linewidth=0.8, zorder=6,
                     label=f'k={K_max_val} (end)')

        n_class = int((pooled_labels[k_list[0]] == c).sum()) if k_list[0] in pooled_labels else '?'
        ax_c.set_title(f'Class {c}  (n≈{n_class})', fontweight='bold', fontsize=10)
        ax_c.set_xlabel('Mean Entropy', fontsize=8)
        ax_c.set_ylabel('Mean P(Correct)', fontsize=8)
        ax_c.grid(True, alpha=0.3)
        ax_c.set_xlim(left=0); ax_c.set_ylim(0, 1)
        ax_c.legend(fontsize=6, loc='upper right', framealpha=0.85)
        ax_last_c = ax_c

    for extra in range(num_classes, traj_nrows * traj_ncols):
        tr, tc = extra // traj_ncols, extra % traj_ncols
        if tr < traj_nrows:
            fig_c.add_subplot(gs_c[tr, tc]).axis('off')

    if ax_last_c is not None:
        sm = plt.cm.ScalarMappable(cmap=depth_cmap, norm=depth_norm)
        sm.set_array([])
        cbar_c = fig_c.colorbar(sm, ax=ax_last_c, shrink=1.0, pad=0.04,
                                aspect=15, location='right')
        cbar_c.set_label('Depth k', fontsize=11)
        cbar_c.set_ticks(k_list)

    _fig_h = 4.5 * traj_nrows + 1.0
    top_c   = min(0.94, 0.97 - 0.90 / _fig_h)
    y_title = min(0.98, top_c + 0.06 + 0.30 / _fig_h)
    fig_c.subplots_adjust(top=top_c, bottom=0.06, left=0.07, right=0.96,
                          hspace=0.3, wspace=0.2)
    fig_c.suptitle(
        f'{model}/{dataset}, K={K}, seeds=[{seeds_str}], {splits_str} ({sides_str})\n'
        f'Per-Node Mean Entropy vs Mean Correct-Class Probability, by Class',
        fontsize=13, fontweight='bold', x=0.5, y=y_title
    )
    classes_path = figures_dir / f'{stem}_classes.png'
    fig_c.savefig(classes_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {classes_path}")
    plt.close(fig_c)



def main():
    parser = argparse.ArgumentParser(description='Plot entropy vs correct-class probability or correctness')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=str, default='0', help='Seed, "all", or comma-separated list like "0,1,3"')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'all'],
                        help='"val", "test", or "all" (pools val+test). Use with --split_idx for single-split datasets.')
    parser.add_argument('--split_idx', type=int, default=None,
                        help='If set, only load data from this split index (e.g. 0 for split0). '
                             'Works with multi-split datasets like Roman-Empire. '
                             'Combine with --split val/test/all to control which side(s) are included.')
    parser.add_argument('--plot_type', type=str, default='probability',
                       choices=['probability', 'correctness'],
                       help='Plot type: probability (per-class) or correctness (binary)')
    parser.add_argument('--loss-type', type=str, default='ce_only',
                       help='Loss type directory name (e.g. ce_only, ce_plus_R_band-1.0to0.0). '
                            'Used to locate pernode arrays and suffix output filenames.')

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
    
    loss_type = args.loss_type

    if args.plot_type == 'correctness':
        # Correctness plot (binary: correct/incorrect)
        if seed_mode == 'single':
            print(f"Creating entropy vs correctness plot for {args.dataset}/{args.model}")
            print(f"K={args.K}, seed={seed}, split={args.split}, loss_type={loss_type}")
            plot_entropy_vs_correctness(args.dataset, args.model, args.K, seed, config, args.split,
                                        split_idx=args.split_idx, loss_type=loss_type)
        else:
            # Aggregated across seeds
            print(f"Creating aggregated entropy vs correctness plot for {args.dataset}/{args.model}")
            print(f"K={args.K}, seeds={seeds}, split={args.split}, loss_type={loss_type}")
            plot_entropy_vs_correctness_aggregated(args.dataset, args.model, args.K, seeds, config, args.split, seed_mode,
                                                   split_idx=args.split_idx, loss_type=loss_type)

    else:
        # Probability plot (per-class)
        use_allsplits = (args.split == 'all') or (args.split_idx is not None)
        if use_allsplits:
            # Multi-split datasets (e.g. Roman-Empire): pool across split files
            split_indices = [args.split_idx] if args.split_idx is not None else None  # None = all splits
            split_sides   = ['val', 'test'] if args.split == 'all' else [args.split]
            # allsplits always needs a list of seeds
            seeds_for_allsplits = seeds if seeds is not None else [seed]
            print(f"Creating all-splits entropy vs probability plot for {args.dataset}/{args.model}")
            print(f"K={args.K}, seeds={seeds_for_allsplits}, split_indices={split_indices}, split_sides={split_sides}, loss_type={loss_type}")
            plot_entropy_vs_prob_allsplits(
                args.dataset, args.model, args.K, seeds_for_allsplits, config,
                split_indices=split_indices, split_sides=split_sides,
                loss_type=loss_type
            )
        elif seed_mode == 'single':
            print(f"Creating entropy vs probability plot for {args.dataset}/{args.model}")
            print(f"K={args.K}, seed={seed}, split={args.split}, loss_type={loss_type}")
            plot_entropy_vs_prob(args.dataset, args.model, args.K, seed, config, args.split,
                                 loss_type=loss_type)
        else:
            # Aggregated plot across seeds
            print(f"Creating aggregated entropy vs probability plot for {args.dataset}/{args.model}")
            print(f"K={args.K}, seeds={seeds}, split={args.split}, loss_type={loss_type}")
            plot_entropy_vs_prob_aggregated(args.dataset, args.model, args.K, seeds, config, args.split, seed_mode,
                                            loss_type=loss_type)

    print("\nDone!")


if __name__ == '__main__':
    main()

