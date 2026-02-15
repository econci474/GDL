"""
Unified interactive visualization: Everything in one shareable HTML!
Toggle: Dataset, Model, K, Seed/Averaging, Training Method, Class, Correctness
"""

import argparse
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import torch
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def entropy_from_probs(probs, eps=1e-10):
    """Compute entropy from probability distributions."""
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def load_and_aggregate_data(datasets, models, K_values, seeds, split='val', beta=0.5):
    """
    Load all data for specified configurations and seeds.
    Returns organized data structure with averaging support.
    
    NOW FIXED: Loads from results/arrays/{dataset}_{model}_K{K}_seed{seed}_pernode.npz
    """
    print(f"\n{'='*70}")
    print(f"LOADING DATA FOR INTERACTIVE VISUALIZATION")
    print(f"{'='*70}")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"K values: {K_values}")
    print(f"Seeds: {seeds}")
    print(f"Split: {split}")
    print(f"{'='*70}\n")
    
    # Structure: data[dataset][model][K][method][seed_key][k] = probs
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))))
    
    # Track metadata
    metadata = defaultdict(lambda: {'labels': None, 'num_classes': 0})
    
    for dataset in datasets:
        print(f"?? Loading {dataset}...")
        
        # Load dataset info (only once per dataset)
        try:
            from src.datasets import load_dataset
            graph_data, _, _ = load_dataset(dataset)
            labels = graph_data.y.numpy()
            
            if split == 'val':
                mask = graph_data.val_mask.numpy()
            else:
                mask = graph_data.test_mask.numpy()
            
            plot_labels = labels[mask]
            metadata[dataset]['labels'] = plot_labels
            metadata[dataset]['num_classes'] = len(np.unique(labels))
            print(f"  ? Loaded dataset: {len(plot_labels)} {split} nodes, {metadata[dataset]['num_classes']} classes")
            
        except Exception as e:
            print(f"  ? Failed to load dataset: {e}")
            continue
        
        for model in models:
            for K in K_values:
                for seed in seeds:
                    # Load per-node arrays from results/arrays/
                    arrays_path = Path(cfg.results_dir) / 'arrays' / f'{dataset}_{model}_K{K}_seed{seed}_pernode.npz'
                    
                    if not arrays_path.exists():
                        print(f"  ??  {dataset}/{model}/K={K}/seed={seed}: File not find (skipping)")
                        continue
                    
                    try:
                        data = np.load(arrays_path)
                        loaded_ks = []
                        
                        for k in range(K + 1):
                            # Get probabilities for this depth
                            probs_key = f'{split}_probs_{k}' if f'{split}_probs_{k}' in data else f'p_{split}_{k}'
                            
                            if probs_key in data:
                                probs = data[probs_key]
                                # Store as probe method (only data we have)
                                all_data[dataset][model][K]['probe'][f'seed_{seed}'][k] = probs
                                loaded_ks.append(k)
                        
                        if loaded_ks:
                            print(f"  ? {dataset}/{model}/K={K}/seed={seed}: Loaded probe k={loaded_ks}")
                    
                    except Exception as e:
                        print(f"  ? {dataset}/{model}/K={K}/seed={seed}: Failed - {type(e).__name__}: {e}")
    
    # Now compute averaged versions
    print(f"\n?? Computing seed averages...")
    for dataset in all_data:
        for model in all_data[dataset]:
            for K in all_data[dataset][model]:
                for method in all_data[dataset][model][K]:
                    # Collect all seeds for averaging
                    seed_keys = [k for k in all_data[dataset][model][K][method].keys() if k.startswith('seed_')]
                    
                    if len(seed_keys) > 1:
                        # Average across all available seeds
                        for k in range(K + 1):
                            probs_list = []
                            for seed_key in seed_keys:
                                if k in all_data[dataset][model][K][method][seed_key]:
                                    probs_list.append(all_data[dataset][model][K][method][seed_key][k])
                            
                            if len(probs_list) > 0:
                                avg_probs = np.mean(probs_list, axis=0)
                                all_data[dataset][model][K][method]['averaged'][k] = avg_probs
                        print(f"  ? {dataset}/{model}/K={K}/{method}: Averaged {len(seed_keys)} seeds")
    
    print(f"? Data loading complete\n")
    return all_data, metadata


def create_unified_interactive(datasets=None, models=None, K_values=None, 
                                seeds=None, split='val', beta=0.5):
    """
    Create single shareable HTML with ALL configurations.
    Dropdowns: Dataset, Model, K, Seed, Training Method, Class, Correctness
    """
    # Defaults
    if datasets is None:
        datasets = ['Cora', 'PubMed', 'Roman-empire', 'Minesweeper']
    if models is None:
        models = ['GCN']
    if K_values is None:
        K_values = list(range(9))
    if seeds is None:
        seeds = [0, 1, 2, 3]
    
    # Load all data
    all_data, metadata = load_and_aggregate_data(datasets, models, K_values, seeds, split, beta)
    
    # Create figure
    fig = go.Figure()
    
    # Color palette
    colors_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                  '#c49c94', '#f7b6d2', '#c7c7c7']
    
    # Method display names
    method_names = {
        'probe': 'Linear Probe',
        'exponential': f'Exponential (ß={beta})',
        'weighted': 'Class-Weighted'
    }
    
    # Build all traces
    trace_idx = 0
    trace_info = []  # Store metadata for visibility control
    
    print(f"\n?? Building traces...")
    print(f"  Datasets to process: {datasets}")
    
    for dataset in datasets:
        if dataset not in metadata or metadata[dataset]['labels'] is None:
            continue
        
        plot_labels = metadata[dataset]['labels']
        num_classes = metadata[dataset]['num_classes']
        
        for model in models:
            for K in K_values:
                if K not in all_data[dataset][model]:
                    continue
                
                for method_key in ['probe', 'exponential', 'weighted']:
                    if method_key not in all_data[dataset][model][K]:
                        continue
                    
                    method_name = method_names[method_key]
                    
                    # Process each seed option (individual seeds + averaged)
                    seed_options = sorted([k for k in all_data[dataset][model][K][method_key].keys()])
                    
                    for seed_key in seed_options:
                        for k in range(K + 1):
                            if k not in all_data[dataset][model][K][method_key][seed_key]:
                                continue
                            
                            probs = all_data[dataset][model][K][method_key][seed_key][k]
                            
                            # Calculate metrics
                            H = entropy_from_probs(probs)
                            p_correct = probs[np.arange(len(probs)), plot_labels.astype(int)]
                            pred_labels = np.argmax(probs, axis=1)
                            is_correct = (pred_labels == plot_labels)
                            
                            # Create traces for each class and correctness
                            for c in range(num_classes):
                                class_mask = plot_labels == c
                                
                                if class_mask.sum() == 0:
                                    continue
                                
                                # Correct nodes
                                correct_mask = class_mask & is_correct
                                if correct_mask.sum() > 0:
                                    visible = (dataset == datasets[0] and model == models[0] and 
                                             K == K_values[0] and seed_key == seed_options[0] and
                                             k == 0 and method_key == 'probe')
                                    
                                    fig.add_trace(go.Scatter(
                                        x=H[correct_mask],
                                        y=p_correct[correct_mask],
                                        mode='markers',
                                        name=f'C{c} (?)',
                                        marker=dict(
                                            color=colors_hex[c % len(colors_hex)],
                                            size=7,
                                            opacity=0.7
                                        ),
                                        hovertemplate=f'<b>{dataset}/{model}/K={K}/k={k}/{seed_key}</b><br>' +
                                                     f'{method_name} | Class {c} (Correct)<br>' +
                                                     'H: %{x:.3f} | P(correct): %{y:.3f}<extra></extra>',
                                        visible=visible,
                                        showlegend=(dataset == datasets[0] and model == models[0] and 
                                                  K == K_values[0] and k == 0 and seed_key == seed_options[0] and
                                                  method_key == 'probe'),  # Only show legend for first config
                                        legendgroup=f'class_{c}',  # Group by class for toggling
                                        legendgrouptitle_text=None
                                    ))
                                    
                                    trace_info.append({
                                        'dataset': dataset, 'model': model, 'K': K, 'k': k,
                                        'seed_key': seed_key, 'method': method_key,
                                        'class': c, 'correctness': 'correct'
                                    })
                                    
                                    if len(trace_info) == 1:
                                        print(f"  ? First trace created: {dataset}/{model}/K={K}/k={k}/{seed_key}/{method_key}/class={c}")
                                
                                # Incorrect nodes
                                incorrect_mask = class_mask & ~is_correct
                                if incorrect_mask.sum() > 0:
                                    visible = (dataset == datasets[0] and model == models[0] and 
                                             K == K_values[0] and seed_key == seed_options[0] and
                                             k == 0 and method_key == 'probe')
                                    
                                    fig.add_trace(go.Scatter(
                                        x=H[incorrect_mask],
                                        y=p_correct[incorrect_mask],
                                        mode='markers',
                                        name=f'C{c} (?)',
                                        marker=dict(
                                            color=colors_hex[c % len(colors_hex)],
                                            size=7,
                                            opacity=0.25,
                                            symbol='x'
                                        ),
                                        hovertemplate=f'<b>{dataset}/{model}/K={K}/k={k}/{seed_key}</b><br>' +
                                                     f'{method_name} | Class {c} (Incorrect)<br>' +
                                                     'H: %{x:.3f} | P(correct): %{y:.3f}<extra></extra>',
                                        visible=visible,
                                        showlegend=(dataset == datasets[0] and model == models[0] and 
                                                  K == K_values[0] and k == 0 and seed_key == seed_options[0] and
                                                  method_key == 'probe'),  # Only show legend for first config
                                        legendgroup=f'class_{c}',  # Group by class for toggling
                                        legendgrouptitle_text=None
                                    ))
                                    
                                    trace_info.append({
                                        'dataset': dataset, 'model': model, 'K': K, 'k': k,
                                        'seed_key': seed_key, 'method': method_key,
                                        'class': c, 'correctness': 'incorrect'
                                    })
    
    # Create control buttons using update pattern
    # We'll create buttons that update visibility based on selected filters
    
    # Build unique options for each dimension
    unique_datasets = sorted(set(t['dataset'] for t in trace_info))
    unique_models = sorted(set(t['model'] for t in trace_info))
    unique_Ks = sorted(set(t['K'] for t in trace_info))
    unique_ks = sorted(set(t['k'] for t in trace_info))
    unique_seeds = sorted(set(t['seed_key'] for t in trace_info))
    unique_methods = ['probe', 'exponential', 'weighted']
    
    # Create interactive dropdown menus
    print(f"\n???  Creating interactive controls...")
    print(f"  Datasets: {unique_datasets}")
    print(f"  Models: {unique_models}")
    print(f"  K values: {unique_Ks}")
    print(f"  Depth k: {unique_ks}")
    print(f"  Seeds: {unique_seeds}")
    print(f"  Methods: {unique_methods}")
    
    # Helper function to create visibility array for a given selection
    def make_visibility(selected_dataset, selected_model, selected_K, selected_k, selected_method, selected_seed):
        """Return visibility array for all traces based on selections."""
        return [
            (t['dataset'] == selected_dataset and 
             t['model'] == selected_model and 
             t['K'] == selected_K and 
             t['k'] == selected_k and 
             t['method'] == selected_method and 
             t['seed_key'] == selected_seed)
            for t in trace_info
        ]
    
    # Build dropdown menus for each control dimension
    # We'll use a hierarchical approach: Dataset -> Model -> K -> k -> Method -> Seed
    
    updatemenus = []
    
    # 1. Dataset dropdown
    dataset_buttons = []
    for ds in unique_datasets:
        # When dataset changes, keep other selections or default to first available
        dataset_buttons.append(dict(
            label=ds,
            method='update',
            args=[{'visible': make_visibility(ds, unique_models[0], unique_Ks[0], 0, 'probe', unique_seeds[0])}]
        ))
    
    updatemenus.append(dict(
        buttons=dataset_buttons,
        direction='down',
        showactive=True,
        x=0.01,
        xanchor='left',
        y=1.15,
        yanchor='top',
        bgcolor='#ffffff',
        bordercolor='#333',
        font=dict(size=11)
    ))
    
    # 2. Model dropdown  
    model_buttons = []
    for mdl in unique_models:
        model_buttons.append(dict(
            label=mdl,
            method='update',
            args=[{'visible': make_visibility(unique_datasets[0], mdl, unique_Ks[0], 0, 'probe', unique_seeds[0])}]
        ))
    
    updatemenus.append(dict(
        buttons=model_buttons,
        direction='down',
        showactive=True,
        x=0.15,
        xanchor='left',
        y=1.15,
        yanchor='top',
        bgcolor='#ffffff',
        bordercolor='#333',
        font=dict(size=11)
    ))
    
    # 3. K_max dropdown
    K_buttons = []
    for K_val in unique_Ks:
        K_buttons.append(dict(
            label=f'K={K_val}',
            method='update',
            args=[{'visible': make_visibility(unique_datasets[0], unique_models[0], K_val, 0, 'probe', unique_seeds[0])}]
        ))
    
    updatemenus.append(dict(
        buttons=K_buttons,
        direction='down',
        showactive=True,
        x=0.28,
        xanchor='left',
        y=1.15,
        yanchor='top',
        bgcolor='#ffffff',
        bordercolor='#333',
        font=dict(size=11)
    ))
    
    # 4. Create SLIDER for depth k (for smooth animation through layers)
    # This is better than dropdown because it maintains context from other selections
    sliders = []
    slider_steps = []
    
    for k_val in unique_ks:
        step = dict(
            method='update',
            args=[{'visible': make_visibility(unique_datasets[0], unique_models[0], unique_Ks[0], k_val, 'probe', unique_seeds[0])}],
            label=f'k={k_val}'
        )
        slider_steps.append(step)
    
    sliders.append(dict(
        active=0,
        yanchor='top',
        y=1.05,
        xanchor='left',
        x=0.42,
        currentvalue=dict(
            prefix='Layer: ',
            visible=True,
            xanchor='left'
        ),
        pad=dict(b=10, t=50),
        len=0.3,
        steps=slider_steps
    ))
    
    # 5. Method dropdown (moved right to make room for slider)
    method_buttons = []
    for method in unique_methods:
        method_buttons.append(dict(
            label=method.capitalize(),
            method='update',
            args=[{'visible': make_visibility(unique_datasets[0], unique_models[0], unique_Ks[0], 0, method, unique_seeds[0])}]
        ))
    
    updatemenus.append(dict(
        buttons=method_buttons,
        direction='down',
        showactive=True,
        x=0.75,  # Moved right
        xanchor='left',
        y=1.15,
        yanchor='top',
        bgcolor='#ffffff',
        bordercolor='#333',
        font=dict(size=11)
    ))
    
    # 6. Seed dropdown (moved right)
    seed_buttons = []
    for sd in unique_seeds:
        label = 'Average' if sd == 'averaged' else sd.replace('seed_', 'Seed ')
        seed_buttons.append(dict(
            label=label,
            method='update',
            args=[{'visible': make_visibility(unique_datasets[0], unique_models[0], unique_Ks[0], 0, 'probe', sd)}]
        ))
    
    updatemenus.append(dict(
        buttons=seed_buttons,
        direction='down',
        showactive=True,
        x=0.88,  # Moved right
        xanchor='left',
        y=1.15,
        yanchor='top',
        bgcolor='#ffffff',
        bordercolor='#333',
        font=dict(size=11)
    ))
    
    print(f"  ? Created {len(updatemenus)} dropdown menus + 1 slider")
    
    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Unified Interactive GNN Visualization</b><br>' +
                 f'<sub>Shareable HTML | {len(trace_info)} traces loaded | ' +
                 f'{len(datasets)} datasets × {len(models)} models × {len(K_values)} K values × {len(seeds)+1} seed options</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(title='<b>Predictive Entropy</b>', range=[0, 3], showgrid=True),
        yaxis=dict(title='<b>P(Correct Class)</b>', range=[0, 1], showgrid=True),
        height=750,
        width=1200,
        showlegend=True,
        legend=dict(
            title='<b>Classes</b>',
            yanchor="top", y=0.99,
            xanchor="left", x=1.01,
            font=dict(size=10)
        ),
        hovermode='closest',
        template='plotly_white',
        font=dict(family='Arial', size=12),
        updatemenus=updatemenus,  # Add the dropdown controls
        sliders=sliders  # Add the layer slider
    )
    
    # Add instructions
    fig.add_annotation(
        text="<b>?? Tip:</b> Select options from dropdowns below to filter the visualization. " +
             "HTML is fully self-contained and shareable!",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=11, color='#555'),
        align='center',
        bgcolor='#f0f0f0',
        bordercolor='#ccc',
        borderwidth=1,
        borderpad=8
    )
    
    # Save
    output_dir = Path(cfg.figures_dir) / 'interactive'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'unified_interactive_{split}.html'
    
    fig.write_html(
        output_path,
        config={'displayModeBar': True, 'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']}
    )
    
    print(f"\n{'='*70}")
    print(f"? UNIFIED INTERACTIVE VISUALIZATION CREATED")
    print(f"{'='*70}")
    print(f"?? Output: {output_path}")
    print(f"?? Total traces: {len(trace_info):,}")
    print(f"?? File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\n? SHAREABLE: Yes! Send this HTML file to anyone.")
    print(f"   - No internet required")
    print(f"   - Works in any browser (Chrome, Firefox, Edge, Safari)")
    print(f"   - All data embedded in the file")
    print(f"\n?? Next: Double-click the HTML file to open in browser!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Create unified interactive visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All defaults (4 datasets, K=0-8, seeds 0-3)
  python -m src.plot_unified_interactive
  
  # Specific datasets
  python -m src.plot_unified_interactive --datasets Cora PubMed
  
  # Specific K range
  python -m src.plot_unified_interactive --K-values 0 1 2 3
  
  # Single seed only
  python -m src.plot_unified_interactive --seeds 0
        """
    )
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Datasets to include (default: Cora PubMed Roman-empire Minesweeper)')
    parser.add_argument('--models', type=str, nargs='+', default=['GCN'],
                       help='Models to include (default: GCN)')
    parser.add_argument('--K-values', type=int, nargs='+', default=None,
                       help='K values to include (default: 0-8)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Seeds to include (default: 0 1 2 3, plus averaged)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--beta', type=float, default=0.5)
    
    args = parser.parse_args()
    
    datasets = args.datasets if args.datasets else ['Cora', 'PubMed', 'Roman-empire', 'Minesweeper']
    K_values = args.K_values if args.K_values else list(range(9))
    seeds = args.seeds if args.seeds else [0, 1, 2, 3]
    
    create_unified_interactive(
        datasets=datasets,
        models=args.models,
        K_values=K_values,
        seeds=seeds,
        split=args.split,
        beta=args.beta
    )


if __name__ == '__main__':
    main()

