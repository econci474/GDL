"""
Interactive Plotly visualization comparing Linear Probes vs Classifier Heads.
Toggle between training methods, classes, correctness, and depth (K).
"""

import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def entropy_from_probs(probs, eps=1e-10):
    """Compute entropy from probability distributions."""
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def load_probe_data(dataset, model, K, seed, split='val'):
    """Load linear probe results."""
    probe_path = Path(cfg.runs_dir) / dataset / model / f'seed_{seed}' / f'K_{K}' / 'probe_results.pt'
    
    if not probe_path.exists():
        return None
    
    probe_results = torch.load(probe_path)
    
    # Extract probabilities for each layer
    probs_list = []
    for k in range(K + 1):
        if k in probe_results:
            layer_probes = probe_results[k]
            best_C = list(layer_probes.keys())[0]
            probe_data = layer_probes[best_C]
            if f'{split}_probs' in probe_data:
                probs_list.append(probe_data[f'{split}_probs'])
            else:
                probs_list.append(None)
        else:
            probs_list.append(None)
    
    return probs_list


def load_classifier_data(dataset, model, K, seed, loss_type, split='val'):
    """Load classifier head outputs."""
    base_dir = Path(cfg.classifier_heads_dir) / loss_type / dataset / model / f'seed_{seed}' / f'K_{K}'
    probs_path = base_dir / 'layer_probs.npz'
    
    if not probs_path.exists():
        return None
    
    probs_data = np.load(probs_path)
    
    probs_list = []
    for k in range(K + 1):
        key = f'{split}_probs_{k}'
        if key in probs_data:
            probs_list.append(probs_data[key])
        else:
            probs_list.append(None)
    
    return probs_list


def create_interactive_comparison(dataset, model, K, seed, config, split='val', beta=0.5):
    """
    Create interactive Plotly visualization with toggles for:
    - Training method (Linear Probe, Exponential, Class-Weighted)
    - Individual classes
    - Correct/Incorrect
    - Max K to display
    """
    # Load all three training methods
    probe_probs_list = load_probe_data(dataset, model, K, seed, split)
    exp_probs_list = load_classifier_data(dataset, model, K, seed, 'exponential', split)
    weighted_probs_list = load_classifier_data(dataset, model, K, seed, 'class-weighted', split)
    
    if probe_probs_list is None and exp_probs_list is None and weighted_probs_list is None:
        print(f"? No data found for {dataset}/{model}/K={K}/seed={seed}")
        return
    
    # Load dataset for labels
    from src.datasets import load_dataset
    graph_data, _, _ = load_dataset(dataset)
    labels = graph_data.y.numpy()
    num_classes = len(np.unique(labels))
    
    if split == 'val':
        mask = graph_data.val_mask.numpy()
    else:
        mask = graph_data.test_mask.numpy()
    
    plot_indices = np.where(mask)[0]
    plot_labels = labels[plot_indices]
    
    # Prepare data for all methods
    methods_data = {}
    
    if probe_probs_list:
        methods_data['Linear Probe'] = probe_probs_list
    if exp_probs_list:
        methods_data[f'Exponential (ß={beta})'] = exp_probs_list
    if weighted_probs_list:
        methods_data['Class-Weighted'] = weighted_probs_list
    
    # Color mapping
    colors_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                  '#c49c94', '#f7b6d2', '#c7c7c7']
    
    # Create subplot structure
    ncols = min(3, K + 1)
    nrows = int(np.ceil((K + 1) / ncols))
    
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[f'Depth k={k}' for k in range(K + 1)],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Add traces for each method, class, and layer
    trace_idx = 0
    trace_metadata = []  # Store metadata for each trace
    
    for method_name, probs_list in methods_data.items():
        for k in range(K + 1):
            row = k // ncols + 1
            col = k % ncols + 1
            
            probs = probs_list[k]
            if probs is None:
                continue
            
            # Calculate metrics
            H = entropy_from_probs(probs)
            p_correct = probs[np.arange(len(probs)), plot_labels.astype(int)]
            pred_labels = np.argmax(probs, axis=1)
            is_correct = (pred_labels == plot_labels)
            
            # Add traces for each class
            for c in range(num_classes):
                class_mask = plot_labels == c
                
                if class_mask.sum() == 0:
                    continue
                
                # Correct nodes
                class_correct_mask = class_mask & is_correct
                if class_correct_mask.sum() > 0:
                    visible = (method_name == list(methods_data.keys())[0])  # Only first method visible initially
                    
                    fig.add_trace(
                        go.Scatter(
                            x=H[class_correct_mask],
                            y=p_correct[class_correct_mask],
                            mode='markers',
                            name=f'{method_name} | C{c} (?)',
                            legendgroup=f'{method_name}_c{c}',
                            marker=dict(
                                color=colors_hex[c % len(colors_hex)],
                                size=6,
                                opacity=0.7,
                                line=dict(width=0)
                            ),
                            hovertemplate=f'<b>{method_name} | Class {c} (Correct)</b><br>' +
                                         'Entropy: %{x:.3f}<br>' +
                                         'P(Correct): %{y:.3f}<extra></extra>',
                            visible=visible
                        ),
                        row=row, col=col
                    )
                    
                    trace_metadata.append({
                        'method': method_name,
                        'class': c,
                        'correctness': 'correct',
                        'k': k
                    })
                    trace_idx += 1
                
                # Incorrect nodes
                class_incorrect_mask = class_mask & ~is_correct
                if class_incorrect_mask.sum() > 0:
                    visible = (method_name == list(methods_data.keys())[0])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=H[class_incorrect_mask],
                            y=p_correct[class_incorrect_mask],
                            mode='markers',
                            name=f'{method_name} | C{c} (?)',
                            legendgroup=f'{method_name}_c{c}',
                            marker=dict(
                                color=colors_hex[c % len(colors_hex)],
                                size=6,
                                opacity=0.3,
                                symbol='x',
                                line=dict(width=1, color='darkgray')
                            ),
                            hovertemplate=f'<b>{method_name} | Class {c} (Incorrect)</b><br>' +
                                         'Entropy: %{x:.3f}<br>' +
                                         'P(Correct): %{y:.3f}<extra></extra>',
                            visible=visible
                        ),
                        row=row, col=col
                    )
                    
                    trace_metadata.append({
                        'method': method_name,
                        'class': c,
                        'correctness': 'incorrect',
                        'k': k
                    })
                    trace_idx += 1
    
    # Create dropdown buttons for filtering
    buttons_method = []
    
    # Method selection buttons
    for method_name in methods_data.keys():
        visible = []
        for meta in trace_metadata:
            visible.append(meta['method'] == method_name)
        
        buttons_method.append(dict(
            label=method_name,
            method='update',
            args=[{'visible': visible}]
        ))
    
    # Class filter buttons
    buttons_class = [dict(
        label='All Classes',
        method='update',
        args=[{'visible': [True] * len(trace_metadata)}]
    )]
    
    for c in range(num_classes):
        visible = []
        for meta in trace_metadata:
            visible.append(meta['class'] == c)
        
        buttons_class.append(dict(
            label=f'Class {c} Only',
            method='update',
            args=[{'visible': visible}]
        ))
    
    # Correctness filter buttons
    buttons_correctness = [
        dict(
            label='Show All',
            method='update',
            args=[{'visible': [True] * len(trace_metadata)}]
        ),
        dict(
            label='Correct Only',
            method='update',
            args=[{'visible': [meta['correctness'] == 'correct' for meta in trace_metadata]}]
        ),
        dict(
            label='Incorrect Only',
            method='update',
            args=[{'visible': [meta['correctness'] == 'incorrect' for meta in trace_metadata]}]
        )
    ]
    
    # Update axes
    for k in range(K + 1):
        row = k // ncols + 1
        col = k % ncols + 1
        
        fig.update_xaxes(title_text='Entropy', row=row, col=col, range=[0, None])
        fig.update_yaxes(title_text='P(Correct)', row=row, col=col, range=[0, 1])
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
    
    # Layout with multiple dropdown menus
    fig.update_layout(
        title=dict(
            text=f'{dataset}/{model} (K={K}, seed={seed}, {split} set)<br>' +
                 'Interactive: Entropy vs Correct-Class Probability',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=400 * nrows,
        width=550 * ncols,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            font=dict(size=8)
        ),
        updatemenus=[
            # Method selector
            dict(
                buttons=buttons_method,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.18,
                yanchor="top",
                bgcolor='#e8f4f8',
                bordercolor='#2c3e50',
                font=dict(size=11, color='black'),
                active=0
            ),
            # Class selector
            dict(
                buttons=buttons_class,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.25,
                xanchor="left",
                y=1.18,
                yanchor="top",
                bgcolor='#f0e68c',
                bordercolor='#2c3e50',
                font=dict(size=11, color='black'),
                active=0
            ),
            # Correctness selector
            dict(
                buttons=buttons_correctness,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.45,
                xanchor="left",
                y=1.18,
                yanchor="top",
                bgcolor='#ffcccb',
                bordercolor='#2c3e50',
                font=dict(size=11, color='black'),
                active=0
            )
        ],
        hovermode='closest'
    )
    
    # Add labels for dropdowns
    fig.add_annotation(
        text="<b>Training Method:</b>",
        xref="paper", yref="paper",
        x=0.01, y=1.21,
        showarrow=False,
        font=dict(size=12, color='black')
    )
    fig.add_annotation(
        text="<b>Class Filter:</b>",
        xref="paper", yref="paper",
        x=0.25, y=1.21,
        showarrow=False,
        font=dict(size=12, color='black')
    )
    fig.add_annotation(
        text="<b>Correctness:</b>",
        xref="paper", yref="paper",
        x=0.45, y=1.21,
        showarrow=False,
        font=dict(size=12, color='black')
    )
    
    # Save as HTML
    figures_dir = Path(config['figures_dir']) / dataset / model / f'K_{K}'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = figures_dir / f'{dataset}_{model}_k{K}_seed{seed}_{split}_interactive_comparison.html'
    fig.write_html(output_path)
    
    print(f"? Interactive comparison saved to:")
    print(f"  {output_path}")
    print(f"\n?? Open in your browser to interact!")
    print(f"  - Toggle between: {', '.join(methods_data.keys())}")
    print(f"  - Filter by class (0-{num_classes-1})")
    print(f"  - Show correct/incorrect only")


def main():
    parser = argparse.ArgumentParser(description='Create interactive comparison: Probes vs Classifier Heads')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--beta', type=float, default=0.5, help='Beta for exponential classifier')
    
    args = parser.parse_args()
    
    # Convert config to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print(f"\n{'='*60}")
    print(f"Creating interactive comparison plot")
    print(f"  Dataset: {args.dataset}, Model: {args.model}")
    print(f"  K={args.K}, seed={args.seed}, split={args.split}")
    print(f"{'='*60}\n")
    
    create_interactive_comparison(args.dataset, args.model, args.K, args.seed, 
                                  config, args.split, args.beta)
    
    print("\n? Done!")


if __name__ == '__main__':
    main()

