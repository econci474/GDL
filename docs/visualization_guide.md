# Visualization System for Probe vs Classifier Analysis

## Overview
The `plot_probes_vs_classifier_heads.py` script generates comprehensive visualizations comparing linear probes (trained on frozen embeddings) vs classifier heads (trained jointly with exponential/class-weighted losses).

## Plot Types Generated

### 1. **Scatter Plot: Entropy vs Correct-Class Probability**
- **Layout:** 3 rows × (K+1) columns
  - Row 1: Linear Probe
  - Row 2: Exponential Classifier (β=0.5)
  - Row 3: Class-Weighted Classifier
  - Each column: One layer depth (k=0 to K)
- **Each point:** One validation node
- **Colors:** Different classes with node counts in legend
- **Shows:** How prediction uncertainty relates to correctness at each depth

### 2. **Per-Class Entropy Heatmap**
- **Layout:** 3 rows (one per training method)
- **Heatmap:** Rows = classes (with node counts), Columns = layers
- **Values:** Mean predictive entropy per class per layer
- **Colors:** Lower entropy = higher confidence (more saturated colors)
- **Useful for:** Identifying which classes are harder to classify at different depths

### 3. **Comprehensive Comparison**
- **Top row:** Accuracy, Mean Entropy, Mean Max Probability across layers
- **Bottom row:** Confidence distributions at k=0, k=1, k=2
- **All 3 methods compared** with different markers and colors
- **Shows:** Overall performance trends across depth

## Usage

### Generate all plots for all configurations:
```bash
python -m src.plot_probes_vs_classifier_heads --dataset all
```

### Generate for specific configuration:
```bash
python -m src.plot_probes_vs_classifier_heads --dataset Cora --K 2 --seed 0
```

### Generate for specific dataset, all K and seeds:
```bash
python -m src.plot_probes_vs_classifier_heads --dataset PubMed --K -1 --seed -1
```

## Arguments
- `--dataset`: Dataset name or "all" (Cora, PubMed, Roman-empire, Minesweeper)
- `--model`: Model architecture (default: GCN)
- `--K`: Number of layers or -1 for all K ∈ [0,8]
- `--seed`: Random seed or -1 for all seeds ∈ [0,1,2,3]
- `--beta`: Beta parameter for exponential classifier (default: 0.5)

## Output
All plots saved to: `results/figures/probe_vs_classifier/`

Filenames:
- `{dataset}_{model}_k{K}_seed{seed}_scatter.pdf`
- `{dataset}_{model}_k{K}_seed{seed}_entropy_heatmap.pdf`
- `{dataset}_{model}_k{K}_seed{seed}_comprehensive.pdf`

## Prerequisites
Before running, ensure you have:
1. ✅ Trained GNN models with embeddings extracted
2. ✅ Probe results from `python -m src.probe`
3. ✅ Classifier head results from training pipeline
4. ✅ Extracted classifier outputs from `python -m src.extract_classifier_outputs`

## Expected Data Locations
```
results/
├── runs/{dataset}/{model}/seed_{seed}/K_{K}/
│   └── probe_results.pt
└── classifier_heads/
    ├── exponential/{dataset}/{model}/seed_{seed}/K_{K}/
    │   └── layer_probs.npz
    └── class-weighted/{dataset}/{model}/seed_{seed}/K_{K}/
        └── layer_probs.npz
```

## Progress Tracking
The script shows:
- Configuration being processed
- Each plot type generated
- Progress: X/Total | Elapsed time | Estimated remaining time
