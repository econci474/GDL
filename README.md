# Entropy-Guided Depth Selection in GNNs

Research codebase for investigating predictive entropy dynamics across message-passing depth in Graph Neural Networks.

## Setup

### Installation

```bash
# Create and activate the gdl environment (already set up)
conda activate gdl

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
entropy-selection/
├── config.yaml              # Hyperparameters
├── requirements.txt         # Dependencies
├── results/
│   ├── tables/             # CSV outputs
│   ├── figures/            # Plots
│   └── runs/               # Checkpoints & embeddings
└── src/
    ├── datasets.py         # Dataset loaders
    ├── models.py           # GCN & GAT models
    ├── metrics.py          # Entropy & NLL calculations
    ├── train_gnn.py        # Training script
    ├── extract_embeddings.py  # Layer-wise embedding extraction
    ├── probe.py            # Linear probing
    ├── depth_selection.py  # Depth selection via val NLL
    ├── controls.py         # Negative controls
    ├── plots.py            # Visualization
    └── utils.py            # Utilities (set_seed, etc.)
```

## Usage

### 1. Train a model

```bash
python -m src.train_gnn --dataset Cora --model GCN --K 8 --seed 0
```

### 2. Extract layer-wise embeddings

```bash
python -m src.extract_embeddings --dataset Cora --model GCN --K 8 --seed 0
```

### 3. Run linear probing at each depth

```bash
python -m src.probe --dataset Cora --model GCN --K 8 --seed 0
```

### 4. Select optimal depth

```bash
python -m src.depth_selection --dataset Cora --model GCN
```

### 5. Generate plots

```bash
python -m src.plots --dataset Cora --model GCN
```

## Datasets

- **Homophilous**: Cora, PubMed
- **Heterophilous**: Roman-empire, Minesweeper

All datasets use standard benchmark splits from PyTorch Geometric.

## Models

- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network

Both models support layer-wise embedding extraction via `forward_with_embeddings()`.

## Git Repository

```bash
# Initialize git (if not already done)
git init
git remote add origin git@github.com:econci474/GDL.git

# First commit
git add .
git commit -m "Initial commit: project structure"
git push -u origin main
```

## Development Status

**Current**: Minimal prototype with Cora + GCN
**Next**: Full pipeline validation, then scale to all datasets and models
