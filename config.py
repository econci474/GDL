"""Configuration for GNN entropy experiments."""

# Model Configuration
K_max = 8
hidden_dim = 64
dropout = None  # No dropout

# Training
lr = 0.001 # need to tune with grid search [0.01, 0.001]
weight_decay = 0.0005
max_epochs = 500
patience = 100

# Experiments
seeds = [0, 1, 2, 3]
datasets = ['Cora', 'PubMed', 'Roman-empire', 'Minesweeper']
models = ['GCN', 'GAT', 'GraphSAGE']

# GAT specific
gat_heads = 8

# Probing
probe_C_values = [0.01, 0.1, 1, 10, 100]

# Depth selection
lambda_values = [0, 0.01, 0.1, 1]

# Paths
results_dir = 'results'
runs_dir = 'results/runs'
tables_dir = 'results/tables'
figures_dir = 'results/figures'
classifier_heads_dir = 'results/classifier_heads'

# Classifier head training
beta_default = 0.5  # Depth decay parameter for exponential weighting
