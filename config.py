"""Configuration for GNN entropy experiments."""

# Model Configuration
K_max = 8
hidden_dim = 64
dropout = None  # No dropout

# Training 
lr = 0.01  # Updated for Roman-empire retraining
weight_decay = 0.0005
max_epochs = 700  # Updated for Roman-empire retraining
patience = 10

#

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

#Classifier head training with regularisation 
exponential_decay = True # set to False for linear decay
beta = 0.5 # decay parameter for exponential weighting
lambda_R = 1.0          # Weight of curvature regularizer
R_mode = 'hard'         # 'hard' or 'smooth'
entropy_floor = None    # or 0.2, 0.3, etc.
per_class_R = False     # True to weight R by class

# Band penalty configuration
band_lower = -1.5       # Lower bound for curvature band penalty
band_upper = 0.5        # Upper bound for curvature band penalty
