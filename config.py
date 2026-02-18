"""Configuration for GNN entropy experiments."""

# ── Experiments ────────────────────────────────────────────────────
seeds = [0, 1, 2, 3]
models = ['GCN', 'GAT', 'GraphSAGE']

# Dataset classification
homophilous_datasets   = ['Cora', 'PubMed']
heterophilous_datasets = ['Roman-empire', 'Minesweeper', 'Squirrel']
datasets = homophilous_datasets + heterophilous_datasets

# ── Model configuration ────────────────────────────────────────────
K_max    = 8
dropout_input  = 0.5   # Applied to raw input features (k=0) before first conv
dropout_middle = None  # Applied between intermediate conv layers (None = disabled)

# GAT specific
gat_heads = 8

# Probing
probe_C_values = [0.01, 0.1, 1, 10, 100]

# ── Defaults (used when not sweeping) ──────────────────────────────
# Homophilous datasets (Cora, PubMed)
defaults_homophilous = {
    'hidden_dim':   64,
    'lr':           0.01,
    'weight_decay': 5e-4,
    'max_epochs':   500,
    'patience':     50,
}

# Heterophilous datasets (Roman-empire, Minesweeper, Squirrel)
defaults_heterophilous = {
    'hidden_dim':   256,
    'lr':           0.05,
    'weight_decay': 5e-4,
    'max_epochs':   1000,
    'patience':     100,
}

# ── Hyperparameter sweep grids ─────────────────────────────────────
# Standard GNN (train_gnn.py) — homophilous
sweep_homophilous = {
    'hidden_dim':   [16, 64],
    'lr':           [0.01, 0.005],
    'weight_decay': [0.0, 1e-4, 5e-4],
    'max_epochs':   [500],
    'patience':     [50],
}

# Standard GNN (train_gnn.py) — heterophilous
sweep_heterophilous = {
    'hidden_dim':   [256, 512],
    'lr':           [0.05, 0.1],
    'weight_decay': [0.0, 5e-4, 5e-3, 1e-2],
    'max_epochs':   [1000],
    'patience':     [100],
}

# Classifier head additional sweep params (applied on top of the above)
# loss_type options:
#   'ce_only'            — unweighted CE, no regulariser
#   'weighted_ce'        — class-weighted CE, no regulariser
#   'ce_plus_R'          — unweighted CE + curvature regulariser
#   'weighted_ce_plus_R' — class-weighted CE + curvature regulariser
#   'R_only'             — curvature regulariser only (ablation)
sweep_entropy = {
    'loss_type':     ['ce_only', 'weighted_ce', 'ce_plus_R', 'weighted_ce_plus_R', 'R_only'],
    'beta':          [0.1, 0.5, 1.0],
    'lambda_R':      [0.01, 0.1, 1.0, 5.0, 10.0],  # ignored for ce_only / weighted_ce
    'entropy_floor': [None, 0.05, 0.1, 0.2], # ignored for ce_only / weighted_ce
    'per_class_R':   [False, True], # ignored for ce_only/ weighted_ce
    # band swept as coupled (lower, upper) pairs
    'band':          [(-1.0, 0.0), (-1.5, 0.25), (-2.0, 0.5)],
}

# ── Classifier head defaults ───────────────────────────────────────
exponential_decay = True   # set to False for linear decay
beta              = 0.5    # depth decay parameter
lambda_R          = 1.0    # weight of curvature regulariser
R_mode            = 'smooth'
entropy_floor     = None
per_class_R       = False
band_lower        = -1.5
band_upper        = 0.5

# ── Paths ──────────────────────────────────────────────────────────
results_dir          = 'results'
runs_dir             = 'results/runs'
tables_dir           = 'results/tables'
figures_dir          = 'results/figures'
classifier_heads_dir = 'results/classifier_heads'
