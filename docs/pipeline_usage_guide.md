# Pipeline Usage Guide

Two end-to-end analysis pipelines exist in `src/`. They share the same GNN backbone architecture but differ in **how depth is probed**: a frozen linear classifier (linear probe) vs jointly trained classifier heads with entropy regularisation (classifier head sweep).

---

## Pipeline 1 — Linear Probe Analysis

```
train_gnn  →  [training_diagnostics]  →  extract_embeddings  →  probe
          →  [aggregate_probe_splits]  →  separability_metrics
                                       →  plot_node_entropy_vs_prob
```

Used to study embedding separability at each depth using a frozen linear probe trained post-hoc on frozen GNN embeddings.

---

### Step 1 — Train the base GNN

[`src/train_gnn.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/train_gnn.py)

Trains a CE-only GNN (no entropy regularisation). Saves `best.pt` and `train_log.csv`.

```bash
# Homophilous dataset (Cora, PubMed)
python -m src.train_gnn --dataset Cora --model GCN --K 4 --seed 0

# Heterophilous dataset — single split
python -m src.train_gnn --dataset Roman-empire --model GCN --K 4 --seed 0 \
    --split-mode first --split-id 0

# Heterophilous dataset — all 10 splits
python -m src.train_gnn --dataset Roman-empire --model GCN --K 4 --seed 0 \
    --split-mode all
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | Dataset name (Cora, PubMed, Roman-empire, Squirrel, Minesweeper) |
| `--model` | required | GCN, GAT, or GraphSAGE |
| `--K` | 8 | Number of GNN layers |
| `--seed` | 0 | Random seed |
| `--split-mode` | auto | `auto` (homophilous), `first` (split 0 only), `all` (all 10 splits) |
| `--split-id` | 0 | Which split to use when `--split-mode first` |
| `--lr`, `--weight-decay`, `--patience`, `--max-epochs`, `--hidden-dim` | config defaults | Hyperparameters (override config.py) |

**Output:** `results/runs/{dataset}/{model}/seed_{seed}/K_{K}/[split_{id}/]`
- `best.pt` — best checkpoint
- `train_log.csv` — per-epoch loss/accuracy

---

### Step 1a (optional) — Training diagnostic plots

[`src/training_diagnostics.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/training_diagnostics.py)

Plots training/validation loss and accuracy curves by epoch to verify convergence before proceeding.

```bash
python -m src.training_diagnostics --dataset Cora --model GCN --K 4

# Heterophilous datasets (per-split view)
python -m src.training_diagnostics --dataset Roman-empire --model GCN --K 4
```

---

### Step 2 — Extract embeddings

[`src/extract_embeddings.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/extract_embeddings.py)

Loads `best.pt`, runs a forward pass, saves layer-wise embeddings.

```bash
python -m src.extract_embeddings --dataset Cora --model GCN --K 4 --seed 0

# Heterophilous (single split)
python -m src.extract_embeddings --dataset Roman-empire --model GCN --K 4 --seed 0
```

**Output:** `results/runs/.../embeddings.pt`

---

### Step 3 — Run linear probe

[`src/probe.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/probe.py)

Fits a logistic regression at each layer on frozen embeddings. Saves per-node probabilities and summary metrics.

```bash
# Homophilous
python -m src.probe --dataset Cora --model GCN --K 4 --seed 0

# Heterophilous — one split at a time
python -m src.probe --dataset Roman-empire --model GCN --K 4 --seed 0 --split-id 0
python -m src.probe --dataset Roman-empire --model GCN --K 4 --seed 0 --split-id 1
# ... repeat for all 10 splits
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--split-id` | -1 (no split) | Which split's embeddings to probe |

**Output:**
- `results/tables/{dataset}_{model}_K{K}_seed{seed}[_split{id}]_probe.csv` — per-layer val/test NLL and accuracy
- `results/arrays/{dataset}_{model}_K{K}_seed{seed}[_split{id}]_pernode.npz` — per-node probabilities

---

### Step 4 (heterophilous only) — Aggregate probe splits

[`src/aggregate_probe_splits.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/aggregate_probe_splits.py)

Averages probe CSV metrics across the 10 splits into a single `*_probe.csv` without a split suffix, so the plotting pipeline can consume it.

```bash
python -m src.aggregate_probe_splits --dataset Roman-empire --model GCN --seed 0
python -m src.aggregate_probe_splits --dataset Roman-empire --model GCN --seed all
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--seed` | all | Specific seed or `all` |

---

### Step 5 — Separability metrics

[`src/separability_metrics.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/separability_metrics.py)

Computes AUROC, Cohen's d, and constrained depth selection (k*) from the probe probabilities.

```bash
# Single seed
python -m src.separability_metrics --dataset Cora --model GCN --K 4 --seed 0

# All seeds aggregated (seed_all)
python -m src.separability_metrics --dataset Cora --model GCN --K 4 --seed all

# Comma-separated seeds
python -m src.separability_metrics --dataset Cora --model GCN --K 4 --seed 0,1,2
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--seed` | 0 | Seed, `all`, or `0,1,2` |
| `--split` | None | `val` or `test` (default: val for selection, test for final) |
| `--eps_acc` | 0.02 | Accuracy tolerance for constrained depth selection |

**Output:** Separability figure saved to `results/figures/{dataset}/{model}/K_{K}/`

---

### Step 6 — Entropy plots by layer, class, and correctness

[`src/plot_node_entropy_vs_prob.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/plot_node_entropy_vs_prob.py)

Scatter plots of per-node entropy vs correct-class probability across depths, broken down by class and correctness.

```bash
# Probability scatter, single seed
python -m src.plot_node_entropy_vs_prob --dataset Cora --model GCN --K 4 \
    --seed 0 --plot_type probability --loss-type ce_only

# Correctness-coloured scatter, all seeds aggregated
python -m src.plot_node_entropy_vs_prob --dataset Cora --model GCN --K 4 \
    --seed all --plot_type correctness --loss-type ce_only

# Heterophilous — specify split index
python -m src.plot_node_entropy_vs_prob --dataset Roman-empire --model GCN --K 4 \
    --seed 0 --plot_type correctness --loss-type ce_only --split_idx 0
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--plot_type` | probability | `probability` (scatter by class) or `correctness` (binary correct/incorrect colouring) |
| `--split` | val | `val`, `test`, or `all` |
| `--split_idx` | None | Split index for heterophilous datasets |
| `--loss-type` | ce_only | Loss type string used to locate `layer_probs.npz` |

---

## Pipeline 2 — Classifier Head Analysis

```
run_sweep  →  evaluate_final  →  [generate_layer_probs]  →  separability_metrics_classifier_heads
                                                          →  plot_node_entropy_vs_prob
                                                          →  plot_curvature_violations
```

Trains GNNs end-to-end with classifier heads at each depth, using cross-entropy + entropy curvature regularisation (R loss). Hyperparameter sweep over bands, λ, and other settings.

---

### Step 1 — Hyperparameter sweep

[`scripts/run_sweep.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/scripts/run_sweep.py)

The main sweep runner. Calls `train_gnn_entropy.py` as a subprocess for each `(dataset, model, loss_type, K, seed, split)` combination. Results are recorded to `sweep_results.csv` via `results_tracker.py`.

Intended to be run from `notebooks/classifier_heads_hyperparameter_sweep_colab.ipynb` on Colab A100.

```bash
# Dry run — preview commands without executing
python scripts/run_sweep.py --datasets Cora PubMed --models GCN \
    --loss-types ce_only ce_plus_R --K-values 0 1 2 3 4 5 6 7 8 \
    --seeds 0 1 2 --dry-run

# Real run with resume support
python scripts/run_sweep.py --datasets Cora PubMed Roman-empire Squirrel \
    --models GCN --loss-types ce_only ce_plus_R --seeds 0 1 2 \
    --skip-existing

# Heterophilous — use all splits
python scripts/run_sweep.py --datasets Roman-empire Squirrel --models GCN \
    --loss-types ce_only ce_plus_R --seeds 0 1 2 --split-mode all
```

**Output per run:**
- `results/classifier_heads/{loss_type}/{dataset}/{model}/seed_{seed}/K_{K}/[split_{id}/]`
  - `best.pt` — best checkpoint
  - `train_log.csv` — per-epoch metrics
  - `layer_probs.npz` — val/test probabilities at each layer (auto-generated on save)

---

### Step 1b (optional) — Classifier head training diagnostics

[`src/plot_classifier_head_diagnostics.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/plot_classifier_head_diagnostics.py)

Plots training/validation loss and accuracy curves from classifier head runs to verify convergence before evaluating.

```bash
python -m src.plot_classifier_head_diagnostics \
    --dataset Cora --model GCN --K 4 \
    --loss-type ce_plus_R_R1.0_smooth_band-1.0to0.0

# Heterophilous — per split view
python -m src.plot_classifier_head_diagnostics \
    --dataset Roman-empire --model GCN --K 4 \
    --loss-type ce_only --split-id 0
```

---

### Step 1c — Train a single configuration manually

[`src/train_gnn_entropy.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/train_gnn_entropy.py)

Called by `run_sweep.py` but can also be run directly.

```bash
python -m src.train_gnn_entropy --dataset Cora --model GCN --K 4 --seed 0 \
    --loss-type ce_plus_R --band-lower -1.0 --band-upper 0.0 --lambda-r 1.0

# CE-only baseline (no regularisation)
python -m src.train_gnn_entropy --dataset Cora --model GCN --K 4 --seed 0 \
    --loss-type ce_only
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--loss-type` | weighted_ce_plus_R | `ce_only`, `ce_plus_R`, or `r_only` |
| `--band-lower` / `--band-upper` | None | δ²H band for R loss (e.g. `-1.0` to `0.0`) |
| `--lambda-r` | None | Weight for R regularisation term |
| `--beta` | None | Entropy smoothing coefficient |
| `--entropy-floor` | None | Minimum entropy floor |
| `--per-class-r` | False | Use per-class R loss |
| `--split-mode` | auto | `auto`, `first`, or `all` |

---

### Step 2 — Evaluate best hyperparameters

[`src/evaluate_final.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/evaluate_final.py)

> [!IMPORTANT]
> Run **once only** after the sweep is complete. Reads `best_hyperparams_*.csv` and evaluates the best configuration on the test set.

```bash
# Evaluate from best-hyperparams CSV (standard usage)
python -m src.evaluate_final --from-best-hyperparams \
    --best-hyperparams-path results/tables/best_hyperparams_GCN_per_layer.csv \
    --dataset Cora --model GCN --seeds all

# Heterophilous — average across all 10 splits
python -m src.evaluate_final --from-best-hyperparams \
    --best-hyperparams-path results/tables/best_hyperparams_GCN_per_layer.csv \
    --dataset Roman-empire --model GCN --split-mode all --seeds all
```

**Key arguments:**

| Argument | Description |
|---|---|
| `--from-best-hyperparams` | Read config from CSV rather than specifying manually |
| `--best-hyperparams-path` | Path to `best_hyperparams_*.csv` |
| `--split-mode` | `first` (default) or `all` (heterophilous) |
| `--seeds` | `all` or specific seeds |
| `--K-values` | `all` or specific K values |

---

### Step 3 — Generate layer probabilities (if missing)

[`scripts/generate_layer_probs.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/scripts/generate_layer_probs.py)

If `layer_probs.npz` was not saved during training, generate it from a `best.pt` checkpoint.

```bash
python scripts/generate_layer_probs.py --dataset Cora --model GCN \
    --loss-type ce_plus_R_R1.0_smooth_band-1.0to0.0 --K 8 --seed 0
```

---

### Step 4 — Separability metrics (classifier heads)

[`src/separability_metrics_classifier_heads.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/separability_metrics_classifier_heads.py)

Computes AUROC and Cohen's d from `layer_probs.npz`. Equivalent to `separability_metrics.py` but reads from the classifier head output directory.

```bash
# Single seed
python -m src.separability_metrics_classifier_heads \
    --dataset Cora --model GCN --K 8 --seed 0 \
    --loss-type ce_plus_R_R1.0_smooth_band-1.0to0.0

# Aggregated across seeds
python -m src.separability_metrics_classifier_heads \
    --dataset Cora --model GCN --K 8 --seed all \
    --loss-type ce_plus_R_R1.0_smooth_band-1.0to0.0

# Heterophilous — specify split
python -m src.separability_metrics_classifier_heads \
    --dataset Roman-empire --model GCN --K 8 --seed 0 \
    --loss-type ce_only --split-id 0
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--loss-type` | required | Loss type directory name |
| `--split` | val | `val` or `test` |
| `--split-id` | None | Split index for heterophilous datasets |
| `--classifier-heads-dir` | config default | Override path to classifier heads results |

---

### Step 5 — Entropy vs probability plots (classifier heads)

[`src/plot_node_entropy_vs_prob.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/plot_node_entropy_vs_prob.py)

Same script as in the probe pipeline, but pointed at classifier head outputs via `--loss-type`.

```bash
python -m src.plot_node_entropy_vs_prob --dataset Cora --model GCN --K 8 \
    --seed all --plot_type correctness \
    --loss-type ce_plus_R_R1.0_smooth_band-1.0to0.0
```

---

### Step 6 — Curvature violation plots

[`scripts/plot_curvature_violations.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/scripts/plot_curvature_violations.py)

Plots δ²H curvature vs depth for each dataset/model/K/seed, highlighting the R-loss band region.

```bash
python scripts/plot_curvature_violations.py --dataset Cora --model GCN --K 8 \
    --loss-type ce_plus_R_R1.0_smooth_band-1.0to0.0 --seed all
```

---

## Shared Utilities

| Script | Purpose |
|---|---|
| [`src/datasets.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/datasets.py) | Dataset loading (Cora, PubMed, Roman-empire, Squirrel, Minesweeper) |
| [`src/models.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/models.py) | GCN, GAT, GraphSAGE model definitions |
| [`src/metrics.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/metrics.py) | Entropy, NLL, accuracy, and correctness-split utilities |
| [`src/train_gnn_entropy.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/train_gnn_entropy.py) | Also contains δ²H curvature computation — used as the R-loss training signal |
| [`src/results_tracker.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/results_tracker.py) | Process-safe CSV appender for sweep results |
| [`src/aggregate_split_results.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/aggregate_split_results.py) | Aggregate test accuracy across splits for heterophilous datasets |
| [`src/controls.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/controls.py) | Random label control experiments |
| [`src/plot_classifier_head_diagnostics.py`](file:///c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/src/plot_classifier_head_diagnostics.py) | Training curve diagnostics for classifier head runs |

---

## Common Patterns

### Running for all K values (K=0..8)
Wrap any script in a shell loop or use `scripts/regen_all_figures.py` which handles all datasets, K values, seeds, and loss types in one pass.

```bash
python scripts/regen_all_figures.py
```

### Heterophilous datasets
Always specify `--split-id 0` (or loop over 0..9) and run `aggregate_probe_splits.py` / `aggregate_split_results.py` afterwards. Roman-empire and Squirrel require 10-split averaging for valid results.

### Loss type naming convention
Loss type strings follow the pattern:
```
ce_only
ce_plus_R_R{lambda_r}_{smooth|hard}_[floor{floor}_]band{lower}to{upper}
r_only_R{lambda_r}_{smooth|hard}_band{lower}to{upper}
```
These strings are both the `--loss-type` argument and the subdirectory name under `results/classifier_heads/`.
