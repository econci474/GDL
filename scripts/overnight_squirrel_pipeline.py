"""
Overnight pipeline: GCN on Squirrel, ce_only (linear probe), K=0..7, seeds 1,2,3, split 0.

Steps per K/seed:
  1. Train     -> results/runs/Squirrel/GCN/seed_S/K_K/split_0/  (best.pt + embeddings.pt)
  2. Probe     -> results/tables/Squirrel_GCN_K{K}_seed{S}_split0_probe.csv
Then per K (after all seeds):
  3. Separability per seed (--seed 1, 2, 3, --split 0)
  4. Separability seed_all (--seed all, --split 0)
"""

import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATASET  = 'Squirrel'
MODEL    = 'GCN'
SEEDS    = [1, 2, 3]
K_VALUES = list(range(0, 8))   # 0..7
SPLIT_ID = 0

py = sys.executable

def run(cmd, desc):
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"  [WARNING] exit {r.returncode} — continuing...")
    return r.returncode


# ── Steps 1 + 2: Train then Probe ────────────────────────────────────────────
for K in K_VALUES:
    for seed in SEEDS:
        emb_path = (ROOT / 'results' / 'runs' / DATASET / MODEL
                    / f'seed_{seed}' / f'K_{K}' / f'split_{SPLIT_ID}' / 'embeddings.pt')

        # --- Train (skip if embeddings already exist) ---
        if emb_path.exists():
            print(f"\n[SKIP train] K={K} seed={seed}  (embeddings.pt exists)")
        else:
            run([py, 'src/train_gnn.py',
                 '--dataset', DATASET, '--model', MODEL,
                 '--K', str(K), '--seed', str(seed),
                 '--split-mode', 'first'],
                f"TRAIN  K={K}  seed={seed}")

        # --- Probe (skip if probe CSV already exists) ---
        probe_csv = (ROOT / 'results' / 'tables'
                     / f'{DATASET}_{MODEL}_K{K}_seed{seed}_split{SPLIT_ID}_probe.csv')
        if probe_csv.exists():
            print(f"[SKIP probe] K={K} seed={seed}  (probe CSV exists)")
        else:
            run([py, '-m', 'src.probe',
                 '--dataset', DATASET, '--model', MODEL,
                 '--K', str(K), '--seed', str(seed),
                 '--split-id', str(SPLIT_ID)],
                f"PROBE  K={K}  seed={seed}")


# ── Step 3+4: Separability metrics & plots ──────────────────────────────────
for K in K_VALUES:
    # Per-seed separability
    for seed in SEEDS:
        run([py, 'src/separability_metrics.py',
             '--dataset', DATASET, '--model', MODEL,
             '--K', str(K),
             '--seed', str(seed),
             '--split', str(SPLIT_ID)],
            f"SEPARABILITY  K={K}  seed={seed}  split={SPLIT_ID}")

    # Aggregated across seeds (seed_all)
    run([py, 'src/separability_metrics.py',
         '--dataset', DATASET, '--model', MODEL,
         '--K', str(K),
         '--seed', 'all',
         '--split', str(SPLIT_ID)],
        f"SEPARABILITY  K={K}  seed=all  split={SPLIT_ID}")


# ── Step 5: Entropy dynamics plots ───────────────────────────────────────────
SEEDS_CSV = ','.join(str(s) for s in SEEDS)   # "1,2,3"

for K in K_VALUES:
    # Per-seed: val_per_class (probability, per-class colour)
    for seed in SEEDS:
        run([py, 'src/plot_node_entropy_vs_prob.py',
             '--dataset', DATASET, '--model', MODEL,
             '--K', str(K), '--seed', str(seed),
             '--split', 'val', '--split_idx', str(SPLIT_ID),
             '--plot_type', 'probability'],
            f"ENTROPY PROB (per-class)  K={K}  seed={seed}")

        # val_per_layer (correctness binary plot)
        run([py, 'src/plot_node_entropy_vs_prob.py',
             '--dataset', DATASET, '--model', MODEL,
             '--K', str(K), '--seed', str(seed),
             '--split', 'val', '--split_idx', str(SPLIT_ID),
             '--plot_type', 'correctness'],
            f"ENTROPY CORRECTNESS (per-layer)  K={K}  seed={seed}")

    # Combined / aggregated across seeds — probability
    run([py, 'src/plot_node_entropy_vs_prob.py',
         '--dataset', DATASET, '--model', MODEL,
         '--K', str(K), '--seed', SEEDS_CSV,
         '--split', 'val', '--split_idx', str(SPLIT_ID),
         '--plot_type', 'probability'],
        f"ENTROPY PROB aggregated  K={K}  seeds={SEEDS_CSV}")

    # Combined / aggregated across seeds — correctness
    run([py, 'src/plot_node_entropy_vs_prob.py',
         '--dataset', DATASET, '--model', MODEL,
         '--K', str(K), '--seed', SEEDS_CSV,
         '--split', 'val', '--split_idx', str(SPLIT_ID),
         '--plot_type', 'correctness'],
        f"ENTROPY CORRECTNESS aggregated  K={K}  seeds={SEEDS_CSV}")


print("\n" + "="*60)
print("OVERNIGHT PIPELINE COMPLETE")
print("="*60)
