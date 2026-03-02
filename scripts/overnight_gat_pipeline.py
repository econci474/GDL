"""
Overnight pipeline: GAT, all 4 datasets, K=0..7, seeds 0/1/2.
- Cora, PubMed     (homophilous): no split dimension, split_mode=auto
- Roman-empire, Squirrel (heterophilous): split_mode=first (split 0 only)

Steps per (dataset, K, seed):
  1. Train     -> results/runs/{dataset}/GAT/seed_{s}/K_{k}/[split_0/]  best.pt + embeddings.pt
  2. Probe     -> results/tables/{dataset}_GAT_K{k}_seed{s}[_split0]_probe.csv
Then per (dataset, K):
  3. Separability  per seed + seed_all
  4. Entropy dynamics  per seed + aggregated (seeds 0,1,2)
"""

import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
MODEL   = 'GAT'
SEEDS   = [0, 1, 2]
K_VALUES = list(range(0, 8))   # 0..7

HOMO_DATASETS   = ['Cora', 'PubMed']
HETERO_DATASETS = ['Roman-empire', 'Squirrel']
ALL_DATASETS    = HOMO_DATASETS + HETERO_DATASETS
SPLIT_ID        = 0

py = sys.executable

def run(cmd, desc):
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"  [WARNING] exit {r.returncode} — continuing...")
    return r.returncode


# ── Steps 1 + 2: Train then Probe ────────────────────────────────────────────
for dataset in ALL_DATASETS:
    is_hetero   = dataset in HETERO_DATASETS
    split_mode  = 'first' if is_hetero else 'auto'
    split_sfx   = f'_split{SPLIT_ID}' if is_hetero else ''

    for K in K_VALUES:
        for seed in SEEDS:
            # Expected embeddings path
            base = ROOT / 'results' / 'runs' / dataset / MODEL / f'seed_{seed}' / f'K_{K}'
            emb_path = (base / f'split_{SPLIT_ID}' / 'embeddings.pt') if is_hetero else (base / 'embeddings.pt')

            # --- Train ---
            if emb_path.exists():
                print(f"\n[SKIP train] {dataset} K={K} seed={seed}")
            else:
                run([py, 'src/train_gnn.py',
                     '--dataset', dataset, '--model', MODEL,
                     '--K', str(K), '--seed', str(seed),
                     '--split-mode', split_mode],
                    f"TRAIN  {dataset}  K={K}  seed={seed}")

            # --- Probe ---
            probe_csv = ROOT / 'results' / 'tables' / f'{dataset}_{MODEL}_K{K}_seed{seed}{split_sfx}_probe.csv'
            if probe_csv.exists():
                print(f"[SKIP probe] {dataset} K={K} seed={seed}")
            else:
                probe_cmd = [py, '-m', 'src.probe',
                             '--dataset', dataset, '--model', MODEL,
                             '--K', str(K), '--seed', str(seed)]
                if is_hetero:
                    probe_cmd += ['--split-id', str(SPLIT_ID)]
                run(probe_cmd, f"PROBE  {dataset}  K={K}  seed={seed}")


# ── Steps 3 + 4: Separability + Entropy dynamics ─────────────────────────────
SEEDS_CSV = ','.join(str(s) for s in SEEDS)   # "0,1,2"

for dataset in ALL_DATASETS:
    is_hetero  = dataset in HETERO_DATASETS
    split_arg  = str(SPLIT_ID) if is_hetero else None   # --split for separability_metrics

    for K in K_VALUES:
        # ── Separability ─────────────────────────────────────────────
        for seed in SEEDS:
            cmd = [py, 'src/separability_metrics.py',
                   '--dataset', dataset, '--model', MODEL,
                   '--K', str(K), '--seed', str(seed)]
            if split_arg:
                cmd += ['--split', split_arg]
            run(cmd, f"SEPARABILITY  {dataset}  K={K}  seed={seed}")

        # seed_all
        cmd = [py, 'src/separability_metrics.py',
               '--dataset', dataset, '--model', MODEL,
               '--K', str(K), '--seed', 'all']
        if split_arg:
            cmd += ['--split', split_arg]
        run(cmd, f"SEPARABILITY  {dataset}  K={K}  seed=all")

        # ── Entropy dynamics ─────────────────────────────────────────
        for plot_type in ['probability', 'correctness']:
            # Per seed
            for seed in SEEDS:
                cmd = [py, 'src/plot_node_entropy_vs_prob.py',
                       '--dataset', dataset, '--model', MODEL,
                       '--K', str(K), '--seed', str(seed),
                       '--split', 'val',
                       '--plot_type', plot_type]
                if is_hetero:
                    cmd += ['--split_idx', str(SPLIT_ID)]
                run(cmd, f"ENTROPY {plot_type}  {dataset}  K={K}  seed={seed}")

            # Aggregated
            cmd = [py, 'src/plot_node_entropy_vs_prob.py',
                   '--dataset', dataset, '--model', MODEL,
                   '--K', str(K), '--seed', SEEDS_CSV,
                   '--split', 'val',
                   '--plot_type', plot_type]
            if is_hetero:
                cmd += ['--split_idx', str(SPLIT_ID)]
            run(cmd, f"ENTROPY {plot_type} aggregated  {dataset}  K={K}  seeds={SEEDS_CSV}")


print("\n" + "="*60)
print("GAT OVERNIGHT PIPELINE COMPLETE")
print("="*60)
