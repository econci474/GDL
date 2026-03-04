"""
regen_all_figures.py
=====================
Regenerate ALL separability and entropy figures for GCN and GAT.

Covers:
  - Models:   GCN, GAT
  - Datasets: Cora, PubMed  (homophilous — no split_idx)
               Roman-empire, Squirrel (heterophilous — split_idx=0)
  - K values: 0..7  (convention used by separability/entropy scripts)
  - Seeds:    0, 1, 2 individually  +  'all' (sep) / '0,1,2' (entropy agg)
  - Plots:    separability_metrics.py
              plot_node_entropy_vs_prob.py   --plot_type probability
              plot_node_entropy_vs_prob.py   --plot_type correctness

All previous figures will be overwritten.

Usage (local or Colab):
    python scripts/regen_all_figures.py [--models GCN GAT] [--datasets Cora ...]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
py   = sys.executable

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--models",   nargs="+", default=["GCN", "GAT"])
parser.add_argument("--datasets", nargs="+",
                    default=["Cora", "PubMed", "Roman-empire", "Squirrel"])
parser.add_argument("--k-values", nargs="+", type=int, default=list(range(0, 8)),
                    help="K values to plot (default: 0..7)")
parser.add_argument("--seeds",    nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--plot-types", nargs="+",
                    default=["probability", "correctness"])
parser.add_argument("--sep-only",     action="store_true",
                    help="Only run separability, skip entropy plots")
parser.add_argument("--entropy-only", action="store_true",
                    help="Only run entropy plots, skip separability")
args = parser.parse_args()

HETERO = {"Roman-empire", "Squirrel"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd, desc):
    print(f"\n{'='*60}\n{desc}\n{'='*60}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    elapsed = (time.time() - t0) / 60
    status = "OK" if r.returncode == 0 else f"WARNING exit {r.returncode}"
    print(f"  [{status}] {elapsed:.1f} min", flush=True)

# ── Main loop ─────────────────────────────────────────────────────────────────
total_start = time.time()

for model in args.models:
    for ds in args.datasets:
        is_hetero = ds in HETERO
        split_idx = 0 if is_hetero else None

        for K in args.k_values:

            # ── Separability ─────────────────────────────────────────────────
            if not args.entropy_only:
                for seed in args.seeds:
                    cmd = [py, "src/separability_metrics.py",
                           "--dataset", ds, "--model", model,
                           "--K", str(K), "--seed", str(seed)]
                    if is_hetero:
                        cmd += ["--split", str(split_idx)]
                    run(cmd, f"SEPARABILITY  {ds} {model}  K={K}  seed={seed}")

                # Aggregated (all seeds)
                cmd = [py, "src/separability_metrics.py",
                       "--dataset", ds, "--model", model,
                       "--K", str(K), "--seed", "all"]
                if is_hetero:
                    cmd += ["--split", str(split_idx)]
                run(cmd, f"SEPARABILITY  {ds} {model}  K={K}  seed=all")

            # ── Entropy dynamics ──────────────────────────────────────────────
            if not args.sep_only:
                for plot_type in args.plot_types:
                    for seed in args.seeds:
                        cmd = [py, "src/plot_node_entropy_vs_prob.py",
                               "--dataset", ds, "--model", model,
                               "--K", str(K), "--seed", str(seed),
                               "--split", "val", "--plot_type", plot_type]
                        if is_hetero:
                            cmd += ["--split_idx", str(split_idx)]
                        run(cmd,
                            f"ENTROPY {plot_type}  {ds} {model}  K={K}  seed={seed}")

                    # Aggregated (all seeds)
                    cmd = [py, "src/plot_node_entropy_vs_prob.py",
                           "--dataset", ds, "--model", model,
                           "--K", str(K), "--seed", "0,1,2",
                           "--split", "val", "--plot_type", plot_type]
                    if is_hetero:
                        cmd += ["--split_idx", str(split_idx)]
                    run(cmd,
                        f"ENTROPY {plot_type} agg  {ds} {model}  K={K}  seeds=0,1,2")

total_min = (time.time() - total_start) / 60
print(f"\n{'='*60}")
print(f"ALL FIGURES DONE in {total_min:.1f} min")
print(f"{'='*60}", flush=True)
