"""Batch regenerate val_layers (and _classes, _trajectories) figures for GCN.

Runs plot_node_entropy_vs_prob.py for all (dataset, K, seed) combinations
that have pernode.npz data.
"""
import subprocess, sys, re
from pathlib import Path

DATASETS_HOMO   = ["Cora", "PubMed"]
DATASETS_HETERO = ["Roman-empire", "Squirrel"]
MODEL = "GCN"
SEEDS = ["all", "0", "1", "2"]

# Discover available (dataset, K) from pernode.npz files
arrays_dir = Path("results/arrays")
available = set()
for f in arrays_dir.glob(f"*_{MODEL}_K*_pernode.npz"):
    m = re.match(r"(.+)_{MODEL}_K(\d+)_seed\d+.*_pernode".replace("{MODEL}", MODEL), f.stem)
    if m:
        available.add((m.group(1), int(m.group(2))))

# Build run list
combos = []
for ds in DATASETS_HOMO + DATASETS_HETERO:
    ks = sorted({K for (d, K) in available if d == ds})
    for K in ks:
        for seed in SEEDS:
            combos.append((ds, K, seed))

total = len(combos)
print(f"Running {total} combinations ...\n")

for i, (ds, K, seed) in enumerate(combos, 1):
    cmd = [
        sys.executable, "src/plot_node_entropy_vs_prob.py",
        "--dataset", ds,
        "--model",   MODEL,
        "--K",       str(K),
        "--seed",    seed,
        "--split",   "val",
    ]
    # Heterophilous datasets: pool split 0 val only
    if ds in DATASETS_HETERO:
        cmd += ["--split_idx", "0"]

    tag = f"{ds}/K={K}/seed={seed}"
    print(f"[{i}/{total}] {tag}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    for line in result.stdout.splitlines():
        if "Saved" in line:
            print(f"  -> {Path(line.split('Saved:')[-1].strip()).name}")
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[-300:]}")

print("\nAll done.")
