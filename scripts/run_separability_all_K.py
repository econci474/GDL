"""
Run separability_metrics.py for all K values (0..8) for specified datasets.
Generates individual seed plots + aggregated mean±std plot for each K.
"""
import subprocess
import sys
from pathlib import Path

DATASETS = ["Cora", "PubMed"]
MODEL    = "GCN"
K_VALUES = list(range(0, 9))   # K=0 through K=8
SEED_ARG = "all"               # runs seeds 0,1,2 individually + aggregated

root = Path(__file__).parent.parent

for dataset in DATASETS:
    for K in K_VALUES:
        print(f"\n{'='*60}")
        print(f"  {dataset} / {MODEL} / K={K}")
        print(f"{'='*60}")
        cmd = [
            sys.executable,
            str(root / "src" / "separability_metrics.py"),
            "--dataset", dataset,
            "--model",   MODEL,
            "--K",       str(K),
            "--seed",    SEED_ARG,
        ]
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            print(f"  [WARNING] Failed for {dataset}/K={K} (exit {result.returncode}), skipping.")

print("\nAll done.")
