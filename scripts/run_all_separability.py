"""Run separability analysis for all datasets and seeds."""

import subprocess
import time
from datetime import datetime

datasets = ['Cora', 'PubMed', 'Roman-empire', 'Minesweeper']
model = 'GCN'
K = 8
seeds = [0, 1, 2, 3]

print("="*60)
print("Running separability analysis for all datasets/seeds")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("="*60)

total = len(datasets) * len(seeds)
completed = 0

for dataset in datasets:
    for seed in seeds:
        print(f"\n[{completed+1}/{total}] Analyzing {dataset} seed={seed}...")
        
        cmd = f'python -m src.separability_metrics --dataset {dataset} --model {model} --K {K} --seed {seed}'
        
        t0 = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        elapsed = time.time() - t0
        
        if result.returncode != 0:
            print(f"  FAILED after {elapsed:.1f}s")
            # Check if it's because no data exists
            if "FileNotFoundError" in result.stderr:
                print(f"  (No data found - skipping)")
            else:
                print(f"  Error: {result.stderr[-200:]}")
        else:
            print(f"  Success in {elapsed:.1f}s")
        
        completed += 1

print(f"\n{'='*60}")
print(f"COMPLETE: {completed}/{total} analyses run")
print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
print('='*60)
