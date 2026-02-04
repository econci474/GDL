"""
Overnight batch execution for all datasets/models/K values.
Runs in gdl environment, saves results locally.
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime
import sys

# Configuration
datasets = ['Cora', 'PubMed', 'Roman-empire', 'Minesweeper']
model = 'GCN'
K_values = list(range(9))  # 0-8
seeds = [0, 1, 2, 3]

# Skip already completed
skip_list = [
    ('Cora', 3, [0, 1, 2]),
    ('Cora', 8, [0, 1, 2, 3]),
    ('PubMed', 3, [0, 1, 2]),  # Probing now, will finish soon
]

def should_skip(dataset, K, seed):
    """Check if this run should be skipped."""
    for skip_dataset, skip_K, skip_seeds in skip_list:
        if dataset == skip_dataset and K == skip_K and seed in skip_seeds:
            return True
    return False

def run_pipeline(dataset, K, seed):
    """Run full pipeline: train -> extract -> probe"""
    print(f"\n{'='*60}")
    print(f"Running: {dataset}/{model} K={K} seed={seed}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print('='*60)
    
    steps = [
        ('train', f'python -m src.train_gnn --dataset {dataset} --model {model} --K {K} --seed {seed}'),
        ('extract', f'python -m src.extract_embeddings --dataset {dataset} --model {model} --K {K} --seed {seed}'),
        ('probe', f'python -m src.probe --dataset {dataset} --model {model} --K {K} --seed {seed}'),
    ]
    
    for step_name, cmd in steps:
        print(f"  [{step_name.upper()}] Starting...")
        t0 = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        elapsed = time.time() - t0
        
        if result.returncode != 0:
            print(f"  [{step_name.upper()}] FAILED after {elapsed:.1f}s")
            print(f"  Error: {result.stderr[-500:]}")  # Last 500 chars
            return False
        else:
            print(f"  [{step_name.upper()}] Success in {elapsed:.1f}s")
    
    return True

# Main execution
log_file = Path('overnight_progress.log')
with open(log_file, 'w') as f:
    f.write(f"Started: {datetime.now()}\n")
    f.write(f"Total runs: {len(datasets) * len(K_values) * len(seeds)}\n\n")

total = len(datasets) * len(K_values) * len(seeds)
completed = 0
failed = 0
skipped = 0

start_time = time.time()

# Prioritize datasets
priority_order = ['Cora', 'PubMed', 'Roman-empire', 'Minesweeper']

for dataset in priority_order:
    for K in K_values:
        for seed in seeds:
            if should_skip(dataset, K, seed):
                print(f"SKIP: {dataset}/{model} K={K} seed={seed}")
                skipped += 1
                continue
            
            success = run_pipeline(dataset, K, seed)
            
            if success:
                completed += 1
            else:
                failed += 1
            
            # Progress update
            processed = completed + failed + skipped
            progress = (processed / total) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / processed) * (total - processed) if processed > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"PROGRESS: {processed}/{total} ({progress:.1f}%)")
            print(f"Completed: {completed}, Failed: {failed}, Skipped: {skipped}")
            print(f"Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
            print('='*60)
            
            # Save checkpoint
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')} | {dataset} K={K} seed={seed} | {'OK' if success else 'FAIL'}\n")
            
            # Flush output
            sys.stdout.flush()

print(f"\n{'='*60}")
print("BATCH COMPLETE")
print(f"Total: {total}, Completed: {completed}, Failed: {failed}, Skipped: {skipped}")
print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
print('='*60)

with open(log_file, 'a') as f:
    f.write(f"\nFinished: {datetime.now()}\n")
    f.write(f"Completed: {completed}/{total} | Failed: {failed} | Skipped: {skipped}\n")
    f.write(f"Total time: {(time.time() - start_time)/3600:.2f} hours\n")

print(f"\nLog saved to: {log_file.absolute()}")
