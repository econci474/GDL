"""
Generate all visualization plots for all datasets and K values.
Includes:
- Entropy vs probability (per-class)
- Entropy vs correctness (binary)
- Depth analysis curves (NLL, entropy, accuracy vs depth)
- Training diagnostics (loss curves, accuracy curves)
- Both individual seeds and aggregated (all seeds + not_seed_2)
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training_diagnostics import plot_training_curves, plot_accuracy_curves
import config as cfg

datasets = ['Cora', 'PubMed']
model = 'GCN'
K_values = list(range(9))  # 0-8
seeds = [0, 1, 2, 3]
plot_types = ['probability', 'correctness']

print("="*60)
print("Generating all visualizations for all K depths")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print(f"K values: {K_values}")
print("="*60)

total_plots = 0
completed = 0

# 1. Generate entropy vs prob/correctness plots for all K values and datasets
print("\n[1/4] Generating entropy plots for all K values...")
for K in K_values:
    print(f"\n{'='*60}")
    print(f"Processing K={K}")
    print('='*60)
    
    for dataset in datasets:
        for plot_type in plot_types:
            # Individual seeds
            for seed in seeds:
                cmd = f'python -m src.plot_node_entropy_vs_prob --dataset {dataset} --model {model} --K {K} --seed {seed} --split val --plot_type {plot_type}'
                
                print(f"  K={K} {dataset} {plot_type} seed={seed}...", end=" ")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✓")
                    completed += 1
                else:
                    print("✗")
                total_plots += 1
            
            # Aggregated (all seeds)
            cmd = f'python -m src.plot_node_entropy_vs_prob --dataset {dataset} --model {model} --K {K} --seed all --split val --plot_type {plot_type}'
            print(f"  K={K} {dataset} {plot_type} seed=all...", end=" ")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓")
                completed += 1
            else:
                print("✗")
            total_plots += 1
            
            # Aggregated (seeds 0,1,3 - not seed 2)
            cmd = f'python -m src.plot_node_entropy_vs_prob --dataset {dataset} --model {model} --K {K} --seed 0,1,3 --split val --plot_type {plot_type}'
            print(f"  K={K} {dataset} {plot_type} seed=0,1,3...", end=" ")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓")
                completed += 1
            else:
                print("✗")
            total_plots += 1



print(f"\n[2/4] Generating depth analysis plots (accuracy, entropy, NLL, correct/incorrect)...")
depth_completed = 0
depth_failed = 0

for K in K_values:
    print(f"\n{'='*60}")
    print(f"Depth Analysis for K={K}")
    print('='*60)
    
    for dataset in datasets:
        print(f"  K={K} {dataset}...", end=" ")
        
        # Generate all depth analysis plots (accuracy, entropy, NLL, correct/incorrect)
        cmd = f'python -m src.plots --dataset {dataset} --model {model} --K {K} --plots all'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓")
            depth_completed += 1
        else:
            print("✗")
            depth_failed += 1


print(f"\n[3/4] Generating separability plots (AUROC, Cohen's d vs depth)...")
sep_completed = 0
sep_failed = 0

for K in K_values:
    print(f"\n{'='*60}")
    print(f"Separability Analysis for K={K}")
    print('='*60)
    
    for dataset in datasets:
        print(f"  K={K} {dataset}...", end=" ")
        
        # Generate separability plots for all seeds
        cmd = f'python -m src.separability_metrics --dataset {dataset} --model {model} --K {K} --seed all'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓")
            sep_completed += 1
        else:
            print("✗")
            sep_failed += 1

print(f"\n[4/4] Generating training diagnostic plots (loss & accuracy curves)...")
diag_completed = 0
diag_failed = 0

# Convert config to dict for training_diagnostics functions
config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}

for K in K_values:
    print(f"\n{'='*60}")
    print(f"Training Diagnostics for K={K}")
    print('='*60)
    
    for dataset in datasets:
        print(f"  K={K} {dataset}...", end=" ")
        
        try:
            # Generate training loss curves
            plot_training_curves(dataset, model, K, config_dict, seeds=seeds)
            
            # Generate accuracy curves
            plot_accuracy_curves(dataset, model, K, config_dict, seeds=seeds)
            
            print("✓")
            diag_completed += 1
        except Exception as e:
            print(f"✗ ({str(e)[:50]})")
            diag_failed += 1

print(f"\n{'='*60}")
print(f"COMPLETE:")
print(f"  Entropy plots: {completed}/{total_plots}")
print(f"  Depth analysis: {depth_completed}/{len(K_values) * len(datasets)} (failed: {depth_failed})")
print(f"  Separability: {sep_completed}/{len(K_values) * len(datasets)} (failed: {sep_failed})")
print(f"  Training diagnostics: {diag_completed}/{len(K_values) * len(datasets)} (failed: {diag_failed})")
print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
print('='*60)



