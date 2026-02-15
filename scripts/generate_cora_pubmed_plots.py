"""
Generate all plots for Cora and PubMed only (all K values 0-8)
Includes:
- Entropy vs probability
- Entropy vs correctness  
- Training diagnostics
- Depth analysis
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

datasets = ['Cora', 'PubMed']  # Only Cora/PubMed - heterophilous don't have embeddings
model = 'GCN'
K_values = list(range(9))  # 0-8
seeds = [0, 1, 2, 3]
plot_types = ['probability', 'correctness']

print("="*70)
print("GENERATING ALL PLOTS - CORA/PUBMED ONLY")
print("="*70)
print(f"Datasets: {datasets}")
print(f"K values: {K_values}")
print(f"Seeds: {seeds}")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("="*70)

total_plots = 0
completed = 0

# Generate entropy vs prob/correctness plots for all K values
print("\n[1] Entropy vs Probability/Correctness Plots")
print("-"*70)

for K in K_values:
    print(f"\nK={K}:")
    for dataset in datasets:
        for plot_type in plot_types:
            # Individual seeds
            for seed in seeds:
                total_plots += 1
                cmd = [
                    sys.executable, 
                    'scripts/plot_node_entropy_vs_prob.py',
                    '--dataset', dataset,
                    '--model', model,
                    '--K', str(K),
                    '--seed', str(seed),
                    '--split', 'val',
                    '--plot_type', plot_type
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    completed += 1
                    print(f"  OK: {dataset} {plot_type} seed={seed}")
                else:
                    print(f"  SKIP: {dataset} {plot_type} seed={seed}")
            
            # Aggregated (all seeds)
            total_plots += 1
            cmd = [
                sys.executable,
                'scripts/plot_node_entropy_vs_prob.py',
                '--dataset', dataset,
                '--model', model,
                '--K', str(K),
                '--split', 'val',
                '--plot_type', plot_type,
                '--aggregate', 'all'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                completed += 1
                print(f"  OK: {dataset} {plot_type} aggregate=all")
            else:
                print(f"  SKIP: {dataset} {plot_type} aggregate=all")
            
            # Aggregated (not_seed_2)
            total_plots += 1
            cmd = [
                sys.executable,
                'scripts/plot_node_entropy_vs_prob.py',
                '--dataset', dataset,
                '--model', model,
                '--K', str(K),
                '--split', 'val',
                '--plot_type', plot_type,
                '--aggregate', 'not_seed_2'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                completed += 1
                print(f"  OK: {dataset} {plot_type} aggregate=not_seed_2")
            else:
                print(f"  SKIP: {dataset} {plot_type} aggregate=not_seed_2")

print(f"\n{'='*70}")
print(f"DONE: {completed}/{total_plots} plots generated")
print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*70}")
