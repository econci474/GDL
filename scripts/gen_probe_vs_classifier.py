"""
Generate probe vs classifier comparison plots for Cora/PubMed
Requires both probe results and classifier outputs
"""
import subprocess
import sys
from pathlib import Path

print("="*70)
print("GENERATING PROBE VS CLASSIFIER PLOTS (Cora + PubMed)")
print("="*70)

datasets = ['Cora', 'PubMed']
model = 'GCN'
success = 0
total = 0

for dataset in datasets:
    print(f"\n📊 {dataset}")
    for K in range(9):
        for seed in range(4):
            total += 1
            cmd = [
                sys.executable, '-m', 'src.plot_probes_vs_classifier_heads',
                '--dataset', dataset,
                '--model', model,
                '--K', str(K),
                '--seed', str(seed)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                success += 1
                print(f"  ✓ K={K} seed={seed}", flush=True)
            else:
                print(f"  ⚠ K={K} seed={seed} - {result.stderr[:100] if result.stderr else 'failed'}", flush=True)

print(f"\n📈 Probe vs Classifier Plots: {success}/{total}")
print(f"Output: results/figures/probe_vs_classifier/")
