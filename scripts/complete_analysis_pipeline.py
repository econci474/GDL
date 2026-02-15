"""
Complete pipeline: Extract K=0 embeddings → Train probes → Generate all plots
"""
import subprocess
import sys
from pathlib import Path

print("="*70)
print("COMPLETE CORA/PUBMED ANALYSIS PIPELINE")
print("="*70)

datasets = ['Cora', 'PubMed']
model = 'GCN'

# Phase 1: Extract K=0 embeddings (should already exist from training)
print("\n[PHASE 1] Checking K=0 Embeddings")
print("-"*70)
for dataset in datasets:
    for seed in range(4):
        emb_path = Path(f'results/runs/{dataset}/{model}/seed_{seed}/K_0/embeddings.pt')
        if emb_path.exists():
            print(f"✓ {dataset} seed={seed}")
        else:
            print(f"✗ {dataset} seed={seed} - MISSING")

# Phase 2: Train ALL probes (K=0 through K=8, all seeds)
print("\n[PHASE 2] Training ALL Probes (72 jobs)")
print("-"*70)
probe_success = 0
probe_total = 0

for dataset in datasets:
    print(f"\n📊 {dataset}")
    for K in range(9):
        for seed in range(4):
            probe_total += 1
            cmd = [
                sys.executable, '-m', 'src.probe',
                '--dataset', dataset,
                '--model', model,
                '--K', str(K),
                '--seed', str(seed)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                probe_success += 1
                print(f"  ✓ K={K} seed={seed}", flush=True)
            else:
                print(f"  ⚠ K={K} seed={seed} - failed", flush=True)

print(f"\n📈 Probes: {probe_success}/{probe_total}")

# Phase 3: Generate all plots
print("\n[PHASE 3] Generating All Plots")
print("-"*70)

# Use the existing plot generation scripts
plot_scripts = [
    'scripts/generate_all_plots.py'
]

for script in plot_scripts:
    if Path(script).exists():
        print(f"\nRunning {script}...")
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ Success")
        else:
            print(f"  ⚠ Failed: {result.stderr[:200] if result.stderr else 'unknown error'}")

# Phase 4: Probe vs Classifier plots
print("\n[PHASE 4] Probe vs Classifier Comparisons")
print("-"*70)
pvc_success = 0
pvc_total = 0

for dataset in datasets:
    for K in range(9):
        for seed in range(9):
            pvc_total += 1
            cmd = [
                sys.executable, '-m', 'src.plot_probes_vs_classifier_heads',
                '--dataset', dataset,
                '--model', model,
                '--K', str(K),
                '--seed', str(seed)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                pvc_success += 1

print(f"📈 Probe vs Classifier: {pvc_success}/{pvc_total}")

# Phase 5: Interactive visualization
print("\n[PHASE 5] Interactive Visualization")
print("-"*70)
cmd = [sys.executable, '-m', 'src.plot_unified_interactive', '--datasets', 'Cora', 'PubMed']
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print("✓ Interactive viz created")
else:
    print(f"⚠ Failed: {result.stderr[:200] if result.stderr else 'unknown'}")

print(f"\n{'='*70}")
print("PIPELINE COMPLETE!")
print(f"{'='*70}")
print("\nResults:")
print(f"  Probes: {probe_success}/{probe_total}")
print(f"  Probe vs Classifier: {pvc_success}/{pvc_total}")
print(f"\nView: results/figures/interactive/unified_interactive_val.html")
