"""
Generate depth analysis plots for Cora and PubMed across all K values
"""
import subprocess
import sys

datasets = ['Cora', 'PubMed']
model = 'GCN'
K_values = list(range(9))  # 0-8

print("="*70)
print("DEPTH ANALYSIS - CORA/PUBMED")
print("="*70)

success = 0
total = 0

for dataset in datasets:
    print(f"\n{dataset}:")
    for K in K_values:
        total += 1
        print(f"  K={K}...", end=" ", flush=True)
        
        cmd = [sys.executable, '-m', 'src.plots', '--dataset', dataset, '--model', model, '--K', str(K), '--plots', 'all']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            success += 1
            print("OK")
        else:
            print("FAILED")
            if result.stderr:
                print(f"    Error: {result.stderr[:150]}")

print(f"\n{'='*70}")
print(f"DONE: {success}/{total} successful")
print(f"{'='*70}")
