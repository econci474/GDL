"""Convenience script to run full pipeline for a dataset/model combination."""

import argparse
import subprocess
import sys


def run_pipeline(dataset, model, K=8, seed='all'):
    """Run the complete pipeline: train, extract, probe, select, plot."""
    
    commands = [
        f"python -m src.train_gnn --dataset {dataset} --model {model} --K {K} --seed {seed}",
        f"python -m src.extract_embeddings --dataset {dataset} --model {model} --K {K} --seed {seed}",
        f"python -m src.probe --dataset {dataset} --model {model} --K {K} --seed {seed}",
        f"python -m src.depth_selection --dataset {dataset} --model {model}",
        f"python -m src.plots --dataset {dataset} --model {model}",
    ]
    
    print(f"\n{'='*70}")
    print(f"🚀 Running Full Pipeline: {dataset} + {model} (K={K}, seed={seed})")
    print(f"{'='*70}\n")
    
    for i, cmd in enumerate(commands, 1):
        step_names = ['Training', 'Extracting Embeddings', 'Probing', 'Depth Selection', 'Plotting']
        print(f"\n{'─'*70}")
        print(f"STEP {i}/5: {step_names[i-1]}")
        print(f"{'─'*70}")
        print(f"$ {cmd}\n")
        
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode != 0:
            print(f"\n❌ Pipeline failed at step {i}: {step_names[i-1]}")
            sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"✅ Pipeline Complete! Results in results/ directory")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full GNN entropy analysis pipeline')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (Cora, PubMed, Roman-empire, Minesweeper)')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (GCN, GAT)')
    parser.add_argument('--K', type=int, default=8,
                       help='Maximum depth')
    parser.add_argument('--seed', type=str, default='all',
                       help='Seed(s) to run: integer or "all"')
    
    args = parser.parse_args()
    
    run_pipeline(args.dataset, args.model, args.K, args.seed)
