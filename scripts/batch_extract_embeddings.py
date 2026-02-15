"""
Batch extraction of embeddings from all available Colab checkpoints.
Finds all best.pt files and extracts embeddings for each.
"""

import subprocess
from pathlib import Path

def main():
    results_dir = Path("results/runs")
    datasets = ["Cora", "PubMed", "Roman-empire", "Minesweeper"]
    model = "GCN"
    
    print("\n" + "="*60)
    print("BATCH EMBEDDINGS EXTRACTION FROM COLAB CHECKPOINTS")
    print("="*60)
    
    # Find all checkpoints (including those in split folders)
    checkpoints = []
    for dataset in datasets:
        dataset_dir = results_dir / dataset / model
        if not dataset_dir.exists():
            continue
            
        # Find all best.pt files
        for ckpt_path in dataset_dir.rglob("best.pt"):
            # Parse path: .../seed_X/K_Y/[split_Z/]best.pt
            parts = ckpt_path.parts
            
            # Find seed and K indices
            try:
                seed_idx = next(i for i, p in enumerate(parts) if p.startswith('seed_'))
                k_idx = next(i for i, p in enumerate(parts) if p.startswith('K_'))
                seed = int(parts[seed_idx].split('_')[1])
                K = int(parts[k_idx].split('_')[1])
                
                # Check if this is in a split folder
                has_split = 'split_' in str(ckpt_path)
                if has_split:
                    split_idx = next(i for i, p in enumerate(parts) if p.startswith('split_'))
                    split_num = int(parts[split_idx].split('_')[1])
                    # Only process split_0 to avoid duplicates
                    if split_num != 0:
                        continue
                
                checkpoints.append((dataset, K, seed, ckpt_path))
            except (StopIteration, ValueError, IndexError):
                print(f"Warning: Could not parse {ckpt_path}")
                continue
    
    print(f"\nFound {len(checkpoints)} checkpoints to process")
    
    # Group by dataset for summary
    from collections import defaultdict
    by_dataset = defaultdict(int)
    for ds, _, _, _ in checkpoints:
        by_dataset[ds] += 1
    
    for ds, count in sorted(by_dataset.items()):
        print(f"  {ds}: {count} checkpoints")
    print()
    
    # Extract embeddings for each
    success_count = 0
    fail_count = 0
    
    for i, (dataset, K, seed, ckpt_path) in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] {dataset} K={K} seed={seed}")
        print("-" * 40)
        
        cmd = [
            "python", "scripts/extract_embeddings_from_checkpoints.py",
            "--dataset", dataset,
            "--model", model,
            "--K", str(K),
            "--seed", str(seed)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                print("✓ Success")
                success_count += 1
            else:
                print(f"✗ Failed: {result.stderr[:200]}")
                fail_count += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            fail_count += 1
    
    print("\n" + "="*60)
    print(f"BATCH EXTRACTION COMPLETE")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
