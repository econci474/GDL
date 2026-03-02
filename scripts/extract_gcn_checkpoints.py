"""
Extract only GCN seed 0/1/2 best.pt files from all zip groups to D:\GCN_eval\classifier_heads\.

Selective extraction: skips GAT, GraphSAGE, train_log.csv, checkpoint_epoch_*.pt.
Only extracts files matching:
  */<dataset>/GCN/seed_{0,1,2}/K_*/best.pt
  */<dataset>/GCN/seed_{0,1,2}/K_*/split_*/best.pt
"""

import zipfile
import re
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
OUT_DIR = Path("D:/GCN_eval/classifier_heads")

# All zip groups to process
C_BASE = Path("c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/colab_GCN_sweep")
D_BASE = Path("D:/1_Entropy_project_colab_sweep")

ZIP_GROUPS = [
    (C_BASE, "ce_only-*"),
    (C_BASE, "ce_plus_R_R1.0_smooth_band-1.0to0.0-*"),
    (C_BASE, "ce_plus_R_R1.0_smooth_band-1.5to0.2-*"),
    (C_BASE, "ce_plus_R_R1.0_smooth_floor0.10_band-1.0to0.0-*"),
    (C_BASE, "ce_plus_R_R1.0_smooth_floor0.10_band-1.5to0.2-*"),
    (C_BASE, "ce_plus_R_R10.0_smooth_band-1.0to0.0-*"),
    (C_BASE, "ce_plus_R_R10.0_smooth_band-1.5to0.2-*"),
    (D_BASE, "ce_plus_R_R10.0_smooth_floor0.10_band-1.0to0.0-*"),
    (D_BASE, "ce_plus_R_R10.0_smooth_floor0.10_band-1.5to0.2-*"),
]

# Pattern: must have GCN and seed 0/1/2 and filename best.pt
GCN_BEST_PATTERN = re.compile(
    r"[^/\\]+/[^/\\]+/GCN/seed_[012]/K_\d+(?:/split_\d+)?/best\.pt$",
    re.IGNORECASE
)

def should_extract(name: str) -> bool:
    """Return True if this file should be extracted."""
    # Normalise separators
    name_norm = name.replace("\\", "/")
    return bool(GCN_BEST_PATTERN.search(name_norm))

def extract_zip(zip_path: Path, out_dir: Path) -> int:
    """Extract matching files from zip_path into out_dir. Returns count extracted."""
    count = 0
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                if not should_extract(info.filename):
                    continue
                target = out_dir / info.filename.replace("\\", "/")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    print(f"    [skip] {info.filename}")
                    count += 1
                    continue
                print(f"    [extract] {info.filename}")
                data = zf.read(info.filename)
                target.write_bytes(data)
                count += 1
    except zipfile.BadZipFile as e:
        print(f"  WARNING: bad zip {zip_path}: {e}")
    return count

# ── Main ─────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

total_files = 0

for base_dir, glob_pattern in ZIP_GROUPS:
    zip_files = sorted(base_dir.glob(f"{glob_pattern}.zip"))
    if not zip_files:
        print(f"\nWARNING: No zips found for {base_dir / glob_pattern}")
        continue

    print(f"\n{'='*60}")
    print(f"Group: {glob_pattern}  ({len(zip_files)} zip files)")
    print(f"{'='*60}")

    group_count = 0
    for zip_path in zip_files:
        print(f"  {zip_path.name}")
        n = extract_zip(zip_path, OUT_DIR)
        group_count += n

    print(f"  -> {group_count} files from this group")
    total_files += group_count

print(f"\n{'='*60}")
print(f"Done! Total best.pt files extracted: {total_files}")

# Count actual files on disk
actual = list(OUT_DIR.rglob("best.pt"))
print(f"Unique best.pt files on disk: {len(actual)}")
print(f"Output directory: {OUT_DIR}")
