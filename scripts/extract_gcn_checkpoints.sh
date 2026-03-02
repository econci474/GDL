#!/usr/bin/env bash
# Extract only GCN seed 0/1/2 best.pt files from all zip groups.
# Output: D:\GCN_eval\classifier_heads\ (used with --results-dir D:\GCN_eval)

set -e

OUT="D:/GCN_eval/classifier_heads"
C_ZIPS="c:/Users/elena/iCloudDrive/Desktop/ACS_MPhil/GDL/Project/entropy-selection/colab_GCN_sweep"
D_ZIPS="D:/1_Entropy_project_colab_sweep"

# 7-zip filter: GCN, seeds 0/1/2, best.pt only (both flat and split subdirs)
FILTERS=(
  -ir!"GCN\\seed_0\\K_*\\best.pt"
  -ir!"GCN\\seed_1\\K_*\\best.pt"
  -ir!"GCN\\seed_2\\K_*\\best.pt"
  -ir!"GCN\\seed_0\\K_*\\split_*\\best.pt"
  -ir!"GCN\\seed_1\\K_*\\split_*\\best.pt"
  -ir!"GCN\\seed_2\\K_*\\split_*\\best.pt"
)

extract_group() {
    local pattern="$1"
    echo ""
    echo "========================================"
    echo "Extracting: $pattern"
    echo "========================================"
    for zip in $pattern; do
        [ -f "$zip" ] || continue
        echo "  -> $zip"
        7z x "$zip" "${FILTERS[@]}" -o"$OUT" -aoa -y 2>&1 | grep -v "^$" | grep -v "^7-Zip" | grep -v "^Copyright" | grep -v "^Scanning"
    done
}

mkdir -p "$OUT"

# ── C: drive zip groups ──────────────────────────────────────────────
extract_group "$C_ZIPS/ce_only-*.zip"
extract_group "$C_ZIPS/ce_plus_R_R1.0_smooth_band-1.0to0.0-*.zip"
extract_group "$C_ZIPS/ce_plus_R_R1.0_smooth_band-1.5to0.2-*.zip"
extract_group "$C_ZIPS/ce_plus_R_R1.0_smooth_floor0.10_band-1.0to0.0-*.zip"
extract_group "$C_ZIPS/ce_plus_R_R1.0_smooth_floor0.10_band-1.5to0.2-*.zip"
extract_group "$C_ZIPS/ce_plus_R_R10.0_smooth_band-1.0to0.0-*.zip"
extract_group "$C_ZIPS/ce_plus_R_R10.0_smooth_band-1.5to0.2-*.zip"

# ── D: drive zip groups ──────────────────────────────────────────────
extract_group "$D_ZIPS/ce_plus_R_R10.0_smooth_floor0.10_band-1.0to0.0-*.zip"
extract_group "$D_ZIPS/ce_plus_R_R10.0_smooth_floor0.10_band-1.5to0.2-*.zip"

echo ""
echo "========================================"
echo "Extraction complete!"
echo "Output: $OUT"
echo "========================================"

# Count extracted files
find "$OUT" -name "best.pt" | wc -l | xargs echo "Total best.pt files:"
