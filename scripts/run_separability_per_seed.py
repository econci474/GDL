"""Run separability metrics for each individual seed and all combinations."""
import subprocess, sys

DATASETS_HOMO   = ["Cora", "PubMed"]
DATASETS_HETERO = ["Roman-empire", "Squirrel"]
SEEDS = [0, 1, 2]
K = 8
HEADS_DIR = "D:/GCN_eval/classifier_heads"

# (dataset, loss_type) pairs
RUNS = []
for ds in DATASETS_HOMO + DATASETS_HETERO:
    RUNS.append((ds, "ce_only"))
    RUNS.append((ds, "ce_plus_R_R10.0_smooth_band-1.0to0.0"))   # band-1.0to0.0
    RUNS.append((ds, "ce_plus_R_R10.0_smooth_floor0.10_band-1.0to0.0"))
    RUNS.append((ds, "ce_plus_R_R10.0_smooth_band-1.5to0.2"))    # band-1.5to0.25

# Remove duplicates while preserving order
seen, unique_runs = set(), []
for r in RUNS:
    if r not in seen:
        seen.add(r)
        unique_runs.append(r)

# Per-dataset the K=8 loss types we actually care about:
DATASET_BAND = {
    # (dataset, band_tag): loss_type_dir
    ("Cora",         "ce_only"):    "ce_only",
    ("Cora",         "band10"):     "ce_plus_R_R10.0_smooth_band-1.0to0.0",
    ("Cora",         "band15"):     "ce_plus_R_R10.0_smooth_band-1.5to0.2",
    ("PubMed",       "ce_only"):    "ce_only",
    ("PubMed",       "band10"):     "ce_plus_R_R10.0_smooth_floor0.10_band-1.0to0.0",
    ("PubMed",       "band15"):     "ce_plus_R_R10.0_smooth_band-1.5to0.2",
    ("Roman-empire", "ce_only"):    "ce_only",
    ("Roman-empire", "band10"):     "ce_plus_R_R1.0_smooth_floor0.10_band-1.0to0.0",
    ("Roman-empire", "band15"):     "ce_plus_R_R10.0_smooth_band-1.5to0.2",
    ("Squirrel",     "ce_only"):    "ce_only",
    ("Squirrel",     "band10"):     "ce_plus_R_R1.0_smooth_floor0.10_band-1.0to0.0",
    ("Squirrel",     "band15"):     "ce_plus_R_R10.0_smooth_band-1.5to0.2",
}

combos = []
for (ds, band), loss_type in DATASET_BAND.items():
    for seed in SEEDS:
        combos.append((ds, loss_type, seed))

total = len(combos)
print(f"Running {total} combinations ...\n")

for i, (ds, lt, seed) in enumerate(combos, 1):
    cmd = [
        sys.executable, "src/separability_metrics_classifier_heads.py",
        "--dataset", ds,
        "--model", "GCN",
        "--K", str(K),
        "--seed", str(seed),
        "--loss-type", lt,
        "--classifier-heads-dir", HEADS_DIR,
    ]
    print(f"[{i}/{total}] {ds}  seed={seed}  {lt}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    # Print last saved line
    for line in result.stdout.splitlines():
        if "Saved" in line or "SKIP" in line or "No data" in line:
            print(f"  -> {line.strip()}")
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[-300:]}")

print("\nAll done.")
