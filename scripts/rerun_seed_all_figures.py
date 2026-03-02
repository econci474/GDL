"""Re-run the seed_all separability figures with the corrected title."""
import subprocess, sys

CLASSIFIER_HEADS_DIR = "D:/GCN_eval/classifier_heads"
MODEL = "GCN"
K = 8

RUNS = [
    ("Cora",         "ce_only"),
    ("Cora",         "ce_plus_R_R10.0_smooth_band-1.0to0.0"),
    ("Cora",         "ce_plus_R_R10.0_smooth_band-1.5to0.2"),
    ("PubMed",       "ce_only"),
    ("PubMed",       "ce_plus_R_R10.0_smooth_floor0.10_band-1.0to0.0"),
    ("PubMed",       "ce_plus_R_R10.0_smooth_band-1.5to0.2"),
    ("Roman-empire", "ce_only"),
    ("Roman-empire", "ce_plus_R_R1.0_smooth_floor0.10_band-1.0to0.0"),
    ("Roman-empire", "ce_plus_R_R10.0_smooth_band-1.5to0.2"),
    ("Squirrel",     "ce_only"),
    ("Squirrel",     "ce_plus_R_R1.0_smooth_floor0.10_band-1.0to0.0"),
    ("Squirrel",     "ce_plus_R_R10.0_smooth_band-1.5to0.2"),
]

for i, (ds, lt) in enumerate(RUNS, 1):
    print(f"[{i}/{len(RUNS)}] {ds}  seed=all  {lt}")
    result = subprocess.run([
        sys.executable, "src/separability_metrics_classifier_heads.py",
        "--dataset", ds, "--model", MODEL, "--K", str(K),
        "--seed", "all", "--loss-type", lt,
        "--classifier-heads-dir", CLASSIFIER_HEADS_DIR,
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in result.stdout.splitlines():
        if "Saved" in line or "SKIP" in line or "No data" in line:
            print(f"  -> {line.strip()}")
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[-200:]}")

print("\nAll done.")
