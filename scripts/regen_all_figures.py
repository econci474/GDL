"""
regen_all_figures.py  (v2 — classifier head pipeline)
=======================================================
Regenerate ALL separability, entropy, and curvature figures using the
trained classifier head checkpoints.

Pipeline (4 phases):
  Phase 0  Read best_hyperparams_{model}_per_layer.csv → (dataset, K) → loss_dir
  Phase 1  extract_classifier_outputs.py  → layer_probs.npz + pernode.npz
  Phase 2  separability_metrics_classifier_heads.py  → separability figures
  Phase 3  plot_node_entropy_vs_prob.py   → entropy figures (probability + correctness)
  Phase 4  plot_curvature_violations.py  → curvature figures (once per model, all datasets)

Usage (Colab):
    python scripts/regen_all_figures.py \\
        --models GCN GAT \\
        --hyperparams-dir /content/drive/MyDrive/GDL/sweep_results \\
        --classifier-heads-dir /content/drive/MyDrive/GDL/sweep_results/classifier_heads

Run a quick test first:
    python scripts/regen_all_figures.py \\
        --models GAT --datasets Cora --k-values 1 \\
        --hyperparams-dir /content/drive/MyDrive/GDL/sweep_results \\
        --classifier-heads-dir /content/drive/MyDrive/GDL/sweep_results/classifier_heads
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
py   = sys.executable

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--models",   nargs="+", default=["GCN", "GAT"])
parser.add_argument("--datasets", nargs="+",
                    default=["Cora", "PubMed", "Roman-empire", "Squirrel"])
parser.add_argument("--k-values", nargs="+", type=int, default=list(range(1, 9)),
                    help="K values to process (default: 1..8)")
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--hyperparams-dir", type=str, default=None,
                    help="Directory containing best_hyperparams_*_per_layer.csv files. "
                         "If omitted, the best-config lookup is skipped and you must "
                         "supply --default-loss-type instead.")
parser.add_argument("--default-loss-type", type=str, default="ce_only",
                    help="Fallback loss_type dir when --hyperparams-dir is not set.")
parser.add_argument("--classifier-heads-dir", type=str, default=None,
                    help="Override cfg.classifier_heads_dir (e.g. on Drive)")
parser.add_argument("--sep-only",      action="store_true")
parser.add_argument("--entropy-only",  action="store_true")
parser.add_argument("--extract-only",  action="store_true")
parser.add_argument("--skip-extract",  action="store_true",
                    help="Skip Phase 1 (extraction), assume layer_probs.npz already exist")
parser.add_argument("--plot-types", nargs="+",
                    default=["probability", "correctness"])
args = parser.parse_args()

HETERO = {"Roman-empire", "Squirrel"}

# ── Helpers ────────────────────────────────────────────────────────────────────
def run(cmd, desc):
    print(f"\n{'='*60}\n{desc}\n{'='*60}", flush=True)
    t0 = time.time()
    r  = subprocess.run(cmd, cwd=ROOT)
    elapsed = (time.time() - t0) / 60
    status  = "OK" if r.returncode == 0 else f"WARNING exit {r.returncode}"
    print(f"  [{status}] {elapsed:.1f} min", flush=True)
    return r.returncode


def ckpt_dir_flag():
    """Return --classifier-heads-dir flag if set."""
    if args.classifier_heads_dir:
        return ["--classifier-heads-dir", args.classifier_heads_dir]
    return []


# ── Phase 0 — Build (dataset, model, K) → loss_dir lookup ────────────────────
import pandas as pd   # noqa: E402  (only needed here)
sys.path.insert(0, str(ROOT))
from src.evaluate_final import _build_loss_dir_candidates  # noqa: E402

def _build_loss_dir_for_row(row):
    """Reconstruct the loss directory name from a best_hyperparams CSV row."""
    import math
    loss_type = row.get("loss_type", "ce_only")
    if loss_type not in {"ce_plus_R", "weighted_ce_plus_R", "R_only"}:
        return loss_type

    config = {
        "lambda_R":     float(row.get("lambda_r",    1.0)),
        "R_mode":       str(row.get("R_mode",      "smooth")),
        "entropy_floor": (float(row["entropy_floor"])
                          if "entropy_floor" in row and
                          not (isinstance(row["entropy_floor"], float)
                               and math.isnan(row["entropy_floor"]))
                          else None),
        "per_class_R":  bool(row.get("per_class_r", False)),
        "band_lower":   float(row.get("band_lower", -1.0)),
        "band_upper":   float(row.get("band_upper",  0.0)),
    }
    return _build_loss_dir_candidates(loss_type, config)[0]


# Load three lookups: overall best, band-1.0to0.0 best, band-1.5to0.25 best
_CSV_SUFFIXES = [
    ("loss_dir_best",   "{model}_per_layer.csv"),
    ("loss_dir_band1",  "{model}_band-1.0to0.0_per_layer.csv"),
    ("loss_dir_band2",  "{model}_band-1.5to0.25_per_layer.csv"),
]
loss_dir_lookups = {tag: {} for tag, _ in _CSV_SUFFIXES}

if args.hyperparams_dir:
    hp_dir = Path(args.hyperparams_dir)
    for tag, csv_pattern in _CSV_SUFFIXES:
        for model in args.models:
            hp_csv = hp_dir / f"best_hyperparams_{csv_pattern.format(model=model)}"
            if not hp_csv.exists():
                print(f"[WARN] {hp_csv.name} not found — skipping {tag}")
                continue
            df_hp = pd.read_csv(hp_csv)
            for _, row in df_hp.iterrows():
                ds  = row["dataset"]
                K   = int(row["K"])
                loss_dir_lookups[tag][(model, ds, K)] = _build_loss_dir_for_row(row)
            print(f"[Phase 0] Loaded {len(df_hp)} rows from {hp_csv.name} → {tag}")


def get_all_loss_dirs(model, ds, K):
    """Return deduplicated ordered list of loss_dirs for this (model, ds, K).
    Always includes ce_only; adds the band-specific bests when available.
    """
    seen = set()
    result = []
    for ldir in [
        "ce_only",
        loss_dir_lookups["loss_dir_band1"].get((model, ds, K)),
        loss_dir_lookups["loss_dir_band2"].get((model, ds, K)),
        loss_dir_lookups["loss_dir_best"].get((model, ds, K),
                                              args.default_loss_type),
    ]:
        if ldir and ldir not in seen:
            seen.add(ldir)
            result.append(ldir)
    return result


# ── Main loop ─────────────────────────────────────────────────────────────────
total_start = time.time()
curvature_loss_dirs_by_model = {m: set() for m in args.models}

for model in args.models:
    for ds in args.datasets:
        is_hetero = ds in HETERO
        split_id  = 0 if is_hetero else None

        for K in args.k_values:
            loss_dirs = get_all_loss_dirs(model, ds, K)
            curvature_loss_dirs_by_model[model].update(loss_dirs)
            label = f"{model}/{ds}/K={K}"

            for loss_dir in loss_dirs:
                label_ld = f"{label} [{loss_dir}]"

                # ── Phase 1: Extract ────────────────────────────────────────────
                if not args.skip_extract and not args.sep_only and not args.entropy_only:
                    for seed in args.seeds:
                        cmd = [py, "src/extract_classifier_outputs.py",
                               "--dataset", ds, "--model", model,
                               "--K", str(K), "--seed", str(seed),
                               "--loss-type", loss_dir]
                        if is_hetero:
                            cmd += ["--split-id", str(split_id)]
                        if args.classifier_heads_dir:
                            cmd += ["--classifier-heads-dir", args.classifier_heads_dir]
                        run(cmd, f"EXTRACT  {label_ld}  seed={seed}")

                if args.extract_only:
                    continue

                # ── Phase 2: Separability ──────────────────────────────────────
                if not args.entropy_only:
                    for seed in args.seeds:
                        cmd = [py, "src/separability_metrics_classifier_heads.py",
                               "--dataset", ds, "--model", model,
                               "--K", str(K), "--seed", str(seed),
                               "--loss-type", loss_dir] + ckpt_dir_flag()
                        if is_hetero:
                            cmd += ["--split-id", str(split_id)]
                        run(cmd, f"SEPARABILITY  {label_ld}  seed={seed}")

                    # Aggregated (all seeds)
                    seeds_str = ",".join(str(s) for s in args.seeds)
                    cmd = [py, "src/separability_metrics_classifier_heads.py",
                           "--dataset", ds, "--model", model,
                           "--K", str(K), "--seed", seeds_str,
                           "--loss-type", loss_dir] + ckpt_dir_flag()
                    if is_hetero:
                        cmd += ["--split-id", str(split_id)]
                    run(cmd, f"SEPARABILITY agg  {label_ld}  seeds={seeds_str}")

                # ── Phase 3: Entropy ────────────────────────────────────────────
                if not args.sep_only:
                    for plot_type in args.plot_types:
                        for seed in args.seeds:
                            cmd = [py, "src/plot_node_entropy_vs_prob.py",
                                   "--dataset", ds, "--model", model,
                                   "--K", str(K), "--seed", str(seed),
                                   "--split", "val", "--plot_type", plot_type]
                            if is_hetero:
                                cmd += ["--split_idx", str(split_id)]
                            run(cmd, f"ENTROPY {plot_type}  {label_ld}  seed={seed}")

                        # Aggregated
                        seeds_str = ",".join(str(s) for s in args.seeds)
                        cmd = [py, "src/plot_node_entropy_vs_prob.py",
                               "--dataset", ds, "--model", model,
                               "--K", str(K), "--seed", seeds_str,
                               "--split", "val", "--plot_type", plot_type]
                        if is_hetero:
                            cmd += ["--split_idx", str(split_id)]
                        run(cmd, f"ENTROPY {plot_type} agg  {label_ld}  seeds={seeds_str}")

# ── Phase 4: Curvature violations (once per model, all datasets) ──────────────
if not args.sep_only and not args.entropy_only and not args.extract_only:
    max_K = max(args.k_values)
    if max_K < 3:
        print(f"\n[Phase 4] Skipping curvature plots — K={max_K} < 3 (need ≥3 layers for δ²H)")
    else:
        datasets_str = ",".join(args.datasets)
        for model in args.models:
            loss_types_str = ",".join(sorted(curvature_loss_dirs_by_model[model]))
            if not loss_types_str:
                continue
            # Use the largest K available for the curvature plot
            cmd = [py, "scripts/plot_curvature_violations.py",
                   "--datasets", datasets_str,
                   "--model", model,
                   "--K", str(max_K),
                   "--seed", ",".join(str(s) for s in args.seeds),
                   "--loss-types", loss_types_str]
            if args.classifier_heads_dir:
                cmd += ["--classifier-heads-dir", args.classifier_heads_dir]
            run(cmd, f"CURVATURE  {model}  K={max_K}  all datasets")

total_min = (time.time() - total_start) / 60
print(f"\n{'='*60}")
print(f"ALL FIGURES DONE in {total_min:.1f} min")
print(f"{'='*60}", flush=True)
