"""
Plot the fraction of nodes where δ²H > 0 (curvature violation) at each layer.

δ²H[k] = H_norm[k+2] - 2*H_norm[k+1] + H_norm[k]   (second finite diff of normalised entropy)
When δ²H > 0 the entropy trajectory is locally convex-up, violating the upper-band constraint.

Supports multiple datasets (one row per dataset) and multiple loss_types (one colour each).

Usage:
    python scripts/plot_curvature_violations.py \\
        --datasets "Cora,PubMed,Roman-empire,Squirrel" \\
        --model GCN --K 8 --seed all \\
        --loss-types "ce_only,ce_plus_R_R1.0_smooth_floor0.10_band-1.0to0.0,ce_plus_R_R10.0_smooth_floor0.10_band-1.0to0.0,ce_plus_R_R10.0_smooth_band-1.5to0.2" \\
        --classifier-heads-dir D:/GCN_eval/classifier_heads
"""

import argparse
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.datasets import load_dataset as load_ds

HETERO_DATASETS = {"Roman-empire", "Squirrel"}

BAND_LABEL_FIXES = {
    "band-1.5to0.2": "band-1.5to0.25",
}

def _pretty(loss_type):
    label = loss_type.replace("ce_plus_R_", "").replace("ce_only", "CE only")
    for old, new in BAND_LABEL_FIXES.items():
        label = label.replace(old, new)
    # Shorten for legend
    label = label.replace("_smooth_", " ").replace("_", " ")
    return label


def calc_entropy_norm(probs, num_classes):
    H = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    return H / math.log(num_classes)


def second_finite_diff(H_kN):
    """H_kN [K+1, N] → [K-1, N]  (curvature at layers 1..K-1)."""
    return H_kN[2:] - 2 * H_kN[1:-1] + H_kN[:-2]


def load_violation_stats(dataset, model, K, seed, loss_type,
                         split_id=None, classifier_heads_dir=None,
                         num_classes=None):
    heads_dir = Path(classifier_heads_dir or cfg.classifier_heads_dir)
    base = heads_dir / loss_type / dataset / model / f"seed_{seed}" / f"K_{K}"
    if split_id is not None:
        base = base / f"split_{split_id}"

    val_path   = base / "layer_probs.npz"
    train_path = base / "layer_probs_train.npz"

    if not val_path.exists():
        raise FileNotFoundError(val_path)

    def stack_from(npz_path, prefix):
        if not npz_path.exists():
            return None
        d = np.load(npz_path)
        layers = []
        for k in range(K + 1):
            key = f"{prefix}_probs_{k}"
            if key not in d:
                return None
            layers.append(d[key])
        return np.stack(layers, axis=0)

    train_stack = stack_from(train_path, "train")
    val_stack   = stack_from(val_path,   "val")

    results = {}
    for name, stack in [("train", train_stack), ("val", val_stack)]:
        if stack is None:
            results[f"frac_{name}"]     = None
            results[f"mean_pos_{name}"] = None
            continue
        nc = stack.shape[-1] if num_classes is None else num_classes
        H_norm = np.stack([calc_entropy_norm(stack[k], nc) for k in range(K + 1)], axis=0)
        delta2 = second_finite_diff(H_norm)
        results[f"frac_{name}"]     = (delta2 > 0).mean(axis=1)
        results[f"mean_pos_{name}"] = delta2.clip(min=0).mean(axis=1)

    # k_vals: center layer of each triplet, i.e. 1..K-1
    results["k_vals"] = list(range(1, K))
    return results


def plot_all(all_dataset_results, datasets, model, K, seeds_used,
             output_dir, loss_types, show_std=True, suffix=""):
    """
    all_dataset_results: {dataset -> {loss_type -> [seed_stats]}}
    Layout: rows = datasets, cols = (violation rate, violation magnitude)
    """
    n_rows = len(datasets)
    colors = plt.cm.tab10(np.arange(10))
    ls_map  = {"train": "-", "val": "--"}
    mk_map  = {"train": "o", "val": "s"}

    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(13, 4 * n_rows),
                             sharex=True)
    # Ensure axes is always 2-D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, dataset in enumerate(datasets):
        ds_results = all_dataset_results.get(dataset, {})

        for col, metric in enumerate(["frac", "mean_pos"]):
            ax = axes[row, col]

            for idx, lt in enumerate(loss_types):
                seed_stats = ds_results.get(lt, [])
                if not seed_stats:
                    continue
                color  = colors[idx % 10]
                legend_label = _pretty(lt)

                for split in ["train", "val"]:
                    key  = f"{metric}_{split}"
                    valid = [s[key] for s in seed_stats if s.get(key) is not None]
                    if not valid:
                        continue
                    arr   = np.stack(valid, axis=0)
                    m     = arr.mean(axis=0)
                    s     = arr.std(axis=0)
                    k_vals = seed_stats[0]["k_vals"]
                    label  = f"{legend_label} ({split})" if row == 0 else None
                    ax.plot(k_vals, m,
                            ls_map[split], color=color, linewidth=1.8,
                            marker=mk_map[split], markersize=4, label=label)
                    if show_std:
                        ax.fill_between(k_vals, m - s, m + s, alpha=0.05, color=color)

            ax.set_xlim(0, K)
            ax.set_xticks(range(0, K + 1))
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
                ax.set_ylabel(f"{dataset}\nFraction δ²H > 0", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("Mean δ²H (viol. nodes)", fontsize=9)

            if row == 0:
                title = ("Curvature Violation Rate (δ²H > 0)"
                         if col == 0 else
                         "Mean Violation Magnitude (δ²H > 0 only)")
                ax.set_title(title, fontsize=11, fontweight="bold")

            if row == n_rows - 1:
                ax.set_xlabel("Depth k", fontsize=10)

    # Build legend manually so all loss_types always appear
    handles = []
    labels  = []
    for idx, lt in enumerate(loss_types):
        color = colors[idx % 10]
        legend_label = _pretty(lt)
        for split in ["train", "val"]:
            h = matplotlib.lines.Line2D(
                [], [], color=color, linestyle=ls_map[split],
                marker=mk_map[split], markersize=5, linewidth=1.8,
            )
            handles.append(h)
            labels.append(f"{legend_label} ({split})")

    seeds_str = ", ".join(str(s) for s in seeds_used)
    fig.suptitle(
        f"Entropy Curvature Violations: {model}, K={K}\n"
        f"Seeds: {seeds_str}, solid=train, dashed=val",
        fontsize=12, fontweight="bold"
    )
    fig.legend(handles, labels,
               loc="lower center", ncol=len(loss_types),
               fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    ds_tag  = "_".join(d.replace("-", "") for d in datasets)
    std_tag = "" if show_std else "_no_std"
    out = output_dir / f"{ds_tag}_{model}_K{K}_curvature_violations{suffix}{std_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",             type=str,
                        default="Roman-empire",
                        help="Comma-separated list of datasets")
    parser.add_argument("--model",                type=str, default="GCN")
    parser.add_argument("--K",                    type=int, default=8)
    parser.add_argument("--seed",                 type=str, default="all")
    parser.add_argument("--loss-types",           type=str, required=True)
    parser.add_argument("--split-id",             type=int, default=None)
    parser.add_argument("--classifier-heads-dir", type=str, default=None)
    parser.add_argument("--no-std",               action="store_true",
                        help="Suppress shaded std bands (saves as _no_std)")
    parser.add_argument("--suffix",               type=str, default="",
                        help="Extra suffix appended to output filename before _no_std")
    args = parser.parse_args()

    if args.classifier_heads_dir:
        cfg.classifier_heads_dir = args.classifier_heads_dir

    seeds = (list(cfg.seeds) if args.seed.lower() == "all"
             else [int(s) for s in args.seed.split(",")])
    datasets   = [d.strip() for d in args.datasets.split(",")]
    loss_types = [lt.strip() for lt in args.loss_types.split(",")]

    output_dir = Path(cfg.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dataset_results = {}
    for dataset in datasets:
        split_id = args.split_id
        if split_id is None and dataset in HETERO_DATASETS:
            split_id = 0

        _, num_classes, _ = load_ds(dataset, root_dir="data",
                                    planetoid_normalize=False,
                                    planetoid_split="public")
        ds_results = {}
        for lt in loss_types:
            seed_stats = []
            for seed in seeds:
                try:
                    stats = load_violation_stats(
                        dataset, args.model, args.K, seed, lt,
                        split_id=split_id,
                        classifier_heads_dir=args.classifier_heads_dir,
                        num_classes=num_classes,
                    )
                    seed_stats.append(stats)
                except FileNotFoundError:
                    pass
            if seed_stats:
                ds_results[lt] = seed_stats
                print(f"  [{dataset}][{lt}]  {len(seed_stats)} seeds")
            else:
                print(f"  [{dataset}][{lt}]  SKIP (no data)")
        all_dataset_results[dataset] = ds_results

    if any(all_dataset_results.values()):
        plot_all(all_dataset_results, datasets, args.model, args.K,
                 seeds, output_dir, loss_types,
                 show_std=not args.no_std,
                 suffix=args.suffix)
    else:
        print("No data found.")


if __name__ == "__main__":
    main()
