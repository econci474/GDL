"""Generate layer_probs.npz from best.pt checkpoints.

Loads each best.pt checkpoint, runs a forward pass with all classifier heads,
and saves layer_probs.npz to the same directory. This file is required by
separability_metrics_classifier_heads.py.

Usage (from project root):
    python scripts/generate_layer_probs.py \\
        --best-hyperparams-path results/best_hyperparams_GCN_per_layer.csv \\
        --loss-filter ce_only \\
        --classifier-heads-dir D:/GCN_eval/classifier_heads

    # Run for all 3 band CSVs and all loss types:
    python scripts/generate_layer_probs.py \\
        --best-hyperparams-path results/best_hyperparams_GCN_per_layer.csv \\
        --classifier-heads-dir D:/GCN_eval/classifier_heads

    python scripts/generate_layer_probs.py \\
        --best-hyperparams-path results/best_hyperparams_GCN_band-1.0to0.0_per_layer.csv \\
        --classifier-heads-dir D:/GCN_eval/classifier_heads

    python scripts/generate_layer_probs.py \\
        --best-hyperparams-path results/best_hyperparams_GCN_band-1.5to0.25_per_layer.csv \\
        --classifier-heads-dir D:/GCN_eval/classifier_heads
"""

import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from src.evaluate_final import build_model, build_loss_dir
from src.datasets import load_dataset as load_ds

HETERO_DATASETS = {"Roman-empire", "Squirrel"}
HOMO_DATASETS   = {"Cora", "PubMed", "Pubmed"}  # tolerate capitalisation


@torch.no_grad()
def generate_layer_probs_for_checkpoint(
    checkpoint_path: Path,
    dataset: str,
    model_name: str,
    K: int,
    split_id: int | None,
    root_dir: str = "data",
) -> bool:
    """
    Load a best.pt checkpoint, run a forward pass, and save layer_probs.npz.

    Keys saved:  val_probs_{k}  and  test_probs_{k}  for k = 0 .. K.

    Returns True on success, False on skip/error.
    """
    out_path       = checkpoint_path.parent / "layer_probs.npz"
    out_train_path = checkpoint_path.parent / "layer_probs_train.npz"
    both_exist = out_path.exists() and out_train_path.exists()
    if both_exist:
        print(f"  [SKIP] already exists: {out_path.name} + layer_probs_train.npz")
        return True

    if not checkpoint_path.exists():
        print(f"  [MISS] checkpoint not found: {checkpoint_path}")
        return False

    # ---- Load dataset -------------------------------------------------------
    data, num_classes, _ = load_ds(
        dataset,
        root_dir=root_dir,
        planetoid_normalize=False,
        planetoid_split="public",
    )

    # Select correct split mask for hetero datasets
    if split_id is not None and data.train_mask.dim() > 1:
        data = data.clone()
        data.train_mask = data.train_mask[:, split_id]
        data.val_mask   = data.val_mask[:, split_id]
        data.test_mask  = data.test_mask[:, split_id]

    # ---- Load model from checkpoint ----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hp = checkpoint.get("hyperparams", {})
    eval_config = {**vars(cfg), **hp}

    model = build_model(model_name, data, num_classes, K, eval_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data = data.to(device)

    # ---- Forward pass through all classifier heads -------------------------
    _, layer_probs = model.forward_with_classifier_head(data)

    val_mask   = data.val_mask.cpu().numpy().astype(bool)
    test_mask  = data.test_mask.cpu().numpy().astype(bool)
    train_mask = data.train_mask.cpu().numpy().astype(bool)

    npz_dict   = {}
    train_dict = {}
    for k, probs_k in enumerate(layer_probs):
        probs_np = probs_k.cpu().numpy()          # [N, num_classes]
        npz_dict[f"val_probs_{k}"]     = probs_np[val_mask]
        npz_dict[f"test_probs_{k}"]    = probs_np[test_mask]
        train_dict[f"train_probs_{k}"] = probs_np[train_mask]

    rel = checkpoint_path.parent.relative_to(Path(cfg.classifier_heads_dir).parent)
    if not out_path.exists():
        np.savez(out_path, **npz_dict)
        print(f"  [SAVED]       {rel / 'layer_probs.npz'}")
    else:
        print(f"  [SKIP val]    {rel / 'layer_probs.npz'} already exists")

    if not out_train_path.exists():
        np.savez(out_train_path, **train_dict)
        print(f"  [SAVED train] {rel / 'layer_probs_train.npz'}")
    else:
        print(f"  [SKIP train]  {rel / 'layer_probs_train.npz'} already exists")

    return True


def main():
    global args  # used inside generate_layer_probs_for_checkpoint

    parser = argparse.ArgumentParser(
        description="Generate layer_probs.npz from best.pt checkpoints"
    )
    parser.add_argument("--best-hyperparams-path", required=True,
                        help="Path to best hyperparams CSV (e.g. results/best_hyperparams_GCN_per_layer.csv)")
    parser.add_argument("--classifier-heads-dir", type=str, default=None,
                        help="Override cfg.classifier_heads_dir (e.g. D:/GCN_eval/classifier_heads)")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--root-dir", type=str, default="data")
    parser.add_argument("--loss-filter", type=str, default=None,
                        help="If set, only process rows with this loss_type (e.g. ce_only)")
    parser.add_argument("--seeds", type=str, default="all",
                        help="Comma-separated seeds or 'all' (default: all from config)")
    parser.add_argument("--split-id", type=int, default=0,
                        help="Split index for heterophilous datasets (default: 0)")
    args = parser.parse_args()

    if args.classifier_heads_dir:
        cfg.classifier_heads_dir = args.classifier_heads_dir

    # ---- Seeds ---------------------------------------------------------------
    if args.seeds.lower() == "all":
        seeds = cfg.seeds
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    # ---- Load hyperparams CSV -----------------------------------------------
    hp_df = pd.read_csv(args.best_hyperparams_path)
    if args.loss_filter:
        hp_df = hp_df[hp_df["loss_type"] == args.loss_filter]

    print(f"\nLoaded {len(hp_df)} rows from {args.best_hyperparams_path}")
    print(f"Seeds: {seeds}  |  Model: {args.model}  |  split_id (hetero): {args.split_id}\n")

    n_ok = 0
    n_fail = 0

    for _, row in hp_df.iterrows():
        dataset    = row["dataset"]
        loss_type  = row["loss_type"]
        K          = int(row["K"])
        config_row = row.to_dict()

        # Normalise row dict: map lowercase CSV keys → what build_loss_dir expects,
        # and convert NaN → None (pd.isna check) so entropy_floor=NaN doesn't
        # produce 'floornan' in the path.
        import math
        def _nan_to_none(v):
            try:
                return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
            except Exception:
                return v

        norm = {k: _nan_to_none(v) for k, v in config_row.items()}
        norm['lambda_R'] = norm.get('lambda_r', norm.get('lambda_R', 1.0)) or 1.0
        norm['R_mode']   = norm.get('R_mode', norm.get('r_mode', 'smooth')) or 'smooth'
        norm['entropy_floor'] = norm.get('entropy_floor')  # already None if NaN

        loss_dir = build_loss_dir(loss_type, norm)

        # Hetero datasets use split_id; homo datasets use no split
        split_id = args.split_id if dataset in HETERO_DATASETS else None

        for seed in seeds:
            base = (
                Path(cfg.classifier_heads_dir)
                / loss_dir / dataset / args.model
                / f"seed_{seed}" / f"K_{K}"
            )
            if split_id is not None:
                base = base / f"split_{split_id}"
            ckpt = base / "best.pt"

            print(f"{dataset}/{args.model}/K={K}/seed={seed}/split={split_id}  [{loss_dir}]")
            ok = generate_layer_probs_for_checkpoint(
                ckpt, dataset, args.model, K, split_id, args.root_dir
            )
            if ok:
                n_ok += 1
            else:
                n_fail += 1

    print(f"\nDone.  {n_ok} generated / skipped,  {n_fail} missing checkpoints.")


if __name__ == "__main__":
    main()
