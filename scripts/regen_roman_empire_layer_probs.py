"""
Re-generate layer_probs.npz for Roman-empire K=8 with train_probs included.
Also generates for the no-floor configs requested for comparison.
Deletes existing npz files first to force re-generation.
"""
import subprocess, sys
from pathlib import Path

HEADS_DIR = Path("D:/GCN_eval/classifier_heads")
DATASET   = "Roman-empire"
MODEL     = "GCN"
K         = 8
SEEDS     = [0, 1, 2]
SPLIT_ID  = 0   # Roman-empire uses split_0

# All loss_type directories to regenerate for Roman-empire K=8
LOSS_TYPES = [
    "ce_only",
    "ce_plus_R_R1.0_smooth_floor0.10_band-1.0to0.0",
    "ce_plus_R_R10.0_smooth_floor0.10_band-1.0to0.0",
    "ce_plus_R_R1.0_smooth_band-1.0to0.0",          # no floor
    "ce_plus_R_R10.0_smooth_band-1.0to0.0",          # no floor  
    "ce_plus_R_R1.0_smooth_floor0.10_band-1.5to0.2",
    "ce_plus_R_R10.0_smooth_band-1.5to0.2",
]

import sys
sys.path.insert(0, ".")
import math, numpy as np, torch
import config as cfg

def delete_and_regen(loss_type, seed):
    base = HEADS_DIR / loss_type / DATASET / MODEL / f"seed_{seed}" / f"K_{K}" / f"split_{SPLIT_ID}"
    npz  = base / "layer_probs.npz"
    ckpt = base / "best.pt"

    if not ckpt.exists():
        print(f"  [NO CKPT] {loss_type}/seed_{seed}")
        return

    if npz.exists():
        npz.unlink()
        print(f"  [DELETED] {npz.relative_to(HEADS_DIR.parent)}")

    # Re-run generate_layer_probs.py indirectly: we directly load and save
    from src.datasets import load_dataset
    from src.models   import build_model

    data_obj, num_classes, _ = load_dataset(DATASET, root_dir="data",
                                             planetoid_normalize=False,
                                             planetoid_split="public")
    data = data_obj.to("cpu")

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = saved.get("model_state_dict", saved)
    model_cfg = saved.get("config", {})
    K_model = model_cfg.get("K", K)
    model_obj = build_model(model_name=MODEL, num_features=data.x.shape[1],
                             num_classes=num_classes, K=K_model,
                             **{k: v for k, v in model_cfg.items()
                                if k not in ("K","model","dataset","seed","split")})
    model_obj.load_state_dict(state)
    model_obj.eval()

    with torch.no_grad():
        _, layer_probs_list = model_obj.forward_with_classifier_head(data)

    val_mask   = data.val_mask
    test_mask  = data.test_mask
    train_mask = data.train_mask
    # 2-D masks for hetero datasets
    if val_mask.dim() == 2:
        vm  = val_mask[:,   SPLIT_ID].numpy().astype(bool)
        tm  = test_mask[:,  SPLIT_ID].numpy().astype(bool)
        trm = train_mask[:, SPLIT_ID].numpy().astype(bool) if train_mask.dim() == 2 else train_mask.numpy().astype(bool)
    else:
        vm  = val_mask.numpy().astype(bool)
        tm  = test_mask.numpy().astype(bool)
        trm = train_mask.numpy().astype(bool)

    npz_dict = {}
    for ki, probs_k in enumerate(layer_probs_list):
        p = probs_k.cpu().numpy()
        npz_dict[f"val_probs_{ki}"]   = p[vm]
        npz_dict[f"test_probs_{ki}"]  = p[tm]
        npz_dict[f"train_probs_{ki}"] = p[trm]

    base.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **npz_dict)
    print(f"  [SAVED] {npz.relative_to(HEADS_DIR.parent)}")


if __name__ == "__main__":
    for lt in LOSS_TYPES:
        print(f"\n=== {lt} ===")
        for seed in SEEDS:
            delete_and_regen(lt, seed)
    print("\nDone.")
