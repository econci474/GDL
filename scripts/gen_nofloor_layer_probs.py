"""Generate layer_probs.npz + layer_probs_train.npz for the no-floor configs."""
import sys, math, numpy as np, torch
from pathlib import Path

sys.path.insert(0, '.')
import config as cfg
from src.evaluate_final import build_model
from src.datasets import load_dataset

HEADS_DIR = Path('D:/GCN_eval/classifier_heads')
K = 8
SEEDS = [0, 1, 2]
HETERO = {'Roman-empire', 'Squirrel'}

# Only the configs that need generating
TARGETS = [
    ('Cora',         'ce_plus_R_R1.0_smooth_band-1.0to0.0',  None),
    ('Cora',         'ce_plus_R_R10.0_smooth_band-1.0to0.0', None),
    ('Cora',         'ce_plus_R_R1.0_smooth_band-1.5to0.2',  None),
    ('PubMed',       'ce_plus_R_R1.0_smooth_band-1.0to0.0',  None),
    ('PubMed',       'ce_plus_R_R10.0_smooth_band-1.0to0.0', None),
    ('PubMed',       'ce_plus_R_R1.0_smooth_band-1.5to0.2',  None),
    ('Roman-empire', 'ce_plus_R_R1.0_smooth_band-1.0to0.0',  0),
    ('Roman-empire', 'ce_plus_R_R10.0_smooth_band-1.0to0.0', 0),
    ('Roman-empire', 'ce_plus_R_R1.0_smooth_band-1.5to0.2',  0),
    ('Squirrel',     'ce_plus_R_R1.0_smooth_band-1.0to0.0',  0),
    ('Squirrel',     'ce_plus_R_R10.0_smooth_band-1.0to0.0', 0),
    ('Squirrel',     'ce_plus_R_R1.0_smooth_band-1.5to0.2',  0),
]

# Cache loaded datasets
_ds_cache = {}

def get_data(dataset):
    if dataset not in _ds_cache:
        data_obj, nc, _ = load_dataset(dataset, root_dir='data',
                                        planetoid_normalize=False,
                                        planetoid_split='public')
        _ds_cache[dataset] = (data_obj, nc)
    return _ds_cache[dataset]

for dataset, loss_type, split_id in TARGETS:
    print(f'\n[{dataset}] {loss_type}')
    data_obj, num_classes = get_data(dataset)

    for seed in SEEDS:
        base = HEADS_DIR / loss_type / dataset / 'GCN' / f'seed_{seed}' / f'K_{K}'
        if split_id is not None:
            base = base / f'split_{split_id}'
        ckpt      = base / 'best.pt'
        out_val   = base / 'layer_probs.npz'
        out_train = base / 'layer_probs_train.npz'

        if out_val.exists() and out_train.exists():
            print(f'  seed={seed}: SKIP (both exist)')
            continue
        if not ckpt.exists():
            print(f'  seed={seed}: MISS checkpoint')
            continue

        # Prepare data with correct mask
        d = data_obj.clone()
        if split_id is not None and d.val_mask.dim() == 2:
            d.train_mask = d.train_mask[:, split_id]
            d.val_mask   = d.val_mask[:,   split_id]
            d.test_mask  = d.test_mask[:,  split_id]

        vm  = d.val_mask.numpy().astype(bool)
        tm  = d.test_mask.numpy().astype(bool)
        trm = d.train_mask.numpy().astype(bool)

        saved = torch.load(ckpt, map_location='cpu', weights_only=False)
        hp    = saved.get('hyperparams', {})
        ecfg  = {**vars(cfg), **hp}
        model = build_model('GCN', d, num_classes, K, ecfg)
        model.load_state_dict(saved['model_state_dict'])
        model.eval()

        with torch.no_grad():
            _, layer_probs = model.forward_with_classifier_head(d)

        npz, trn = {}, {}
        for k, p in enumerate(layer_probs):
            pn = p.numpy()
            npz[f'val_probs_{k}']   = pn[vm]
            npz[f'test_probs_{k}']  = pn[tm]
            trn[f'train_probs_{k}'] = pn[trm]

        if not out_val.exists():
            np.savez(out_val, **npz);   print(f'  seed={seed}: SAVED val')
        if not out_train.exists():
            np.savez(out_train, **trn); print(f'  seed={seed}: SAVED train')

print('\nDone.')
