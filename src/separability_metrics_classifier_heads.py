"""Compute separability metrics from classifier head outputs.

This script computes AUROC and Cohen's d for error detection based on entropy,
directly from classifier head layer_probs.npz outputs.  Supports aggregation
across multiple seeds (--seed all) and accepts a custom --classifier-heads-dir.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    from torchmetrics import AUROC
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score as _sklearn_auroc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_auroc_torchmetrics(H, e):
    """AUROC for error detection using entropy as the score.

    Falls back to sklearn if torchmetrics is unavailable.
    H  : entropy scores (higher = more uncertain).
    e  : binary labels  (1 = error, 0 = correct).
    """
    # --- numpy arrays for sklearn path ---
    H_np = H.numpy() if not isinstance(H, np.ndarray) else H
    e_np = e.numpy() if not isinstance(e, np.ndarray) else e

    if len(np.unique(e_np)) < 2:
        return np.nan

    # torchmetrics path
    if TORCHMETRICS_AVAILABLE:
        try:
            H_t = torch.from_numpy(H_np).float()
            e_t = torch.from_numpy(e_np).long()
            auroc_fn = AUROC(task='binary')
            return auroc_fn(H_t, e_t).item()
        except Exception as ex:
            print(f"Warning: torchmetrics AUROC failed: {ex}")

    # sklearn fallback
    if SKLEARN_AVAILABLE:
        try:
            return float(_sklearn_auroc(e_np, H_np))
        except Exception as ex:
            print(f"Warning: sklearn AUROC failed: {ex}")

    return np.nan


def compute_cohens_d(H_wrong, H_correct):
    """Cohen's d = (mean_wrong - mean_correct) / pooled_std."""
    n_w, n_c = len(H_wrong), len(H_correct)
    if n_w < 2 or n_c < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((n_w - 1) * np.var(H_wrong, ddof=1) + (n_c - 1) * np.var(H_correct, ddof=1))
        / (n_w + n_c - 2)
    )
    if pooled_std == 0:
        return np.nan
    return (np.mean(H_wrong) - np.mean(H_correct)) / pooled_std


def calc_entropy(probs):
    """Entropy per node."""
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

HETERO_DATASETS = {"Roman-empire", "Squirrel"}


def _loss_dir_candidates(loss_dir: str) -> list:
    """Return all plausible directory names for a given loss_dir string.

    Handles historical naming inconsistencies:
      - default band (-1.0, 0.0) may or may not have an explicit suffix
      - _floor<val> segment may or may not be present
      - _perclass segment may or may not be present
    Tries all combinations, most-preferred first.
    """
    import re
    m = re.match(
        r'ce_plus_R_R([\d.]+)_smooth'
        r'(?:_(floor[\d.]+))?'
        r'(?:_(perclass))?'
        r'(?:_band([-\d.]+)to([-\d.]+))?$',
        loss_dir
    )
    if not m:
        return [loss_dir]
    lr         = m.group(1)
    floor_part = m.group(2)  # e.g. 'floor0.10' or None
    pc_part    = m.group(3)  # 'perclass' or None
    bl         = float(m.group(4)) if m.group(4) else -1.0
    bu         = float(m.group(5)) if m.group(5) else  0.0

    bare = f'ce_plus_R_R{lr}_smooth'
    bases = [bare]
    if floor_part or pc_part:
        decorated = bare
        if floor_part:
            decorated = f'{decorated}_{floor_part}'
        if pc_part:
            decorated = f'{decorated}_{pc_part}'
        bases = [decorated, bare]  # prefer decorated; fall back to bare

    if bl == -1.0 and bu == 0.0:  # default band
        candidates = []
        for b in bases:
            candidates += [b, f'{b}_band{bl:.1f}to{bu:.1f}', f'{b}_band{bl:.2f}to{bu:.2f}']
    else:
        candidates = []
        for b in bases:
            candidates += [f'{b}_band{bl:.1f}to{bu:.1f}', f'{b}_band{bl:.2f}to{bu:.2f}', b]

    # Deduplicate while preserving order
    seen, deduped = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _find_probs_path(loss_type, dataset, model, K, seed, split_id,
                     classifier_heads_dir=None):
    """Return the first existing layer_probs.npz path, trying all candidate dirs."""
    heads = Path(classifier_heads_dir) if classifier_heads_dir else Path(cfg.classifier_heads_dir)
    for cand in _loss_dir_candidates(loss_type):
        base = heads / cand / dataset / model / f'seed_{seed}' / f'K_{K}'
        if split_id is not None:
            base = base / f'split_{split_id}'
        p = base / 'layer_probs.npz'
        if p.exists():
            return p
    # Return primary path for the error message
    base = heads / _loss_dir_candidates(loss_type)[0] / dataset / model / f'seed_{seed}' / f'K_{K}'
    if split_id is not None:
        base = base / f'split_{split_id}'
    return base / 'layer_probs.npz'

# Mapping of known directory-name truncations → human-readable labels.
# (build_loss_dir used :.1f which rounds 0.25 → "0.2" via banker's rounding)
_BAND_LABEL_FIXES = {
    "band-1.5to0.2":  "band-1.5to0.25",
    "band-1.0to0.0":  "band-1.0to0.0",   # already correct
}


def _pretty_loss_label(loss_type: str) -> str:
    """Return a human-readable version of a loss_type directory name."""
    label = loss_type
    for old, new in _BAND_LABEL_FIXES.items():
        label = label.replace(old, new)
    return label


def compute_separability_from_classifier_outputs(dataset, model, K, seed, loss_type,
                                                  split='val',
                                                  classifier_heads_dir=None,
                                                  split_id=None):
    """
    Compute separability metrics from layer_probs.npz for a single (seed, K).

    Returns (df_aggregate, df_per_class, num_classes).
    """
    from src.datasets import load_dataset as load_ds
    data_obj, num_classes, _ = load_ds(
        dataset, root_dir='data', planetoid_normalize=False, planetoid_split='public'
    )

    labels = data_obj.y.numpy()
    # For heterophilous datasets the masks are 2-D [N x 10]; select the right split column
    val_mask_raw   = data_obj.val_mask
    test_mask_raw  = data_obj.test_mask
    train_mask_raw = data_obj.train_mask
    if split_id is not None and val_mask_raw.dim() == 2:
        vm = val_mask_raw[:, split_id].numpy()
        tm = test_mask_raw[:, split_id].numpy()
        trm = train_mask_raw[:, split_id].numpy() if train_mask_raw.dim() == 2 else train_mask_raw.numpy()
    else:
        vm  = val_mask_raw.numpy()   if val_mask_raw.dim()   == 1 else val_mask_raw[:,   0].numpy()
        tm  = test_mask_raw.numpy()  if test_mask_raw.dim()  == 1 else test_mask_raw[:,  0].numpy()
        trm = train_mask_raw.numpy() if train_mask_raw.dim() == 1 else train_mask_raw[:, 0].numpy()
    if split == 'val':
        split_mask = vm
    elif split == 'test':
        split_mask = tm
    else:  # 'train'
        split_mask = trm
    labels_split = labels[split_mask]

    probs_path = _find_probs_path(loss_type, dataset, model, K, seed,
                                   split_id, classifier_heads_dir)
    if not probs_path.exists():
        raise FileNotFoundError(f"Classifier outputs not found: {probs_path}")

    probs_data = np.load(probs_path)
    results, per_class_metrics = [], []

    for k in range(K + 1):
        probs_k   = probs_data[f'{split}_probs_{k}']
        preds_k   = probs_k.argmax(axis=1)
        correct   = (preds_k == labels_split).astype(int)
        entropy_k = calc_entropy(probs_k)
        H_correct = entropy_k[correct == 1]
        H_wrong   = entropy_k[correct == 0]

        per_class_data = {'k': k}
        for c in range(num_classes):
            mask = (labels_split == c)
            n    = mask.sum()
            if n > 0:
                cp = probs_k[mask]; cc = (preds_k[mask] == c).astype(int)
                per_class_data[f'class_{c}_accuracy']  = cc.mean()
                per_class_data[f'class_{c}_entropy']   = calc_entropy(cp).mean()
                per_class_data[f'class_{c}_n_total']   = n
                per_class_data[f'class_{c}_n_correct'] = cc.sum()
                per_class_data[f'class_{c}_n_wrong']   = n - cc.sum()
            else:
                for col in ['accuracy', 'entropy']:
                    per_class_data[f'class_{c}_{col}'] = np.nan
                for col in ['n_total', 'n_correct', 'n_wrong']:
                    per_class_data[f'class_{c}_{col}'] = 0

        per_class_metrics.append(per_class_data)
        results.append({
            'k':                    k,
            'accuracy':             correct.mean(),
            'auroc':                compute_auroc_torchmetrics(entropy_k, 1 - correct),
            'cohens_d':             compute_cohens_d(H_wrong, H_correct),
            'mean_entropy':         entropy_k.mean(),
            'mean_entropy_correct': H_correct.mean() if len(H_correct) > 0 else np.nan,
            'mean_entropy_wrong':   H_wrong.mean()   if len(H_wrong)   > 0 else np.nan,
            'n_correct': len(H_correct),
            'n_wrong':   len(H_wrong),
        })

    return pd.DataFrame(results), pd.DataFrame(per_class_metrics), num_classes


# ---------------------------------------------------------------------------
# Plotting  — layout matches separability_metrics.py exactly
# ---------------------------------------------------------------------------

def _plot_mean_std(ax, kv, m, s, color, label, marker='o'):
    ax.plot(kv, m, f'{marker}-', label=label, color=color, linewidth=2)
    ax.fill_between(kv, m - s, m + s, alpha=0.2, color=color)


def plot_separability_vs_k(df, df_per_class, num_classes, dataset, model, K,
                            seeds_used, loss_type, output_dir):
    """
    6-panel figure — same layout as separability_metrics.py:
        [0,0] AUROC              [0,1] Cohen's d
        [1,0] Validation Acc     [1,1] Mean Entropy correct vs incorrect
        [2,0] Per-Class Acc      [2,1] Per-Class Entropy
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    colors = plt.cm.tab10(np.arange(max(num_classes, 10)))

    layers   = df['k'].values
    have_agg = 'auroc_mean' in df.columns     # True when multiple seeds were aggregated

    def get_m_s(col):
        if have_agg:
            return df[f'{col}_mean'].values, df[f'{col}_std'].fillna(0).values
        return df[col].values, np.zeros(len(df))

    seed_lbl = (f"Mean (n={len(seeds_used)} seeds)" if len(seeds_used) > 1
                else f"seed {seeds_used[0]}")

    # [0,0]  AUROC
    ax = axes[0, 0]
    _plot_mean_std(ax, layers, *get_m_s('auroc'), 'tab:blue', seed_lbl)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set(xlabel='Depth k', ylabel='AUROC',
           title='Error Detection AUROC vs Depth', ylim=[0, 1])
    ax.set_xlim(layers[0] - 0.3, layers[-1] + 0.3)  # prevent NaN-driven autoscale
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # [0,1]  Cohen's d
    ax = axes[0, 1]
    _plot_mean_std(ax, layers, *get_m_s('cohens_d'), 'tab:orange', seed_lbl)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set(xlabel='Depth k', ylabel="Cohen's d",
           title="Entropy Separability (Cohen's d) vs Depth")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # [1,0]  Validation Accuracy
    ax = axes[1, 0]
    _plot_mean_std(ax, layers, *get_m_s('accuracy'), 'tab:green', seed_lbl)
    ax.set(xlabel='Depth k', ylabel='Accuracy',
           title='Validation Accuracy vs Depth', ylim=[0, 1])
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # [1,1]  Mean Entropy correct vs incorrect
    ax = axes[1, 1]
    _plot_mean_std(ax, layers, *get_m_s('mean_entropy_correct'), 'tab:blue', 'Correct', 'o')
    _plot_mean_std(ax, layers, *get_m_s('mean_entropy_wrong'),   'tab:red',  'Incorrect', 's')
    ax.set(xlabel='Depth k', ylabel='Mean Entropy',
           title='Mean Entropy: Correct vs Incorrect')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # Per-class panels
    if df_per_class is not None:
        pc_k      = df_per_class['k'].values
        have_pc   = 'class_0_accuracy_mean' in df_per_class.columns
        have_pc_r = (not have_pc) and 'class_0_accuracy' in df_per_class.columns

        n_by_c      = {}
        n_correct_c = {}
        n_wrong_c   = {}
        for c in range(num_classes):
            col  = f'class_{c}_n_total'
            colm = f'class_{c}_n_total_mean' if have_pc else col
            if colm in df_per_class.columns:
                n_by_c[c] = int(df_per_class[colm].iloc[-1])
            elif col in df_per_class.columns:
                n_by_c[c] = int(df_per_class[col].iloc[-1])
            else:
                n_by_c[c] = 0

            # n_correct / n_wrong at the last depth row (k=K)
            for tag, store in [('n_correct', n_correct_c), ('n_wrong', n_wrong_c)]:
                raw_col  = f'class_{c}_{tag}'
                mean_col = f'class_{c}_{tag}_mean' if have_pc else raw_col
                if mean_col in df_per_class.columns:
                    store[c] = int(round(df_per_class[mean_col].iloc[-1]))
                elif raw_col in df_per_class.columns:
                    store[c] = int(df_per_class[raw_col].iloc[-1])
                else:
                    store[c] = 0

        sc = sorted(range(num_classes), key=lambda c: n_by_c.get(c, 0))

        for panel_col, metric in [(0, 'accuracy'), (1, 'entropy')]:
            ax = axes[2, panel_col]
            for c in sc:
                col_name = f'class_{c}_{metric}'
                if have_pc:
                    m = df_per_class[f'{col_name}_mean'].values
                    s = df_per_class[f'{col_name}_std'].fillna(0).values
                elif have_pc_r:
                    m = df_per_class[col_name].values
                    s = np.zeros_like(m)
                else:
                    continue
                n        = n_by_c.get(c, 0)
                n_ok     = n_correct_c.get(c, 0)
                n_bad    = n_wrong_c.get(c, 0)
                lbl = f'C{c} (n={n}: {n_ok}[OK]/{n_bad}[FAIL])'
                ax.plot(pc_k, m, 'o-', linewidth=1.5, markersize=5,
                        label=lbl, color=colors[c])
                ax.fill_between(pc_k, m - s, m + s, alpha=0.12, color=colors[c])

            if metric == 'accuracy':
                ax.set(xlabel='Depth k', ylabel='Per-Class Accuracy',
                       title='Per-Class Validation Accuracy by Depth', ylim=[0, 1.05])
                ax.grid(True, alpha=0.3)
            else:
                ax.set(xlabel='Depth k', ylabel='Per-Class Mean Entropy',
                       title='Per-Class Mean Entropy by Depth')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9, ncol=1, loc='best')
    else:
        axes[2, 0].set_visible(False)
        axes[2, 1].set_visible(False)

    seeds_str = ", ".join(str(s) for s in seeds_used)
    if len(seeds_used) == 1:
        subtitle = f"Seed {seeds_used[0]}"
    else:
        subtitle = f"Mean +/- Std across Seeds ({seeds_str})"
    label = _pretty_loss_label(loss_type)
    fig.suptitle(
        f"Classifier Head Analysis: {model}, {dataset}, {label}\n{subtitle}",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    out = output_dir / f'{dataset}_{model}_k{K}_seed_all_{loss_type}_separability_vs_k_per_class.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute separability metrics from classifier head outputs'
    )
    parser.add_argument('--dataset',              type=str, required=True)
    parser.add_argument('--model',                type=str, default='GCN')
    parser.add_argument('--K',                    type=int, default=8)
    parser.add_argument('--seed',                 type=str, default='0',
                        help='Seed value, comma-separated list, or "all"')
    parser.add_argument('--loss-type',            type=str, required=True,
                        help='Loss-type directory name (e.g. ce_only)')
    parser.add_argument('--split',                type=str, default='val',
                        choices=['val', 'test'])
    parser.add_argument('--classifier-heads-dir', type=str, default=None,
                        help='Override cfg.classifier_heads_dir '
                             '(e.g. D:/GCN_eval/classifier_heads)')
    parser.add_argument('--split-id', type=int, default=None,
                        help='Split index for heterophilous datasets. '
                             'If not set, auto-detected from dataset name.')
    args = parser.parse_args()

    if args.classifier_heads_dir:
        cfg.classifier_heads_dir = args.classifier_heads_dir

    seeds = (list(cfg.seeds) if args.seed.lower() == 'all'
             else [int(s) for s in args.seed.split(',')])

    print(f'\n{"="*60}')
    print(f'Separability Metrics: {args.model} on {args.dataset}')
    print(f'  K={args.K}  seeds={seeds}  loss_type={args.loss_type}  split={args.split}')
    print(f'{"="*60}\n')

    # Auto-detect split_id for heterophilous datasets
    split_id = args.split_id
    if split_id is None and args.dataset in HETERO_DATASETS:
        split_id = 0
        print(f'  [INFO] Heterophilous dataset detected — using split_id={split_id}')

    all_dfs, all_pc_dfs, num_classes = [], [], None

    for seed in seeds:
        print(f'  Processing seed {seed} ...')
        try:
            df, df_pc, nc = compute_separability_from_classifier_outputs(
                args.dataset, args.model, args.K, seed,
                args.loss_type, args.split, args.classifier_heads_dir,
                split_id=split_id
            )
            df['seed'] = seed
            df_pc['seed'] = seed
            all_dfs.append(df)
            all_pc_dfs.append(df_pc)
            num_classes = nc
        except FileNotFoundError as exc:
            print(f'  [SKIP] {exc}')

    if not all_dfs:
        print('No data found — check that layer_probs.npz files exist.')
        return

    output_dir = Path(cfg.figures_dir) / args.dataset / args.model / f'K_{args.K}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate across seeds if multiple
    if len(all_dfs) == 1:
        df_plot    = all_dfs[0]
        df_pc_plot = all_pc_dfs[0]
    else:
        combined  = pd.concat(all_dfs, ignore_index=True)
        agg_cols  = ['auroc', 'cohens_d', 'accuracy',
                     'mean_entropy_correct', 'mean_entropy_wrong']
        agg = combined.groupby('k')[agg_cols].agg(['mean', 'std']).reset_index()
        agg.columns = ['k'] + [f'{c}_{s}' for c, s in agg.columns[1:]]
        df_plot = agg

        comb_pc   = pd.concat(all_pc_dfs, ignore_index=True)
        pc_acc    = [c for c in comb_pc.columns
                     if c.startswith('class_') and c.endswith('_accuracy')]
        pc_ent    = [c for c in comb_pc.columns
                     if c.startswith('class_') and c.endswith('_entropy')]
        pc_n      = [c for c in comb_pc.columns
                     if c.startswith('class_') and c.endswith('_n_total')]
        pc_agg = comb_pc.groupby('k')[pc_acc + pc_ent].agg(['mean', 'std']).reset_index()
        pc_agg.columns = ['k'] + [f'{c}_{s}' for c, s in pc_agg.columns[1:]]
        for col in pc_n:
            pc_agg[col] = int(all_pc_dfs[0][col].iloc[0])
        df_pc_plot = pc_agg

    # Save CSV
    tables_dir = Path(cfg.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / (f'{args.dataset}_{args.model}_K{args.K}'
                              f'_seeds{"_".join(str(s) for s in seeds)}'
                              f'_{args.loss_type}_separability.csv')
    df_plot.to_csv(csv_path, index=False)
    print(f'\nCSV saved -> {csv_path}')

    plot_separability_vs_k(
        df_plot, df_pc_plot, num_classes,
        args.dataset, args.model, args.K,
        seeds, args.loss_type, output_dir
    )

    print(f'\n{"="*60}\nDone\n{"="*60}')


if __name__ == '__main__':
    main()
