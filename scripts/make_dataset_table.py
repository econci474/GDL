"""
make_dataset_table.py  -- Generate a publication-quality PNG table of dataset statistics.

Style: transposed (datasets as columns), minimal academic / booktabs style.
Outputs: results/figures/dataset_summary_table.png
"""

import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.datasets import load_dataset


# ------------------------------------------------------------------
# Adjusted homophily (Platonov et al., 2023)
# h_adj = (h - sum_k p_k^2) / (1 - sum_k p_k^2)
# ------------------------------------------------------------------
def adjusted_homophily(edge_index, y, num_nodes):
    src, dst = edge_index
    same = (y[src] == y[dst]).float().mean().item()
    num_classes = int(y.max().item()) + 1
    class_counts = torch.bincount(y, minlength=num_classes).float()
    p = class_counts / num_nodes
    sum_p2 = (p ** 2).sum().item()
    if abs(1 - sum_p2) < 1e-8:
        return 1.0
    return (same - sum_p2) / (1 - sum_p2)


# ------------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------------
DATASETS = [
    ('Cora',         'cora'),
    ('PubMed',       'pubmed'),
    ('Roman-empire', 'roman-empire'),
    ('Squirrel',     'squirrel'),
]

records = []
for display_name, name in DATASETS:
    print(f'Loading {display_name}...')
    data, num_classes, kind = load_dataset(name, root_dir='data')
    N = data.num_nodes
    tm = data.train_mask[:, 0] if data.train_mask.dim() > 1 else data.train_mask
    vm = data.val_mask[:, 0]   if data.val_mask.dim()   > 1 else data.val_mask
    xm = data.test_mask[:, 0]  if data.test_mask.dim()  > 1 else data.test_mask
    h_adj = adjusted_homophily(data.edge_index, data.y, N)
    records.append({
        'name':     display_name,
        'h_adj':    h_adj,
        'train':    int(tm.sum()),
        'val':      int(vm.sum()),
        'test':     int(xm.sum()),
        'classes':  num_classes,
    })

# Sort descending by adjusted homophily
records.sort(key=lambda r: r['h_adj'], reverse=True)
print('\nOrder:', [r['name'] for r in records])


# ------------------------------------------------------------------
# Build table data
# Row labels (left column) and values per dataset (remaining columns)
# ------------------------------------------------------------------
ROW_LABELS = ['Adj Hom', '#Nodes (Train)', '#Nodes (Val)', '#Nodes (Test)', '#Classes']

def fmt(rec, row):
    if row == 'Adj Hom':
        return f"{rec['h_adj']:.3f}"
    elif row == '#Nodes (Train)':
        return f"{rec['train']:,}"
    elif row == '#Nodes (Val)':
        return f"{rec['val']:,}"
    elif row == '#Nodes (Test)':
        return f"{rec['test']:,}"
    elif row == '#Classes':
        return str(rec['classes'])

# table[i][j] = cell text, where i=row index, j=col index (0=row label, 1..N=datasets)
col_headers = [r['name'] for r in records]
table = []
for row_label in ROW_LABELS:
    row = [row_label] + [fmt(r, row_label) for r in records]
    table.append(row)

n_rows = len(table)
n_cols = len(col_headers) + 1  # +1 for row label column


# ------------------------------------------------------------------
# Render with matplotlib in booktabs style
# ------------------------------------------------------------------
import matplotlib.font_manager as fm

# Computer Modern Roman for regular text (matplotlib built-in)
# STIXGeneral gives reliable bold serif on all platforms
plt.rcParams.update({
    'font.family': 'STIXGeneral',
    'font.size':   10,
})

# Column widths (inches): row-label col, then one per dataset
COL_W = [1.5] + [1.25] * len(col_headers)
ROW_H = 0.34   # row height in inches
HEAD_H = 0.42  # header row height

fig_w = sum(COL_W) + 0.1
fig_h = HEAD_H + n_rows * ROW_H + 0.1

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis('off')

# Coordinate helpers
def col_x(j):
    return sum(COL_W[:j])

def row_y(i):
    """y of bottom of row i (0=top data row, going downward)."""
    return fig_h - HEAD_H - (i + 1) * ROW_H

# ---- Horizontal rules (booktabs: toprule, midrule, bottomrule) ----
RULE_TOP    = fig_h
RULE_MID    = fig_h - HEAD_H
RULE_BOT    = fig_h - HEAD_H - n_rows * ROW_H

def hrule(y, lw):
    ax.plot([0, fig_w], [y, y], color='black', linewidth=lw,
            solid_capstyle='butt', transform=ax.transData, clip_on=False)

hrule(RULE_TOP, 1.4)  # toprule
hrule(RULE_MID, 0.8)  # midrule (after header)
hrule(RULE_BOT, 1.4)  # bottomrule

# ---- Column headers (dataset names) ----
for j, ds_name in enumerate(col_headers):
    cx = col_x(j + 1) + COL_W[j + 1] / 2
    cy = RULE_MID + HEAD_H / 2
    ax.text(cx, cy, ds_name, ha='center', va='center',
            fontsize=10, fontweight='bold')

# ---- Row label column is left-aligned, data cells centred ----
for i, row_cells in enumerate(table):
    y_center = row_y(i) + ROW_H / 2
    is_hom_row = (ROW_LABELS[i] == 'Adj Hom')

    # Row label (left-aligned)
    ax.text(col_x(0) + 0.08, y_center, row_cells[0],
            ha='left', va='center', fontsize=9.5)

    # Data cells
    for j, val in enumerate(row_cells[1:]):
        cx = col_x(j + 1) + COL_W[j + 1] / 2
        # fw = 'bold' if is_hom_row else 'normal'
        ax.text(cx, y_center, val, ha='center', va='center',
                fontsize=9.5, fontweight="normal")

out_dir = Path('results/figures')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'dataset_summary_table.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nSaved -> {out_path}')
