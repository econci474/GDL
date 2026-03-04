"""
make_r_only_with_ce_table.py
============================
Like make_summary_table.py --r-only, but with a CE only baseline column
prepended to each dataset group.  Significance markers (* p<0.05, paired
t-test vs CE only) are added to R_only cells.

Layout per dataset (3 sub-columns):
  [CE only | R only (-1, 0) | R only (-1.5, 0.25)]

CE only data is loaded from the standard final_results_{MODEL}_per_layer.csv.
R only data comes from the two R-only CSVs in --r-only-dir.

Usage (local):
    python scripts/make_r_only_with_ce_table.py \\
        --model GCN \\
        --ce-dir results/comparison_tables \\
        --r-only-dir tables_R_only \\
        --out-dir tables_R_only

Usage (Colab):
    !python scripts/make_r_only_with_ce_table.py \\
        --model GAT \\
        --ce-dir /content/drive/MyDrive/GDL/sweep_results \\
        --r-only-dir /content/drive/MyDrive/GDL/tables_R_only \\
        --out-dir /content/drive/MyDrive/GDL/tables_R_only
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Args ------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="R-only table with CE only baseline column and significance markers"
)
parser.add_argument("--model",       type=str, default="GCN",
                    help="Model name: GCN or GAT")
parser.add_argument("--ce-dir",      type=str, default="results/comparison_tables",
                    help="Directory containing final_results_{MODEL}_per_layer.csv")
parser.add_argument("--r-only-dir",  type=str, default="tables_R_only",
                    help="Directory containing final_results_R_only_{MODEL}_band*.csv")
parser.add_argument("--out-dir",     type=str, default=None,
                    help="Output directory for PNG (default: same as --r-only-dir)")
args = parser.parse_args()

MODEL     = args.model
CE_DIR    = Path(args.ce_dir)
RONLY_DIR = Path(args.r_only_dir)
OUT_DIR   = Path(args.out_dir) if args.out_dir else RONLY_DIR

# -- CSV paths -------------------------------------------------------------
# Always use per-layer CE csv so K-specific best hyperparams are used
CSV_CE     = CE_DIR    / f"final_results_{MODEL}_per_layer.csv"
CSV_BAND10 = RONLY_DIR / f"final_results_R_only_{MODEL}_band-1.0to0.0.csv"
CSV_BAND15 = RONLY_DIR / f"final_results_R_only_{MODEL}_band-1.5to0.25.csv"
OUT_PATH   = OUT_DIR   / f"test_acc_summary_table_R_only_with_CE_{MODEL}.png"

DATASETS = ["Cora", "PubMed", "Roman-empire", "Squirrel"]
K_VALUES = list(range(2, 9))   # R_only starts at K=2

CONFIGS = [
    ("ce_only", None,  None,  "CE only"),
    ("R_only",  -1.0,  0.0,   "R only\n(-1, 0)"),
    ("R_only",  -1.5,  0.25,  "R only\n(-1.5, 0.25)"),
]

# -- Load & aggregate -------------------------------------------------------
def load_and_agg(path):
    if path is None or not path.exists():
        print(f"[WARN] Not found: {path}")
        return None
    df = pd.read_csv(path)
    df["test_acc_pct"] = df["test_acc"] * 100
    grp_cols = ["dataset", "loss_type", "K"]
    if "band_lower" in df.columns:
        grp_cols += ["band_lower", "band_upper"]
    agg = (
        df.groupby(grp_cols, dropna=False)["test_acc_pct"]
        .agg(["mean", "std"])
        .reset_index()
    )
    if "band_lower" not in agg.columns:
        agg["band_lower"] = None
        agg["band_upper"] = None
    agg["cell"] = agg.apply(lambda r: f"{r['mean']:.1f} ± {r['std']:.1f}", axis=1)
    return agg


def load_raw(path):
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def get_per_seed_acc(raw_df, ds, loss_type, K):
    """Mean-over-splits per-seed test accuracy, sorted by seed."""
    if raw_df is None:
        return None
    mask = ((raw_df["dataset"] == ds) &
            (raw_df["loss_type"] == loss_type) &
            (raw_df["K"] == K))
    rows = raw_df[mask]
    if rows.empty:
        return None
    return rows.groupby("seed")["test_acc"].mean().sort_index().values


def is_significant(baseline_accs, treatment_accs, alpha=0.05):
    """Paired t-test: True if treatment significantly differs from baseline."""
    if baseline_accs is None or treatment_accs is None:
        return False
    n = min(len(baseline_accs), len(treatment_accs))
    if n < 2:
        return False
    try:
        _, p = ttest_rel(treatment_accs[:n], baseline_accs[:n])
        return bool(p < alpha)
    except Exception:
        return False


agg_ce     = load_and_agg(CSV_CE)
agg_band10 = load_and_agg(CSV_BAND10)
agg_band15 = load_and_agg(CSV_BAND15)

raw_ce     = load_raw(CSV_CE)
raw_band10 = load_raw(CSV_BAND10)
raw_band15 = load_raw(CSV_BAND15)


def lookup(ds, loss_type, band_lower, band_upper, K):
    """Return formatted cell string with optional * significance marker."""
    if loss_type == "ce_only":
        # Baseline — no significance marker
        if agg_ce is None:
            return "-"
        src  = agg_ce
        mask = (src["dataset"] == ds) & (src["loss_type"] == "ce_only") & (src["K"] == K)
        hits = src[mask]
        return hits["cell"].values[0] if len(hits) else "-"

    elif band_lower == -1.0 and band_upper == 0.0:
        if agg_band10 is None:
            return "-"
        src     = agg_band10
        raw_src = raw_band10
        mask    = ((src["dataset"] == ds) &
                   (src["loss_type"] == loss_type) &
                   (src["K"] == K))
    else:  # band (-1.5, 0.25)
        if agg_band15 is None:
            return "-"
        src     = agg_band15
        raw_src = raw_band15
        mask    = ((src["dataset"] == ds) &
                   (src["loss_type"] == loss_type) &
                   (src["K"] == K))

    hits = src[mask]
    if not len(hits):
        return "-"
    cell = hits["cell"].values[0]

    # Significance test vs CE only baseline
    baseline  = get_per_seed_acc(raw_ce,  ds, "ce_only",  K)
    treatment = get_per_seed_acc(raw_src, ds, loss_type,  K)
    if is_significant(baseline, treatment):
        cell = cell + "*"
    return cell


# -- Build table arrays ----------------------------------------------------
col_headers = [(ds, conf[3]) for ds in DATASETS for conf in CONFIGS]
col_keys    = [(ds, conf[0], conf[1], conf[2]) for ds in DATASETS for conf in CONFIGS]

row_labels = [f"K = {k}" for k in K_VALUES]
table_data = []
for k in K_VALUES:
    row = [lookup(ds, lt, bl, bu, k) for ds, lt, bl, bu in col_keys]
    table_data.append(row)

n_rows      = len(K_VALUES)
n_data_cols = len(col_headers)
n_cfg       = len(CONFIGS)

# -- Layout (inches) -------------------------------------------------------
plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 10})

ROW_LABEL_W = 0.9
COL_W       = 1.35
ROW_H       = 0.34
HEAD_H      = 0.62

fig_w = ROW_LABEL_W + n_data_cols * COL_W + 0.1
fig_h = HEAD_H + n_rows * ROW_H + 0.15

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")


def col_x_center(j):
    return ROW_LABEL_W + j * COL_W + COL_W / 2


def row_y_center(i):
    return fig_h - HEAD_H - i * ROW_H - ROW_H / 2


# -- Horizontal rules (booktabs) -------------------------------------------
RULE_TOP = fig_h
RULE_MID = fig_h - HEAD_H
RULE_BOT = fig_h - HEAD_H - n_rows * ROW_H


def hrule(y, lw):
    ax.plot([0, fig_w], [y, y], color="black", linewidth=lw,
            solid_capstyle="butt", transform=ax.transData, clip_on=False)


hrule(RULE_TOP, 1.4)
hrule(RULE_MID, 0.8)
hrule(RULE_BOT, 1.4)

# -- Column headers --------------------------------------------------------
for ds_idx, ds in enumerate(DATASETS):
    j_start = ds_idx * n_cfg
    j_mid   = j_start + (n_cfg - 1) / 2
    cx_ds   = ROW_LABEL_W + j_mid * COL_W + COL_W / 2
    cy_top  = RULE_MID + HEAD_H * 0.72
    ax.text(cx_ds, cy_top, ds, ha="center", va="center",
            fontsize=9.5, fontweight="bold")

    # Thin vertical separator between dataset groups
    if ds_idx > 0:
        xv = ROW_LABEL_W + ds_idx * n_cfg * COL_W
        ax.plot([xv, xv], [RULE_BOT, RULE_TOP], color="black",
                linewidth=0.5, linestyle=(0, (4, 4)), alpha=0.6)

    # Sub-column labels; shade CE only column lightly
    for conf_idx, (_, _, _, conf_label) in enumerate(CONFIGS):
        j  = j_start + conf_idx
        cx = col_x_center(j)
        cy = RULE_MID + HEAD_H * 0.28

        if conf_idx == 0:  # CE only — light grey background
            ax.add_patch(plt.Rectangle(
                (ROW_LABEL_W + j * COL_W, RULE_BOT),
                COL_W, RULE_TOP - RULE_BOT,
                color="#f0f0f0", zorder=0, transform=ax.transData
            ))

        ax.text(cx, cy, conf_label, ha="center", va="center",
                fontsize=7.8, fontstyle="italic", multialignment="center")

    # Thin vertical rule between CE only and first R only per dataset group
    xv = ROW_LABEL_W + (ds_idx * n_cfg + 1) * COL_W
    ax.plot([xv, xv], [RULE_BOT, RULE_MID], color="black",
            linewidth=0.4, linestyle=(0, (3, 3)), alpha=0.5)

# -- Row labels + data cells -----------------------------------------------
for i, (k_label, row_cells) in enumerate(zip(row_labels, table_data)):
    yc = row_y_center(i)
    ax.text(0.08, yc, k_label, ha="left", va="center", fontsize=9.5)
    for j, val in enumerate(row_cells):
        ax.text(col_x_center(j), yc, val, ha="center", va="center",
                fontsize=8.5)

# -- Title -----------------------------------------------------------------
hetero_note = "hetero: 3 seeds × split 0, homo: 3 seeds"
ax.text(fig_w / 2, fig_h + 0.06,
        f"{MODEL} Test Accuracy (%), R only vs CE only baseline, "
        f"per-layer hyperparams, {hetero_note}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
        transform=ax.transData, clip_on=False)

# -- Footnote --------------------------------------------------------------
ax.text(0, RULE_BOT - 0.06,
        "* p < 0.05 vs CE only (paired t-test)",
        ha="left", va="top", fontsize=7, color="#444444",
        transform=ax.transData, clip_on=False)

# -- Save ------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved -> {OUT_PATH}")
