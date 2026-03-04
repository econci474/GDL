import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

# -- Args ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate booktabs-style test accuracy summary table")
parser.add_argument("--model", type=str, default="GCN",
                    help="Model name, e.g. GCN, GAT, GraphSAGE (default: GCN)")
parser.add_argument("--results-dir", type=str, default=None,
                    help="Directory containing final_results CSVs "
                         "(default: results/comparison_tables)")
parser.add_argument("--out-dir", type=str, default=None,
                    help="Directory for output PNG (default: same as results-dir)")
parser.add_argument("--per-layer", action="store_true", default=False,
                    help="Load per-layer results (final_results_{MODEL}_per_layer.csv)")
parser.add_argument("--r-only", action="store_true", default=False,
                    help="Generate R_only table (2 band configs, K>=2) instead of CE vs CE+R")
args = parser.parse_args()

MODEL = args.model
RESULTS_DIR = Path(args.results_dir) if args.results_dir else Path("results/comparison_tables")
OUT_DIR = Path(args.out_dir) if args.out_dir else RESULTS_DIR

# -- Config ----------------------------------------------------------------
if args.r_only:
    CSV_MAIN   = None
    CSV_BAND10 = RESULTS_DIR / f"final_results_R_only_{MODEL}_band-1.0to0.0.csv"
    CSV_BAND15 = RESULTS_DIR / f"final_results_R_only_{MODEL}_band-1.5to0.25.csv"
    OUT_PATH   = OUT_DIR / f"test_acc_summary_table_R_only_{MODEL}.png"
elif args.per_layer:
    CSV_MAIN   = RESULTS_DIR / f"final_results_{MODEL}_per_layer.csv"
    CSV_BAND10 = RESULTS_DIR / f"final_results_{MODEL}_band-1.0to0.0_per_layer.csv"
    CSV_BAND15 = RESULTS_DIR / f"final_results_{MODEL}_band-1.5to0.25_per_layer.csv"
    OUT_PATH   = OUT_DIR / f"test_acc_summary_table_{MODEL}_per_layer.png"
else:
    CSV_MAIN   = RESULTS_DIR / f"final_results_{MODEL}.csv"
    CSV_BAND10 = RESULTS_DIR / f"final_results_{MODEL}_band-1.0to0.0.csv"
    CSV_BAND15 = RESULTS_DIR / f"final_results_{MODEL}_band-1.5to0.25.csv"
    OUT_PATH   = OUT_DIR / f"test_acc_summary_table_{MODEL}.png"

DATASETS = ["Cora", "PubMed", "Roman-empire", "Squirrel"]
K_VALUES = list(range(2, 9)) if args.r_only else list(range(1, 9))

# Configs per dataset
if args.r_only:
    CONFIGS = [
        ("R_only", -1.0,  0.0,  "R only\n(-1, 0)"),
        ("R_only", -1.5,  0.25, "R only\n(-1.5, 0.25)"),
    ]
else:
    CONFIGS = [
        ("ce_only",   None,  None,  "CE only"),
        ("ce_plus_R", -1.0,  0.0,   "CE+R\n(-1, 0)"),
        ("ce_plus_R", -1.5,  0.25,  "CE+R\n(-1.5, 0.25)"),
    ]

# -- Load & aggregate ------------------------------------------------------
def load_and_agg(path):
    df = pd.read_csv(path)
    df["test_acc_pct"] = df["test_acc"] * 100
    grp_cols = ["dataset", "loss_type", "K"]
    if "band_lower" in df.columns:
        grp_cols += ["band_lower", "band_upper"]
    agg = (
        df.groupby(grp_cols)["test_acc_pct"]
        .agg(["mean", "std"])
        .reset_index()
    )
    if "band_lower" not in agg.columns:
        agg["band_lower"] = None
        agg["band_upper"] = None
    agg["cell"] = agg.apply(lambda r: f"{r['mean']:.1f} \u00b1 {r['std']:.1f}", axis=1)
    return agg

agg_main   = load_and_agg(CSV_MAIN) if CSV_MAIN and CSV_MAIN.exists() else None
agg_band10 = load_and_agg(CSV_BAND10) if CSV_BAND10 and CSV_BAND10.exists() else None
agg_band15 = load_and_agg(CSV_BAND15) if CSV_BAND15 and CSV_BAND15.exists() else None

def lookup(ds, loss_type, band_lower, band_upper, K):
    """Return formatted cell string, or '-' if not found."""
    if loss_type == "ce_only":
        if agg_main is None: return "-"
        src = agg_main
        mask = (src.dataset == ds) & (src.loss_type == "ce_only") & (src.K == K)
    elif band_lower == -1.0 and band_upper == 0.0:
        if agg_band10 is None:
            return "-"
        src = agg_band10
        mask = (src.dataset == ds) & (src.loss_type == loss_type) & (src.K == K)
    else:  # band (-1.5, 0.25)
        if agg_band15 is None:
            return "-"
        src = agg_band15
        mask = (src.dataset == ds) & (src.loss_type == loss_type) & (src.K == K)
    hits = src[mask]
    return hits["cell"].values[0] if len(hits) else "-"

# -- Build table arrays ----------------------------------------------------
# col_headers: (line1=dataset, line2=config label) for each (dataset, config)
col_headers = [(ds, conf[3]) for ds in DATASETS for conf in CONFIGS]
col_keys    = [(ds, conf[0], conf[1], conf[2]) for ds in DATASETS for conf in CONFIGS]

row_labels = [f"K = {k}" for k in K_VALUES]
table_data = []
for k in K_VALUES:
    row = [lookup(ds, lt, bl, bu, k) for ds, lt, bl, bu in col_keys]
    table_data.append(row)

n_rows     = len(K_VALUES)
n_data_cols = len(col_headers)

# -- Layout (inches) -------------------------------------------------------
plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 10})

ROW_LABEL_W = 0.9
COL_W       = 1.35   # slightly narrower to fit 12 columns
ROW_H       = 0.34
HEAD_H      = 0.62   # taller header for two-line labels

fig_w = ROW_LABEL_W + n_data_cols * COL_W + 0.1
fig_h = HEAD_H + n_rows * ROW_H + 0.15

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

# -- Coordinate helpers ----------------------------------------------------
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
# Dataset name centred over its 3 sub-columns, config label in header row
n_cfg = len(CONFIGS)
for ds_idx, ds in enumerate(DATASETS):
    # Dataset name above the three sub-columns
    j_start = ds_idx * n_cfg
    j_mid   = j_start + (n_cfg - 1) / 2
    cx_ds   = ROW_LABEL_W + j_mid * COL_W + COL_W / 2
    cy_top  = RULE_MID + HEAD_H * 0.72
    ax.text(cx_ds, cy_top, ds, ha="center", va="center",
            fontsize=9.5, fontweight="bold")

    # Thin vertical separator between dataset groups (except before first)
    if ds_idx > 0:
        xv = ROW_LABEL_W + ds_idx * n_cfg * COL_W
        ax.plot([xv, xv], [RULE_BOT, RULE_TOP], color="black",
                linewidth=0.5, linestyle=(0, (4, 4)), alpha=0.6)

    # Sub-column labels
    for conf_idx, (_, _, _, conf_label) in enumerate(CONFIGS):
        j = j_start + conf_idx
        cx = col_x_center(j)
        cy = RULE_MID + HEAD_H * 0.28
        ax.text(cx, cy, conf_label, ha="center", va="center",
                fontsize=7.8, fontstyle="italic", multialignment="center")

# -- Row labels + data cells -----------------------------------------------
for i, (k_label, row_cells) in enumerate(zip(row_labels, table_data)):
    yc = row_y_center(i)
    ax.text(0.08, yc, k_label, ha="left", va="center", fontsize=9.5)
    for j, val in enumerate(row_cells):
        ax.text(col_x_center(j), yc, val, ha="center", va="center",
                fontsize=8.5)

# -- Title -----------------------------------------------------------------
hetero_note = "hetero: 3 seeds x split 0, homo: 3 seeds"
if args.r_only:
    title_suffix = f"R only, per-layer hyperparams, {hetero_note}"
elif args.per_layer:
    title_suffix = f"per-layer hyperparams, {hetero_note}"
else:
    title_suffix = f"mean +/- std, seeds 0-2"
ax.text(fig_w / 2, fig_h + 0.06,
        f"{MODEL} Test Accuracy (%), {title_suffix}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
        transform=ax.transData, clip_on=False)

# -- Save ------------------------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved -> {OUT_PATH}")
