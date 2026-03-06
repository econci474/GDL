# Interactive Visualization System

## Quick Start

### Generate Interactive HTML
```bash
# All defaults (4 datasets, K=0-8, seeds 0-3 + averaged)
python -m src.plot_unified_interactive

# Specific datasets
python -m src.plot_unified_interactive --datasets Cora PubMed

# Specific K range  
python -m src.plot_unified_interactive --K-values 0 1 2 3

# Single seed
python -m src.plot_unified_interactive --seeds 0
```

**Output:** `results/figures/interactive/unified_interactive_{split}.html`

### View the HTML
Just **double-click** the HTML file → Opens in your browser!

---

## What You Get

### Single Shareable HTML File
- ✅ **100% Self-Contained** - All data embedded
- ✅ **No Internet Required** - Works offline
- ✅ **Shareable** - Email it, put on Dropbox, send to collaborators
- ✅ **Universal** - Opens in Chrome, Firefox, Edge, Safari

### Interactive Controls

**Dataset Selection:** Toggle between Cora, PubMed, Roman-empire, Minesweeper

**Model Selection:** GCN, GAT, GraphSAGE (if loaded)

**Depth (K):** Maximum layers (0-8)

**Layer (k):** Specific layer to view (0 to K)

**Seed Options:**
- `seed_0`, `seed_1`, `seed_2`, `seed_3` (individual runs)
- `averaged` (mean across all seeds)

**Training Method:**
- Linear Probe (frozen embeddings)
- Exponential Classifier (β=0.5)
- Class-Weighted Classifier

**Class Filter:** All classes or individual class

**Correctness:** All / Correct only / Incorrect only

---

## Static Plots with Trajectories

For publication-quality PDFs with trajectory panels:

```bash
# Aggregated with trajectory summary panel
python -m src.plot_node_entropy_vs_prob --dataset Cora --model GCN --K 5 --seed all
```

**Output:** `*_with_trajectories.pdf`

**Trajectory panel shows:**
- Class mean positions at each layer
- Connecting lines showing evolution k=0 → K
- Helps identify class-specific depth patterns

---

## File Sizes & Performance

| Configuration | Approx. Size | Load Time |
|--------------|--------------|-----------|
| 1 dataset, K=0-2 | ~2-5 MB | Instant |
| 4 datasets, K=0-8 | ~20-50 MB | 1-2 sec |
| Full (all data) | ~50-100 MB | 2-5 sec |

**Tip:** Use specific `--datasets` and `--K-values` to keep file size manageable for sharing.

---

## Sharing the HTML

### Method 1: Email
- Attach the HTML file directly (if < 25 MB)
- For larger files, use file transfer services

### Method 2: Cloud Storage
- Upload to Dropbox/Google Drive/OneDrive
- Share link - recipients just download and open

### Method 3: GitHub
- Commit to repository
- Use GitHub Pages to host (or download raw)

### Method 4: Direct Send
- Copy file to shared network drive
- USB stick for offline sharing

**No setup required for recipients** - they just open it in a browser!

---

## Example Workflows

### For Presentation
```bash
# Create comprehensive interactive
python -m src.plot_unified_interactive --datasets Cora PubMed

# Share HTML with collaborators
# They can explore all configurations during meeting
```

### For Publication
```bash
# Generate static PDFs with trajectories
python -m src.plot_node_entropy_vs_prob --dataset Cora --K 5 --seed all
python -m src.plot_node_entropy_vs_prob --dataset PubMed --K 5 --seed all

# Include in paper as figures
```

### For Analysis
```bash
# Full interactive for yourself
python -m src.plot_unified_interactive

# Explore patterns across all configurations
# Filter to interesting cases
# Generate targeted static plots for those cases
```

---

## FAQ

**Q: Can I edit the HTML after generation?**  
A: The HTML is read-only. Re-run the script with different parameters to regenerate.

**Q: Does the recipient need Python?**  
A: No! Just a web browser.

**Q: Can I embed this in a website?**  
A: Yes! Just upload the HTML file and link to it, or use iframe embedding.

**Q: Why is the file large?**  
A: All node-level data is embedded for interactivity. Use `--datasets` filter to reduce size.

**Q: Can I see multiple layers at once?**  
A: Not currently - use layer (k) dropdown to switch. For multi-layer view, use static plots.

**Q: What about seed averaging?**  
A: Select `averaged` from the seed dropdown to see mean probabilities across all seeds.
