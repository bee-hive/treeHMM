"""Step 3b: Check what the DINO PCs encode before trusting any DINO arm.

The risk this guards against: if the leading PCs track *when* a patch was
taken rather than *what it looks like*, a low-k HMM will happily split the
movie into early and late instead of alive and dying.  This pipeline
normalizes phase contrast once per crop with fixed percentiles specifically
to avoid that, so this script is the check that the fix worked.

Each PC is correlated against:
  - frame index                 (illumination / confluence drift over time)
  - crop identity               (batch effect between crops and wells)
  - d_circularity               (the acute death signature)
  - dilated_t_cell_neighbors    (T-cell context)

Writes dino_pc_confounds.{csv,png} and prints the table.  Run after Step 3.

Usage (OccidentAnalysis or cs229Dino -- numpy/pandas/matplotlib only):
    conda run -n OccidentAnalysis python check_dino_confounds.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
out_base_dir = cfg["output_base_dir"]
feature_names = cfg["emission_feature_names"]
idx = {name: i for i, name in enumerate(feature_names)}


def correlation_ratio(values, groups):
    """Eta: the fraction of variance in `values` explained by group identity.

    Args:
        values (np.ndarray): (N,) continuous observations.
        groups (np.ndarray): (N,) integer group labels.

    Returns:
        float: eta in [0, 1]; 0 means the grouping explains nothing.
    """
    overall = values.mean()
    ss_between = sum(
        len(values[groups == g]) * (values[groups == g].mean() - overall) ** 2
        for g in np.unique(groups))
    ss_total = ((values - overall) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


print("=" * 60)
print("Step 3b: DINO PC confound check")
print("=" * 60)

pcs_list, frame_list, crop_list, diag_list = [], [], [], []
for crop_idx, crop in enumerate(crop_ids):
    out_dir = os.path.join(out_base_dir, crop)
    pcs = np.load(os.path.join(out_dir, "cancer_dino_pca.npy"))
    valid = np.load(os.path.join(out_dir, "cancer_dino_valid_mask.npy"))
    diag = np.load(os.path.join(out_dir, "cancer_emissions_array.npy"))
    frame_grid = np.repeat(np.arange(pcs.shape[0])[:, None], pcs.shape[1], axis=1)

    pcs_list.append(pcs[valid])
    frame_list.append(frame_grid[valid])
    crop_list.append(np.full(int(valid.sum()), crop_idx))
    diag_list.append(diag[valid])

pcs = np.concatenate(pcs_list, axis=0)
frames = np.concatenate(frame_list)
crops = np.concatenate(crop_list)
diags = np.concatenate(diag_list, axis=0)
n_pcs = pcs.shape[1]
print(f"{pcs.shape[0]} valid cell-frames, {n_pcs} PCs\n")

rows = []
for p in range(n_pcs):
    v = pcs[:, p]
    rows.append({
        "pc": f"PC{p}",
        "|r| vs frame": abs(np.corrcoef(v, frames)[0, 1]),
        "eta vs crop": correlation_ratio(v, crops),
        "|r| vs d_circularity": abs(np.corrcoef(v, diags[:, idx["d_circularity"]])[0, 1]),
        "|r| vs t_cell_contact": abs(
            np.corrcoef(v, diags[:, idx["dilated_t_cell_neighbors"]])[0, 1]),
    })
table = pd.DataFrame(rows)
pd.set_option("display.width", 140)
print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
table.to_csv(os.path.join(out_base_dir, "dino_pc_confounds.csv"), index=False)

worst_time = table["|r| vs frame"].max()
worst_batch = table["eta vs crop"].max()
best_signal = table["|r| vs d_circularity"].max()
print(f"\nStrongest PC correlation with FRAME INDEX    : {worst_time:.3f}")
print(f"Strongest PC association with CROP IDENTITY : {worst_batch:.3f}")
print(f"Strongest PC correlation with d_circularity : {best_signal:.3f}")

# Absolute thresholds, so "the least bad number wins" cannot be mistaken for
# a clean result when every association is weak.
if worst_time > 0.30:
    print("\nWARNING [time]: the PCs track frame index.  A low-k fit risks\n"
          "splitting early-vs-late rather than alive-vs-dying; the signature\n"
          "in states.png is a single monotonic switch through the movie.")
if worst_batch > 0.40:
    print("\nWARNING [batch]: the PCs are strongly associated with crop identity.\n"
          "Since crops map onto conditions, a state defined mostly by these PCs\n"
          "may separate wells rather than phenotypes -- which would invalidate\n"
          "any downstream per-condition death comparison built on it.")
if best_signal < 0.10:
    print("\nWARNING [signal]: no PC is linearly associated with the acute shape\n"
          "signature.  The DINO arms may still work through a nonlinear or\n"
          "multivariate combination, but the single-PC evidence is absent -- do\n"
          "not assume the embedding has captured the death transition.")

fig, axes = plt.subplots(1, 4, figsize=(16, 3.2), tight_layout=True)
cols = [("|r| vs frame", "tab:red", "PC vs frame index (confound)"),
        ("eta vs crop", "tab:orange", "PC vs crop identity (batch)"),
        ("|r| vs d_circularity", "tab:green", "PC vs d_circularity (signal)"),
        ("|r| vs t_cell_contact", "tab:blue", "PC vs T-cell contact (context)")]
x = np.arange(n_pcs)
for ax, (col, colour, title) in zip(axes, cols):
    ax.bar(x, table[col], color=colour)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xlabel("DINO PC")
    ax.set_ylabel("|correlation|")
    ax.set_ylim(0, 1)
plt.savefig(os.path.join(out_base_dir, "dino_pc_confounds.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {os.path.join(out_base_dir, 'dino_pc_confounds.png')}")
