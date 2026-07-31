"""Step 3b: Check what the DINO PCs actually encode before trusting the HMM.

``get_rgb_image_with_nuclei`` min-max normalizes each 600x600 frame as a whole,
so every patch cut from a given frame shares that frame's normalization.  If
illumination or contrast drifts over the 250 frames, the leading PCs can end up
encoding *when* a patch was taken rather than *what it looks like* -- and a
2-state HMM will then happily split the movie into "early" and "late" instead of
"alone" and "aggregated".

This script quantifies that risk before the fit, by correlating each PC against:
  - frame index                 (illumination / confluence drift over time)
  - well identity               (batch effect between wells)
  - nuclei_neighbors_30px       (the signal we actually want)

Writes ``dino_pc_confounds.png`` and prints a table.  Run it after Step 3.

Usage (OccidentAnalysis or cs229Dino -- needs numpy/pandas/matplotlib only):
    conda run -n OccidentAnalysis python check_dino_confounds.py
"""

import os
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

wells: List[str] = cfg["wells"]
out_base_dir: str = cfg["output_base_dir"]

with open(os.path.join(out_base_dir, wells[0], "nuclei_emissions_names.txt")) as fh:
    feature_names = [l.strip() for l in fh if l.strip()]
agg_idx = feature_names.index("nuclei_neighbors_30px")


def correlation_ratio(values: np.ndarray, groups: np.ndarray) -> float:
    """Eta: the fraction of variance in ``values`` explained by group identity.

    Args:
        values (np.ndarray): (N,) continuous observations.
        groups (np.ndarray): (N,) integer group labels.

    Returns:
        float: eta in [0, 1]; 0 means the grouping explains nothing.
    """
    overall = values.mean()
    ss_between = 0.0
    for g in np.unique(groups):
        vals = values[groups == g]
        ss_between += len(vals) * (vals.mean() - overall) ** 2
    ss_total = ((values - overall) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


print("=" * 60)
print("Step 3b: DINO PC confound check")
print("=" * 60)

pcs_list, frame_list, well_list, agg_list = [], [], [], []
for well_idx, well in enumerate(wells):
    out_dir = os.path.join(out_base_dir, well)
    pcs = np.load(os.path.join(out_dir, "nuclei_dino_pca.npy"))          # (T, N, P)
    valid = np.load(os.path.join(out_dir, "nuclei_dino_valid_mask.npy"))  # (T, N)
    diag = np.load(os.path.join(out_dir, "nuclei_emissions_array.npy"))
    num_frames = pcs.shape[0]
    frame_grid = np.repeat(np.arange(num_frames)[:, None], pcs.shape[1], axis=1)

    pcs_list.append(pcs[valid])
    frame_list.append(frame_grid[valid])
    well_list.append(np.full(int(valid.sum()), well_idx))
    agg_list.append(diag[:, :, agg_idx][valid])

pcs = np.concatenate(pcs_list, axis=0)
frames = np.concatenate(frame_list)
well_ids = np.concatenate(well_list)
agg = np.concatenate(agg_list)
n_pcs = pcs.shape[1]
print(f"{pcs.shape[0]} valid cell-frames, {n_pcs} PCs\n")

rows = []
for p in range(n_pcs):
    v = pcs[:, p]
    rows.append({
        "pc": f"PC{p}",
        "|r| vs frame": abs(np.corrcoef(v, frames)[0, 1]),
        "eta vs well": correlation_ratio(v, well_ids),
        "|r| vs neighbors30": abs(np.corrcoef(v, agg)[0, 1]),
    })
table = pd.DataFrame(rows)
pd.set_option("display.width", 120)
print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
table.to_csv(os.path.join(out_base_dir, "dino_pc_confounds.csv"), index=False)

worst_time = table["|r| vs frame"].max()
best_agg = table["|r| vs neighbors30"].max()
print(f"\nStrongest PC correlation with FRAME INDEX : {worst_time:.3f}")
print(f"Strongest PC correlation with NEIGHBOR COUNT: {best_agg:.3f}")
if worst_time > best_agg:
    print("\nWARNING: the PCs track time more strongly than they track local\n"
          "crowding.  A 2-state fit is at real risk of splitting early-vs-late\n"
          "rather than alone-vs-aggregated.  Check state_counts.png for a single\n"
          "monotonic switch part-way through the movie -- that is the signature.")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), tight_layout=True)
x = np.arange(n_pcs)
axes[0].bar(x, table["|r| vs frame"], color="tab:red")
axes[0].set_title("PC vs frame index (confound)")
axes[1].bar(x, table["eta vs well"], color="tab:orange")
axes[1].set_title("PC vs well identity (batch)")
axes[2].bar(x, table["|r| vs neighbors30"], color="tab:green")
axes[2].set_title("PC vs nuclei within 30 px (signal)")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in x])
    ax.set_xlabel("DINO PC")
    ax.set_ylabel("|correlation|")
    ax.set_ylim(0, 1)
plt.savefig(os.path.join(out_base_dir, "dino_pc_confounds.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {os.path.join(out_base_dir, 'dino_pc_confounds.png')}")
