"""Overlay videos coloring each cancer nucleus by its DINO PC1 mode.

Port of `scripts/cancer_dino_hmm/create_dino_mode_overlay_videos.py` to the
600x600 wells. Pools every valid cell-frame's PC1 across all wells, splits it
with a global 1-D 2-means fit, and paints each nucleus mask by which mode it
falls into.

This is independent of the HMM fit and of the `min_t` filter -- every nucleus
with a valid embedding is colored -- so it is written once into the shared
embedding directory and symlinked into each variant.

Saved per well:
    {output_base_dir}/{well}/nuclei_dino_pc1_mode_overlay_video.mp4

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python create_dino_mode_overlay_videos.py
"""

import io
import os
import pickle
import sys
import zipfile
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

wells: List[str] = cfg["wells"]
out_base_dir: str = cfg["output_base_dir"]
fps: int = cfg.get("video_fps", 10)
figsize = tuple(cfg.get("video_figsize", [6, 6]))

if not cfg.get("use_dino", True):
    sys.exit("use_dino is false; there are no DINO embeddings to color by.")

runs_dir = Path(cfg["snakemake_runs_dir"])
reg_dir = runs_dir / cfg["registration_run"] / "output"
caliban_dir = runs_dir / cfg["caliban_run"] / "output"

raw_shape = cfg["raw_video_shape"]
center_h, center_w = cfg["video_center_extract"]
frame_start, frame_end = cfg["frame_start"], cfg["frame_end"]
phase_channel = cfg["phase_channel"]
Y_OFFSET = (raw_shape[1] - center_h) // 2
X_OFFSET = (raw_shape[2] - center_w) // 2

sys.path.insert(0, cfg.get("imaging_pipeline_dir",
                           "/gladstone/engelhardt/lab/jadjasu/LiveCellUmbrella/MarsonImagingPipeline"))
from scripts.utils.PlottingUtils import plt_to_mp4

SMALL = 7
plt.rc('font', size=SMALL)
plt.rc('axes', titlesize=8, labelsize=SMALL)
plt.rc('legend', fontsize=SMALL)

MODE_COLORS = ['tab:orange', 'tab:purple']
NO_EMBED_COLOR = (0.5, 0.5, 0.5, 1.0)
NO_EMBED_VAL = -1


def fit_two_modes_1d(x: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """1-D 2-means; returns the two cluster centers sorted ascending.

    Dependency-light stand-in for a 2-component mixture so this runs in the
    Occident env. Assignment is nearest-center.

    Args:
        x (np.ndarray): 1-D values.
        n_iter (int): maximum Lloyd iterations.

    Returns:
        np.ndarray: the two centers, ascending.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    c = np.array([x.min(), x.max()], dtype=np.float64)
    for _ in range(n_iter):
        assign = (np.abs(x - c[1]) < np.abs(x - c[0])).astype(int)
        new_c = c.copy()
        for k in (0, 1):
            if np.any(assign == k):
                new_c[k] = x[assign == k].mean()
        if np.allclose(new_c, c):
            c = new_c
            break
        c = new_c
    return np.sort(c)


def load_nuclei_masks(well: str) -> np.ndarray:
    """Load the Caliban nucleus label mask for a well.

    Args:
        well (str): well name.

    Returns:
        np.ndarray: (T, H, W) int label mask.
    """
    with zipfile.ZipFile(caliban_dir / well / "Caliban" / "deepcell_tracks.zip") as z:
        masks = tifffile.imread(io.BytesIO(z.read("y.ome.tiff")))
    return masks[..., 0] if masks.ndim == 4 else masks


def load_tcell_masks(well: str):
    """Load SAM3 tracks restricted to T cells, or None if unavailable.

    Args:
        well (str): well name.

    Returns:
        Optional[np.ndarray]: (T, H, W) mask, non-zero where a T cell is present.
    """
    sam3 = runs_dir / cfg.get("sam3_run", "Finetuned_Arrietty_Crops_250t_150xy") \
        / "output" / well / "full_tracks.tiff"
    typing = runs_dir / cfg.get("cell_typing_run", "Finetuned_CellTyping_250t_600xy") \
        / "output" / well / "cell_type_dict.pkl"
    if not (sam3.exists() and typing.exists()):
        return None
    tracks = np.asarray(tifffile.imread(sam3))
    with open(typing, "rb") as fh:
        type_dict = pickle.load(fh)
    tcell_ids = np.array([int(k) for k, v in type_dict.items() if v == "t_cell"])
    keep = np.zeros(int(tracks.max()) + 1, dtype=bool)
    keep[tcell_ids[tcell_ids <= tracks.max()]] = True
    return np.where(keep[tracks], tracks, 0)


print("=" * 60)
print("Finding the two PC1 modes (global, across all wells)")
print("=" * 60)

pc1_by_well, valid_by_well, pooled = {}, {}, []
for well in wells:
    out_dir = Path(out_base_dir) / well
    pca = np.load(out_dir / "nuclei_dino_pca.npy")
    valid = np.load(out_dir / "nuclei_dino_valid_mask.npy")
    pc1_by_well[well] = pca[..., 0]
    valid_by_well[well] = valid
    pooled.append(pca[..., 0][valid])

pooled = np.concatenate(pooled)
centers = fit_two_modes_1d(pooled)
boundary = centers.mean()
print(f"Pooled valid cell-frames: {pooled.shape[0]}")
print(f"Mode 0 center (lower PC1):  {centers[0]:.3f}")
print(f"Mode 1 center (higher PC1): {centers[1]:.3f}")
print(f"Split point: {boundary:.3f}")
n0 = int(np.sum(pooled < boundary))
print(f"Assignment: mode 0 = {n0}  |  mode 1 = {pooled.shape[0] - n0}")

mode_cmap = ListedColormap(MODE_COLORS)
mode_norm = BoundaryNorm([0.5, 1.5, 2.5], mode_cmap.N)

for well in wells:
    print("\n" + "=" * 60)
    print(f"Well {well}")
    print("=" * 60)

    out_dir = Path(out_base_dir) / well
    cell_ids = np.load(out_dir / "nuclei_cell_ids.npy")
    pc1 = pc1_by_well[well]
    valid = valid_by_well[well]

    nuc = load_nuclei_masks(well)
    tcell = load_tcell_masks(well)
    num_frames = nuc.shape[0]

    reg = tifffile.memmap(reg_dir / well / "registered.tiff")
    phase = np.asarray(reg[frame_start:frame_end,
                           Y_OFFSET:Y_OFFSET + center_h,
                           X_OFFSET:X_OFFSET + center_w, phase_channel])

    max_id = int(nuc.max())
    mode_tracks = np.zeros_like(nuc, dtype=np.int32)
    for t in range(num_frames):
        lut = np.zeros(max_id + 1, dtype=np.int32)
        present = valid[t]
        lut[cell_ids[present]] = np.where(pc1[t, present] >= boundary, 2, 1)
        ids_t = np.unique(nuc[t])
        ids_t = ids_t[ids_t > 0]
        lut[ids_t[lut[ids_t] == 0]] = NO_EMBED_VAL
        mode_tracks[t] = lut[nuc[t]]

    def frame_fn(t, _phase=phase, _mode=mode_tracks, _tcell=tcell, _well=well):
        plt.imshow(_phase[t], cmap="gray")
        if _tcell is not None:
            plt.imshow(np.where(_tcell[t] != 0, 1, 0), cmap="Blues", alpha=0.30)

        none_mask = (_mode[t] == NO_EMBED_VAL)
        if none_mask.any():
            rgba = np.zeros((*none_mask.shape, 4))
            rgba[none_mask] = NO_EMBED_COLOR
            rgba[..., 3] *= 0.55
            plt.imshow(rgba)

        plt.imshow(np.ma.masked_less_equal(_mode[t], 0),
                   cmap=mode_cmap, norm=mode_norm, alpha=0.60)

        handles = [
            mpatches.Patch(color=MODE_COLORS[0], label=f"mode 0 (PC1 < {boundary:.1f})"),
            mpatches.Patch(color=MODE_COLORS[1], label=f"mode 1 (PC1 >= {boundary:.1f})"),
            mpatches.Patch(color="grey", label="no embedding"),
            mpatches.Patch(color="tab:blue", label="T cell"),
        ]
        plt.legend(title="DINO PC1 mode", handles=handles, loc="upper left")
        plt.title(f"Well {_well}, frame {t + 1}\ncancer-nucleus DINO PC1 mode\n"
                  f"mode centers: {centers[0]:.1f} / {centers[1]:.1f}")
        plt.axis("off")

    video_path = out_dir / "nuclei_dino_pc1_mode_overlay_video.mp4"
    print(f"  rendering {num_frames} frames -> {video_path}")
    plt_to_mp4(frame_fn, list(range(num_frames)), str(video_path),
               fps=fps, figsize=figsize)
    print(f"  done: {video_path}")

print("\n" + "=" * 60)
print("All PC1-mode overlay videos complete.")
print("=" * 60)
