"""
Extra step: Create overlay videos coloring each cancer cell by which mode of the
1st DINO principal component its embedding falls into.

The 1st PC of the (jointly-fit) DINOv2 PCA has a clearly bimodal distribution
across all valid cancer cell-frames.  This script:
  1) Pools every valid cell-frame's PC1 value across ALL crops and finds the two
     modes with a global, dependency-light 1D 2-means fit (so the split is
     comparable across crops, like the jointly-fit PCA/HMM).
  2) Per cell-frame, assigns the cell to whichever mode center its PC1 value is
     CLOSER to (mode 0 = lower center, mode 1 = higher center).
  3) Renders an mp4 per crop coloring each cancer nucleus by its mode; cells that
     lack a valid embedding that frame (no nucleus patch) are drawn in grey.

This is independent of the AR-HMM fit and the min_t filter -- every cancer cell
with a valid embedding is colored.

Outputs (into {output_base_dir}/{crop}/):
    cancer_dino_pc1_mode_overlay_video.mp4

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python create_dino_mode_overlay_videos.py
"""

import os
import sys
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------------------------------------------------------------------------
# Load shared configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
cvat_base_dir = cfg["cvat_base_dir"]
caliban_base_dir = cfg["caliban_base_dir"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
out_base_dir = cfg["output_base_dir"]
video_fps = cfg["video_fps"]
video_figsize = tuple(cfg["video_figsize"])

if not cfg.get("use_dino", True):
    sys.exit("use_dino is false; there are no DINO embeddings to color by.")

sys.path.insert(0, cfg["imaging_pipeline_dir"])
import tifffile
from scripts.utils.PlottingUtils import plt_to_mp4

SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 7, 8, 10
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Mode colors (distinct from the blue T-cell layer and grey "no embedding").
MODE_COLORS = ['tab:orange', 'tab:purple']  # mode 0 (lower PC1), mode 1 (higher PC1)
NO_EMBED_COLOR = (0.5, 0.5, 0.5, 1.0)


def load_nuclei_tracks(crop):
    nuc = tifffile.imread(os.path.join(caliban_base_dir, f"{crop}.tiff"))
    if nuc.ndim == 4:
        nuc = nuc[..., 0]
    return nuc


def fit_two_modes_1d(x, n_iter=100):
    """1D 2-means: return the two cluster centers (sorted ascending).

    Dependency-light stand-in for a 2-component fit so this runs in the
    Occident env (no sklearn needed).  Assignment is nearest-center, i.e.
    exactly "which mode is the value closer to".
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    # Initialize at the extremes so the two centers land on opposite modes.
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


# ---------------------------------------------------------------------------
# Pass 1: pool every valid PC1 value across all crops and find the two modes.
# ---------------------------------------------------------------------------
print("=" * 60)
print("Finding the two PC1 modes (global, across all crops)")
print("=" * 60)

pc1_by_crop = {}
valid_by_crop = {}
pooled = []
for crop in crop_ids:
    out_dir = os.path.join(out_base_dir, crop)
    pca = np.load(os.path.join(out_dir, "cancer_dino_pca.npy"))           # (T, N, n_pcs)
    valid = np.load(os.path.join(out_dir, "cancer_dino_valid_mask.npy"))  # (T, N)
    pc1 = pca[..., 0]
    pc1_by_crop[crop] = pc1
    valid_by_crop[crop] = valid
    pooled.append(pc1[valid])

pooled = np.concatenate(pooled)
centers = fit_two_modes_1d(pooled)
boundary = centers.mean()  # nearest-center split point
print(f"Pooled valid cell-frames: {pooled.shape[0]}")
print(f"Mode 0 center (lower PC1): {centers[0]:.3f}")
print(f"Mode 1 center (higher PC1): {centers[1]:.3f}")
print(f"Split point (closer-to boundary): {boundary:.3f}")
n0 = int(np.sum(pooled < boundary))
print(f"Assignment: mode 0 = {n0}  |  mode 1 = {pooled.shape[0] - n0}")

# Discrete colormap: index 1 -> mode 0, index 2 -> mode 1 (0 reserved for bg).
mode_cmap = ListedColormap(MODE_COLORS)
mode_norm = BoundaryNorm([0.5, 1.5, 2.5], mode_cmap.N)


# ---------------------------------------------------------------------------
# Pass 2: render one mode-colored overlay video per crop.
# ---------------------------------------------------------------------------
for crop in crop_ids:
    print("=" * 60)
    print(f"Processing crop: {crop}")
    print("=" * 60)
    well_id = crop.split("_")[0]

    raw_tiff = tifffile.imread(os.path.join(cvat_base_dir, well_id, crop, 'crop.tiff'))
    cancer_tracks = load_nuclei_tracks(crop)                       # (T,H,W) nucleus IDs
    t_cell_tracks = tifffile.imread(
        os.path.join(type_sep_tracks_dir, crop, 'tracks.tiff'))[..., 0]
    T = cancer_tracks.shape[0]

    cancer_cell_ids = np.load(os.path.join(out_base_dir, crop, 'cancer_cell_ids.npy'))
    id_to_col = {int(cid): i for i, cid in enumerate(cancer_cell_ids)}
    pc1 = pc1_by_crop[crop]
    valid = valid_by_crop[crop]

    # Build a spatial mode overlay: 0=bg/none, 1=mode0, 2=mode1, -1=no-embedding.
    NO_EMBED_VAL = -1
    mode_tracks = np.zeros_like(cancer_tracks, dtype=np.int32)
    for t_idx in range(T):
        frame = cancer_tracks[t_idx]
        for cid in np.unique(frame[frame > 0]):
            cid = int(cid)
            col = id_to_col.get(cid)
            region = frame == cid
            if col is not None and valid[t_idx, col]:
                mode = 1 if pc1[t_idx, col] >= boundary else 0
                mode_tracks[t_idx][region] = mode + 1
            else:
                mode_tracks[t_idx][region] = NO_EMBED_VAL

    def frame_fn(t, _raw=raw_tiff, _cancer=cancer_tracks, _mode=mode_tracks,
                 _tcell=t_cell_tracks, _crop=crop):
        bg = _raw[t, ..., 1] if _raw.ndim == 4 else _raw[t]
        plt.imshow(bg, cmap="gray")

        # T cells as a secondary blue layer
        tcell_frame = np.where(_tcell[t] != 0, 1, 0)
        plt.imshow(tcell_frame, cmap='Blues', alpha=0.30)

        # Cancer nucleus IDs labelled
        cancer_frame = _cancer[t]
        for track_id in np.unique(cancer_frame[cancer_frame > 0]):
            yx = np.argwhere(cancer_frame == track_id).mean(axis=0)
            plt.text(yx[1], yx[0], str(int(track_id)), color='white',
                     fontsize=SMALL_SIZE, fontweight='bold', ha='center', va='center')

        mode_frame = _mode[t]

        # Cells without a valid embedding -> grey
        none_mask = (mode_frame == NO_EMBED_VAL).astype(float)
        none_rgba = np.zeros((*none_mask.shape, 4))
        none_rgba[none_mask == 1] = NO_EMBED_COLOR
        none_rgba[..., 3] *= 0.55
        plt.imshow(none_rgba)

        # Mode-colored cells (mask background/non-mode pixels so only nuclei paint)
        mode_masked = np.ma.masked_less_equal(mode_frame, 0)
        plt.imshow(mode_masked, cmap=mode_cmap, norm=mode_norm, alpha=0.55)

        handles = [
            mpatches.Patch(color=MODE_COLORS[0], label=f'mode 0 (PC1 < {boundary:.1f})'),
            mpatches.Patch(color=MODE_COLORS[1], label=f'mode 1 (PC1 >= {boundary:.1f})'),
            mpatches.Patch(color='grey', label='no embedding'),
            mpatches.Patch(color='tab:blue', label='T cell'),
        ]
        plt.legend(title='DINO PC1 mode', handles=handles, loc='upper left')
        plt.title(f'Crop {_crop}, frame {t + 1}\ncancer-cell DINO PC1 mode\n'
                  f'mode centers: {centers[0]:.1f} / {centers[1]:.1f}')
        plt.axis('off')

    video_dir = os.path.join(out_base_dir, crop)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, 'cancer_dino_pc1_mode_overlay_video.mp4')
    print(f"  Rendering video to {video_path} ...")
    plt_to_mp4(frame_fn, list(range(T)), video_path, fps=video_fps, figsize=video_figsize)
    print(f"  Done: {video_path}\n")

print("=" * 60)
print("All PC1-mode overlay videos complete!")
print("=" * 60)
