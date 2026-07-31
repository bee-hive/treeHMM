"""
Step 5: Create HMM overlay videos for the cancer phase-cell crops.

For each crop:
  1) Load the crop background image, cancer PHASE tracks (channel 1 of the
     type-separated tracks), and the T-cell tracks (channel 0, secondary layer)
  2) Reconstruct the same min_t cell filter used during fitting
  3) Load per-crop cancer_state_assignments.npy
  4) Color each cancer phase cell by its HMM state (warmup frames in grey)
  5) Render an mp4

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python create_cancer_phase_overlay_videos.py
"""

import os
import sys
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Load shared configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
cvat_base_dir = cfg["cvat_base_dir"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
out_base_dir = cfg["output_base_dir"]
min_t = cfg["min_t"]
num_lags = cfg["num_lags"]
model_features = ', '.join(cfg["model_features"])
use_dino = cfg.get("use_dino", True)
video_fps = cfg["video_fps"]
video_figsize = tuple(cfg["video_figsize"])

sys.path.insert(0, cfg["imaging_pipeline_dir"])
import tifffile
from scripts.utils.PlottingUtils import (
    create_fixed_colormap,
    create_fixed_norm,
    plt_to_mp4,
)

SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 7, 8, 10
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.use14corefonts'] = True

feat_str = model_features + (', dino_pc[0:%d]' % cfg["n_dino_pcs"] if use_dino else '')


def load_cancer_phase_tracks(crop):
    """Load cancer phase tracks (channel 1 of the type-separated tracks) -> (T,H,W)."""
    tr = tifffile.imread(os.path.join(type_sep_tracks_dir, crop, "tracks.tiff"))
    return tr[..., 1]


for crop in crop_ids:
    print("=" * 60)
    print(f"Processing crop: {crop}")
    print("=" * 60)
    well_id = crop.split("_")[0]

    # 1. Load background, cancer phase tracks, T-cell tracks
    raw_tiff = tifffile.imread(os.path.join(cvat_base_dir, well_id, crop, 'crop.tiff'))
    cancer_tracks = load_cancer_phase_tracks(crop)                 # (T,H,W) phase IDs == cancer cells
    t_cell_tracks = tifffile.imread(
        os.path.join(type_sep_tracks_dir, crop, 'tracks.tiff'))[..., 0]
    T = cancer_tracks.shape[0]

    # 2. Reconstruct the min_t cell filter (same as fitting)
    all_cell_ids = np.sort(np.unique(cancer_tracks[cancer_tracks > 0]))
    id_to_col_initial = {int(cid): i for i, cid in enumerate(all_cell_ids)}
    active_mask = np.zeros((T, len(all_cell_ids)), dtype=bool)
    for t_idx in range(T):
        for cid in np.unique(cancer_tracks[t_idx][cancer_tracks[t_idx] > 0]):
            active_mask[t_idx, id_to_col_initial[int(cid)]] = True
    durations = np.sum(active_mask, axis=0)
    keep_ids = all_cell_ids[durations >= min_t]
    cancer_tracks[~np.isin(cancer_tracks, keep_ids)] = 0
    id_to_column_index = {int(cid): i for i, cid in enumerate(keep_ids)}
    print(f"  Filtered {len(all_cell_ids)} -> {len(keep_ids)} cancer cells (min_t={min_t})")

    # 3. Load state assignments
    state_assignments = np.load(
        os.path.join(out_base_dir, crop, 'cancer_state_assignments.npy'))
    assert len(keep_ids) == state_assignments.shape[1], (
        f"Mismatch for {crop}: {len(keep_ids)} cells vs {state_assignments.shape[1]} "
        f"state columns. Check min_t matches the value used during fitting.")
    num_states_found = int(state_assignments.max()) + 1

    # Warmup frames: first max(1, num_lags) active frames per cell
    warmup_n = max(1, num_lags)
    warmup_frames = set()
    for cid in keep_ids:
        active_t = [t for t in range(T) if np.any(cancer_tracks[t] == cid)]
        for i in range(min(warmup_n, len(active_t))):
            warmup_frames.add((active_t[i], int(cid)))

    # 4. Build spatial state overlay: 0=bg, -1=warmup, 1..K=states
    WARMUP_VAL = -1
    state_tracks = np.zeros_like(cancer_tracks, dtype=np.int32)
    for t_idx in range(T):
        for cid in keep_ids:
            cid = int(cid)
            if not np.any(cancer_tracks[t_idx] == cid):
                continue
            col = id_to_column_index[cid]
            if (t_idx, cid) in warmup_frames:
                state_tracks[t_idx][cancer_tracks[t_idx] == cid] = WARMUP_VAL
            else:
                state_tracks[t_idx][cancer_tracks[t_idx] == cid] = state_assignments[t_idx, col] + 1

    cmap = create_fixed_colormap(num_states_found)
    norm = create_fixed_norm(num_states_found)
    warmup_color = (0.5, 0.5, 0.5, 1.0)

    def frame_fn(t, _raw=raw_tiff, _cancer=cancer_tracks, _state=state_tracks,
                 _tcell=t_cell_tracks, _sa=state_assignments, _cmap=cmap, _norm=norm,
                 _crop=crop, _ns=num_states_found):
        bg = _raw[t, ..., 1] if _raw.ndim == 4 else _raw[t]
        plt.imshow(bg, cmap="gray")

        # T cells as a secondary blue layer
        tcell_frame = np.where(_tcell[t] != 0, 1, 0)
        plt.imshow(tcell_frame, cmap='Blues', alpha=0.30)

        # Cancer phase-cell IDs labelled
        cancer_frame = _cancer[t]
        for track_id in np.unique(cancer_frame[cancer_frame > 0]):
            yx = np.argwhere(cancer_frame == track_id).mean(axis=0)
            plt.text(yx[1], yx[0], str(int(track_id)), color='white',
                     fontsize=SMALL_SIZE, fontweight='bold', ha='center', va='center')

        state_frame = _state[t]
        warmup_mask = (state_frame == WARMUP_VAL).astype(float)
        warmup_rgba = np.zeros((*warmup_mask.shape, 4))
        warmup_rgba[warmup_mask == 1] = warmup_color
        warmup_rgba[..., 3] *= 0.55
        plt.imshow(warmup_rgba)

        state_masked = np.where(state_frame > 0, state_frame, 0)
        plt.imshow(state_masked, cmap=_cmap, norm=_norm, alpha=0.55)

        handles = [mpatches.Patch(color=_cmap(i), label=str(i - 1))
                   for i in range(1, _ns + 1)]
        handles.append(mpatches.Patch(color='grey', label='warmup'))
        handles.append(mpatches.Patch(color='tab:blue', label='T cell'))
        plt.legend(title='state', handles=handles, loc='upper left')
        plt.title(f'Crop {_crop}, frame {t + 1}\ncancer phase-cell AR-HMM states\n'
                  f'num_lags: {num_lags}, features: {feat_str}')
        plt.axis('off')

    video_dir = os.path.join(out_base_dir, crop)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, 'cancer_phase_HMM_overlay_video.mp4')
    print(f"  Rendering video to {video_path} ...")
    plt_to_mp4(frame_fn, list(range(T)), video_path, fps=video_fps, figsize=video_figsize)
    print(f"  Done: {video_path}\n")

print("=" * 60)
print("All crop videos complete!")
print("=" * 60)
