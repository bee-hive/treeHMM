"""
Create HMM overlay videos for all ground-truth T cell crops.

For each crop the script:
  1) Loads the raw TIFF, CVAT tracks, and cell type dictionary
  2) Filters tracks by cell type (cancer vs. T cell)
  3) Applies the same time-duration filter used during model fitting
  4) Loads the per-crop state assignments from fit_arhmm.py
  5) Maps state assignments onto the T cell mask for a spatial overlay
  6) Renders an mp4 video with the overlay

Usage (AnalysisEnv):
    conda run -n AnalysisEnv python create_overlay_videos.py
"""

import os
import sys
import pickle
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.ma import masked_where

# ---------------------------------------------------------------------------
# Load shared configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
cvat_base_dir = cfg["cvat_base_dir"]
out_base_dir = cfg["output_base_dir"]
min_t = cfg["min_t"]
num_lags = cfg["num_lags"]
model_features = ', '.join(cfg["model_features"])
video_fps = cfg["video_fps"]
video_figsize = tuple(cfg["video_figsize"])

# ---------------------------------------------------------------------------
# Make MarsonImagingPipeline importable
# ---------------------------------------------------------------------------
sys.path.insert(0, cfg["imaging_pipeline_dir"])
import tifffile
from scripts.utils.PlottingUtils import (
    create_fixed_colormap,
    create_fixed_norm,
    plt_to_mp4,
)

# ---------------------------------------------------------------------------
# Matplotlib styling (matches the original notebook)
# ---------------------------------------------------------------------------
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.use14corefonts'] = True


# ---------------------------------------------------------------------------
# Helper: filter tracks by cell type
# ---------------------------------------------------------------------------
def filter_tracks_by_type(track_type, tracks, type_dict):
    """Filter tracks to only contain tracks of the desired cell type."""
    valid_ids = [
        cell_id for cell_id, cell_type in type_dict.items()
        if cell_type == track_type
    ]
    filtered_tracks = tracks.copy()
    filtered_tracks[~np.isin(filtered_tracks, valid_ids)] = 0
    return filtered_tracks


# ---------------------------------------------------------------------------
# Main loop over crops
# ---------------------------------------------------------------------------
for crop in crop_ids:
    print("=" * 60)
    print(f"Processing crop: {crop}")
    print("=" * 60)

    well_id = crop.split("_")[0]

    # 1. Load CVAT data
    raw_tiff = tifffile.imread(
        os.path.join(cvat_base_dir, well_id, crop, 'crop.tiff')
    )
    cvat_tracks = tifffile.imread(
        os.path.join(cvat_base_dir, well_id, crop, 'ALL_tracks.tiff')
    )
    cell_type_dict = pickle.load(
        open(os.path.join(cvat_base_dir, well_id, crop,
                          'full_cell_type_dict.pkl'), "rb")
    )

    # NOTE: all frames are kept — AR warmup is reflected in state_assignments
    T = raw_tiff.shape[0]

    # 2. Separate tracks by cell type
    type_tracks = {}
    for ct in ['cancer', 't_cell']:
        type_tracks[ct] = filter_tracks_by_type(ct, cvat_tracks, cell_type_dict)

    t_cell_tracks = type_tracks['t_cell']

    # 3. Filter tracks by time to match AR-HMM state assignments
    all_cell_ids = np.unique(t_cell_tracks[t_cell_tracks > 0])
    all_cell_ids.sort()
    id_to_col_initial = {int(cid): i for i, cid in enumerate(all_cell_ids)}

    num_cells_initial = len(all_cell_ids)
    active_mask = np.zeros((T, num_cells_initial), dtype=bool)

    for t_idx in range(T):
        frame_ids = np.unique(t_cell_tracks[t_idx])
        frame_ids = frame_ids[frame_ids > 0]
        for cid in frame_ids:
            active_mask[t_idx, id_to_col_initial[int(cid)]] = True

    durations = np.sum(active_mask, axis=0)
    keep_ids = all_cell_ids[durations >= min_t]

    t_cell_tracks[~np.isin(t_cell_tracks, keep_ids)] = 0

    t_cell_ids = keep_ids
    id_to_column_index = {
        int(cell_id): index for index, cell_id in enumerate(t_cell_ids)
    }

    print(f"  Filtered {num_cells_initial} -> {len(t_cell_ids)} cells (min_t={min_t})")

    # 4. Load state assignments
    state_assignments = np.load(
        os.path.join(out_base_dir, crop, 't_cell_state_assignments.npy')
    )

    assert len(t_cell_ids) == state_assignments.shape[1], (
        f"Mismatch for crop {crop}: {len(t_cell_ids)} T cell IDs vs "
        f"{state_assignments.shape[1]} columns in state_assignments. "
        f"Check that min_t={min_t} matches the value used during model fitting."
    )
    print(f"  state_assignments shape: {state_assignments.shape}")

    # Determine warmup frames (first `num_lags` active frames per cell)
    warmup_frames = set()  # set of (t_idx, cell_id) tuples
    if num_lags > 0:
        for cell_id in t_cell_ids:
            active_t = [t_idx for t_idx in range(T)
                        if np.any(t_cell_tracks[t_idx] == cell_id)]
            for i in range(min(num_lags, len(active_t))):
                warmup_frames.add((active_t[i], cell_id))

    # 5. Build the spatial state-assignment overlay
    # Use state+1 for valid states, -1 for warmup frames (rendered grey)
    WARMUP_VAL = -1
    state_assignment_tracks = np.zeros_like(t_cell_tracks, dtype=np.int32)
    for t_idx in range(t_cell_tracks.shape[0]):
        for cell_id in t_cell_ids:
            if cell_id == 0:
                continue
            if not np.any(t_cell_tracks[t_idx] == cell_id):
                continue
            col = id_to_column_index[cell_id]
            if (t_idx, cell_id) in warmup_frames:
                # Warmup frame — mark with a special value
                state_assignment_tracks[t_idx][t_cell_tracks[t_idx] == cell_id] = WARMUP_VAL
            else:
                state = state_assignments[t_idx, col]
                state_assignment_tracks[t_idx][t_cell_tracks[t_idx] == cell_id] = state + 1

    # 6. Define per-frame plotting function and render video
    num_states_found = int(state_assignments.max()) + 1
    cmap = create_fixed_colormap(num_states_found)
    norm = create_fixed_norm(num_states_found)

    # Build a colourmap that also includes grey for warmup frames.
    # state_assignment_tracks uses: 0 = background, -1 = warmup, 1..K = states
    import matplotlib.colors as mcolors
    warmup_color = (0.5, 0.5, 0.5, 1.0)  # grey

    def create_HMM_overlay_video(
        t,
        _raw=raw_tiff,
        _tcell=t_cell_tracks,
        _state_tracks=state_assignment_tracks,
        _type_tracks=type_tracks,
        _sa=state_assignments,
        _cmap=cmap,
        _norm=norm,
        _crop=crop,
        _warmup_color=warmup_color,
    ):
        plt.imshow(_raw[t, ..., 1], cmap="gray")

        label_frame = _tcell[t]
        state_frame = _state_tracks[t]

        for track_id in np.unique(label_frame[label_frame > 0]):
            yx = np.argwhere(label_frame == track_id).mean(axis=0)
            plt.text(
                yx[1], yx[0], str(int(track_id)),
                color='white', fontsize=SMALL_SIZE,
                fontweight='bold', ha='center', va='center',
            )

        cancer_frame = _type_tracks['cancer'][t]
        cancer_frame = np.where(cancer_frame != 0, 1, 0)
        plt.imshow(cancer_frame, cmap='Reds', alpha=0.35)

        # Render warmup pixels in grey
        warmup_mask = (state_frame == WARMUP_VAL).astype(float)
        warmup_rgba = np.zeros((*warmup_mask.shape, 4))
        warmup_rgba[warmup_mask == 1] = _warmup_color
        warmup_rgba[..., 3] *= 0.45
        plt.imshow(warmup_rgba)

        # Render state-assigned pixels (values >= 1) with the fixed colormap
        state_frame_masked = np.where(state_frame > 0, state_frame, 0)
        plt.imshow(state_frame_masked, cmap=_cmap, norm=_norm, alpha=0.35)

        handles = [
            mpatches.Patch(color=_cmap(i), label=str(i - 1))
            for i in range(1, num_states_found + 1)
        ]
        if num_lags > 0:
            handles.append(mpatches.Patch(color='grey', label='warmup'))
        handles.append(mpatches.Patch(color='darkred', label='cancer'))
        plt.legend(title='state', handles=handles, loc='upper left')

        plt.title(f'Crop {_crop}, frame {t + 1}\nT cell track AR-HMM states\nnum_lags: {num_lags}, features: {model_features}')
        plt.axis('off')

    video_dir = os.path.join(out_base_dir, crop)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, 't_cell_HMM_overlay_video.mp4')

    print(f"  Rendering video to {video_path} ...")
    plt_to_mp4(
        create_HMM_overlay_video,
        list(range(T)),
        video_path,
        fps=video_fps,
        figsize=video_figsize,
    )
    print(f"  Done: {video_path}\n")

print("=" * 60)
print("All crop videos complete!")
print("=" * 60)
