"""Step 5: Overlay videos of cancer nuclei painted by inferred HMM state.

Mirrors `scripts/cancer_dino_hmm/create_cancer_overlay_videos.py`: the phase
image is the greyscale background, T cells are a translucent blue layer, and
each cancer nucleus MASK is filled with its state color. Nuclei dropped by the
`min_t` filter (so they have no inferred state) are drawn grey.

Unlike the 150x150 reference, per-nucleus ID text is off by default -- a 600x600
frame holds ~300 nuclei and the labels are unreadable. Enable with
`video_label_ids: true`.

Saved per well:
    {output_base_dir}/{well}/nuclei_HMM_overlay_video.mp4

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python create_nuclei_overlay_videos.py
"""

import io
import os
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

out_base_dir: str = cfg["output_base_dir"]
video_wells: List[str] = cfg.get("video_wells", cfg["wells"])
num_states: int = cfg["num_states"]
min_t: int = cfg["min_t"]
fps: int = cfg.get("video_fps", 10)
figsize = tuple(cfg.get("video_figsize", [6, 6]))
label_ids: bool = cfg.get("video_label_ids", False)

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

NO_STATE_VAL = -1
NO_STATE_COLOR = (0.5, 0.5, 0.5, 1.0)


def load_nuclei_masks(well: str) -> np.ndarray:
    """Load the Caliban nucleus label mask for a well.

    Args:
        well (str): well name.

    Returns:
        np.ndarray: (T, H, W) int label mask; 0 is background.
    """
    zpath = caliban_dir / well / "Caliban" / "deepcell_tracks.zip"
    with zipfile.ZipFile(zpath) as z:
        masks = tifffile.imread(io.BytesIO(z.read("y.ome.tiff")))
    if masks.ndim == 4:
        masks = masks[..., 0]
    return masks


def load_tcell_masks(well: str) -> np.ndarray:
    """Load SAM3 tracks restricted to T cells, for the blue overlay layer.

    Returns an all-zero array if the SAM3 / cell-typing runs are unavailable,
    so the video still renders without the T-cell layer.

    Args:
        well (str): well name.

    Returns:
        np.ndarray: (T, H, W) int mask; non-zero where a T cell is present.
    """
    import pickle
    sam3 = runs_dir / cfg.get("sam3_run", "Finetuned_Arrietty_Crops_250t_150xy") \
        / "output" / well / "full_tracks.tiff"
    typing = runs_dir / cfg.get("cell_typing_run", "Finetuned_CellTyping_250t_600xy") \
        / "output" / well / "cell_type_dict.pkl"
    if not (sam3.exists() and typing.exists()):
        print(f"  [{well}] SAM3/cell-typing not found; skipping T-cell layer")
        return None
    tracks = np.asarray(tifffile.imread(sam3))
    with open(typing, "rb") as fh:
        type_dict = pickle.load(fh)
    tcell_ids = np.array([int(k) for k, v in type_dict.items() if v == "t_cell"])
    keep = np.zeros(int(tracks.max()) + 1, dtype=bool)
    keep[tcell_ids[tcell_ids <= tracks.max()]] = True
    return np.where(keep[tracks], tracks, 0)


print("=" * 60)
print("Step 5: Rendering HMM state overlay videos")
print("=" * 60)

state_cmap = ListedColormap([plt.get_cmap("viridis", num_states)(s)
                             for s in range(num_states)])
state_norm = BoundaryNorm(np.arange(0.5, num_states + 1.5), state_cmap.N)

for well in video_wells:
    print("\n" + "-" * 60)
    print(f"Well {well}")

    out_dir = Path(out_base_dir) / well
    states_path = out_dir / "nuclei_state_assignments.npy"
    if not states_path.exists():
        print(f"  no state assignments; run the fit first. Skipping.")
        continue

    centroids = np.load(out_dir / "nuclei_centroids.npy")
    cell_ids = np.load(out_dir / "nuclei_cell_ids.npy")
    states = np.load(states_path)

    active_all = ~np.isnan(centroids[:, :, 0])
    kept = np.where(active_all.sum(axis=0) >= min_t)[0]
    if states.shape[1] != len(kept):
        raise RuntimeError(
            f"{well}: {states.shape[1]} state columns vs {len(kept)} kept tracks")
    kept_ids = cell_ids[kept]
    active = active_all[:, kept]

    nuc = load_nuclei_masks(well)
    tcell = load_tcell_masks(well)
    num_frames = nuc.shape[0]

    reg = tifffile.memmap(reg_dir / well / "registered.tiff")
    phase = np.asarray(reg[frame_start:frame_end,
                           Y_OFFSET:Y_OFFSET + center_h,
                           X_OFFSET:X_OFFSET + center_w, phase_channel])

    # Per-frame lookup: label id -> state+1 (0 = background, -1 = filtered out)
    max_id = int(nuc.max())
    state_tracks = np.zeros_like(nuc, dtype=np.int32)
    for t in range(num_frames):
        lut = np.zeros(max_id + 1, dtype=np.int32)
        present = kept_ids[active[t]]
        lut[present] = states[t, active[t]] + 1
        # kept-but-inactive and never-kept ids both fall through to NO_STATE
        all_ids = np.unique(nuc[t])
        all_ids = all_ids[all_ids > 0]
        missing = all_ids[lut[all_ids] == 0]
        lut[missing] = NO_STATE_VAL
        state_tracks[t] = lut[nuc[t]]

    def frame_fn(t, _phase=phase, _state=state_tracks, _tcell=tcell,
                 _nuc=nuc, _well=well):
        plt.imshow(_phase[t], cmap="gray")
        if _tcell is not None:
            plt.imshow(np.where(_tcell[t] != 0, 1, 0), cmap="Blues", alpha=0.30)

        none_mask = (_state[t] == NO_STATE_VAL)
        if none_mask.any():
            rgba = np.zeros((*none_mask.shape, 4))
            rgba[none_mask] = NO_STATE_COLOR
            rgba[..., 3] *= 0.55
            plt.imshow(rgba)

        plt.imshow(np.ma.masked_less_equal(_state[t], 0),
                   cmap=state_cmap, norm=state_norm, alpha=0.60)

        if label_ids:
            frame = _nuc[t]
            for tid in np.unique(frame[frame > 0]):
                yx = np.argwhere(frame == tid).mean(axis=0)
                plt.text(yx[1], yx[0], str(int(tid)), color="white",
                         fontsize=SMALL, fontweight="bold", ha="center", va="center")

        handles = [mpatches.Patch(color=state_cmap(s), label=f"State {s}")
                   for s in range(num_states)]
        handles += [mpatches.Patch(color="grey", label="no state (min_t filtered)"),
                    mpatches.Patch(color="tab:blue", label="T cell")]
        plt.legend(title="HMM state", handles=handles, loc="upper left")
        plt.title(f"Well {_well}, frame {t + 1}\ncancer-nucleus HMM state")
        plt.axis("off")

    video_path = out_dir / "nuclei_HMM_overlay_video.mp4"
    print(f"  rendering {num_frames} frames -> {video_path}")
    plt_to_mp4(frame_fn, list(range(num_frames)), str(video_path),
               fps=fps, figsize=figsize)
    print(f"  done: {video_path}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
