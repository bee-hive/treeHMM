"""Step 5: Render overlay videos of nuclei colored by inferred HMM state.

Each frame shows the phase/RFP composite of the 600x600 crop with a marker at
every present nucleus, colored by its state.  Watching these is the real test
of whether the two states correspond to "left alone" versus "in an aggregate" --
the quantitative check lives in ``state_vs_aggregation.png``.

Rendered only for the wells listed in ``video_wells``; all 12 is slow.

Saved per well:
    {output_base_dir}/{well}/nuclei_HMM_overlay_video.mp4

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python create_nuclei_overlay_videos.py
"""

import os
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import yaml
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

out_base_dir: str = cfg["output_base_dir"]
video_wells: List[str] = cfg.get("video_wells", cfg["wells"])
num_states: int = cfg["num_states"]
fps: int = cfg.get("video_fps", 10)
figsize = tuple(cfg.get("video_figsize", [6, 6]))
alpha: float = cfg["dino_rgb_alpha"]
patch_px: int = cfg["dino_patch_px"]

reg_dir = Path(cfg["snakemake_runs_dir"]) / cfg["registration_run"] / "output"
raw_shape = cfg["raw_video_shape"]
center_h, center_w = cfg["video_center_extract"]
frame_start, frame_end = cfg["frame_start"], cfg["frame_end"]
rfp_channel, phase_channel = cfg["rfp_channel"], cfg["phase_channel"]

Y_OFFSET = (raw_shape[1] - center_h) // 2
X_OFFSET = (raw_shape[2] - center_w) // 2


def get_rgb_image_with_nuclei(
    phase_image: np.ndarray,
    nuclei_image: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """Composite phase and RFP nuclei frames into an (H, W, 3) uint8 RGB.

    Args:
        phase_image (np.ndarray): (H, W) phase channel.
        nuclei_image (np.ndarray): (H, W) RFP nuclei channel.
        alpha (float): weight of the nuclei channel in the red plane.

    Returns:
        np.ndarray: (H, W, 3) uint8 RGB composite.
    """
    def _norm(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        lo, hi = float(np.min(img)), float(np.max(img))
        if hi - lo < 1e-8:
            return np.zeros_like(img)
        return (img - lo) / (hi - lo)

    phase = _norm(phase_image) * 255.0
    nuclei = _norm(nuclei_image) * 255.0
    rgb = np.stack([phase] * 3, axis=-1) * (1.0 - alpha)
    rgb[..., 0] += nuclei * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


print("=" * 60)
print("Step 5: Rendering state overlay videos")
print("=" * 60)

state_cmap = plt.get_cmap("viridis", num_states)
half = patch_px // 2

for well in video_wells:
    print("\n" + "-" * 60)
    print(f"Well {well}")

    out_dir = os.path.join(out_base_dir, well)
    centroids = np.load(os.path.join(out_dir, "nuclei_centroids.npy"))
    states_path = os.path.join(out_dir, "nuclei_state_assignments.npy")
    if not os.path.exists(states_path):
        print(f"  no state assignments at {states_path}; run Step 4 first. Skipping.")
        continue
    states = np.load(states_path)

    # Step 4 drops short tracks, so the state array is narrower than the
    # centroid array.  Re-derive which columns survived by track duration.
    num_frames, num_cells = centroids.shape[:2]
    if states.shape[1] != num_cells:
        active_all = ~np.isnan(centroids[:, :, 0])
        durations = active_all.sum(axis=0)
        kept = np.where(durations >= cfg["min_t"])[0]
        if len(kept) != states.shape[1]:
            raise RuntimeError(
                f"{well}: cannot align {states.shape[1]} state columns to "
                f"{len(kept)} kept tracks (min_t={cfg['min_t']})")
        centroids = centroids[:, kept, :]
        num_cells = len(kept)

    reg = tifffile.memmap(reg_dir / well / "registered.tiff")
    crop = np.asarray(
        reg[frame_start:frame_end,
            Y_OFFSET:Y_OFFSET + center_h,
            X_OFFSET:X_OFFSET + center_w, :]
    )

    out_path = os.path.join(out_dir, "nuclei_HMM_overlay_video.mp4")
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="marson_nuclei_dino"))
    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=state_cmap(s),
               markeredgecolor='white', markersize=7, label=f'State {s}')
        for s in range(num_states)
    ]

    with writer.saving(fig, out_path, dpi=150):
        for t in range(num_frames):
            ax.clear()
            rgb = get_rgb_image_with_nuclei(
                crop[t, ..., phase_channel], crop[t, ..., rfp_channel], alpha=alpha)
            ax.imshow(rgb)
            present = np.where(~np.isnan(centroids[t, :, 0]))[0]
            if len(present):
                ax.scatter(centroids[t, present, 1], centroids[t, present, 0],
                           c=[state_cmap(s) for s in states[t, present]],
                           s=22, edgecolors='white', linewidths=0.4)
            ax.set_title(f'{well}  frame {t}  (n={len(present)} nuclei)')
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(handles=legend_handles, loc='upper right', framealpha=0.8)
            writer.grab_frame()
            if t % 50 == 0:
                print(f"  frame {t}/{num_frames}")
    plt.close(fig)
    print(f"  saved -> {out_path}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
