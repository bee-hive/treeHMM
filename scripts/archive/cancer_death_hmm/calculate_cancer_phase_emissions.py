"""Step 1: Per-cancer-cell, per-frame features from the phase masks.

Cancer cells are defined by their PHASE masks -- channel 1 of the
type-separated CVAT tracks (channel 0 = T cells).  Each cancer CVAT track
is one cancer cell, and the CVAT track ID is the ID space the ground-truth
death annotations use, so columns here index those annotations directly.

Ten features are computed and saved.  Every arm in config.yml selects a
subset to feed the model; the remainder stay available as
annotation-independent diagnostics for state profiling and evaluation.

    0. area                      phase-mask pixel count
    1. circularity               4*pi*area / perimeter^2, largest component
    2. velocity                  centroid displacement, px/frame
    3. t_cell_neighbors_20px     T cells within radius_px of the centroid
    4. dilated_t_cell_neighbors  T cells touching the mask dilated by disk(r)
    5. d_area_frac               (area_t - area_prev) / area_prev
    6. d_circularity             circularity_t - circularity_prev
    7. win_std_log_area          trailing rolling std of log(area)
    8. win_std_circularity       trailing rolling std of circularity
    9. win_std_displacement      trailing rolling std of per-frame displacement

Features 5-6 make the single-frame death transition an explicit
observable; features 7-9 make the elevated post-death instability an
explicit observable, so a death state can persist rather than firing on
one frame.  Deltas and windows are computed over a cell's ACTIVE frames
(gaps are skipped, not treated as zeros).

Saved per crop, into {output_base_dir}/{crop}/:
    cancer_cell_ids.npy         (N,)     sorted CVAT track IDs == column order
    cancer_phase_centroids.npy  (T,N,2)  crop-local (y,x); NaN where absent
    cancer_emissions_array.npy  (T,N,10) the features above
    cancer_emissions_names.txt           one feature name per line

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python calculate_cancer_phase_emissions.py
"""

import os
import sys
from pathlib import Path

import yaml
import numpy as np
import tifffile
from skimage.measure import label as sklabel, regionprops

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
out_base_dir = cfg["output_base_dir"]
feature_names = cfg["emission_feature_names"]
radius_px = cfg["radius_px"]
dilate_radius = cfg["dilate_radius"]
window_frames = cfg["window_frames"]

sys.path.insert(0, cfg["imaging_pipeline_dir"])
from scripts.utils.StatUtils import (  # noqa: E402
    calculate_centroids_per_frame_dict,
    get_cell_neighbors,
)

EXPECTED = [
    "area", "circularity", "velocity", "t_cell_neighbors_20px",
    "dilated_t_cell_neighbors", "d_area_frac", "d_circularity",
    "win_std_log_area", "win_std_circularity", "win_std_displacement",
]
assert feature_names == EXPECTED, (
    f"This script computes exactly {EXPECTED}; "
    f"config emission_feature_names = {feature_names}")

IDX = {name: i for i, name in enumerate(feature_names)}


def circularity_of(mask):
    """4*pi*area / perimeter^2 for the largest connected component of `mask`.

    Args:
        mask (np.ndarray): (H, W) boolean single-cell mask.

    Returns:
        float: circularity, or NaN if the mask is empty or has no perimeter.
            A perfect disc scores 1.0; discretization can push small masks
            slightly above 1.0, which is left uncorrected.
    """
    props = regionprops(sklabel(mask.astype(int)))
    if not props:
        return np.nan
    region = max(props, key=lambda p: p.area)
    perimeter = region.perimeter
    if perimeter <= 0:
        return np.nan
    return float(4.0 * np.pi * region.area / (perimeter ** 2))


def trailing_std(series, active, window):
    """Trailing rolling std of `series` over a cell's active frames.

    At frame t the window spans the active frames in [t-window+1, t], so the
    statistic rises on the frame the change happens and stays elevated for
    `window` frames afterwards -- which is what lets a death state persist.

    Args:
        series (np.ndarray): (T,) values, NaN where the cell is absent.
        active (np.ndarray): (T,) bool, True where the cell is present.
        window (int): trailing window length in frames.

    Returns:
        np.ndarray: (T,) rolling std; 0.0 where fewer than two active frames
            fall in the window (nothing to measure yet).
    """
    out = np.zeros(len(series), dtype=float)
    for t in np.where(active)[0]:
        lo = max(0, t - window + 1)
        vals = series[lo:t + 1][active[lo:t + 1]]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= 2:
            out[t] = float(np.std(vals))
    return out


def count_tcell_neighbors(centroids, tcell_centroids_dict, radius, T, N):
    """Count T-cell centroids within `radius` px of each cancer centroid.

    Args:
        centroids (np.ndarray): (T, N, 2) crop-local (y, x), NaN where absent.
        tcell_centroids_dict (dict): {t: {tcell_id: (y, x)}}.
        radius (float): distance threshold in pixels.
        T (int): number of frames.
        N (int): number of cancer columns.

    Returns:
        np.ndarray: (T, N) counts, 0 where the cell is absent.
    """
    counts = np.zeros((T, N), dtype=float)
    for t in range(T):
        tcells = tcell_centroids_dict.get(t, {})
        if not tcells:
            continue
        tc = np.array(list(tcells.values()), dtype=float)
        cent_t = centroids[t]
        present = ~np.isnan(cent_t[:, 0])
        if not present.any():
            continue
        d = np.linalg.norm(cent_t[present, None, :] - tc[None, :, :], axis=2)
        counts[t, present] = np.sum(d <= radius, axis=1)
    return counts


print("=" * 60)
print("Step 1: Cancer phase-cell features")
print("=" * 60)

for crop in crop_ids:
    print("\n" + "-" * 60)
    print(f"Crop {crop}")

    tracks = tifffile.imread(os.path.join(type_sep_tracks_dir, crop, "tracks.tiff"))
    t_cell_tracks, cancer_tracks = tracks[..., 0], tracks[..., 1]
    T = cancer_tracks.shape[0]

    cancer_cell_ids = np.sort(np.unique(cancer_tracks[cancer_tracks > 0]))
    N = len(cancer_cell_ids)
    id_to_col = {int(cid): i for i, cid in enumerate(cancer_cell_ids)}
    print(f"  T={T}, num_cancer={N}, num_t_cell_ids="
          f"{len(np.unique(t_cell_tracks[t_cell_tracks > 0]))}")

    emissions = np.zeros((T, N, len(feature_names)), dtype=float)
    centroids = np.full((T, N, 2), np.nan, dtype=float)
    active = np.zeros((T, N), dtype=bool)

    # --- Per-frame area, circularity, centroid, dilated T-cell contact ---
    for t in range(T):
        c_frame = cancer_tracks[t]
        tc_frame = t_cell_tracks[t]
        for cid in np.unique(c_frame[c_frame > 0]):
            col = id_to_col[int(cid)]
            mask = c_frame == cid
            active[t, col] = True
            emissions[t, col, IDX["area"]] = float(mask.sum())
            emissions[t, col, IDX["circularity"]] = circularity_of(mask)
            ys, xs = np.nonzero(mask)
            centroids[t, col] = (ys.mean(), xs.mean())
            emissions[t, col, IDX["dilated_t_cell_neighbors"]] = len(
                get_cell_neighbors(mask, tc_frame, dilate_radius, exclude_ids=None))

    # --- T cells within radius_px of the centroid ---
    tcell_centroids_dict = calculate_centroids_per_frame_dict(t_cell_tracks)
    emissions[:, :, IDX["t_cell_neighbors_20px"]] = count_tcell_neighbors(
        centroids, tcell_centroids_dict, radius_px, T, N)

    # --- Per-cell temporal features over ACTIVE frames only ---
    # Deltas are taken against the previous ACTIVE frame, so a tracking gap
    # produces one wide delta rather than a spurious drop to zero and back.
    for col in range(N):
        act = active[:, col]
        frames = np.where(act)[0]
        if len(frames) == 0:
            continue

        area = np.where(act, emissions[:, col, IDX["area"]], np.nan)
        circ = np.where(act, emissions[:, col, IDX["circularity"]], np.nan)
        disp = np.zeros(T, dtype=float)

        for i in range(1, len(frames)):
            t, prev = frames[i], frames[i - 1]
            if area[prev] > 0:
                emissions[t, col, IDX["d_area_frac"]] = (
                    (area[t] - area[prev]) / area[prev])
            if not (np.isnan(circ[t]) or np.isnan(circ[prev])):
                emissions[t, col, IDX["d_circularity"]] = circ[t] - circ[prev]
            step = float(np.linalg.norm(centroids[t, col] - centroids[prev, col]))
            # Normalize by the gap so a 2-frame gap is not read as a 2x jump.
            disp[t] = step / float(t - prev)

        emissions[:, col, IDX["velocity"]] = disp

        log_area = np.where(act & (area > 0), np.log(np.where(area > 0, area, 1.0)), np.nan)
        emissions[:, col, IDX["win_std_log_area"]] = trailing_std(log_area, act, window_frames)
        emissions[:, col, IDX["win_std_circularity"]] = trailing_std(circ, act, window_frames)
        emissions[:, col, IDX["win_std_displacement"]] = trailing_std(
            np.where(act, disp, np.nan), act, window_frames)

    # Circularity can be NaN for degenerate masks; the model cannot take NaN.
    # Zero is outside the plausible circularity range so it stays identifiable.
    n_nan = int(np.isnan(emissions).sum())
    if n_nan:
        print(f"  WARNING: {n_nan} NaN feature values (degenerate masks) -> 0.0")
        emissions = np.nan_to_num(emissions, nan=0.0)

    out_dir = os.path.join(out_base_dir, crop)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "cancer_cell_ids.npy"), cancer_cell_ids)
    np.save(os.path.join(out_dir, "cancer_phase_centroids.npy"), centroids)
    np.save(os.path.join(out_dir, "cancer_emissions_array.npy"), emissions)
    with open(os.path.join(out_dir, "cancer_emissions_names.txt"), "w") as fh:
        fh.write("\n".join(feature_names) + "\n")

    act_vals = emissions[active]
    print(f"  emissions {emissions.shape}, {int(active.sum())} active cell-frames")
    for i, name in enumerate(feature_names):
        v = act_vals[:, i]
        print(f"    {name:<26s} mean={v.mean():9.3f}  "
              f"[{v.min():9.3f}, {v.max():9.3f}]")
    print(f"  saved -> {out_dir}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
