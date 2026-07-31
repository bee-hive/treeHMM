"""
Step 1: Calculate cancer-PHASE-cell (non-DINO) emissions.

Cancer cells are defined by their PHASE masks: channel 1 of the regenerated,
type-separated CVAT tracks (channel 0 = T cells, channel 1 = cancer).  Each
cancer CVAT track == one cancer cell.

Per cancer cell per frame we compute four scalar features:
    0. velocity                 - phase-cell velocity (Euclidean centroid displacement)
    1. area                     - phase-cell area (mask pixel count)
    2. t_cell_neighbors_20px    - number of T cells within radius_px of the phase centroid
    3. dilated_t_cell_neighbors - number of T cells whose mask touches the cancer
                                  phase mask after dilating it by disk(dilate_radius)

Saved per crop (into {output_base_dir}/{crop}/):
    cancer_cell_ids.npy        (N,)      canonical sorted phase track IDs == column ordering
    cancer_phase_centroids.npy (T,N,2)   crop-local (y,x) per cancer column; NaN where absent
    cancer_emissions_array.npy (T,N,4)   [velocity, area, t_cell_neighbors_20px, dilated_t_cell_neighbors]
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

# ---------------------------------------------------------------------------
# Load shared configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
out_base_dir = cfg["output_base_dir"]
feature_names = cfg["emission_feature_names"]
radius_px = cfg["radius_px"]
dilate_radius = cfg["dilate_radius"]

# MarsonImagingPipeline utilities
sys.path.insert(0, cfg["imaging_pipeline_dir"])
from scripts.utils.StatUtils import (
    calculate_centroids_per_frame_dict,
    compute_cell_velocities_per_frame_dict,
    calculate_area,
    get_cell_neighbors,
)

# Time factor (units per frame) for velocity scaling
with open(cfg["experiment_config_path"], "r") as f:
    experiment_config = yaml.safe_load(f)
unit_per_frame = experiment_config.get("TIME_FACTOR", 1)

assert feature_names == [
    "velocity", "area", "t_cell_neighbors_20px", "dilated_t_cell_neighbors"
], (
    f"This script computes exactly [velocity, area, t_cell_neighbors_20px, "
    f"dilated_t_cell_neighbors]; config emission_feature_names = {feature_names}"
)


def load_type_sep_tracks(crop):
    """Load the type-separated tracks for a crop -> (T,H,W,2): ch0=T cell, ch1=cancer phase."""
    tr = tifffile.imread(os.path.join(type_sep_tracks_dir, crop, "tracks.tiff"))
    return tr[..., 0], tr[..., 1]


def count_tcell_neighbors(centroids, tcell_centroids_dict, radius, T, N):
    """Count T-cell centroids within `radius` of each cancer phase-cell centroid.

    Args:
        centroids: (T, N, 2) crop-local (y,x) per cancer column (NaN where absent).
        tcell_centroids_dict: {t: {tcell_id: (y,x)}}.
        radius: distance threshold (pixels).
    Returns:
        (T, N) float array of neighbor counts (0 where the cell is absent).
    """
    counts = np.zeros((T, N), dtype=float)
    for t in range(T):
        tcells = tcell_centroids_dict.get(t, {})
        if len(tcells) == 0:
            continue
        tc = np.array(list(tcells.values()), dtype=float)  # (M, 2)
        cent_t = centroids[t]                              # (N, 2)
        present = ~np.isnan(cent_t[:, 0])
        if not present.any():
            continue
        d = np.linalg.norm(cent_t[present, None, :] - tc[None, :, :], axis=2)
        counts[t, present] = np.sum(d <= radius, axis=1)
    return counts


def count_dilated_tcell_neighbors(cancer_tracks, t_cell_tracks, dilate_r, T, N, id_to_col):
    """Count T cells touching each cancer phase mask dilated by disk(dilate_r).

    For each frame and cancer cell, dilate its phase mask by a disk of radius
    `dilate_r` and count the unique non-zero T-cell IDs overlapping the dilated
    region (reusing StatUtils.get_cell_neighbors).  exclude_ids is None because
    T cells and cancer cells live in separate channels / label spaces.
    """
    counts = np.zeros((T, N), dtype=float)
    for t in range(T):
        c_frame = cancer_tracks[t]
        tc_frame = t_cell_tracks[t]
        for cid in np.unique(c_frame[c_frame > 0]):
            cmask = c_frame == cid
            neighbors = get_cell_neighbors(cmask, tc_frame, dilate_r, exclude_ids=None)
            counts[t, id_to_col[int(cid)]] = len(neighbors)
    return counts


# ============================================================
print("=" * 60)
print("Step 1: Calculating cancer phase-cell emissions")
print("=" * 60)

for crop in crop_ids:
    print("\n" + "-" * 60)
    print(f"Crop {crop}")

    t_cell_tracks, cancer_tracks = load_type_sep_tracks(crop)
    T = cancer_tracks.shape[0]

    cancer_cell_ids = np.sort(np.unique(cancer_tracks[cancer_tracks > 0]))
    N = len(cancer_cell_ids)
    id_to_col = {int(cid): i for i, cid in enumerate(cancer_cell_ids)}
    print(f"  T={T}, num_cancer (phase)={N}, num_t_cell_ids="
          f"{len(np.unique(t_cell_tracks[t_cell_tracks > 0]))}")

    # --- Phase-cell centroids (T, N, 2), NaN where absent ---
    centroids_dict = calculate_centroids_per_frame_dict(cancer_tracks)
    centroids = np.full((T, N, 2), np.nan, dtype=float)
    for t, id_to_yx in centroids_dict.items():
        for cid, (y, x) in id_to_yx.items():
            centroids[t, id_to_col[int(cid)]] = (y, x)

    emissions = np.zeros((T, N, len(feature_names)), dtype=float)

    # --- Feature 0: phase-cell velocity ---
    velocities_dict = compute_cell_velocities_per_frame_dict(
        cancer_tracks, unit_per_frame=unit_per_frame
    )
    for t, id_to_v in velocities_dict.items():
        for cid, v in id_to_v.items():
            emissions[t, id_to_col[int(cid)], 0] = v

    # --- Feature 1: phase-cell area (mask pixel count) ---
    for t in range(T):
        c_frame = cancer_tracks[t]
        for cid in np.unique(c_frame[c_frame > 0]):
            emissions[t, id_to_col[int(cid)], 1] = calculate_area(c_frame == cid)

    # --- Feature 2: T cells within radius_px of the phase centroid ---
    tcell_centroids_dict = calculate_centroids_per_frame_dict(t_cell_tracks)
    emissions[:, :, 2] = count_tcell_neighbors(
        centroids, tcell_centroids_dict, radius_px, T, N
    )

    # --- Feature 3: T cells touching the dilated phase mask ---
    emissions[:, :, 3] = count_dilated_tcell_neighbors(
        cancer_tracks, t_cell_tracks, dilate_radius, T, N, id_to_col
    )

    # --- Save ---
    out_dir = os.path.join(out_base_dir, crop)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "cancer_cell_ids.npy"), cancer_cell_ids)
    np.save(os.path.join(out_dir, "cancer_phase_centroids.npy"), centroids)
    np.save(os.path.join(out_dir, "cancer_emissions_array.npy"), emissions)
    with open(os.path.join(out_dir, "cancer_emissions_names.txt"), "w") as fh:
        fh.write("\n".join(feature_names) + "\n")

    print(f"  emissions shape:  {emissions.shape}")
    print(f"  velocity range:   [{emissions[:, :, 0].min():.3f}, {emissions[:, :, 0].max():.3f}]")
    print(f"  area range:       [{emissions[:, :, 1].min():.0f}, {emissions[:, :, 1].max():.0f}]")
    print(f"  nbr(20px) range:  [{emissions[:, :, 2].min():.0f}, {emissions[:, :, 2].max():.0f}]")
    print(f"  dilated nbr range:[{emissions[:, :, 3].min():.0f}, {emissions[:, :, 3].max():.0f}]")
    print(f"  saved -> {out_dir}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
