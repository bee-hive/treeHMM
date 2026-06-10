"""
Step 1: Calculate cancer-cell (non-DINO) emissions.

Cancer cells are taken DIRECTLY from the Caliban ground-truth nuclei tracks
(each nucleus track == one cancer cell).  T cells come from the regenerated,
type-separated tracks (channel 0).

Per cancer cell per frame we compute two scalar features:
    0. velocity              - cancer-nucleus velocity (Euclidean centroid displacement)
    1. t_cell_neighbors_20px - number of T cells within radius_px of the nucleus centroid

Saved per crop (into {output_base_dir}/{crop}/):
    cancer_cell_ids.npy          (N,)      canonical sorted nucleus IDs  == column ordering
    cancer_nucleus_centroids.npy (T,N,2)   crop-local (y,x) per cancer column; NaN where absent
    cancer_emissions_array.npy   (T,N,2)   [velocity, t_cell_neighbors_20px]
    cancer_emissions_names.txt             one feature name per line

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python calculate_cancer_emissions.py
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
caliban_base_dir = cfg["caliban_base_dir"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
out_base_dir = cfg["output_base_dir"]
feature_names = cfg["emission_feature_names"]
radius_px = cfg["radius_px"]

# MarsonImagingPipeline utilities
sys.path.insert(0, cfg["imaging_pipeline_dir"])
from scripts.utils.StatUtils import (
    calculate_centroids_per_frame_dict,
    compute_cell_velocities_per_frame_dict,
)

# Time factor (units per frame) for velocity scaling
with open(cfg["experiment_config_path"], "r") as f:
    experiment_config = yaml.safe_load(f)
unit_per_frame = experiment_config.get("TIME_FACTOR", 1)

assert feature_names == ["velocity", "t_cell_neighbors_20px"], (
    f"This script computes exactly [velocity, t_cell_neighbors_20px]; "
    f"config emission_feature_names = {feature_names}"
)


def load_nuclei_tracks(crop):
    """Load Caliban nuclei tracks for a crop -> (T, H, W) int nucleus IDs."""
    nuc = tifffile.imread(os.path.join(caliban_base_dir, f"{crop}.tiff"))
    if nuc.ndim == 4:
        nuc = nuc[..., 0]
    return nuc


def count_tcell_neighbors(nucleus_centroids, tcell_centroids_dict, radius, T, N, id_to_col):
    """Count T-cell centroids within `radius` of each cancer-nucleus centroid.

    Args:
        nucleus_centroids: (T, N, 2) crop-local (y,x) per cancer column (NaN where absent).
        tcell_centroids_dict: {t: {tcell_id: (y,x)}}.
        radius: distance threshold (pixels).
    Returns:
        (T, N) float array of neighbor counts (0 where the nucleus is absent).
    """
    counts = np.zeros((T, N), dtype=float)
    for t in range(T):
        tcells = tcell_centroids_dict.get(t, {})
        if len(tcells) == 0:
            continue
        tc = np.array(list(tcells.values()), dtype=float)  # (M, 2)
        cent_t = nucleus_centroids[t]                       # (N, 2)
        present = ~np.isnan(cent_t[:, 0])
        if not present.any():
            continue
        # pairwise distances: (n_present, M)
        d = np.linalg.norm(cent_t[present, None, :] - tc[None, :, :], axis=2)
        counts[t, present] = np.sum(d <= radius, axis=1)
    return counts


# ============================================================
print("=" * 60)
print("Step 1: Calculating cancer-cell emissions")
print("=" * 60)

for crop in crop_ids:
    print("\n" + "-" * 60)
    print(f"Crop {crop}")

    nuc = load_nuclei_tracks(crop)
    t_cell_tracks = tifffile.imread(
        os.path.join(type_sep_tracks_dir, crop, "tracks.tiff")
    )[..., 0]
    T = nuc.shape[0]

    cancer_cell_ids = np.sort(np.unique(nuc[nuc > 0]))
    N = len(cancer_cell_ids)
    id_to_col = {int(cid): i for i, cid in enumerate(cancer_cell_ids)}
    print(f"  T={T}, num_cancer (nuclei)={N}, num_t_cell_ids="
          f"{len(np.unique(t_cell_tracks[t_cell_tracks > 0]))}")

    # --- Nucleus centroids (T, N, 2), NaN where absent ---
    centroids_dict = calculate_centroids_per_frame_dict(nuc)
    nucleus_centroids = np.full((T, N, 2), np.nan, dtype=float)
    for t, id_to_yx in centroids_dict.items():
        for nid, (y, x) in id_to_yx.items():
            nucleus_centroids[t, id_to_col[int(nid)]] = (y, x)

    # --- Feature 0: nucleus velocity ---
    velocities_dict = compute_cell_velocities_per_frame_dict(
        nuc, unit_per_frame=unit_per_frame
    )
    emissions = np.zeros((T, N, len(feature_names)), dtype=float)
    for t, id_to_v in velocities_dict.items():
        for nid, v in id_to_v.items():
            emissions[t, id_to_col[int(nid)], 0] = v

    # --- Feature 1: T cells within radius_px of the nucleus centroid ---
    tcell_centroids_dict = calculate_centroids_per_frame_dict(t_cell_tracks)
    emissions[:, :, 1] = count_tcell_neighbors(
        nucleus_centroids, tcell_centroids_dict, radius_px, T, N, id_to_col
    )

    # --- Save ---
    out_dir = os.path.join(out_base_dir, crop)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "cancer_cell_ids.npy"), cancer_cell_ids)
    np.save(os.path.join(out_dir, "cancer_nucleus_centroids.npy"), nucleus_centroids)
    np.save(os.path.join(out_dir, "cancer_emissions_array.npy"), emissions)
    with open(os.path.join(out_dir, "cancer_emissions_names.txt"), "w") as fh:
        fh.write("\n".join(feature_names) + "\n")

    print(f"  emissions shape:  {emissions.shape}")
    print(f"  velocity range:   [{emissions[:, :, 0].min():.3f}, {emissions[:, :, 0].max():.3f}]")
    print(f"  neighbors range:  [{emissions[:, :, 1].min():.0f}, {emissions[:, :, 1].max():.0f}]")
    print(f"  saved -> {out_dir}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
