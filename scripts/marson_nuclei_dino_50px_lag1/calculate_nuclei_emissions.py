"""Step 1: Build per-nucleus centroids and diagnostic scalar features.

Cancer cells ARE the Caliban nuclei tracks.  Everything needed is already in
``SimplePlotsV3/stats/{well}/cell_data.parquet`` (nuclei rows carry raw Caliban
label IDs), so no TIFF is touched here.

The scalar features computed here are, by default, NOT fed to the model -- the
emissions are DINO PCs only.  They exist so that the question "did the two
states separate isolated cells from aggregates?" can be answered against data
the model never saw.  ``nuclei_neighbors_30px`` is the primary validator.

Saved per well (into ``{output_base_dir}/{well}/``):
    nuclei_cell_ids.npy       (N,)      sorted Caliban IDs == column ordering
    nuclei_centroids.npy      (T, N, 2) crop-local (y, x); NaN where absent
    nuclei_emissions_array.npy(T, N, F) diagnostic scalar features
    nuclei_emissions_names.txt          one feature name per line

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python calculate_nuclei_emissions.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

wells: List[str] = cfg["wells"]
feature_names: List[str] = cfg["emission_feature_names"]
out_base_dir: str = cfg["output_base_dir"]
nuclei_radii: List[int] = cfg["nuclei_neighbor_radii"]
t_cell_radius: float = cfg["t_cell_radius_px"]
nearest_cap: float = cfg["nearest_dist_cap_px"]

stats_dir = (
    Path(cfg["snakemake_runs_dir"])
    / cfg["analysis_run"]
    / "output"
    / "SimplePlotsV3"
    / "stats"
)

EXPECTED_FEATURES = [
    "velocity",
    "area",
    "circularity",
    "nuclei_neighbors_30px",
    "nuclei_neighbors_50px",
    "nearest_nucleus_dist",
    "t_cell_neighbors_20px",
]
if feature_names != EXPECTED_FEATURES:
    raise ValueError(
        f"This script computes exactly {EXPECTED_FEATURES}; "
        f"config emission_feature_names = {feature_names}"
    )
if nuclei_radii != [30, 50]:
    raise ValueError(
        f"Feature names hardcode the 30/50 px radii; got nuclei_neighbor_radii={nuclei_radii}"
    )


def build_dense_arrays(
    nuclei: pd.DataFrame,
    id_to_col: Dict[int, int],
    num_frames: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Scatter per-frame nucleus rows into dense (T, N, ...) arrays.

    Args:
        nuclei (pd.DataFrame): nuclei rows of cell_data.parquet.
        id_to_col (Dict[int, int]): Caliban ID -> column index.
        num_frames (int): number of frames T.

    Returns:
        Tuple[np.ndarray, np.ndarray]: centroids (T, N, 2) with NaN where the
            nucleus is absent, and a (T, N, 3) array of
            [velocity, area, circularity] with 0 where absent.
    """
    num_cells = len(id_to_col)
    centroids = np.full((num_frames, num_cells, 2), np.nan, dtype=float)
    scalars = np.zeros((num_frames, num_cells, 3), dtype=float)

    cols = nuclei["cell_id"].map(id_to_col).to_numpy()
    frames = nuclei["frame"].to_numpy()
    centroids[frames, cols, 0] = nuclei["centroid_y"].to_numpy()
    centroids[frames, cols, 1] = nuclei["centroid_x"].to_numpy()
    # velocity is NaN on each track's first frame by construction; store 0 there.
    scalars[frames, cols, 0] = np.nan_to_num(nuclei["velocity"].to_numpy())
    scalars[frames, cols, 1] = np.nan_to_num(nuclei["area"].to_numpy())
    scalars[frames, cols, 2] = np.nan_to_num(nuclei["circularity"].to_numpy())
    return centroids, scalars


def count_self_neighbors(
    centroids: np.ndarray,
    radii: List[int],
    nearest_cap: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Count other nuclei within each radius, and distance to the nearest one.

    Self-matches are excluded.  Frames where a nucleus is absent stay at 0
    (counts) / ``nearest_cap`` (distance).

    Args:
        centroids (np.ndarray): (T, N, 2) crop-local (y, x), NaN where absent.
        radii (List[int]): radii in pixels.
        nearest_cap (float): value used when a frame holds fewer than 2 nuclei.

    Returns:
        Tuple[np.ndarray, np.ndarray]: counts (T, N, len(radii)) and nearest
            distances (T, N).
    """
    num_frames, num_cells = centroids.shape[:2]
    counts = np.zeros((num_frames, num_cells, len(radii)), dtype=float)
    nearest = np.full((num_frames, num_cells), nearest_cap, dtype=float)

    for t in range(num_frames):
        present = ~np.isnan(centroids[t, :, 0])
        if present.sum() < 2:
            continue
        pts = centroids[t, present]
        tree = cKDTree(pts)
        for r_idx, radius in enumerate(radii):
            # -1 removes the self-match that query_ball_point always returns.
            n_within = np.array(
                [len(hits) for hits in tree.query_ball_point(pts, r=radius)],
                dtype=float,
            ) - 1.0
            counts[t, present, r_idx] = n_within
        # k=2 -> [self (distance 0), nearest other]
        dists, _ = tree.query(pts, k=2)
        nearest[t, present] = dists[:, 1]
    return counts, nearest


def count_cross_neighbors(
    centroids: np.ndarray,
    other_df: pd.DataFrame,
    radius: float
) -> np.ndarray:
    """Count centroids of another population within ``radius`` of each nucleus.

    Args:
        centroids (np.ndarray): (T, N, 2) nucleus centroids, NaN where absent.
        other_df (pd.DataFrame): rows with frame / centroid_y / centroid_x.
        radius (float): distance threshold in pixels.

    Returns:
        np.ndarray: (T, N) counts, 0 where the nucleus is absent.
    """
    num_frames, num_cells = centroids.shape[:2]
    counts = np.zeros((num_frames, num_cells), dtype=float)
    by_frame = {f: g for f, g in other_df.groupby("frame")}

    for t in range(num_frames):
        group = by_frame.get(t)
        if group is None or len(group) == 0:
            continue
        present = ~np.isnan(centroids[t, :, 0])
        if not present.any():
            continue
        other_pts = group[["centroid_y", "centroid_x"]].to_numpy(dtype=float)
        tree = cKDTree(other_pts)
        hits = tree.query_ball_point(centroids[t, present], r=radius)
        counts[t, present] = [len(h) for h in hits]
    return counts


print("=" * 60)
print("Step 1: Building nuclei centroids + diagnostic features")
print("=" * 60)

for well in wells:
    print("\n" + "-" * 60)
    print(f"Well {well}")

    well_stats = stats_dir / well
    cell_data = pd.read_parquet(well_stats / "cell_data.parquet")
    with open(well_stats / "metadata.json") as fh:
        metadata = json.load(fh)
    num_frames = metadata["num_frames"]

    nuclei = cell_data[cell_data["cell_type"] == "nuclei"]
    t_cells = cell_data[cell_data["cell_type"] == "t_cell"]

    cell_ids = np.sort(nuclei["cell_id"].unique())
    id_to_col = {int(cid): i for i, cid in enumerate(cell_ids)}
    num_cells = len(cell_ids)
    print(f"  T={num_frames}, num_nuclei={num_cells}, "
          f"nuclei cell-frames={len(nuclei)}, t_cell rows={len(t_cells)}")

    centroids, scalars = build_dense_arrays(nuclei, id_to_col, num_frames)
    nuc_counts, nearest = count_self_neighbors(centroids, nuclei_radii, nearest_cap)
    t_counts = count_cross_neighbors(centroids, t_cells, t_cell_radius)

    emissions = np.concatenate(
        [scalars, nuc_counts, nearest[..., None], t_counts[..., None]], axis=-1
    )
    if emissions.shape[-1] != len(feature_names):
        raise RuntimeError(
            f"assembled {emissions.shape[-1]} features but named {len(feature_names)}"
        )

    out_dir = os.path.join(out_base_dir, well)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "nuclei_cell_ids.npy"), cell_ids)
    np.save(os.path.join(out_dir, "nuclei_centroids.npy"), centroids)
    np.save(os.path.join(out_dir, "nuclei_emissions_array.npy"), emissions)
    with open(os.path.join(out_dir, "nuclei_emissions_names.txt"), "w") as fh:
        fh.write("\n".join(feature_names) + "\n")

    present = ~np.isnan(centroids[:, :, 0])
    print(f"  emissions shape: {emissions.shape}")
    print(f"  mean nuclei within 30px (active only): "
          f"{nuc_counts[:, :, 0][present].mean():.2f}")
    print(f"  mean nuclei within 50px (active only): "
          f"{nuc_counts[:, :, 1][present].mean():.2f}")
    print(f"  median nearest-nucleus dist (px):      "
          f"{np.median(nearest[present]):.1f}")
    print(f"  saved -> {out_dir}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
