"""
Calculate T cell emissions for all ground-truth crops.

Computes 3 features for every time frame a T cell is present across six
video crops:
    0. velocity          - instantaneous velocity
    1. cancer_contact    - binary indicator of cancer-cell contact
    2. t_cell_neighbors  - number of T-cell neighbours

Each crop's emissions are saved as a 3-D numpy array
(T x num_cells x 3) and a companion feature-name file.

Usage (AnalysisEnv):
    conda run -n AnalysisEnv python calculate_emissions.py
"""

import os
import sys
import pickle
from pathlib import Path

import yaml
import tifffile
import numpy as np

# ---------------------------------------------------------------------------
# Load shared configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
conditions_dict = cfg["conditions_dict"]
FEATURE_NAMES = cfg["emission_feature_names"]
cvat_base_dir = cfg["cvat_base_dir"]
caliban_base_dir = cfg["caliban_base_dir"]
out_base_dir = cfg["output_base_dir"]

# ---------------------------------------------------------------------------
# Add the pipeline repo to sys.path so its packages are importable
# ---------------------------------------------------------------------------
sys.path.insert(0, cfg["imaging_pipeline_dir"])

from scripts.utils.StatUtils import (
    compute_cell_velocities_per_frame_dict,
    compute_cell_cell_neighbor_dict,
    compute_all_cell_cell_neighbor_dict,
)
from scripts.utils.CellTypingUtils import filter_tracks

# Load external experiment config for TIME_FACTOR
with open(cfg["experiment_config_path"], "r") as f:
    experiment_config = yaml.safe_load(f)


# ============================================================
# Step 1: Load tracks
# ============================================================
def get_gt_info_per_well(conditions):
    """Load CVAT tracks, nuclear (Caliban) tracks, and cell-type dicts."""
    cvat_tracks_per_well = {}
    nuclear_tracks_per_well = {}
    cell_type_dict_per_well = {}

    for condition in conditions:
        for crop in conditions[condition]:
            well_id = crop.split("_")[0]
            cvat_tracks_per_well[crop] = tifffile.imread(
                os.path.join(cvat_base_dir, well_id, crop, "ALL_tracks.tiff")
            )
            caliban_tracks = tifffile.imread(
                os.path.join(caliban_base_dir, f"{crop}.tiff")
            )
            caliban_tracks = caliban_tracks[..., 0]
            nuclear_tracks_per_well[crop] = caliban_tracks
            cell_type_dict_per_well[crop] = pickle.load(
                open(
                    os.path.join(cvat_base_dir, well_id, crop, "full_cell_type_dict.pkl"),
                    "rb",
                )
            )
    return cvat_tracks_per_well, nuclear_tracks_per_well, cell_type_dict_per_well


print("=" * 60)
print("Step 1: Loading tracks")
print("=" * 60)

cvat_tracks_per_well, nuclear_tracks_per_well, cell_type_dict_per_well = (
    get_gt_info_per_well(conditions_dict)
)


# ============================================================
# Step 2: Build per-type track arrays
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Building per-type track arrays")
print("=" * 60)

type_tracks_per_well = {}
for cell_type in ["cancer", "t_cell", "nuclei"]:
    type_tracks_per_well[cell_type] = {}
    if cell_type == "nuclei":
        type_tracks_per_well[cell_type] = nuclear_tracks_per_well.copy()
    else:
        type_tracks_per_well[cell_type] = {
            well: filter_tracks(
                cell_type, cvat_tracks_per_well[well], cell_type_dict_per_well[well],
            )
            for well in cvat_tracks_per_well.keys()
        }

t_cell_tracks = type_tracks_per_well["t_cell"]


# ============================================================
# Step 3: Calculate statistics
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Calculating T cell velocities")
print("=" * 60)

t_cell_velocities_per_frame = {
    well: compute_cell_velocities_per_frame_dict(
        t_cell_tracks[well], unit_per_frame=experiment_config.get("TIME_FACTOR", 1)
    )
    for well in t_cell_tracks.keys()
}

print("\n" + "=" * 60)
print("Step 3b: Calculating type-specific neighbors")
print("=" * 60)

cancer_type_specific_neighbors_per_frame = {}
t_cell_type_specific_neighbors_per_frame = {}

for well in cvat_tracks_per_well.keys():
    well_t = type_tracks_per_well["t_cell"][well]
    well_c = type_tracks_per_well["cancer"][well]

    cn, tn = compute_cell_cell_neighbor_dict(well_t, well_c)

    cancer_type_specific_neighbors_per_frame[well] = cn
    t_cell_type_specific_neighbors_per_frame[well] = tn

print("\n" + "=" * 60)
print("Step 3c: Calculating type-agnostic neighbors")
print("=" * 60)

cancer_all_neighbors_per_frame = {}
t_cell_all_neighbors_per_frame = {}

for well in cvat_tracks_per_well.keys():
    well_t = type_tracks_per_well["t_cell"][well]
    well_c = type_tracks_per_well["cancer"][well]

    cn, tn = compute_all_cell_cell_neighbor_dict(well_t, well_c)

    cancer_all_neighbors_per_frame[well] = cn
    t_cell_all_neighbors_per_frame[well] = tn


# ============================================================
# Step 4: Create and save emissions arrays
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Creating emissions arrays per crop")
print("=" * 60)

for crop in crop_ids:
    example_t_cell_tracks = type_tracks_per_well["t_cell"][crop]
    example_velocities = t_cell_velocities_per_frame[crop]
    example_type_neighbors = t_cell_type_specific_neighbors_per_frame[crop]
    example_all_neighbors = t_cell_all_neighbors_per_frame[crop]

    t_cell_ids = np.unique(example_t_cell_tracks[example_t_cell_tracks > 0])
    id_to_col = {cell_id: idx for idx, cell_id in enumerate(t_cell_ids)}

    emissions_array = np.zeros((50, len(t_cell_ids), 3))

    # Feature 0: velocity
    for frame, velocities in example_velocities.items():
        for cell_id, velocity in velocities.items():
            if cell_id in id_to_col:
                emissions_array[frame, id_to_col[cell_id], 0] = velocity

    # Feature 1: binary cancer contact, Feature 2: t_cell_neighbors
    for frame in example_all_neighbors.keys():
        all_nbrs = example_all_neighbors[frame]
        type_nbrs = example_type_neighbors[frame]
        for cell_id, neighbors in all_nbrs.items():
            if cell_id in id_to_col:
                col = id_to_col[cell_id]
                cancer_n = type_nbrs.get(cell_id, [0])[0]
                emissions_array[frame, col, 1] = 1 if cancer_n > 0 else 0
                emissions_array[frame, col, 2] = neighbors - cancer_n

    # Drop t=0 frame (velocity undefined between t=-1 and t=0)
    emissions_array = emissions_array[1:, ...]
    print(f"  {crop}: emissions_array.shape = {emissions_array.shape}")

    out_dir = os.path.join(out_base_dir, crop)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "t_cell_emissions_array.npy"), emissions_array)

print("\nDone!")
