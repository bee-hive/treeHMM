"""
Calculate T cell emissions for all ground-truth crops.

Computes 3 features for every time frame a T cell is present across six
video crops:
    0. velocity          – instantaneous velocity
    1. cancer_neighbors  – number of cancer-cell neighbours
    2. t_cell_neighbors  – number of T-cell neighbours

Each crop's emissions are saved as a 3-D numpy array
(T × num_cells × 3) and a companion feature-name file.

Usage (AnalysisEnv):
    conda run -n AnalysisEnv python calculating_example_t_cell_emissions.py
"""

import os
import sys
import pickle
from pathlib import Path

import yaml
import tifffile
import numpy as np

# ---------------------------------------------------------------------------
# Add the pipeline repo to sys.path so its packages are importable
# ---------------------------------------------------------------------------
sys.path.insert(0, "/gladstone/engelhardt/lab/adamw/MarsonImagingPipeline")

from scripts.utils.StatUtils import (
    compute_cell_velocities_per_frame_dict,
    compute_cell_cell_contact_dict,
    compute_all_cell_cell_contact_dict,
    compute_cell_cell_neighbor_dict,
    compute_all_cell_cell_neighbor_dict,
)
from scripts.utils.CellTypingUtils import filter_tracks

# ============================================================
# Configuration
# ============================================================
config_path = Path(
    "/gladstone/engelhardt/lab/jadjasu/LiveCellUmbrella/"
    "MarsonImagingPipeline/snakemake_configs/experiment.yml"
)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

cvat_base_dir = (
    "/gladstone/engelhardt/lab/adamw/MarsonImagingPipeline/"
    "data/ground_truth_tracking_annotations/cvat_annotations/TCR-T/"
)
caliban_base_dir = (
    "/gladstone/engelhardt/lab/MarsonLabIncucyteData/groundTruthCalibanTracks/"
)

crop_ids = [
    "B4_t50t100y200y350x750x900",
    "B8_t50t100y200y350x750x900",
    "E4_t50t100y200y350x750x900",
    "B4_t250t300y200y350x750x900",
    "B8_t250t300y200y350x750x900",
    "E4_t250t300y200y350x750x900",
]

conditions_dict = {
    "SH":    ["B4_t50t100y200y350x750x900", "B4_t250t300y200y350x750x900"],
    "RASA2": ["E4_t50t100y200y350x750x900", "E4_t250t300y200y350x750x900"],
    "CUL5":  ["B8_t50t100y200y350x750x900", "B8_t250t300y200y350x750x900"],
}

# Ordered list of the emission features (axis-2 of the emissions array)
FEATURE_NAMES = ["velocity", "cancer_neighbors", "t_cell_neighbors"]

out_base_dir = "/gladstone/engelhardt/lab/adamw/treeHMM/notebooks/data/example_gt_crop"


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

            # get nuclear tracks
            caliban_tracks = tifffile.imread(
                os.path.join(caliban_base_dir, f"{crop}.tiff")
            )
            caliban_tracks = caliban_tracks[..., 0]
            nuclear_tracks_per_well[crop] = caliban_tracks

            # get cell type dict
            cell_type_dict_per_well[crop] = pickle.load(
                open(
                    os.path.join(
                        cvat_base_dir, well_id, crop, "full_cell_type_dict.pkl"
                    ),
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
                cell_type,
                cvat_tracks_per_well[well],
                cell_type_dict_per_well[well],
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
        t_cell_tracks[well], unit_per_frame=config.get("TIME_FACTOR", 1)
    )
    for well in t_cell_tracks.keys()
}

print("\n" + "=" * 60)
print("Step 3b: Calculating type-specific interactions")
print("=" * 60)

cancer_type_specific_contacts_per_frame, t_cell_type_specific_contacts_per_frame = {}, {}
cancer_type_specific_neighbors_per_frame, t_cell_type_specific_neighbors_per_frame = {}, {}

for well in cvat_tracks_per_well.keys():
    well_t_cell_tracks = type_tracks_per_well["t_cell"][well]
    well_cancer_tracks = type_tracks_per_well["cancer"][well]

    well_cancer_contacts, well_t_cell_contacts = compute_cell_cell_contact_dict(
        well_t_cell_tracks, well_cancer_tracks
    )
    well_cancer_neighbors, well_t_cell_neighbors = compute_cell_cell_neighbor_dict(
        well_t_cell_tracks, well_cancer_tracks
    )

    cancer_type_specific_contacts_per_frame[well] = well_cancer_contacts
    t_cell_type_specific_contacts_per_frame[well] = well_t_cell_contacts

    cancer_type_specific_neighbors_per_frame[well] = well_cancer_neighbors
    t_cell_type_specific_neighbors_per_frame[well] = well_t_cell_neighbors

print("\n" + "=" * 60)
print("Step 3c: Calculating type-agnostic interactions")
print("=" * 60)

cancer_all_contacts_per_frame, t_cell_all_contacts_per_frame = {}, {}
cancer_all_neighbors_per_frame, t_cell_all_neighbors_per_frame = {}, {}

for well in cvat_tracks_per_well.keys():
    well_t_cell_tracks = type_tracks_per_well["t_cell"][well]
    well_cancer_tracks = type_tracks_per_well["cancer"][well]

    well_cancer_contacts, well_t_cell_contacts = compute_all_cell_cell_contact_dict(
        well_t_cell_tracks, well_cancer_tracks
    )
    well_cancer_neighbors, well_t_cell_neighbors = compute_all_cell_cell_neighbor_dict(
        well_t_cell_tracks, well_cancer_tracks
    )

    cancer_all_contacts_per_frame[well] = well_cancer_contacts
    t_cell_all_contacts_per_frame[well] = well_t_cell_contacts

    cancer_all_neighbors_per_frame[well] = well_cancer_neighbors
    t_cell_all_neighbors_per_frame[well] = well_t_cell_neighbors


# ============================================================
# Step 4: Create and save emissions arrays
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Creating emissions arrays per crop")
print("=" * 60)

for crop in crop_ids:
    # subset to the desired tracks
    example_t_cell_tracks = type_tracks_per_well["t_cell"][crop]

    example_t_cell_velocities_per_frame = t_cell_velocities_per_frame[crop]
    example_t_cell_type_specific_neighbors_per_frame = (
        t_cell_type_specific_neighbors_per_frame[crop]
    )
    example_t_cell_all_neighbors_per_frame = t_cell_all_neighbors_per_frame[crop]

    t_cell_ids = np.unique(example_t_cell_tracks[example_t_cell_tracks > 0])
    id_to_column_index = {cell_id: index for index, cell_id in enumerate(t_cell_ids)}

    # shape: (num_frames, num_t_cells, num_emission_features)
    emissions_array = np.zeros((50, len(t_cell_ids), 3))

    # Feature 0: velocity
    for frame, velocities in example_t_cell_velocities_per_frame.items():
        for cell_id, velocity in velocities.items():
            if cell_id in id_to_column_index:
                column_index = id_to_column_index[cell_id]
                emissions_array[frame, column_index, 0] = velocity

    # Features 1 & 2: cancer_neighbors and t_cell_neighbors
    for frame in example_t_cell_all_neighbors_per_frame.keys():
        all_neighbors = example_t_cell_all_neighbors_per_frame[frame]
        type_specific_neighbors = example_t_cell_type_specific_neighbors_per_frame[
            frame
        ]

        for cell_id, neighbors in all_neighbors.items():
            if cell_id in id_to_column_index:
                column_index = id_to_column_index[cell_id]

                cancer_neighbors_list = type_specific_neighbors.get(cell_id, [0])
                cancer_neighbors = cancer_neighbors_list[0]

                t_cell_neighbors = neighbors - cancer_neighbors

                emissions_array[frame, column_index, 1] = cancer_neighbors
                emissions_array[frame, column_index, 2] = t_cell_neighbors

    # Drop the t=0 frame because velocity is undefined between t=-1 and t=0
    emissions_array = emissions_array[1:, ...]
    print(f"  {crop}: emissions_array.shape = {emissions_array.shape}")

    # Save the emissions array and feature names for this crop
    out_dir = os.path.join(out_base_dir, crop)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "t_cell_emissions_array.npy"), emissions_array)

    # Save feature names alongside the emissions array
    with open(os.path.join(out_dir, "t_cell_emissions_names.txt"), "w") as f:
        for name in FEATURE_NAMES:
            f.write(name + "\n")

print("\nDone!")
