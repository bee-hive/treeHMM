# CVAT Ground-Truth Crop Analysis Pipeline

This folder contains a three-step pipeline that computes T cell behavioural features,
fits a joint Autoregressive Hidden Markov Model (AR-HMM) across six ground-truth
video crops, and generates overlay videos of the resulting state assignments.

## Quick Start

```bash
bash run_pipeline.sh
```

This single command runs all three steps in order with the correct conda
environments.  You can also run each step individually (see below).

## Pipeline Steps

| Step | Script | Conda Env | Description |
|------|--------|-----------|-------------|
| 1 | `calculate_emissions.py` | `AnalysisEnv` | Computes per-cell features (velocity, cancer neighbors, T cell neighbors) and saves them as `.npy` emission arrays. |
| 2 | `fit_arhmm.py` | `treeHMM_env` | Loads emissions from all crops, fits a single joint AR-HMM, saves per-crop state assignments and summary plots. |
| 3 | `create_overlay_videos.py` | `AnalysisEnv` | Renders `.mp4` overlay videos for each crop showing HMM states colour-coded on top of the raw microscopy data. |

### Running Steps Individually

```bash
# Step 1
conda run --no-capture-output -n AnalysisEnv python calculate_emissions.py

# Step 2
conda run --no-capture-output -n treeHMM_env python fit_arhmm.py

# Step 3
conda run --no-capture-output -n AnalysisEnv python create_overlay_videos.py
```

## Configuration

All shared parameters live in **`config.yml`**, including:

- `crop_ids` – list of the six crop identifiers
- `conditions_dict` – mapping from experimental condition (SH / RASA2 / CUL5) to crop IDs
- `emission_feature_names` – ordered feature names for the emissions array
- `num_states`, `min_t`, `num_lags` – AR-HMM model hyperparameters
- `cvat_base_dir`, `caliban_base_dir` – paths to input tracking data
- `output_base_dir` – root output directory
- `video_fps`, `video_figsize` – video rendering settings

## Inputs

| Data | Path |
|------|------|
| CVAT tracking annotations | `cvat_base_dir/{well_id}/{crop_id}/ALL_tracks.tiff` |
| CVAT cell type dictionaries | `cvat_base_dir/{well_id}/{crop_id}/full_cell_type_dict.pkl` |
| Caliban nuclear tracks | `caliban_base_dir/{crop_id}.tiff` |
| Raw crop TIFFs | `cvat_base_dir/{well_id}/{crop_id}/crop.tiff` |

## Outputs

All outputs are written to `treeHMM/analysis/cvat_gt_crops/`:

```
analysis/cvat_gt_crops/
├── state_assignments.png        # joint heatmap with crop colorbar
├── state_counts.png             # stacked area: state fractions over time per crop
├── feature_distributions.png    # per-state feature violin/histograms
├── cancer_contact_heatmap.png   # heatmap of cancer_contact feature values
├── learned_transition_matrix.png      # heatmap of the learned transition matrix
├── observed_transition_matrices.png   # per-crop observed transition matrices
├── B4_t50t100y200y350x750x900/
│   ├── t_cell_emissions_array.npy         # (T × N × M) emission features
│   ├── t_cell_emissions_names.txt         # feature name list
│   ├── t_cell_state_assignments.npy       # (T × N) state assignments
│   └── t_cell_HMM_overlay_video.mp4       # overlay video
├── B8_t50t100y200y350x750x900/
│   └── ...
└── ...   (one sub-directory per crop)
```
