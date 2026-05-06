"""
Jointly fit AR-HMM across all ground-truth T cell crops.

This script is equivalent to tarHMM_model_dryrun_t_cell_example.ipynb
but fits one HMM jointly across all six crops.

Outputs:
1) One plot of masked_state_assignments across all T cells with crop colorbar
2) Feature distribution plots per state (all crops combined)
3) Per-crop t_cell_state_assignments.npy files
"""

import pickle
import numpy as np
import tifffile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import os
import sys

sys.path.insert(0, '/gladstone/engelhardt/lab/adamw/treeHMM')

# ============================================================
# Configuration
# ============================================================
crop_ids = [
    'B4_t50t100y200y350x750x900',
    'B8_t50t100y200y350x750x900',
    'E4_t50t100y200y350x750x900',
    'B4_t250t300y200y350x750x900',
    'B8_t250t300y200y350x750x900',
    'E4_t250t300y200y350x750x900',
]

cvat_base_dir = '/gladstone/engelhardt/lab/adamw/MarsonImagingPipeline/data/ground_truth_tracking_annotations/cvat_annotations/TCR-T/'
emissions_base_dir = '/gladstone/engelhardt/lab/adamw/treeHMM/notebooks/data/example_gt_crop'
out_base_dir = '/gladstone/engelhardt/lab/adamw/treeHMM/notebooks/data/example_gt_crop'

num_states = 4


# ============================================================
# Helper functions
# ============================================================
def filter_tracks_by_type(track_type, tracks, type_dict):
    """Filter tracks to only contain tracks of the desired cell type."""
    valid_ids = [cell_id for cell_id, cell_type in type_dict.items() if cell_type == track_type]
    filtered_tracks = tracks.copy()
    filtered_tracks[~np.isin(filtered_tracks, valid_ids)] = 0
    return filtered_tracks


def filter_tracks_by_time(data, emissions, min_t=10):
    """
    Filters 'data' and 'emissions' objects to retain only cells present
    in at least min_t unique time frames.

    Returns:
        tuple: (filtered_data, filtered_emissions, keep_indices)
    """
    durations = np.sum(data['active_mask'], axis=0)
    keep_indices = np.where(durations >= min_t)[0]

    if len(keep_indices) == 0:
        print(f"Warning: No cells found with duration >= {min_t} frames.")
        T, _, D = emissions.shape
        empty_data = {k: np.zeros((T, 0), dtype=v.dtype) if isinstance(v, np.ndarray) else v
                      for k, v in data.items()}
        return empty_data, np.zeros((T, 0, D)), keep_indices

    filtered_emissions = emissions[:, keep_indices, :]

    filtered_data = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            filtered_data[key] = val[:, keep_indices]
        else:
            filtered_data[key] = val

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    max_old_idx = data['parent_indices'].max()
    lookup = np.full(max_old_idx + 1, -1, dtype=np.int32)
    for old_idx, new_idx in old_to_new.items():
        lookup[old_idx] = new_idx

    curr_parents = filtered_data['parent_indices']
    curr_active = filtered_data['active_mask']
    new_parents = np.copy(curr_parents)

    rows, cols = np.where(curr_active)
    new_parents[rows, cols] = lookup[curr_parents[rows, cols]]

    orphans = (new_parents == -1) & curr_active
    if np.any(orphans):
        filtered_data['is_new_root_mask'][orphans] = True
        o_rows, o_cols = np.where(orphans)
        new_parents[o_rows, o_cols] = o_cols

    filtered_data['parent_indices'] = new_parents

    print(f"Filtered {data['active_mask'].shape[1]} cells down to {len(keep_indices)} cells.")
    return filtered_data, filtered_emissions, keep_indices


# ============================================================
# Step 1: Generate model inputs from all crops
# ============================================================
print("=" * 60)
print("Step 1: Building data dicts for each crop")
print("=" * 60)

all_data_list = []
crop_num_cells = []
crop_labels = []

for crop_idx, crop in enumerate(crop_ids):
    well_id = crop.split("_")[0]

    cvat_tracks = tifffile.imread(os.path.join(cvat_base_dir, well_id, crop, 'ALL_tracks.tiff'))
    cell_type_dict = pickle.load(open(os.path.join(cvat_base_dir, well_id, crop, 'full_cell_type_dict.pkl'), "rb"))

    type_tracks_per_well = {}
    for cell_type in ['cancer', 't_cell']:
        type_tracks_per_well[cell_type] = filter_tracks_by_type(cell_type, cvat_tracks, cell_type_dict)

    t_cell_tracks = type_tracks_per_well['t_cell']
    print(f"Crop {crop}: t_cell_tracks shape = {t_cell_tracks.shape}")

    data = {}
    T = t_cell_tracks.shape[0]
    all_cell_ids = np.unique(t_cell_tracks[t_cell_tracks > 0])
    all_cell_ids.sort()
    num_cells = len(all_cell_ids)
    id_to_col = {int(cid): i for i, cid in enumerate(all_cell_ids)}

    print(f"  T={T}, num_cells={num_cells}")

    active_mask = np.zeros((T, num_cells), dtype=bool)
    is_division_mask = np.zeros((T, num_cells), dtype=bool)
    is_new_root_mask = np.zeros((T, num_cells), dtype=bool)
    parent_indices = np.zeros((T, num_cells), dtype=np.int32)

    for t in range(T):
        t_cell_frame = t_cell_tracks[t]
        frame_ids = np.unique(t_cell_frame[t_cell_frame > 0])
        for cid in frame_ids:
            active_mask[t, id_to_col[int(cid)]] = True

    for cid, col in id_to_col.items():
        active_frames = np.where(active_mask[:, col])[0]
        if len(active_frames) == 0:
            print(f"Warning: Cell ID {cid} (col {col}) is never active. Skipping.")
            continue
        first_frame = active_frames[0]
        is_new_root_mask[first_frame, col] = True
        parent_indices[first_frame, col] = col
        for t in active_frames[1:]:
            parent_indices[t, col] = col

    data["parent_indices"] = parent_indices
    data['active_mask'] = active_mask
    data["is_division_mask"] = is_division_mask
    data["is_new_root_mask"] = is_new_root_mask

    all_data_list.append(data)
    crop_num_cells.append(num_cells)
    crop_labels.extend([crop_idx] * num_cells)

    print(f"  active_mask shape:       {active_mask.shape}")
    print(f"  Division events (daughters): {is_division_mask.sum()}")
    print(f"  Root cells:                  {is_new_root_mask.sum()}")
    print()


# ============================================================
# Step 2: Concatenate data across crops
# ============================================================
print("=" * 60)
print("Step 2: Concatenating data across crops")
print("=" * 60)

T = all_data_list[0]['active_mask'].shape[0]
total_cells = sum(crop_num_cells)
crop_labels = np.array(crop_labels)

print(f"Total T cells across all crops: {total_cells}")
print(f"T = {T}")

combined_active_mask = np.zeros((T, total_cells), dtype=bool)
combined_is_division_mask = np.zeros((T, total_cells), dtype=bool)
combined_is_new_root_mask = np.zeros((T, total_cells), dtype=bool)
combined_parent_indices = np.zeros((T, total_cells), dtype=np.int32)

cell_offset = 0
for crop_data, nc in zip(all_data_list, crop_num_cells):
    combined_active_mask[:, cell_offset:cell_offset+nc] = crop_data['active_mask']
    combined_is_division_mask[:, cell_offset:cell_offset+nc] = crop_data['is_division_mask']
    combined_is_new_root_mask[:, cell_offset:cell_offset+nc] = crop_data['is_new_root_mask']
    combined_parent_indices[:, cell_offset:cell_offset+nc] = crop_data['parent_indices'] + cell_offset
    cell_offset += nc

combined_data = {
    'active_mask': combined_active_mask,
    'is_division_mask': combined_is_division_mask,
    'is_new_root_mask': combined_is_new_root_mask,
    'parent_indices': combined_parent_indices,
}

print(f"Combined active_mask shape: {combined_active_mask.shape}")


# ============================================================
# Step 3: Load and concatenate emissions
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Loading pre-computed emissions for all crops")
print("=" * 60)

all_emissions_parts = []

for crop_idx, crop in enumerate(crop_ids):
    emissions_crop = np.load(os.path.join(emissions_base_dir, crop, 't_cell_emissions_array.npy'))
    all_emissions_parts.append(emissions_crop)
    print(f"Crop {crop}: emissions shape = {emissions_crop.shape}")

emissions = np.concatenate(all_emissions_parts, axis=1)
print(f"\nCombined emissions shape: {emissions.shape}")


# ============================================================
# Step 4: Filter tracks by time
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Filtering tracks by time")
print("=" * 60)

num_cells_before = combined_data['active_mask'].shape[1]
combined_data, emissions, kept_indices = filter_tracks_by_time(combined_data, emissions, min_t=0)

kept_mask = np.zeros(num_cells_before, dtype=bool)
kept_mask[kept_indices] = True
crop_labels = crop_labels[kept_mask]

print(f"After filtering: emissions shape = {emissions.shape}")
print(f"After filtering: active_mask shape = {combined_data['active_mask'].shape}")
print(f"Crop labels shape: {crop_labels.shape}")


# ============================================================
# Step 5: Fit AR-HMM jointly across all crops
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Fitting AR-HMM")
print("=" * 60)

import jax.numpy as jnp
import jax.random as jr
from models.tarhmm import tARHMM

emission_dim = emissions.shape[-1]
arhmm = tARHMM(num_states, emission_dim, num_lags=1)

key = jr.PRNGKey(0)
params, props = arhmm.initialize(key=key)

emissions_jnp = jnp.array(emissions)
inputs = jnp.zeros_like(emissions_jnp)

batched_emissions = emissions_jnp[None, ...]
batched_inputs = inputs[None, ...]
batched_parent_indices = jnp.array(combined_data['parent_indices'])[None, ...]
batched_is_division_mask = jnp.array(combined_data['is_division_mask'])[None, ...]
batched_active_mask = jnp.array(combined_data['active_mask'])[None, ...]
batched_is_new_root_mask = jnp.array(combined_data['is_new_root_mask'])[None, ...]

fitted_params, lps = arhmm.fit_em(params, props, batched_emissions, inputs=batched_inputs,
    parent_indices=batched_parent_indices,
    is_division_mask=batched_is_division_mask,
    active_mask=batched_active_mask,
    is_new_root_mask=batched_is_new_root_mask)

print("Fitted params:")
print(fitted_params)


# ============================================================
# Step 6: Compute posterior and state assignments
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Computing posterior")
print("=" * 60)

from models.tarhmm import tree_hmm_two_filter_smoother

input_fwd = arhmm._inference_args(params, emissions_jnp, inputs,
    combined_data['parent_indices'], combined_data['is_division_mask'],
    combined_data['active_mask'], combined_data['is_new_root_mask'])

posterior = tree_hmm_two_filter_smoother(*input_fwd)

state_assignments = jnp.argmax(posterior.smoothed_probs, axis=-1)
max_probs = jnp.max(posterior.smoothed_probs, axis=-1)

masked_state_assignments = jnp.where(jnp.isnan(max_probs.T), jnp.nan, state_assignments.T)

print(f"state_assignments shape: {state_assignments.shape}")
print(f"masked_state_assignments shape: {masked_state_assignments.shape}")


# ============================================================
# Output 1: State assignments across all T cells with crop colorbar
# ============================================================
print("\n" + "=" * 60)
print("Output 1: State assignments heatmap with crop colorbar")
print("=" * 60)

num_cells_total = masked_state_assignments.shape[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [1, 30]},
                         sharey=True)

num_crops = len(crop_ids)
crop_cmap = plt.cm.get_cmap('tab10', num_crops)
crop_color_array = np.array(crop_labels).reshape(-1, 1)

axes[0].imshow(crop_color_array, aspect='auto', interpolation='none',
               cmap=crop_cmap, vmin=0, vmax=num_crops-1, origin='upper')
axes[0].set_xticks([])
axes[0].set_ylabel('Cell')
axes[0].set_title('Crop')

legend_elements = [Patch(facecolor=crop_cmap(i), label=crop_ids[i]) for i in range(num_crops)]

im = axes[1].imshow(np.array(masked_state_assignments), aspect='auto', interpolation='none',
                     cmap='viridis', origin='upper')
axes[1].set_xlabel('Time')
axes[1].set_title('State Assignments (argmax of smoothed_probs) - All Crops')
cbar = plt.colorbar(im, ax=axes[1], ticks=range(num_states))
cbar.set_label('State')

fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=8, title='Crop ID', bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig(os.path.join(out_base_dir, 'all_crops_state_assignments.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved state assignments plot.")


# ============================================================
# Output 2: Feature distributions per state (all crops combined)
# ============================================================
print("\n" + "=" * 60)
print("Output 2: Feature distributions per state")
print("=" * 60)

states_flat = np.array(masked_state_assignments.T).flatten()
emissions_flat = np.array(emissions.reshape(-1, emissions.shape[-1]))

feature_names = ['velocity', 'cancer_neighbors', 't_cell_neighbors']
plot_type = ['violin', 'hist', 'hist']
df = pd.DataFrame(emissions_flat, columns=feature_names)
df['state'] = states_flat

df_clean = df.dropna(subset=['state']).copy()
df_clean['state'] = df_clean['state'].astype(int)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)

for i, feature in enumerate(feature_names):
    if plot_type[i] == 'violin':
        sns.violinplot(
            data=df_clean, x='state', y=feature, hue='state',
            ax=axes[i], palette='viridis', legend=False
        )
        axes[i].set_ylabel(f'{feature.replace("_", " ").title()}')
        axes[i].set_xlabel('AR-HMM State')
    elif plot_type[i] == 'hist':
        sns.histplot(
            data=df_clean, x=feature, hue='state',
            ax=axes[i], palette='viridis',
            common_norm=False, stat='density', element='step',
            discrete=True, legend=False
        )
        axes[i].set_xlabel(f'{feature.replace("_", " ").title()}')
    sns.despine(ax=axes[i])

plt.savefig(os.path.join(out_base_dir, 'all_crops_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved feature distribution plot.")


# ============================================================
# Output 3: Save per-crop state assignments
# ============================================================
print("\n" + "=" * 60)
print("Output 3: Saving per-crop state assignments")
print("=" * 60)

for crop_idx, crop in enumerate(crop_ids):
    cell_indices = np.where(crop_labels == crop_idx)[0]
    crop_state_assignments = np.array(state_assignments[:, cell_indices])

    out_dir = os.path.join(out_base_dir, crop)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 't_cell_state_assignments.npy')
    np.save(out_path, crop_state_assignments)
    print(f"Saved {out_path} with shape {crop_state_assignments.shape}")

print("\nDone!")
