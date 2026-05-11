"""
Jointly fit AR-HMM across all ground-truth T cell crops.

Fits one AR-HMM jointly across all crops defined in config.yml.

Outputs (all under output_base_dir):
  1) state_assignments.png              - heatmap with crop colorbar
  2) feature_distributions.png          - per-state feature distributions
  3) {crop_id}/t_cell_state_assignments.npy - per-crop state arrays
  4) state_counts.png                   - stacked area: state fractions over time per crop
  5) learned_transition_matrix.png      - heatmap of the learned transition matrix
  6) observed_transition_matrices.png   - per-crop observed transition matrices

Usage (treeHMM_env):
    conda run -n treeHMM_env python fit_arhmm.py
"""

import os
import sys
import pickle
from pathlib import Path

import yaml
import numpy as np
import tifffile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Load shared configuration
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
cvat_base_dir = cfg["cvat_base_dir"]
out_base_dir = cfg["output_base_dir"]
num_states = cfg["num_states"]
min_t = cfg["min_t"]
num_lags = cfg["num_lags"]
all_feature_names = cfg["emission_feature_names"]
model_features = cfg["model_features"]

sys.path.insert(0, cfg["treehmm_dir"])

# ---------------------------------------------------------------------------
# Matplotlib styling (matches the original notebook)
# ---------------------------------------------------------------------------
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.use14corefonts'] = True


# ============================================================
# Helper functions
# ============================================================
def filter_tracks_by_type(track_type, tracks, type_dict):
    """Filter tracks to only contain tracks of the desired cell type."""
    valid_ids = [cid for cid, ctype in type_dict.items() if ctype == track_type]
    filtered = tracks.copy()
    filtered[~np.isin(filtered, valid_ids)] = 0
    return filtered


def filter_tracks_by_time(data, emissions, min_t=10):
    """Filter data and emissions to retain cells present >= min_t frames.

    Returns:
        tuple: (filtered_data, filtered_emissions, keep_indices)
    """
    durations = np.sum(data['active_mask'], axis=0)
    keep_indices = np.where(durations >= min_t)[0]

    if len(keep_indices) == 0:
        print(f"Warning: No cells found with duration >= {min_t} frames.")
        T, _, D = emissions.shape
        empty_data = {
            k: np.zeros((T, 0), dtype=v.dtype) if isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }
        return empty_data, np.zeros((T, 0, D)), keep_indices

    filtered_emissions = emissions[:, keep_indices, :]
    filtered_data = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            filtered_data[key] = val[:, keep_indices]
        else:
            filtered_data[key] = val

    old_to_new = {old: new for new, old in enumerate(keep_indices)}
    max_old = data['parent_indices'].max()
    lookup = np.full(max_old + 1, -1, dtype=np.int32)
    for old, new in old_to_new.items():
        lookup[old] = new

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

    print(f"Filtered {data['active_mask'].shape[1]} -> {len(keep_indices)} cells.")
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

    cvat_tracks = tifffile.imread(
        os.path.join(cvat_base_dir, well_id, crop, 'ALL_tracks.tiff')
    )
    cell_type_dict = pickle.load(
        open(os.path.join(cvat_base_dir, well_id, crop, 'full_cell_type_dict.pkl'), "rb")
    )

    # remove the t=0 time frame (removed from emissions too)
    cvat_tracks = cvat_tracks[1:, ...]

    type_tracks = {}
    for ct in ['cancer', 't_cell']:
        type_tracks[ct] = filter_tracks_by_type(ct, cvat_tracks, cell_type_dict)

    t_cell_tracks = type_tracks['t_cell']
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
        frame_ids = np.unique(t_cell_tracks[t][t_cell_tracks[t] > 0])
        for cid in frame_ids:
            active_mask[t, id_to_col[int(cid)]] = True

    for cid, col in id_to_col.items():
        active_frames = np.where(active_mask[:, col])[0]
        if len(active_frames) == 0:
            print(f"Warning: Cell ID {cid} (col {col}) is never active.")
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

    print(f"  active_mask shape:  {active_mask.shape}")
    print(f"  Division events:    {is_division_mask.sum()}")
    print(f"  Root cells:         {is_new_root_mask.sum()}")
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

combined_active = np.zeros((T, total_cells), dtype=bool)
combined_div = np.zeros((T, total_cells), dtype=bool)
combined_root = np.zeros((T, total_cells), dtype=bool)
combined_parent = np.zeros((T, total_cells), dtype=np.int32)

cell_offset = 0
for crop_data, nc in zip(all_data_list, crop_num_cells):
    s = slice(cell_offset, cell_offset + nc)
    combined_active[:, s] = crop_data['active_mask']
    combined_div[:, s] = crop_data['is_division_mask']
    combined_root[:, s] = crop_data['is_new_root_mask']
    combined_parent[:, s] = crop_data['parent_indices'] + cell_offset
    cell_offset += nc

combined_data = {
    'active_mask': combined_active,
    'is_division_mask': combined_div,
    'is_new_root_mask': combined_root,
    'parent_indices': combined_parent,
}

print(f"Combined active_mask shape: {combined_active.shape}")


# ============================================================
# Step 3: Load and concatenate emissions (subsetting to model_features)
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Loading pre-computed emissions for all crops")
print("=" * 60)

# Read the saved feature-name ordering from the first crop to determine
# which column indices correspond to the requested model_features.
names_path = os.path.join(out_base_dir, crop_ids[0], 't_cell_emissions_names.txt')
if os.path.exists(names_path):
    with open(names_path, 'r') as fh:
        saved_feature_names = [line.strip() for line in fh if line.strip()]
else:
    # Fall back to the full list in config if the names file hasn't been
    # generated yet (backwards-compatible).
    saved_feature_names = list(all_feature_names)

# Validate that every requested model feature exists in the saved array.
for feat in model_features:
    if feat not in saved_feature_names:
        raise ValueError(
            f"model_features entry '{feat}' not found in saved emission "
            f"features {saved_feature_names}. Re-run calculate_emissions.py "
            f"to include this feature."
        )

feature_indices = [saved_feature_names.index(f) for f in model_features]
print(f"Subsetting emissions to model_features: {model_features}")
print(f"  -> column indices: {feature_indices}")

all_emissions = []
for crop in crop_ids:
    e = np.load(os.path.join(out_base_dir, crop, 't_cell_emissions_array.npy'))
    # Subset to the requested model features
    e = e[:, :, feature_indices]
    all_emissions.append(e)
    print(f"Crop {crop}: emissions shape = {e.shape}")

emissions = np.concatenate(all_emissions, axis=1)
print(f"\nCombined emissions shape: {emissions.shape}")


# ============================================================
# Step 4: Filter tracks by time
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Filtering tracks by time")
print("=" * 60)

n_before = combined_data['active_mask'].shape[1]
combined_data, emissions, kept_indices = filter_tracks_by_time(
    combined_data, emissions, min_t=min_t
)

kept_mask = np.zeros(n_before, dtype=bool)
kept_mask[kept_indices] = True
crop_labels = crop_labels[kept_mask]

print(f"After filtering: emissions shape = {emissions.shape}")
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
arhmm = tARHMM(num_states, emission_dim, num_lags=num_lags)

key = jr.PRNGKey(0)
params, props = arhmm.initialize(key=key)

emissions_jnp = jnp.array(emissions)
inputs = jnp.zeros_like(emissions_jnp)

batched_emissions = emissions_jnp[None, ...]
batched_inputs = inputs[None, ...]
batched_parent = jnp.array(combined_data['parent_indices'])[None, ...]
batched_div = jnp.array(combined_data['is_division_mask'])[None, ...]
batched_active = jnp.array(combined_data['active_mask'])[None, ...]
batched_root = jnp.array(combined_data['is_new_root_mask'])[None, ...]

fitted_params, lps = arhmm.fit_em(
    params, props, batched_emissions, inputs=batched_inputs,
    parent_indices=batched_parent,
    is_division_mask=batched_div,
    active_mask=batched_active,
    is_new_root_mask=batched_root,
)

# ============================================================
# Step 6: Compute posterior and state assignments
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Computing posterior")
print("=" * 60)

from models.tarhmm import tree_hmm_two_filter_smoother

input_fwd = arhmm._inference_args(
    params, emissions_jnp, inputs,
    combined_data['parent_indices'], combined_data['is_division_mask'],
    combined_data['active_mask'], combined_data['is_new_root_mask'],
)

posterior = tree_hmm_two_filter_smoother(*input_fwd)

state_assignments = jnp.argmax(posterior.smoothed_probs, axis=-1)
max_probs = jnp.max(posterior.smoothed_probs, axis=-1)
masked_state_assignments = jnp.where(
    jnp.isnan(max_probs.T), jnp.nan, state_assignments.T
)

print(f"state_assignments shape: {state_assignments.shape}")
print(f"masked_state_assignments shape: {masked_state_assignments.shape}")


# ============================================================
# Output 1: State assignments heatmap with crop colorbar
# ============================================================
print("\n" + "=" * 60)
print("Output 1: State assignments heatmap with crop colorbar")
print("=" * 60)

fig, axes = plt.subplots(
    1, 2, figsize=(14, 8),
    gridspec_kw={'width_ratios': [1, 30]}, sharey=True,
)

num_crops = len(crop_ids)
crop_cmap = plt.cm.get_cmap('tab10', num_crops)
crop_arr = np.array(crop_labels).reshape(-1, 1)

axes[0].imshow(crop_arr, aspect='auto', interpolation='none',
               cmap=crop_cmap, vmin=0, vmax=num_crops - 1, origin='upper')
axes[0].set_xticks([])
axes[0].set_ylabel('Cell')
axes[0].set_title('Crop')

legend_elements = [Patch(facecolor=crop_cmap(i), label=crop_ids[i])
                   for i in range(num_crops)]

im = axes[1].imshow(np.array(masked_state_assignments), aspect='auto',
                     interpolation='none', cmap='viridis', origin='upper')
axes[1].set_xlabel('Time')
axes[1].set_title('State Assignments (argmax of smoothed_probs) - All Crops')
cbar = plt.colorbar(im, ax=axes[1], ticks=range(num_states))
cbar.set_label('State')

fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           title='Crop ID', bbox_to_anchor=(0.5, -0.1))

plt.tight_layout()
os.makedirs(out_base_dir, exist_ok=True)
plt.savefig(os.path.join(out_base_dir, 'state_assignments.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved state assignments plot.")


# ============================================================
# Output 2: Feature distributions per state
# ============================================================
print("\n" + "=" * 60)
print("Output 2: Feature distributions per state")
print("=" * 60)

states_flat = np.array(masked_state_assignments.T).flatten()
emissions_flat = np.array(emissions.reshape(-1, emissions.shape[-1]))

num_features = len(model_features)
df = pd.DataFrame(emissions_flat, columns=model_features)
df['state'] = states_flat
df_clean = df.dropna(subset=['state']).copy()
df_clean['state'] = df_clean['state'].astype(int)

# Choose plot type per feature: continuous features get violin plots,
# discrete / binary features get histograms.
_discrete_features = {'cancer_contact', 't_cell_neighbors'}

fig, axes = plt.subplots(
    1, num_features,
    figsize=(3 * num_features, 3),
    tight_layout=True,
    squeeze=False,
)
axes = axes.flatten()

for i, feat in enumerate(model_features):
    if feat in _discrete_features:
        sns.histplot(data=df_clean, x=feat, hue='state',
                     ax=axes[i], palette='viridis',
                     common_norm=False, stat='density', element='step',
                     discrete=True, legend=False)
        axes[i].set_xlabel(feat.replace("_", " ").title())
    else:
        sns.violinplot(data=df_clean, x='state', y=feat, hue='state',
                       ax=axes[i], palette='viridis', legend=False)
        axes[i].set_ylabel(feat.replace("_", " ").title())
        axes[i].set_xlabel('AR-HMM State')
    sns.despine(ax=axes[i])

plt.savefig(os.path.join(out_base_dir, 'feature_distributions.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved feature distribution plot.")


# ============================================================
# Output 3: Save per-crop state assignments
# ============================================================
print("\n" + "=" * 60)
print("Output 3: Saving per-crop state assignments")
print("=" * 60)

for crop_idx, crop in enumerate(crop_ids):
    cell_idx = np.where(crop_labels == crop_idx)[0]
    crop_states = np.array(state_assignments[:, cell_idx])

    crop_dir = os.path.join(out_base_dir, crop)
    os.makedirs(crop_dir, exist_ok=True)

    out_path = os.path.join(crop_dir, 't_cell_state_assignments.npy')
    np.save(out_path, crop_states)
    print(f"Saved {out_path} with shape {crop_states.shape}")


# ============================================================
# Output 4: Fraction of cells per state over time (per crop)
# ============================================================
print("\n" + "=" * 60)
print("Output 4: State fraction over time (per crop)")
print("=" * 60)

state_colors = plt.cm.get_cmap('viridis', num_states)
state_color_list = [state_colors(s) for s in range(num_states)]

fig, axes = plt.subplots(2, 3, figsize=(9, 6), tight_layout=True)
axes_flat = axes.flatten()

for crop_idx, crop in enumerate(crop_ids):
    ax = axes_flat[crop_idx]
    cell_mask = crop_labels == crop_idx
    # state_assignments: (T, num_cells), active_mask: (T, num_cells)
    crop_states = np.array(state_assignments[:, cell_mask])     # (T, n_crop_cells)
    crop_active = np.array(combined_data['active_mask'][:, cell_mask])  # (T, n_crop_cells)

    T_len = crop_states.shape[0]
    fractions = np.zeros((num_states, T_len))
    for t in range(T_len):
        active_at_t = crop_active[t]
        n_active = active_at_t.sum()
        if n_active == 0:
            continue
        states_at_t = crop_states[t, active_at_t]
        for s in range(num_states):
            fractions[s, t] = np.sum(states_at_t == s) / n_active

    time_axis = np.arange(T_len)
    ax.stackplot(time_axis, fractions, colors=state_color_list,
                 labels=[f'State {s}' for s in range(num_states)])
    ax.set_title(crop)
    ax.set_ylim(0, 1)
    if crop_idx >= 3:
        ax.set_xlabel('Time')
    if crop_idx % 3 == 0:
        ax.set_ylabel('Fraction of cells')

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=num_states,
           bbox_to_anchor=(0.5, -0.02))
plt.suptitle('Fraction of T cells per state over time')
plt.savefig(os.path.join(out_base_dir, 'state_counts.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved state_counts.png")


# ============================================================
# Output 5: Learned transition matrix
# ============================================================
print("\n" + "=" * 60)
print("Output 5: Learned transition matrix")
print("=" * 60)

# fitted_params.transitions.transition_matrix: (K, K) where row = state at t, col = state at t+1
trans_matrix = np.array(fitted_params.transitions.transition_matrix)
print(f"Transition matrix shape: {trans_matrix.shape}")

fig, ax = plt.subplots(figsize=(3, 3))
im = ax.imshow(trans_matrix, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(num_states))
ax.set_yticks(range(num_states))
ax.set_xticklabels([f'State {s}' for s in range(num_states)])
ax.set_yticklabels([f'State {s}' for s in range(num_states)])
ax.set_xlabel('State at $t+1$')
ax.set_ylabel('State at $t$')
ax.set_title('Learned Transition Matrix')

# Annotate cells with values
for i in range(num_states):
    for j in range(num_states):
        val = trans_matrix[i, j]
        text_color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                color=text_color)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Transition probability')
plt.tight_layout()
plt.savefig(os.path.join(out_base_dir, 'learned_transition_matrix.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved learned_transition_matrix.png")


# ============================================================
# Output 6: Observed transition matrices (per crop)
# ============================================================
print("\n" + "=" * 60)
print("Output 6: Observed transition matrices (per crop)")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(9, 6), tight_layout=True)
axes_flat = axes.flatten()

for crop_idx, crop in enumerate(crop_ids):
    ax = axes_flat[crop_idx]
    cell_mask = crop_labels == crop_idx
    crop_states = np.array(state_assignments[:, cell_mask])     # (T, n_crop_cells)
    crop_active = np.array(combined_data['active_mask'][:, cell_mask])  # (T, n_crop_cells)

    # Count transitions
    obs_trans = np.zeros((num_states, num_states))
    T_len = crop_states.shape[0]
    n_cells = crop_states.shape[1]
    for c in range(n_cells):
        for t in range(T_len - 1):
            if crop_active[t, c] and crop_active[t + 1, c]:
                s_from = crop_states[t, c]
                s_to = crop_states[t + 1, c]
                obs_trans[s_from, s_to] += 1

    # Normalise rows to get probabilities
    row_sums = obs_trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    obs_trans_prob = obs_trans / row_sums

    im = ax.imshow(obs_trans_prob, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(num_states))
    ax.set_yticks(range(num_states))
    ax.set_xticklabels([f'{s}' for s in range(num_states)])
    ax.set_yticklabels([f'{s}' for s in range(num_states)])
    ax.set_xlabel('State at $t+1$')
    ax.set_ylabel('State at $t$')
    ax.set_title(f'{crop}\nObserved Transition Matrix')

    for i in range(num_states):
        for j in range(num_states):
            val = obs_trans_prob[i, j]
            text_color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=text_color)

plt.savefig(os.path.join(out_base_dir, 'observed_transition_matrices.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved observed_transition_matrices.png")


print("\nDone!")
