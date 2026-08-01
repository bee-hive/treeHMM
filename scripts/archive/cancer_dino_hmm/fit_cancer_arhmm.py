"""
Step 4: Jointly fit the tree-AR-HMM across cancer cells in all crops.

Cancer cells come from the Caliban nuclei tracks (each nucleus == one cancer cell).
Emissions = z-scored non-DINO features (velocity, t_cell_neighbors_20px) concatenated
with the top-`n_dino_pcs` PCA'd DINOv2 embeddings (if use_dino).

Defaults: num_lags=0 (Gaussian HMM, no AR), allow_divisions=false (independent
per-cell chains).  Both are config-switchable.

Outputs (under output_base_dir), mirroring the T-cell pipeline:
  1) state_assignments.png
  2) feature_distributions.png            (12 panels: velocity, neighbors, 10 DINO PCs)
  3) {crop}/cancer_state_assignments.npy
  4) state_counts.png
  5) learned_transition_matrix.png
  6) observed_transition_matrices.png
  7) t_cell_neighbors_heatmap.png
  +  nondino_standardization.npz, final_feature_names.txt

Usage (treeHMM_env):
    conda run -n treeHMM_env python fit_cancer_arhmm.py
"""

import os
import sys
import pickle
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
num_states = cfg["num_states"]
min_t = cfg["min_t"]
num_lags = cfg["num_lags"]
model_features = cfg["model_features"]
use_dino = cfg.get("use_dino", True)
n_dino_pcs = cfg["n_dino_pcs"]
allow_divisions = cfg.get("allow_divisions", False)
standardize_nondino = cfg.get("standardize_nondino", True)

if num_lags not in (0, 1):
    raise ValueError(f"num_lags must be 0 or 1 (compute_inputs does not support >1); got {num_lags}")

sys.path.insert(0, cfg["treehmm_dir"])

# ---------------------------------------------------------------------------
# Matplotlib styling (matches the existing pipeline)
# ---------------------------------------------------------------------------
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 7, 8, 10
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
# Helper: time-based cell filter (mirrors fit_arhmm.filter_tracks_by_time)
# ============================================================
def filter_tracks_by_time(data, emissions, min_t=10):
    """Filter data/emissions to retain cells present >= min_t frames; remap parents."""
    durations = np.sum(data['active_mask'], axis=0)
    keep_indices = np.where(durations >= min_t)[0]

    if len(keep_indices) == 0:
        print(f"Warning: No cells found with duration >= {min_t} frames.")
        T, _, D = emissions.shape
        empty = {k: (np.zeros((T, 0), dtype=v.dtype) if isinstance(v, np.ndarray) else v)
                 for k, v in data.items()}
        return empty, np.zeros((T, 0, D)), keep_indices

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


def load_nuclei_tracks(crop):
    nuc = np.asarray(__import__('tifffile').imread(
        os.path.join(caliban_base_dir, f"{crop}.tiff")))
    if nuc.ndim == 4:
        nuc = nuc[..., 0]
    return nuc


# ============================================================
# Step 1: Build per-crop masks from the Caliban nuclei tracks
# ============================================================
print("=" * 60)
print("Step 1: Building data dicts (cancer = nuclei tracks)")
print("=" * 60)

all_data_list, crop_num_cells, crop_labels = [], [], []

for crop_idx, crop in enumerate(crop_ids):
    nuc = load_nuclei_tracks(crop)
    T = nuc.shape[0]

    cancer_cell_ids = np.load(os.path.join(out_base_dir, crop, "cancer_cell_ids.npy"))
    derived = np.sort(np.unique(nuc[nuc > 0]))
    assert np.array_equal(cancer_cell_ids, derived), (
        f"cancer_cell_ids.npy disagrees with nuclei tracks for {crop} "
        f"(alignment contract violated)")
    num_cells = len(cancer_cell_ids)
    id_to_col = {int(cid): i for i, cid in enumerate(cancer_cell_ids)}

    active_mask = np.zeros((T, num_cells), dtype=bool)
    is_division_mask = np.zeros((T, num_cells), dtype=bool)
    is_new_root_mask = np.zeros((T, num_cells), dtype=bool)
    parent_indices = np.zeros((T, num_cells), dtype=np.int32)

    for t in range(T):
        frame_ids = np.unique(nuc[t][nuc[t] > 0])
        for cid in frame_ids:
            active_mask[t, id_to_col[int(cid)]] = True

    for cid, col in id_to_col.items():
        active_frames = np.where(active_mask[:, col])[0]
        if len(active_frames) == 0:
            continue
        for t in active_frames:
            parent_indices[t, col] = col
        is_new_root_mask[active_frames[0], col] = True

    # Optional: wire division lineage from the regenerated CVAT cancer graph.
    if allow_divisions:
        from divisions_utils import apply_division_lineage  # local helper module
        apply_division_lineage(
            crop, nuc, cancer_cell_ids, id_to_col, active_mask,
            is_division_mask, is_new_root_mask, parent_indices,
            type_sep_tracks_dir, out_base_dir)

    all_data_list.append(dict(
        parent_indices=parent_indices, active_mask=active_mask,
        is_division_mask=is_division_mask, is_new_root_mask=is_new_root_mask))
    crop_num_cells.append(num_cells)
    crop_labels.extend([crop_idx] * num_cells)
    print(f"  {crop}: T={T}, num_cancer={num_cells}, "
          f"divisions={int(is_division_mask.sum())}, roots={int(is_new_root_mask.sum())}")


# ============================================================
# Step 2: Concatenate masks across crops
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Concatenating data across crops")
print("=" * 60)

T = all_data_list[0]['active_mask'].shape[0]
total_cells = sum(crop_num_cells)
crop_labels = np.array(crop_labels)

combined_active = np.zeros((T, total_cells), dtype=bool)
combined_div = np.zeros((T, total_cells), dtype=bool)
combined_root = np.zeros((T, total_cells), dtype=bool)
combined_parent = np.zeros((T, total_cells), dtype=np.int32)

offset = 0
for d, nc in zip(all_data_list, crop_num_cells):
    s = slice(offset, offset + nc)
    combined_active[:, s] = d['active_mask']
    combined_div[:, s] = d['is_division_mask']
    combined_root[:, s] = d['is_new_root_mask']
    combined_parent[:, s] = d['parent_indices'] + offset
    offset += nc

combined_data = dict(active_mask=combined_active, is_division_mask=combined_div,
                     is_new_root_mask=combined_root, parent_indices=combined_parent)
print(f"Combined active_mask shape: {combined_active.shape}")


# ============================================================
# Step 3: Load emissions (non-DINO subset + standardize) and concat DINO PCs
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Assembling emissions")
print("=" * 60)

# Read saved non-DINO feature order from the first crop
with open(os.path.join(out_base_dir, crop_ids[0], 'cancer_emissions_names.txt')) as fh:
    saved_feature_names = [l.strip() for l in fh if l.strip()]
for feat in model_features:
    if feat not in saved_feature_names:
        raise ValueError(f"model_features '{feat}' not in saved features {saved_feature_names}")
feature_indices = [saved_feature_names.index(f) for f in model_features]
print(f"Non-DINO model_features {model_features} -> column indices {feature_indices}")

nondino_list, dino_list = [], []
for crop in crop_ids:
    e = np.load(os.path.join(out_base_dir, crop, 'cancer_emissions_array.npy'))[:, :, feature_indices]
    nondino_list.append(e)
    if use_dino:
        dino_list.append(np.load(os.path.join(out_base_dir, crop, 'cancer_dino_pca.npy')))

nondino = np.concatenate(nondino_list, axis=1).astype(float)   # (T, total, n_nondino)

# --- z-score non-DINO features jointly over ACTIVE cell-frames ---
if standardize_nondino:
    active = combined_data['active_mask']
    flat = nondino[active]                       # (n_active, n_nondino)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    nondino = (nondino - mean) / std
    np.savez(os.path.join(out_base_dir, 'nondino_standardization.npz'),
             mean=mean, std=std, features=np.array(model_features))
    print(f"Standardized non-DINO features: mean={mean}, std={std}")

if use_dino:
    dino = np.concatenate(dino_list, axis=1).astype(float)     # (T, total, n_dino_pcs)
    emissions = np.concatenate([nondino, dino], axis=-1)
    final_feature_names = list(model_features) + [f'dino_pc_{i}' for i in range(dino.shape[-1])]
else:
    emissions = nondino
    final_feature_names = list(model_features)

with open(os.path.join(out_base_dir, 'final_feature_names.txt'), 'w') as fh:
    fh.write("\n".join(final_feature_names) + "\n")
print(f"Final emissions shape: {emissions.shape}  (emission_dim={emissions.shape[-1]})")


# ============================================================
# Step 4: Filter tracks by time
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Filtering tracks by time")
print("=" * 60)

n_before = combined_data['active_mask'].shape[1]
combined_data, emissions, kept_indices = filter_tracks_by_time(combined_data, emissions, min_t=min_t)
kept_mask = np.zeros(n_before, dtype=bool)
kept_mask[kept_indices] = True
crop_labels = crop_labels[kept_mask]
print(f"After filtering: emissions {emissions.shape}, crop_labels {crop_labels.shape}")


# ============================================================
# Step 4b: AR warmup (mask first max(1, num_lags) active frames per cell)
# ============================================================
print("\n" + "=" * 60)
print("Step 4b: Applying AR warmup")
print("=" * 60)

warmup_frames = max(1, num_lags)
active = combined_data['active_mask']
root = combined_data['is_new_root_mask']
for col in range(active.shape[1]):
    af = np.where(active[:, col])[0]
    if len(af) <= warmup_frames:
        continue
    for i in range(warmup_frames):
        active[af[i], col] = False
        root[af[i], col] = False
    root[af[warmup_frames], col] = True
print(f"Warmup of {warmup_frames} frame(s); inferred time-points = {active.sum()}")


# ============================================================
# Step 5: Fit AR-HMM jointly
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Fitting AR-HMM")
print("=" * 60)

import jax.numpy as jnp
import jax.random as jr
from models.tarhmm import tARHMM, tree_hmm_two_filter_smoother

emission_dim = emissions.shape[-1]
arhmm = tARHMM(num_states, emission_dim, num_lags=num_lags)
params, props = arhmm.initialize(key=jr.PRNGKey(0))

emissions_jnp = jnp.array(emissions)
inputs = arhmm.compute_inputs(
    emissions_jnp,
    jnp.array(combined_data['parent_indices']),
    jnp.array(combined_data['is_division_mask']),
    jnp.array(combined_data['is_new_root_mask']),
    jnp.array(combined_data['active_mask']),
)

fitted_params, lps = arhmm.fit_em(
    params, props, emissions_jnp[None, ...], inputs=inputs[None, ...],
    parent_indices=jnp.array(combined_data['parent_indices'])[None, ...],
    is_division_mask=jnp.array(combined_data['is_division_mask'])[None, ...],
    active_mask=jnp.array(combined_data['active_mask'])[None, ...],
    is_new_root_mask=jnp.array(combined_data['is_new_root_mask'])[None, ...],
)
print(f"Final log prob: {float(lps[-1]):.2f}  (delta first->last: {float(lps[-1] - lps[0]):.2f})")


# ============================================================
# Step 6: Posterior and state assignments
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Computing posterior")
print("=" * 60)

inference_args = arhmm._inference_args(
    fitted_params, emissions_jnp, inputs,
    combined_data['parent_indices'], combined_data['is_division_mask'],
    combined_data['active_mask'], combined_data['is_new_root_mask'])
posterior = tree_hmm_two_filter_smoother(*inference_args)

state_assignments = jnp.argmax(posterior.smoothed_probs, axis=-1)
max_probs = jnp.max(posterior.smoothed_probs, axis=-1)
masked_state_assignments = jnp.where(jnp.isnan(max_probs.T), jnp.nan, state_assignments.T)
print(f"state_assignments shape: {state_assignments.shape}")

os.makedirs(out_base_dir, exist_ok=True)
num_crops = len(crop_ids)


# ============================================================
# Output 1: State assignments heatmap with crop colorbar
# ============================================================
print("\nOutput 1: state assignments heatmap")
fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [1, 30]}, sharey=True)
crop_cmap = plt.cm.get_cmap('tab10', num_crops)
crop_arr = np.array(crop_labels).reshape(-1, 1)
axes[0].imshow(crop_arr, aspect='auto', interpolation='none', cmap=crop_cmap,
               vmin=0, vmax=num_crops - 1, origin='upper')
axes[0].set_xticks([]); axes[0].set_ylabel('Cell'); axes[0].set_title('Crop')
legend_elements = [Patch(facecolor=crop_cmap(i), label=crop_ids[i]) for i in range(num_crops)]
im = axes[1].imshow(np.array(masked_state_assignments), aspect='auto',
                    interpolation='none', cmap='viridis', origin='upper')
axes[1].set_xlabel('Time')
axes[1].set_title('State Assignments (argmax smoothed_probs) - cancer cells')
cbar = plt.colorbar(im, ax=axes[1], ticks=range(num_states)); cbar.set_label('State')
fig.legend(handles=legend_elements, loc='lower center', ncol=3, title='Crop ID',
           bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()
plt.savefig(os.path.join(out_base_dir, 'state_assignments.png'), dpi=300, bbox_inches='tight')
plt.close()


# ============================================================
# Output 2: Feature distributions per state (12 panels)
# ============================================================
print("Output 2: feature distributions")
states_flat = np.array(masked_state_assignments.T).flatten()
emissions_flat = np.array(emissions.reshape(-1, emissions.shape[-1]))
# Show non-DINO features in their real units (invert the z-score) so e.g. the
# T-cell-neighbor histogram reflects actual counts, not standardized values.
if standardize_nondino:
    n_nd = len(model_features)
    emissions_flat[:, :n_nd] = emissions_flat[:, :n_nd] * std + mean
df = pd.DataFrame(emissions_flat, columns=final_feature_names)
df['state'] = states_flat
df_clean = df.dropna(subset=['state']).copy()
df_clean['state'] = df_clean['state'].astype(int)

_discrete = {'t_cell_neighbors_20px'}
n_feat = len(final_feature_names)
ncols = 4
nrows = int(np.ceil(n_feat / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                         tight_layout=True, squeeze=False)
axes = axes.flatten()
for i, feat in enumerate(final_feature_names):
    ax = axes[i]
    if feat in _discrete:
        sns.histplot(data=df_clean, x=feat, hue='state', ax=ax, palette='viridis',
                     common_norm=False, stat='density', element='step', discrete=True, legend=False)
        ax.set_xlabel(feat.replace("_", " ").title())
    else:
        sns.violinplot(data=df_clean, x='state', y=feat, hue='state', ax=ax,
                       palette='viridis', legend=False)
        ax.set_ylabel(feat.replace("_", " ").title()); ax.set_xlabel('State')
    sns.despine(ax=ax)
for j in range(n_feat, len(axes)):
    axes[j].axis('off')
plt.savefig(os.path.join(out_base_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()


# ============================================================
# Output 3: Save per-crop state assignments
# ============================================================
print("Output 3: per-crop state assignments")
for crop_idx, crop in enumerate(crop_ids):
    cell_idx = np.where(crop_labels == crop_idx)[0]
    crop_states = np.array(state_assignments[:, cell_idx])
    crop_dir = os.path.join(out_base_dir, crop)
    os.makedirs(crop_dir, exist_ok=True)
    np.save(os.path.join(crop_dir, 'cancer_state_assignments.npy'), crop_states)


# ============================================================
# Output 4: State fraction over time (per crop)
# ============================================================
print("Output 4: state fractions over time")
state_colors = plt.cm.get_cmap('viridis', num_states)
state_color_list = [state_colors(s) for s in range(num_states)]
fig, axes = plt.subplots(2, 3, figsize=(9, 6), tight_layout=True)
axes_flat = axes.flatten()
for crop_idx, crop in enumerate(crop_ids):
    ax = axes_flat[crop_idx]
    cell_mask = crop_labels == crop_idx
    crop_states = np.array(state_assignments[:, cell_mask])
    crop_active = np.array(combined_data['active_mask'][:, cell_mask])
    T_len = crop_states.shape[0]
    fractions = np.zeros((num_states, T_len))
    for t in range(T_len):
        at = crop_active[t]
        n_active = at.sum()
        if n_active == 0:
            continue
        sat = crop_states[t, at]
        for s in range(num_states):
            fractions[s, t] = np.sum(sat == s) / n_active
    ax.stackplot(np.arange(T_len), fractions, colors=state_color_list,
                 labels=[f'State {s}' for s in range(num_states)])
    ax.set_title(crop); ax.set_ylim(0, 1)
    if crop_idx >= 3:
        ax.set_xlabel('Time')
    if crop_idx % 3 == 0:
        ax.set_ylabel('Fraction of cells')
for j in range(num_crops, len(axes_flat)):
    axes_flat[j].axis('off')
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=num_states, bbox_to_anchor=(0.5, -0.02))
plt.suptitle('Fraction of cancer cells per state over time')
plt.savefig(os.path.join(out_base_dir, 'state_counts.png'), dpi=300, bbox_inches='tight')
plt.close()


# ============================================================
# Output 5: Learned transition matrix
# ============================================================
print("Output 5: learned transition matrix")
trans_matrix = np.array(fitted_params.transitions.transition_matrix)
fig, ax = plt.subplots(figsize=(3, 3))
im = ax.imshow(trans_matrix, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(num_states)); ax.set_yticks(range(num_states))
ax.set_xticklabels([f'State {s}' for s in range(num_states)])
ax.set_yticklabels([f'State {s}' for s in range(num_states)])
ax.set_xlabel('State at $t+1$'); ax.set_ylabel('State at $t$')
ax.set_title('Learned Transition Matrix')
for i in range(num_states):
    for j in range(num_states):
        val = trans_matrix[i, j]
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                color='white' if val > 0.5 else 'black')
cbar = plt.colorbar(im, ax=ax); cbar.set_label('Transition probability')
plt.tight_layout()
plt.savefig(os.path.join(out_base_dir, 'learned_transition_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()


# ============================================================
# Output 6: Observed transition matrices (per crop)
# ============================================================
print("Output 6: observed transition matrices")
fig, axes = plt.subplots(2, 3, figsize=(9, 6), tight_layout=True)
axes_flat = axes.flatten()
for crop_idx, crop in enumerate(crop_ids):
    ax = axes_flat[crop_idx]
    cell_mask = crop_labels == crop_idx
    crop_states = np.array(state_assignments[:, cell_mask])
    crop_active = np.array(combined_data['active_mask'][:, cell_mask])
    obs = np.zeros((num_states, num_states))
    T_len, n_cells = crop_states.shape
    for c in range(n_cells):
        for t in range(T_len - 1):
            if crop_active[t, c] and crop_active[t + 1, c]:
                obs[crop_states[t, c], crop_states[t + 1, c]] += 1
    rs = obs.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    obs_p = obs / rs
    im = ax.imshow(obs_p, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(num_states)); ax.set_yticks(range(num_states))
    ax.set_xticklabels([f'{s}' for s in range(num_states)])
    ax.set_yticklabels([f'{s}' for s in range(num_states)])
    ax.set_xlabel('State at $t+1$'); ax.set_ylabel('State at $t$')
    ax.set_title(f'{crop}\nObserved Transition Matrix')
    for i in range(num_states):
        for j in range(num_states):
            val = obs_p[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color='white' if val > 0.5 else 'black')
for j in range(num_crops, len(axes_flat)):
    axes_flat[j].axis('off')
plt.savefig(os.path.join(out_base_dir, 'observed_transition_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()


# ============================================================
# Output 7: t_cell_neighbors heatmap with crop colorbar
# ============================================================
print("Output 7: t_cell_neighbors heatmap")
with open(os.path.join(out_base_dir, crop_ids[0], 'cancer_emissions_names.txt')) as fh:
    full_names = [l.strip() for l in fh if l.strip()]
nbr_idx = full_names.index('t_cell_neighbors_20px')

full_list = [np.load(os.path.join(out_base_dir, c, 'cancer_emissions_array.npy')) for c in crop_ids]
full_emissions = np.concatenate(full_list, axis=1)[:, kept_indices, :]
nbr_vals = full_emissions[:, :, nbr_idx].T  # (cells, T)
active_np = np.array(combined_data['active_mask']).T
nbr_masked = np.where(active_np, nbr_vals, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [1, 30]}, sharey=True)
crop_arr = np.array(crop_labels).reshape(-1, 1)
axes[0].imshow(crop_arr, aspect='auto', interpolation='none', cmap=crop_cmap,
               vmin=0, vmax=num_crops - 1, origin='upper')
axes[0].set_xticks([]); axes[0].set_ylabel('Cell'); axes[0].set_title('Crop')
legend_elements = [Patch(facecolor=crop_cmap(i), label=crop_ids[i]) for i in range(num_crops)]
im = axes[1].imshow(nbr_masked, aspect='auto', interpolation='none', cmap='magma', origin='upper')
axes[1].set_xlabel('Time'); axes[1].set_title('T cells within 20px of cancer nucleus - All Crops')
cbar = plt.colorbar(im, ax=axes[1]); cbar.set_label('T-cell neighbors (20px)')
fig.legend(handles=legend_elements, loc='lower center', ncol=3, title='Crop ID',
           bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()
plt.savefig(os.path.join(out_base_dir, 't_cell_neighbors_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
