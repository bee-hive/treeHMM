"""Step 4: Fit the tree-HMM over cancer nuclei, jointly across all wells.

Cancer cells ARE the Caliban nuclei tracks.  Emissions are, by default, the
whitened DINOv2 PCs of a 30x30 patch centered on each nucleus -- no scalar
feature reaches the model.  Division lineage from ``{well}_div.pkl`` is wired
into the tree via direct Caliban ID lookup (the daughters' first active frame
IS ``div.frame``; the parent's last active frame is ``div.frame - 1``).

The scalar features from Step 1 are used ONLY as held-out diagnostics.  The
headline question -- did the two states separate isolated cells from cells in
aggregates? -- is answered by ``state_vs_aggregation.png`` and the AUROC in
``state_summary.json``, both computed against ``nuclei_neighbors_30px``, which
the model never saw.

Outputs (under ``output_base_dir``):
    {well}/nuclei_state_assignments.npy   (T, N_well) argmax smoothed states
    state_vs_aggregation.png              PRIMARY VALIDATION
    diagnostic_features_by_state.png      all 7 held-out scalars per state
    dino_pcs_by_state.png                 what the model actually saw
    state_assignments.png                 per-cell state over time
    state_counts.png                      state fractions over time, per condition
    transition_matrices.png               learned P_std and P_div
    condition_occupancy.png               state occupancy by condition
    state_summary.json                    headline numbers
    final_feature_names.txt, nondino_standardization.npz (if model_features)

Usage (treeHMM_env):
    conda run -n treeHMM_env python fit_nuclei_hmm.py
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.patches import Patch

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

wells: List[str] = cfg["wells"]
conditions_dict: Dict[str, List[str]] = cfg["conditions_dict"]
out_base_dir: str = cfg["output_base_dir"]
num_states: int = cfg["num_states"]
num_lags: int = cfg["num_lags"]
min_t: int = cfg["min_t"]
model_features: List[str] = cfg["model_features"]
use_dino: bool = cfg.get("use_dino", True)
allow_divisions: bool = cfg.get("allow_divisions", True)
ar_warmup: bool = cfg.get("ar_warmup", False)
standardize_nondino: bool = cfg.get("standardize_nondino", True)
init_method: str = cfg.get("init_method", "kmeans")
init_stickiness: float = cfg.get("init_stickiness", 0.9)
num_em_iters: int = cfg.get("num_em_iters", 50)
em_seed: int = cfg.get("em_seed", 0)

if num_lags not in (0, 1):
    raise ValueError(f"num_lags must be 0 or 1; got {num_lags}")
if not use_dino and not model_features:
    raise ValueError("use_dino is false and model_features is empty -- no emissions")

caliban_dir = Path(cfg["snakemake_runs_dir"]) / cfg["caliban_run"] / "output"
well_to_condition = {w: c for c, ws in conditions_dict.items() for w in ws}

SMALL, MEDIUM, BIG = 7, 8, 10
plt.rc('font', size=SMALL)
plt.rc('axes', titlesize=MEDIUM, labelsize=SMALL)
plt.rc('xtick', labelsize=SMALL)
plt.rc('ytick', labelsize=SMALL)
plt.rc('legend', fontsize=SMALL)
plt.rc('figure', titlesize=BIG)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.use14corefonts'] = True


def filter_tracks_by_time(
    data: Dict[str, np.ndarray],
    emissions: np.ndarray,
    min_t: int = 10
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Drop short tracks and remap parent indices consistently.

    Adapted from ``scripts/cancer_dino_hmm/fit_cancer_arhmm.py`` with one fix:
    when a division child is orphaned because its parent was filtered out, the
    upstream version promotes it to a new root but leaves ``is_division_mask``
    set, so inference would apply the division kernel to a self-parent edge.
    Here the division flag is cleared alongside the promotion.

    Args:
        data (Dict[str, np.ndarray]): the four (T, C) mask arrays.
        emissions (np.ndarray): (T, C, D) emissions.
        min_t (int): minimum number of active frames to keep a cell.

    Returns:
        Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]: filtered masks,
            filtered emissions, and the kept column indices.
    """
    durations = np.sum(data['active_mask'], axis=0)
    keep_indices = np.where(durations >= min_t)[0]

    if len(keep_indices) == 0:
        raise RuntimeError(f"No cells with duration >= {min_t} frames.")

    filtered_emissions = emissions[:, keep_indices, :]
    filtered_data = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            filtered_data[key] = val[:, keep_indices]
        else:
            filtered_data[key] = val

    max_old = data['parent_indices'].max()
    lookup = np.full(max_old + 1, -1, dtype=np.int32)
    for new, old in enumerate(keep_indices):
        lookup[old] = new

    curr_parents = filtered_data['parent_indices']
    curr_active = filtered_data['active_mask']
    new_parents = np.copy(curr_parents)
    rows, cols = np.where(curr_active)
    new_parents[rows, cols] = lookup[curr_parents[rows, cols]]

    orphans = (new_parents == -1) & curr_active
    n_orphan_divs = int((orphans & filtered_data['is_division_mask']).sum())
    if np.any(orphans):
        filtered_data['is_new_root_mask'][orphans] = True
        filtered_data['is_division_mask'][orphans] = False
        o_rows, o_cols = np.where(orphans)
        new_parents[o_rows, o_cols] = o_cols

    filtered_data['parent_indices'] = new_parents
    print(f"  Filtered {data['active_mask'].shape[1]} -> {len(keep_indices)} cells "
          f"(min_t={min_t}); {n_orphan_divs} division edges dropped to roots.")
    return filtered_data, filtered_emissions, keep_indices


def build_well_masks(
    centroids: np.ndarray,
    cell_ids: np.ndarray,
    well: str
) -> Dict[str, np.ndarray]:
    """Build the four tree-HMM masks for one well from nucleus centroids.

    Each nucleus is its own chain (self-parent, new root at its first active
    frame).  Division edges are then overlaid from ``{well}_div.pkl`` when
    ``allow_divisions`` is set.

    Args:
        centroids (np.ndarray): (T, N, 2) centroids, NaN where absent.
        cell_ids (np.ndarray): (N,) sorted Caliban IDs matching the columns.
        well (str): well name, used to locate the divisions pickle.

    Returns:
        Dict[str, np.ndarray]: active_mask, parent_indices, is_division_mask,
            is_new_root_mask.
    """
    num_frames, num_cells = centroids.shape[:2]
    id_to_col = {int(cid): i for i, cid in enumerate(cell_ids)}

    active_mask = ~np.isnan(centroids[:, :, 0])
    is_division_mask = np.zeros((num_frames, num_cells), dtype=bool)
    is_new_root_mask = np.zeros((num_frames, num_cells), dtype=bool)
    parent_indices = np.zeros((num_frames, num_cells), dtype=np.int32)

    cols = np.arange(num_cells)
    parent_indices[:, :] = cols[None, :]
    for col in cols:
        active_frames = np.where(active_mask[:, col])[0]
        if len(active_frames) == 0:
            continue
        is_new_root_mask[active_frames[0], col] = True

    if allow_divisions:
        div_path = caliban_dir / well / "Caliban" / f"{well}_div.pkl"
        with open(div_path, "rb") as fh:
            divisions = pickle.load(fh)

        applied, skipped = 0, 0
        for _, row in divisions.iterrows():
            parent = int(row["parent"])
            frame = int(row["frame"])
            if parent not in id_to_col:
                skipped += 2
                continue
            p_col = id_to_col[parent]
            # The parent's last active frame is div.frame - 1 in the common
            # case, but it occasionally vanishes a frame or two earlier.
            if frame == 0 or not active_mask[frame - 1, p_col]:
                skipped += 2
                continue
            for key in ("daughter_1", "daughter_2"):
                daughter = int(row[key])
                if daughter not in id_to_col:
                    skipped += 1
                    continue
                d_col = id_to_col[daughter]
                if d_col == p_col or not active_mask[frame, d_col]:
                    skipped += 1
                    continue
                is_division_mask[frame, d_col] = True
                is_new_root_mask[frame, d_col] = False
                parent_indices[frame, d_col] = p_col
                applied += 1
        print(f"  [divisions] {well}: applied {applied}, skipped {skipped} "
              f"(of {2 * len(divisions)} daughter edges)")

    return dict(active_mask=active_mask, parent_indices=parent_indices,
                is_division_mask=is_division_mask,
                is_new_root_mask=is_new_root_mask)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum identity.

    Args:
        scores (np.ndarray): continuous score per observation.
        labels (np.ndarray): boolean, True for the positive class.

    Returns:
        float: AUROC, or NaN if either class is empty.
    """
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ============================================================
print("=" * 60)
print("Step 1: Building masks per well")
print("=" * 60)

all_masks, well_num_cells, well_labels = [], [], []
for well_idx, well in enumerate(wells):
    out_dir = os.path.join(out_base_dir, well)
    cell_ids = np.load(os.path.join(out_dir, "nuclei_cell_ids.npy"))
    centroids = np.load(os.path.join(out_dir, "nuclei_centroids.npy"))
    masks = build_well_masks(centroids, cell_ids, well)
    all_masks.append(masks)
    well_num_cells.append(len(cell_ids))
    well_labels.extend([well_idx] * len(cell_ids))
    print(f"  {well}: T={centroids.shape[0]}, num_nuclei={len(cell_ids)}, "
          f"divisions={int(masks['is_division_mask'].sum())}, "
          f"roots={int(masks['is_new_root_mask'].sum())}")

num_frames = all_masks[0]['active_mask'].shape[0]
total_cells = sum(well_num_cells)
well_labels = np.array(well_labels)


# ============================================================
print("\n" + "=" * 60)
print("Step 2: Concatenating across wells")
print("=" * 60)

combined = dict(
    active_mask=np.zeros((num_frames, total_cells), dtype=bool),
    is_division_mask=np.zeros((num_frames, total_cells), dtype=bool),
    is_new_root_mask=np.zeros((num_frames, total_cells), dtype=bool),
    parent_indices=np.zeros((num_frames, total_cells), dtype=np.int32),
)
offset = 0
for masks, n_cells in zip(all_masks, well_num_cells):
    sl = slice(offset, offset + n_cells)
    combined['active_mask'][:, sl] = masks['active_mask']
    combined['is_division_mask'][:, sl] = masks['is_division_mask']
    combined['is_new_root_mask'][:, sl] = masks['is_new_root_mask']
    combined['parent_indices'][:, sl] = masks['parent_indices'] + offset
    offset += n_cells
print(f"Combined active_mask: {combined['active_mask'].shape}, "
      f"total divisions={int(combined['is_division_mask'].sum())}")


# ============================================================
print("\n" + "=" * 60)
print("Step 3: Assembling emissions")
print("=" * 60)

with open(os.path.join(out_base_dir, wells[0], 'nuclei_emissions_names.txt')) as fh:
    saved_feature_names = [l.strip() for l in fh if l.strip()]

# Full diagnostic scalars, kept aside; NEVER fed to the model unless named in
# model_features.
diag_list = [np.load(os.path.join(out_base_dir, w, 'nuclei_emissions_array.npy'))
             for w in wells]
diagnostics = np.concatenate(diag_list, axis=1).astype(float)

mean = std = None
if model_features:
    for feat in model_features:
        if feat not in saved_feature_names:
            raise ValueError(f"model_features '{feat}' not in {saved_feature_names}")
    idx = [saved_feature_names.index(f) for f in model_features]
    nondino = diagnostics[:, :, idx].copy()
    if standardize_nondino:
        flat = nondino[combined['active_mask']]
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        nondino = (nondino - mean) / std
        np.savez(os.path.join(out_base_dir, 'nondino_standardization.npz'),
                 mean=mean, std=std, features=np.array(model_features))
    print(f"Scalar model_features {model_features} -> indices {idx}")
else:
    nondino = np.zeros((num_frames, total_cells, 0), dtype=float)
    print("model_features is empty -> emissions are DINO PCs only")

if use_dino:
    dino = np.concatenate(
        [np.load(os.path.join(out_base_dir, w, 'nuclei_dino_pca.npy')) for w in wells],
        axis=1).astype(float)
    emissions = np.concatenate([nondino, dino], axis=-1)
    final_feature_names = list(model_features) + \
        [f'dino_pc_{i}' for i in range(dino.shape[-1])]
else:
    emissions = nondino
    final_feature_names = list(model_features)

with open(os.path.join(out_base_dir, 'final_feature_names.txt'), 'w') as fh:
    fh.write("\n".join(final_feature_names) + "\n")
print(f"Emissions: {emissions.shape} (emission_dim={emissions.shape[-1]})")


# ============================================================
print("\n" + "=" * 60)
print("Step 4: Filtering short tracks")
print("=" * 60)

n_before = total_cells
combined, emissions, kept_indices = filter_tracks_by_time(combined, emissions, min_t)
diagnostics = diagnostics[:, kept_indices, :]
well_labels = well_labels[kept_indices]
print(f"After filtering: emissions {emissions.shape}, "
      f"divisions retained={int(combined['is_division_mask'].sum())}")

if ar_warmup:
    print("\nApplying AR warmup (skipping division children)")
    warmup = max(1, num_lags)
    active, root, is_div = (combined['active_mask'], combined['is_new_root_mask'],
                            combined['is_division_mask'])
    skipped_div = 0
    for col in range(active.shape[1]):
        af = np.where(active[:, col])[0]
        if len(af) <= warmup:
            continue
        if is_div[af[0], col]:
            # Masking this frame would destroy the division edge; leave it.
            skipped_div += 1
            continue
        for i in range(warmup):
            active[af[i], col] = False
            root[af[i], col] = False
        root[af[warmup], col] = True
    print(f"  warmup={warmup} frame(s); {skipped_div} division children skipped; "
          f"inferred cell-frames={int(active.sum())}")


# ============================================================
print("\n" + "=" * 60)
print("Step 5: Fitting the HMM")
print("=" * 60)

import sys
sys.path.insert(0, cfg["treehmm_dir"])
import jax.numpy as jnp
import jax.random as jr
from models.tarhmm import tARHMM, tree_hmm_two_filter_smoother

emission_dim = emissions.shape[-1]
arhmm = tARHMM(num_states, emission_dim, num_lags=num_lags)

sticky = (jnp.eye(num_states) * init_stickiness
          + (1.0 - jnp.eye(num_states)) * (1.0 - init_stickiness) / max(num_states - 1, 1))
emissions_jnp = jnp.array(emissions)

init_kwargs = dict(key=jr.PRNGKey(em_seed), method=init_method,
                   transition_matrix=sticky, division_transition_matrix=sticky)
if init_method == "kmeans":
    init_kwargs["emissions"] = emissions_jnp
params, props = arhmm.initialize(**init_kwargs)

parent_j = jnp.array(combined['parent_indices'])
div_j = jnp.array(combined['is_division_mask'])
active_j = jnp.array(combined['active_mask'])
root_j = jnp.array(combined['is_new_root_mask'])

inputs = arhmm.compute_inputs(emissions_jnp, parent_j, div_j, root_j, active_j)
fitted_params, lps = arhmm.fit_em(
    params, props, emissions_jnp[None, ...], inputs=inputs[None, ...],
    parent_indices=parent_j[None, ...], is_division_mask=div_j[None, ...],
    active_mask=active_j[None, ...], is_new_root_mask=root_j[None, ...],
    num_iters=num_em_iters)
lps = np.array(lps)
print(f"Log prob: {lps[0]:.1f} -> {lps[-1]:.1f} (delta {lps[-1] - lps[0]:.1f})")


# ============================================================
print("\n" + "=" * 60)
print("Step 6: Posterior and diagnostics")
print("=" * 60)

posterior = tree_hmm_two_filter_smoother(*arhmm._inference_args(
    fitted_params, emissions_jnp, inputs, parent_j, div_j, active_j, root_j))
states = np.array(jnp.argmax(posterior.smoothed_probs, axis=-1))  # (T, C)
active_np = combined['active_mask']

os.makedirs(out_base_dir, exist_ok=True)
for well_idx, well in enumerate(wells):
    cols = np.where(well_labels == well_idx)[0]
    np.save(os.path.join(out_base_dir, well, 'nuclei_state_assignments.npy'),
            states[:, cols])

state_colors = plt.get_cmap('viridis', num_states)
state_color_list = [state_colors(s) for s in range(num_states)]

# --- Flatten active cell-frames for all downstream diagnostics ---
flat_states = states[active_np]
flat_diag = diagnostics[active_np]
flat_emis = emissions[active_np]
diag_df = pd.DataFrame(flat_diag, columns=saved_feature_names)
diag_df['state'] = flat_states

AGG = 'nuclei_neighbors_30px'
agg_counts = diag_df[AGG].to_numpy()

# Orient labels so that state 0 is the LESS aggregated one, purely for reading
# the plots; the saved .npy assignments keep the model's own numbering.
mean_by_state = [agg_counts[flat_states == s].mean() if (flat_states == s).any()
                 else np.nan for s in range(num_states)]
print(f"Mean {AGG} by state: "
      + ", ".join(f"state {s}={m:.2f}" for s, m in enumerate(mean_by_state)))


# ---------------- PRIMARY VALIDATION ----------------
print("\nOutput: state_vs_aggregation.png (primary validation)")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), tight_layout=True)

sns.violinplot(data=diag_df, x='state', y=AGG, hue='state', ax=axes[0],
               palette='viridis', legend=False, cut=0)
axes[0].set_title(f'{AGG} by state\n(held out from the model)')
axes[0].set_xlabel('State'); axes[0].set_ylabel('Nuclei within 30 px')

sns.histplot(data=diag_df, x=AGG, hue='state', ax=axes[1], palette='viridis',
             common_norm=False, stat='density', element='step', discrete=True)
axes[1].set_title('Distribution overlap')
axes[1].set_xlabel('Nuclei within 30 px'); axes[1].set_xlim(-0.5, 12.5)

sns.violinplot(data=diag_df, x='state', y='nearest_nucleus_dist', hue='state',
               ax=axes[2], palette='viridis', legend=False, cut=0)
axes[2].set_title('Distance to nearest nucleus')
axes[2].set_xlabel('State'); axes[2].set_ylabel('Pixels')
axes[2].set_ylim(0, 80)
for ax in axes:
    sns.despine(ax=ax)
plt.savefig(os.path.join(out_base_dir, 'state_vs_aggregation.png'),
            dpi=300, bbox_inches='tight')
plt.close()

summary = {
    'num_states': num_states,
    'num_lags': num_lags,
    'emission_dim': int(emission_dim),
    'model_features': list(model_features),
    'use_dino': bool(use_dino),
    'dino_patch_px': cfg['dino_patch_px'],
    'allow_divisions': bool(allow_divisions),
    'n_cells_after_filter': int(emissions.shape[1]),
    'n_active_cell_frames': int(active_np.sum()),
    'n_division_edges': int(combined['is_division_mask'].sum()),
    'log_prob_first': float(lps[0]),
    'log_prob_last': float(lps[-1]),
    f'mean_{AGG}_by_state': [float(m) for m in mean_by_state],
    'state_occupancy': [float((flat_states == s).mean()) for s in range(num_states)],
}
if num_states == 2:
    hi = int(np.nanargmax(mean_by_state))
    summary['aggregate_state'] = hi
    summary['auroc_neighbors30_predicts_aggregate_state'] = float(
        auroc(agg_counts, flat_states == hi))
    summary['auroc_nearest_dist_predicts_isolated_state'] = float(
        auroc(diag_df['nearest_nucleus_dist'].to_numpy(), flat_states != hi))
    print(f"  AUROC({AGG} -> state {hi}) = "
          f"{summary['auroc_neighbors30_predicts_aggregate_state']:.3f}")

with open(os.path.join(out_base_dir, 'state_summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)


# ---------------- All held-out diagnostics ----------------
print("Output: diagnostic_features_by_state.png")
n_feat = len(saved_feature_names)
ncols = 4
nrows = int(np.ceil(n_feat / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.8 * nrows),
                         tight_layout=True, squeeze=False)
axes = axes.flatten()
for i, feat in enumerate(saved_feature_names):
    sns.violinplot(data=diag_df, x='state', y=feat, hue='state', ax=axes[i],
                   palette='viridis', legend=False, cut=0)
    axes[i].set_ylabel(feat.replace('_', ' ')); axes[i].set_xlabel('State')
    sns.despine(ax=axes[i])
for j in range(n_feat, len(axes)):
    axes[j].axis('off')
plt.suptitle('Held-out scalar features by inferred state')
plt.savefig(os.path.join(out_base_dir, 'diagnostic_features_by_state.png'),
            dpi=300, bbox_inches='tight')
plt.close()


# ---------------- What the model actually saw ----------------
print("Output: dino_pcs_by_state.png")
emis_df = pd.DataFrame(flat_emis, columns=final_feature_names)
emis_df['state'] = flat_states
n_show = min(len(final_feature_names), 12)
ncols = 4
nrows = int(np.ceil(n_show / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.6 * nrows),
                         tight_layout=True, squeeze=False)
axes = axes.flatten()
for i in range(n_show):
    feat = final_feature_names[i]
    sns.violinplot(data=emis_df, x='state', y=feat, hue='state', ax=axes[i],
                   palette='viridis', legend=False, cut=0)
    axes[i].set_xlabel('State'); sns.despine(ax=axes[i])
for j in range(n_show, len(axes)):
    axes[j].axis('off')
plt.suptitle('Model emissions by inferred state')
plt.savefig(os.path.join(out_base_dir, 'dino_pcs_by_state.png'),
            dpi=300, bbox_inches='tight')
plt.close()


# ---------------- State assignments heatmap ----------------
print("Output: state_assignments.png")
masked_states = np.where(active_np.T, states.T.astype(float), np.nan)
fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [1, 30]}, sharey=True)
well_cmap = plt.get_cmap('tab20', len(wells))
axes[0].imshow(well_labels.reshape(-1, 1), aspect='auto', interpolation='none',
               cmap=well_cmap, vmin=0, vmax=len(wells) - 1, origin='upper')
axes[0].set_xticks([]); axes[0].set_ylabel('Nucleus'); axes[0].set_title('Well')
im = axes[1].imshow(masked_states, aspect='auto', interpolation='none',
                    cmap='viridis', origin='upper')
axes[1].set_xlabel('Frame')
axes[1].set_title('State assignments (argmax smoothed posterior); '
                  'white = track not present')
cbar = plt.colorbar(im, ax=axes[1], ticks=range(num_states))
cbar.set_label('State')
fig.legend(handles=[Patch(facecolor=well_cmap(i), label=wells[i])
                    for i in range(len(wells))],
           loc='lower center', ncol=6, title='Well', bbox_to_anchor=(0.5, -0.08))
plt.tight_layout()
plt.savefig(os.path.join(out_base_dir, 'state_assignments.png'),
            dpi=300, bbox_inches='tight')
plt.close()


# ---------------- State fractions over time, by condition ----------------
print("Output: state_counts.png")
condition_names = list(conditions_dict.keys())
fig, axes = plt.subplots(1, len(condition_names),
                         figsize=(4 * len(condition_names), 3), tight_layout=True)
axes = np.atleast_1d(axes)
for c_idx, condition in enumerate(condition_names):
    cond_wells = [wells.index(w) for w in conditions_dict[condition] if w in wells]
    cell_mask = np.isin(well_labels, cond_wells)
    cond_states = states[:, cell_mask]
    cond_active = active_np[:, cell_mask]
    fractions = np.zeros((num_states, num_frames))
    for t in range(num_frames):
        at = cond_active[t]
        if at.sum() == 0:
            continue
        for s in range(num_states):
            fractions[s, t] = np.mean(cond_states[t, at] == s)
    axes[c_idx].stackplot(np.arange(num_frames), fractions, colors=state_color_list,
                          labels=[f'State {s}' for s in range(num_states)])
    axes[c_idx].set_title(condition); axes[c_idx].set_ylim(0, 1)
    axes[c_idx].set_xlabel('Frame')
    if c_idx == 0:
        axes[c_idx].set_ylabel('Fraction of nuclei')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=num_states,
           bbox_to_anchor=(0.5, -0.08))
plt.suptitle('State occupancy over time by condition')
plt.savefig(os.path.join(out_base_dir, 'state_counts.png'),
            dpi=300, bbox_inches='tight')
plt.close()


# ---------------- Transition matrices ----------------
print("Output: transition_matrices.png")
p_std = np.array(fitted_params.transitions.transition_matrix)
p_div = np.array(fitted_params.division_transitions.transition_matrix)
# Persist the matrices themselves, not just the figure, so downstream scripts
# can re-plot or compare them without refitting.
np.save(os.path.join(out_base_dir, 'learned_P_std.npy'), p_std)
np.save(os.path.join(out_base_dir, 'learned_P_div.npy'), p_div)
fig, axes = plt.subplots(1, 2, figsize=(7, 3), tight_layout=True)
for ax, mat, title in zip(axes, [p_std, p_div],
                          ['Persistence $P_{std}$', 'Division $P_{div}$']):
    im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(num_states)); ax.set_yticks(range(num_states))
    ax.set_xlabel('State at $t+1$'); ax.set_ylabel('State at $t$')
    ax.set_title(title)
    for i in range(num_states):
        for j in range(num_states):
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                    color='white' if mat[i, j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax)
plt.savefig(os.path.join(out_base_dir, 'transition_matrices.png'),
            dpi=300, bbox_inches='tight')
plt.close()


# ---------------- Condition occupancy ----------------
print("Output: condition_occupancy.png")
rows = []
for well_idx, well in enumerate(wells):
    cell_mask = well_labels == well_idx
    ws = states[:, cell_mask][active_np[:, cell_mask]]
    for s in range(num_states):
        rows.append({'well': well, 'condition': well_to_condition.get(well, '?'),
                     'state': s, 'occupancy': float(np.mean(ws == s))})
occ_df = pd.DataFrame(rows)
occ_df.to_csv(os.path.join(out_base_dir, 'condition_occupancy.csv'), index=False)

fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
sns.barplot(data=occ_df, x='condition', y='occupancy', hue='state',
            palette='viridis', ax=ax, errorbar='sd', capsize=0.1)
sns.stripplot(data=occ_df, x='condition', y='occupancy', hue='state',
              palette='viridis', ax=ax, dodge=True, legend=False,
              edgecolor='black', linewidth=0.5, size=4)
ax.set_ylabel('Fraction of active cell-frames'); ax.set_xlabel('')
ax.set_title('State occupancy by condition (points = wells)')
sns.despine(ax=ax)
plt.savefig(os.path.join(out_base_dir, 'condition_occupancy.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print(f"Done -> {out_base_dir}")
print("=" * 60)
