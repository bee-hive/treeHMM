"""Step 4: Fit the death-state HMM for every arm x k x seed.

All arms are fitted over exactly the SAME cells, with the same min_t filter
and the same one-frame AR warmup, so differences between arms are
attributable to the features and lag order alone.  The tree (division) part
is off throughout: every cell is its own parent.

For each (arm, k) the fit is repeated over `em_seeds` and the run with the
best final log probability is kept -- a death state occupies a small
fraction of cell-frames, and which local optimum EM lands in matters more
than usual when the state of interest is rare.

Death-candidate selection is post-hoc, annotation-independent, and applied
identically to every arm, so no arm gets to peek at the labels.  Each state
is scored on the DIAGNOSTIC features (all ten are available regardless of
which fed the model):

    transition_score  = z(mean d_circularity) - z(mean d_area_frac)
    instability_score = mean of z(mean win_std_log_area),
                                z(mean win_std_circularity),
                                z(mean win_std_displacement)
    death_score       = transition_score + instability_score

z(.) standardizes across the k states of that fit.  The transition term is
the acute signature from the ground-truth notes (area drops, circularity
rises); the instability term is the elevated post-transition variance.  A
single-frame death state scores mainly on the first, a durable one mainly
on the second, and a real death state should show both -- which is why they
are summed rather than chosen between.

Outputs, under {output_base_dir}/fits/{arm}/k{K}/:
    state_assignments.npy    (T, n_cells) argmax of the smoothed posterior
    smoothed_probs.npy       (T, n_cells, K)
    state_profile.csv        per-state occupancy + mean of every diagnostic
    fit_summary.yml          log probs, seed used, death state, transition matrix
    states.png               state heatmap + learned transition matrix
Plus a top-level {output_base_dir}/fits/summary.csv over all arms and k.

Usage (treeHMM_env):
    conda run -n treeHMM_env python fit_death_hmm.py
"""

import os
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

crop_ids = cfg["crop_ids"]
type_sep_tracks_dir = cfg["type_sep_tracks_dir"]
out_base_dir = cfg["output_base_dir"]
arms = cfg["arms"]
k_sweep = cfg["num_states_sweep"]
em_seeds = cfg["em_seeds"]
num_em_iters = cfg["num_em_iters"]
init_method = cfg["init_method"]
init_stickiness = cfg["init_stickiness"]
min_t = cfg["min_t"]
standardize = cfg.get("standardize_features", True)
allow_divisions = cfg.get("allow_divisions", False)

if allow_divisions:
    raise NotImplementedError(
        "These experiments deliberately leave the tree part off; "
        "set allow_divisions: false.")

# Optional arm filter: `python fit_death_hmm.py delta_lag0 shape_lag1`
arm_filter = set(sys.argv[1:])
if arm_filter:
    arms = [a for a in arms if a["name"] in arm_filter]
    if not arms:
        sys.exit(f"No arms matched {sorted(arm_filter)}")

sys.path.insert(0, cfg["treehmm_dir"])
fits_dir = os.path.join(out_base_dir, "fits")
os.makedirs(fits_dir, exist_ok=True)

plt.rc("font", size=7)
plt.rc("axes", titlesize=8, labelsize=7)
plt.rc("xtick", labelsize=7)
plt.rc("ytick", labelsize=7)


def filter_tracks_by_time(data, emissions, extras, min_frames):
    """Keep cells active for at least `min_frames`; remap parent indices.

    Args:
        data (dict): mask arrays keyed by name, each (T, n_cells).
        emissions (np.ndarray): (T, n_cells, D).
        extras (list): additional (T, n_cells, ...) arrays to filter alongside.
        min_frames (int): minimum active frames to retain a cell.

    Returns:
        tuple: (filtered_data, filtered_emissions, filtered_extras, keep_indices).
    """
    durations = np.sum(data["active_mask"], axis=0)
    keep = np.where(durations >= min_frames)[0]
    if len(keep) == 0:
        raise RuntimeError(f"No cells with duration >= {min_frames} frames.")

    filtered = {k: (v[:, keep] if isinstance(v, np.ndarray) and v.ndim >= 2 else v)
                for k, v in data.items()}

    lookup = np.full(int(data["parent_indices"].max()) + 1, -1, dtype=np.int32)
    for new, old in enumerate(keep):
        lookup[old] = new
    new_parents = np.copy(filtered["parent_indices"])
    rows, cols = np.where(filtered["active_mask"])
    new_parents[rows, cols] = lookup[filtered["parent_indices"][rows, cols]]

    orphans = (new_parents == -1) & filtered["active_mask"]
    if np.any(orphans):
        filtered["is_new_root_mask"][orphans] = True
        o_rows, o_cols = np.where(orphans)
        new_parents[o_rows, o_cols] = o_cols
    filtered["parent_indices"] = new_parents

    print(f"  min_t={min_frames}: {data['active_mask'].shape[1]} -> {len(keep)} cells")
    return filtered, emissions[:, keep, :], [e[:, keep] for e in extras], keep


# ============================================================
print("=" * 60)
print("Step 4a: Building masks and loading features")
print("=" * 60)

all_masks, crop_num_cells, crop_labels, cell_id_rows = [], [], [], []
for crop_idx, crop in enumerate(crop_ids):
    cancer_tracks = np.asarray(tifffile.imread(
        os.path.join(type_sep_tracks_dir, crop, "tracks.tiff")))[..., 1]
    T = cancer_tracks.shape[0]

    cancer_cell_ids = np.load(os.path.join(out_base_dir, crop, "cancer_cell_ids.npy"))
    derived = np.sort(np.unique(cancer_tracks[cancer_tracks > 0]))
    assert np.array_equal(cancer_cell_ids, derived), (
        f"cancer_cell_ids.npy disagrees with the phase tracks for {crop}")
    n_cells = len(cancer_cell_ids)
    id_to_col = {int(cid): i for i, cid in enumerate(cancer_cell_ids)}

    active = np.zeros((T, n_cells), dtype=bool)
    for t in range(T):
        for cid in np.unique(cancer_tracks[t][cancer_tracks[t] > 0]):
            active[t, id_to_col[int(cid)]] = True

    # Tree off: every cell is its own parent, first active frame is a root.
    parents = np.zeros((T, n_cells), dtype=np.int32)
    roots = np.zeros((T, n_cells), dtype=bool)
    for col in range(n_cells):
        frames = np.where(active[:, col])[0]
        if len(frames) == 0:
            continue
        parents[frames, col] = col
        roots[frames[0], col] = True

    all_masks.append(dict(
        active_mask=active, parent_indices=parents, is_new_root_mask=roots,
        is_division_mask=np.zeros((T, n_cells), dtype=bool)))
    crop_num_cells.append(n_cells)
    crop_labels.extend([crop_idx] * n_cells)
    for cid in cancer_cell_ids:
        cell_id_rows.append((crop, int(cid)))
    print(f"  {crop}: T={T}, cells={n_cells}, active cell-frames={int(active.sum())}")

T = all_masks[0]["active_mask"].shape[0]
total_cells = sum(crop_num_cells)
crop_labels = np.array(crop_labels)

combined = {k: np.zeros((T, total_cells), dtype=v.dtype)
            for k, v in all_masks[0].items()}
offset = 0
for masks, n_cells in zip(all_masks, crop_num_cells):
    sl = slice(offset, offset + n_cells)
    for key in combined:
        combined[key][:, sl] = (masks[key] + offset if key == "parent_indices"
                                else masks[key])
    offset += n_cells

feature_names = cfg["emission_feature_names"]
diagnostics = np.concatenate(
    [np.load(os.path.join(out_base_dir, c, "cancer_emissions_array.npy"))
     for c in crop_ids], axis=1).astype(float)
print(f"Diagnostics: {diagnostics.shape} ({total_cells} cells)")

needs_dino = any(a.get("use_dino") for a in arms)
dino = None
if needs_dino:
    dino = np.concatenate(
        [np.load(os.path.join(out_base_dir, c, "cancer_dino_pca.npy"))
         for c in crop_ids], axis=1).astype(float)
    print(f"DINO PCs: {dino.shape}")


# ============================================================
print("\n" + "=" * 60)
print("Step 4b: Shared cell filter and AR warmup")
print("=" * 60)

extras = [dino] if dino is not None else []
combined, diagnostics, extras, kept = filter_tracks_by_time(
    combined, diagnostics, extras, min_t)
if dino is not None:
    dino = extras[0]
crop_labels = crop_labels[kept]
cell_id_rows = [cell_id_rows[i] for i in kept]

# One warmup frame for every arm, lag-0 included, so the lag-1 arms are not
# scored on a different set of cell-frames than the lag-0 arm.  It also drops
# the first active frame, where deltas and window statistics are undefined.
warmup = 1
active, roots = combined["active_mask"], combined["is_new_root_mask"]
for col in range(active.shape[1]):
    frames = np.where(active[:, col])[0]
    if len(frames) <= warmup:
        continue
    for i in range(warmup):
        active[frames[i], col] = False
        roots[frames[i], col] = False
    roots[frames[warmup], col] = True
n_inferred = int(active.sum())
print(f"  warmup={warmup} frame; inferred cell-frames={n_inferred}")

cell_index = pd.DataFrame(cell_id_rows, columns=["crop", "cvat_cell_id"])
cell_index["column"] = np.arange(len(cell_index))
cell_index["crop_idx"] = crop_labels
cell_index.to_csv(os.path.join(fits_dir, "cell_index.csv"), index=False)

diag_idx = {name: i for i, name in enumerate(feature_names)}
np.save(os.path.join(fits_dir, "diagnostics.npy"), diagnostics)
np.save(os.path.join(fits_dir, "active_mask.npy"), active)


# ============================================================
import jax.numpy as jnp      # noqa: E402
import jax.random as jr      # noqa: E402
from models.tarhmm import tARHMM, tree_hmm_two_filter_smoother  # noqa: E402

TRANSITION_FEATURES = [("d_circularity", +1.0), ("d_area_frac", -1.0)]
INSTABILITY_FEATURES = ["win_std_log_area", "win_std_circularity",
                        "win_std_displacement"]


def zscore_across_states(values):
    """Standardize a length-k vector; all-equal input returns zeros."""
    values = np.asarray(values, dtype=float)
    sd = values.std()
    return np.zeros_like(values) if sd < 1e-12 else (values - values.mean()) / sd


def state_profile(assignments, active_mask, diags, num_states):
    """Mean of every diagnostic feature within each state, plus occupancy.

    Args:
        assignments (np.ndarray): (T, n_cells) integer state per cell-frame.
        active_mask (np.ndarray): (T, n_cells) bool.
        diags (np.ndarray): (T, n_cells, n_features) diagnostic features.
        num_states (int): k.

    Returns:
        pandas.DataFrame: one row per state, indexed by state.
    """
    rows = []
    for s in range(num_states):
        sel = active_mask & (assignments == s)
        n = int(sel.sum())
        row = {"state": s, "n_cell_frames": n,
               "occupancy": n / max(int(active_mask.sum()), 1)}
        for name, i in diag_idx.items():
            row[name] = float(diags[:, :, i][sel].mean()) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


def score_states(profile):
    """Rank states by the annotation-independent death score.

    Args:
        profile (pandas.DataFrame): output of `state_profile`.

    Returns:
        pandas.DataFrame: `profile` with transition/instability/death scores.
    """
    profile = profile.copy()
    transition = np.zeros(len(profile))
    for name, sign in TRANSITION_FEATURES:
        transition += sign * zscore_across_states(profile[name].fillna(0.0).to_numpy())
    instability = np.mean(
        [zscore_across_states(profile[n].fillna(0.0).to_numpy())
         for n in INSTABILITY_FEATURES], axis=0)
    profile["transition_score"] = transition
    profile["instability_score"] = instability
    profile["death_score"] = transition + instability
    return profile


def fit_once(emissions_np, num_states, num_lags, seed):
    """Fit one HMM and return its posterior and final log probability.

    Args:
        emissions_np (np.ndarray): (T, n_cells, D) standardized features.
        num_states (int): k.
        num_lags (int): 0 or 1.
        seed (int): PRNG seed for initialization.

    Returns:
        tuple: (fitted_params, smoothed_probs (T, n_cells, K), log_probs).
    """
    model = tARHMM(num_states, emissions_np.shape[-1], num_lags=num_lags)
    sticky = (jnp.eye(num_states) * init_stickiness
              + (1.0 - jnp.eye(num_states)) * (1.0 - init_stickiness)
              / max(num_states - 1, 1))

    init_kwargs = dict(key=jr.PRNGKey(seed), method=init_method,
                       transition_matrix=sticky, division_transition_matrix=sticky)
    if init_method == "kmeans":
        # Only ACTIVE cell-frames, passed as 2D.  The 3D path in
        # TreeARHMMEmissions.initialize discards padding by vector norm,
        # which stops identifying padding once the features are z-scored.
        init_kwargs["emissions"] = jnp.array(emissions_np[active])
    params, props = model.initialize(**init_kwargs)

    emissions_j = jnp.array(emissions_np)
    parent_j = jnp.array(combined["parent_indices"])
    div_j = jnp.array(combined["is_division_mask"])
    active_j = jnp.array(active)
    root_j = jnp.array(roots)

    inputs = model.compute_inputs(emissions_j, parent_j, div_j, root_j, active_j)
    fitted, lps = model.fit_em(
        params, props, emissions_j[None, ...], inputs=inputs[None, ...],
        parent_indices=parent_j[None, ...], is_division_mask=div_j[None, ...],
        active_mask=active_j[None, ...], is_new_root_mask=root_j[None, ...],
        num_iters=num_em_iters, verbose=False)

    posterior = tree_hmm_two_filter_smoother(*model._inference_args(
        fitted, emissions_j, inputs, parent_j, div_j, active_j, root_j))
    return fitted, np.array(posterior.smoothed_probs), np.array(lps)


# ============================================================
print("\n" + "=" * 60)
print("Step 4c: Fitting arms")
print("=" * 60)

summary_rows = []

for arm in arms:
    name = arm["name"]
    num_lags = arm["num_lags"]
    feats = list(arm["features"])

    cols = [diag_idx[f] for f in feats]
    parts = [diagnostics[:, :, cols]] if cols else []
    final_names = list(feats)
    if arm.get("use_dino"):
        if dino is None:
            raise RuntimeError(f"arm '{name}' needs DINO PCs; run steps 2-3 first.")
        parts.append(dino)
        final_names += [f"dino_pc_{i}" for i in range(dino.shape[-1])]
    if not parts:
        raise ValueError(f"arm '{name}' has no features")
    emissions = np.concatenate(parts, axis=-1).astype(float)

    if standardize:
        flat = emissions[active]
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        emissions = (emissions - mean) / std

    print(f"\n--- arm '{name}': D={emissions.shape[-1]}, lags={num_lags}, "
          f"features={final_names}")

    for k in k_sweep:
        best = None
        for seed in em_seeds:
            fitted, smoothed, lps = fit_once(emissions, k, num_lags, seed)
            ll = float(lps[-1])
            if not np.isfinite(ll):
                print(f"    k={k} seed={seed}: non-finite log prob, skipped")
                continue
            if best is None or ll > best["ll"]:
                best = dict(ll=ll, ll0=float(lps[0]), seed=seed,
                            fitted=fitted, smoothed=smoothed, lps=lps)
        if best is None:
            print(f"    k={k}: all seeds failed")
            continue

        assignments = np.argmax(best["smoothed"], axis=-1)
        profile = score_states(state_profile(assignments, active, diagnostics, k))
        death_state = int(profile["death_score"].idxmax())

        fit_dir = os.path.join(fits_dir, name, f"k{k}")
        os.makedirs(fit_dir, exist_ok=True)
        np.save(os.path.join(fit_dir, "state_assignments.npy"), assignments)
        np.save(os.path.join(fit_dir, "smoothed_probs.npy"), best["smoothed"])
        profile.to_csv(os.path.join(fit_dir, "state_profile.csv"))

        trans = np.array(best["fitted"].transitions.transition_matrix)
        with open(os.path.join(fit_dir, "fit_summary.yml"), "w") as fh:
            yaml.safe_dump(dict(
                arm=name, num_states=k, num_lags=num_lags,
                features=final_names, emission_dim=int(emissions.shape[-1]),
                best_seed=best["seed"], log_prob_first=best["ll0"],
                log_prob_final=best["ll"],
                death_state=death_state,
                death_state_occupancy=float(profile.loc[death_state, "occupancy"]),
                death_state_self_transition=float(trans[death_state, death_state]),
                transition_matrix=trans.tolist()), fh, sort_keys=False)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                                 gridspec_kw={"width_ratios": [3, 1]})
        shown = np.where(active.T, assignments.T, np.nan)
        im = axes[0].imshow(shown, aspect="auto", interpolation="none",
                            cmap="viridis", origin="upper")
        axes[0].set_xlabel("Time (frame)")
        axes[0].set_ylabel("Cell")
        axes[0].set_title(f"{name} k={k}  (death candidate = state {death_state})")
        plt.colorbar(im, ax=axes[0], ticks=range(k), label="State")
        im2 = axes[1].imshow(trans, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_xticks(range(k))
        axes[1].set_yticks(range(k))
        axes[1].set_xlabel("State at $t+1$")
        axes[1].set_ylabel("State at $t$")
        axes[1].set_title("Learned transitions")
        for i in range(k):
            for j in range(k):
                axes[1].text(j, i, f"{trans[i, j]:.2f}", ha="center", va="center",
                             fontsize=6,
                             color="white" if trans[i, j] > 0.5 else "black")
        plt.colorbar(im2, ax=axes[1])
        plt.tight_layout()
        plt.savefig(os.path.join(fit_dir, "states.png"), dpi=200, bbox_inches="tight")
        plt.close()

        row = dict(arm=name, k=k, num_lags=num_lags,
                   emission_dim=int(emissions.shape[-1]),
                   best_seed=best["seed"], log_prob=best["ll"],
                   death_state=death_state,
                   death_occupancy=float(profile.loc[death_state, "occupancy"]),
                   death_self_transition=float(trans[death_state, death_state]),
                   death_score=float(profile.loc[death_state, "death_score"]))
        for feat in ["d_area_frac", "d_circularity", "area", "circularity",
                     "dilated_t_cell_neighbors", "win_std_circularity"]:
            row[f"death_mean_{feat}"] = float(profile.loc[death_state, feat])
        summary_rows.append(row)
        print(f"    k={k}: ll={best['ll']:.1f} (seed {best['seed']}), "
              f"death state {death_state}, occupancy "
              f"{profile.loc[death_state, 'occupancy']:.3f}, "
              f"self-trans {trans[death_state, death_state]:.3f}")

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(fits_dir, "summary.csv")
if arm_filter and os.path.exists(summary_path):
    # Preserve rows for arms not re-fitted in this invocation.
    prior = pd.read_csv(summary_path)
    prior = prior[~prior["arm"].isin(summary["arm"].unique())]
    summary = pd.concat([prior, summary], ignore_index=True)
summary = summary.sort_values(["arm", "k"]).reset_index(drop=True)
summary.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print(f"Fitted {len(summary_rows)} (arm, k) combinations -> {summary_path}")
print("=" * 60)
