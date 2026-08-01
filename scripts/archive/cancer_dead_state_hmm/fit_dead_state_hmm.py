"""Step 2: Fit the dead-state HMM for every arm x k x absorbing-variant x seed.

Every arm is fitted over the same cells with the same min_t filter and the
same one-frame warmup, and each (arm, k) is fitted BOTH with and without the
absorbing constraint, so the constraint's effect is measured rather than
assumed.

The absorbing state is designated by index (the last state, k-1); EM decides
what phenotype lands there.  Whether that phenotype is death is scored
separately, by the same label-blind rule applied to every state:

    transition_score  = z(mean d_circularity) - z(mean d_area_frac)
    persistence_score = mean of z(mean 1 - area_over_running_max),
                                z(mean running_max_abs_d_circularity)
    dead_score        = transition_score + persistence_score

The persistence term replaces `cancer_death_hmm`'s instability term: for a
DEAD state the diagnostic is not "this frame is changing" but "this cell has
already shrunk and rounded and not recovered", which is exactly what the two
memory features encode.  Scores use the full 18-feature diagnostic set
regardless of which features fed the model.

Outputs per fit, under {output_base_dir}/fits/{arm}/{variant}/k{K}/:
    state_assignments.npy, smoothed_probs.npy, state_profile.csv,
    fit_summary.yml, states.png, README.md
Plus {output_base_dir}/fits/summary.csv over all fits.

Usage (treeHMM_env):
    conda run -n treeHMM_env python fit_dead_state_hmm.py [arm ...]
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
absorbing_variants = cfg["absorbing_variants"]
feature_names = cfg["emission_feature_names"]

arm_filter = set(sys.argv[1:])
if arm_filter:
    arms = [a for a in arms if a["name"] in arm_filter]
    if not arms:
        sys.exit(f"No arms matched {sorted(arm_filter)}")

sys.path.insert(0, cfg["treehmm_dir"])
sys.path.insert(0, str(_script_dir))
fits_dir = os.path.join(out_base_dir, "fits")
os.makedirs(fits_dir, exist_ok=True)

plt.rc("font", size=7)
plt.rc("axes", titlesize=8, labelsize=7)
plt.rc("xtick", labelsize=7)
plt.rc("ytick", labelsize=7)

TRANSITION_FEATURES = [("d_circularity", +1.0), ("d_area_frac", -1.0)]
PERSISTENCE_FEATURES = [("area_over_running_max", -1.0),
                        ("running_max_abs_d_circularity", +1.0)]
diag_idx = {name: i for i, name in enumerate(feature_names)}


def filter_tracks_by_time(data, emissions, min_frames):
    """Keep cells active for at least `min_frames`; remap parent indices."""
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
    return filtered, emissions[:, keep, :], keep


print("=" * 60)
print("Step 2a: Building masks and loading features")
print("=" * 60)

all_masks, crop_num_cells, crop_labels, cell_id_rows = [], [], [], []
for crop_idx, crop in enumerate(crop_ids):
    cancer_tracks = np.asarray(tifffile.imread(
        os.path.join(type_sep_tracks_dir, crop, "tracks.tiff")))[..., 1]
    T = cancer_tracks.shape[0]
    cancer_cell_ids = np.load(os.path.join(out_base_dir, crop, "cancer_cell_ids.npy"))
    assert np.array_equal(cancer_cell_ids,
                          np.sort(np.unique(cancer_tracks[cancer_tracks > 0]))), (
        f"cancer_cell_ids.npy disagrees with the phase tracks for {crop}")
    n_cells = len(cancer_cell_ids)
    id_to_col = {int(cid): i for i, cid in enumerate(cancer_cell_ids)}

    active = np.zeros((T, n_cells), dtype=bool)
    for t in range(T):
        for cid in np.unique(cancer_tracks[t][cancer_tracks[t] > 0]):
            active[t, id_to_col[int(cid)]] = True

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
    cell_id_rows.extend((crop, int(cid)) for cid in cancer_cell_ids)
    print(f"  {crop}: cells={n_cells}, active cell-frames={int(active.sum())}")

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

diagnostics = np.concatenate(
    [np.load(os.path.join(out_base_dir, c, "cancer_emissions_array.npy"))
     for c in crop_ids], axis=1).astype(float)
print(f"Diagnostics: {diagnostics.shape}")

print("\n" + "=" * 60)
print("Step 2b: Shared cell filter and warmup")
print("=" * 60)
combined, diagnostics, kept = filter_tracks_by_time(combined, diagnostics, min_t)
crop_labels = crop_labels[kept]
cell_id_rows = [cell_id_rows[i] for i in kept]

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
np.save(os.path.join(fits_dir, "diagnostics.npy"), diagnostics)
np.save(os.path.join(fits_dir, "active_mask.npy"), active)

import jax.numpy as jnp                                    # noqa: E402
import jax.random as jr                                    # noqa: E402
from models.tarhmm import tree_hmm_two_filter_smoother     # noqa: E402
from absorbing import AbsorbingTARHMM, pin_absorbing_row   # noqa: E402


def zscore_across_states(values):
    """Standardize a length-k vector; all-equal input returns zeros."""
    values = np.asarray(values, dtype=float)
    sd = values.std()
    return np.zeros_like(values) if sd < 1e-12 else (values - values.mean()) / sd


def state_profile(assignments, active_mask, diags, num_states):
    """Per-state occupancy and mean of every diagnostic feature."""
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
    """Rank states by the annotation-independent dead score."""
    profile = profile.copy()
    transition = np.zeros(len(profile))
    for name, sign in TRANSITION_FEATURES:
        transition += sign * zscore_across_states(profile[name].fillna(0.0).to_numpy())
    persistence = np.mean(
        [sign * zscore_across_states(profile[name].fillna(0.0).to_numpy())
         for name, sign in PERSISTENCE_FEATURES], axis=0)
    profile["transition_score"] = transition
    profile["persistence_score"] = persistence
    profile["dead_score"] = transition + persistence
    return profile


def fit_once(emissions_np, num_states, num_lags, seed, absorbing_state):
    """Fit one HMM; returns (params, smoothed_probs, log_probs)."""
    model = AbsorbingTARHMM(num_states, emissions_np.shape[-1],
                            num_lags=num_lags, absorbing_state=absorbing_state)
    sticky = (jnp.eye(num_states) * init_stickiness
              + (1.0 - jnp.eye(num_states)) * (1.0 - init_stickiness)
              / max(num_states - 1, 1))
    sticky = pin_absorbing_row(sticky, absorbing_state)

    init_kwargs = dict(key=jr.PRNGKey(seed), method=init_method,
                       transition_matrix=sticky, division_transition_matrix=sticky)
    if init_method == "kmeans":
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


print("\n" + "=" * 60)
print("Step 2c: Fitting")
print("=" * 60)

summary_rows = []
for arm in arms:
    name = arm["name"]
    num_lags = arm["num_lags"]
    feats = list(arm["features"])
    emissions = diagnostics[:, :, [diag_idx[f] for f in feats]].astype(float)

    if standardize:
        flat = emissions[active]
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        emissions = (emissions - mean) / std

    print(f"\n--- arm '{name}': D={emissions.shape[-1]}, lags={num_lags}")

    for use_absorbing in absorbing_variants:
        variant = "absorbing" if use_absorbing else "free"
        for k in k_sweep:
            absorbing_state = (k - 1) if use_absorbing else None
            best = None
            for seed in em_seeds:
                fitted, smoothed, lps = fit_once(
                    emissions, k, num_lags, seed, absorbing_state)
                ll = float(lps[-1])
                if not np.isfinite(ll):
                    continue
                if best is None or ll > best["ll"]:
                    best = dict(ll=ll, ll0=float(lps[0]), seed=seed,
                                fitted=fitted, smoothed=smoothed)
            if best is None:
                print(f"    {variant} k={k}: all seeds failed")
                continue

            assignments = np.argmax(best["smoothed"], axis=-1)
            profile = score_states(state_profile(assignments, active, diagnostics, k))
            dead_state = int(profile["dead_score"].idxmax())
            trans = np.array(best["fitted"].transitions.transition_matrix)
            initial = np.array(best["fitted"].initial.probs)

            fit_dir = os.path.join(fits_dir, name, variant, f"k{k}")
            os.makedirs(fit_dir, exist_ok=True)
            np.save(os.path.join(fit_dir, "state_assignments.npy"), assignments)
            np.save(os.path.join(fit_dir, "smoothed_probs.npy"), best["smoothed"])
            profile.to_csv(os.path.join(fit_dir, "state_profile.csv"))

            info = dict(
                arm=name, variant=variant, num_states=k, num_lags=num_lags,
                features=feats, emission_dim=int(emissions.shape[-1]),
                absorbing_state=absorbing_state,
                best_seed=best["seed"], log_prob_first=best["ll0"],
                log_prob_final=best["ll"],
                dead_state=dead_state,
                dead_state_is_absorbing=bool(absorbing_state is not None
                                             and dead_state == absorbing_state),
                dead_state_occupancy=float(profile.loc[dead_state, "occupancy"]),
                dead_state_self_transition=float(trans[dead_state, dead_state]),
                transition_matrix=trans.tolist(),
                initial_probs=initial.tolist())
            with open(os.path.join(fit_dir, "fit_summary.yml"), "w") as fh:
                yaml.safe_dump(info, fh, sort_keys=False)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                                     gridspec_kw={"width_ratios": [3, 1]})
            shown = np.where(active.T, assignments.T, np.nan)
            im = axes[0].imshow(shown, aspect="auto", interpolation="none",
                                cmap="viridis", origin="upper")
            axes[0].set_xlabel("Time (frame)")
            axes[0].set_ylabel("Cell")
            axes[0].set_title(f"{name} [{variant}] k={k}  "
                              f"(dead candidate = state {dead_state})")
            plt.colorbar(im, ax=axes[0], ticks=range(k), label="State")
            im2 = axes[1].imshow(trans, cmap="Blues", vmin=0, vmax=1)
            axes[1].set_xticks(range(k))
            axes[1].set_yticks(range(k))
            axes[1].set_xlabel("State at $t+1$")
            axes[1].set_ylabel("State at $t$")
            axes[1].set_title("Learned transitions")
            for i in range(k):
                for j in range(k):
                    axes[1].text(j, i, f"{trans[i, j]:.2f}", ha="center",
                                 va="center", fontsize=6,
                                 color="white" if trans[i, j] > 0.5 else "black")
            plt.colorbar(im2, ax=axes[1])
            plt.tight_layout()
            plt.savefig(os.path.join(fit_dir, "states.png"), dpi=200,
                        bbox_inches="tight")
            plt.close()

            summary_rows.append(dict(
                arm=name, variant=variant, k=k, num_lags=num_lags,
                emission_dim=int(emissions.shape[-1]),
                absorbing_state=absorbing_state, best_seed=best["seed"],
                log_prob=best["ll"], dead_state=dead_state,
                dead_state_is_absorbing=info["dead_state_is_absorbing"],
                dead_occupancy=float(profile.loc[dead_state, "occupancy"]),
                dead_self_transition=float(trans[dead_state, dead_state]),
                dead_score=float(profile.loc[dead_state, "dead_score"])))
            print(f"    {variant:9s} k={k}: ll={best['ll']:.1f} "
                  f"(seed {best['seed']}), dead state {dead_state}"
                  f"{' [=absorbing]' if info['dead_state_is_absorbing'] else ''}, "
                  f"occ {profile.loc[dead_state, 'occupancy']:.3f}")

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(fits_dir, "summary.csv")
if arm_filter and os.path.exists(summary_path):
    prior = pd.read_csv(summary_path)
    prior = prior[~prior["arm"].isin(summary["arm"].unique())]
    summary = pd.concat([prior, summary], ignore_index=True)
summary = summary.sort_values(["arm", "variant", "k"]).reset_index(drop=True)
summary.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print(f"Fitted {len(summary_rows)} combinations -> {summary_path}")
print("=" * 60)
