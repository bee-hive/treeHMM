"""Step `fit`: fit the AR-HMM jointly across all crops.

The emission vector is `config.emission_names(cfg)` -- `model.features` in
config order, then the DINO principal components when `model.use_dino_pcs` is
set.  That is the order `tree_input.md` section 3 fixes, and it is what
`fit_summary.yml` records for everything downstream to read.

Preprocessing follows `tree_input.md` section 6, in order:

  1. `lineage.prepare_for_fit` -- drop cells shorter than `cells.min_frames`,
     remapping the masks, then deactivate each cell's leading
     `cells.warmup_frames` frames.  Holding the warmup fixed regardless of lag
     order means a lag-0 and a lag-1 run are scored over exactly the same
     cell-frames.
  2. non-finite scrub, reporting how many zeroed entries fell inside active
     cell-frames rather than swallowing them.
  3. optional standardization over active cell-frames only.
  4. cast to `jnp`, then EM once per seed in `model.em_seeds`, keeping the fit
     with the best final log probability, ties broken by the lowest seed.

Two guards that exist because the model's API invites specific mistakes:

  * `MaskBundle` is the only way this module calls `compute_inputs` and
    `fit_em`.  Those two take the masks in **different orders**, and they are
    same-shaped boolean arrays, so a swap runs happily and returns nonsense.
  * k-means initialization is handed a 2-D array of active cell-frames only.
    Passing the padded `(T, C, D)` tensor lets padding zeros dominate the
    centroids, and once features are standardized the model's norm-based
    padding detection stops working.

After fitting, states are permuted into a canonical order (see
`_canonical_permutation`) so that "state 2" means the same thing across reruns,
seeds and neighbouring `num_states`.

Written into `{run}/fit/`:

    state_assignments.npy   (T, C)      argmax of the smoothed posterior
    state_probs.npy         (T, C, K)   smoothed posterior probabilities
    emissions.npy           (T, C, D)   exactly what the model saw
    diagnostics.npy         (T, C, F)   every cached feature, unstandardized
    active_mask.npy         (T, C)      cell-frames states were inferred for
    masks.npz               parent / division / new-root masks
    cell_index.csv          column -> (crop, cell_id)
    transition_matrix.npy, division_transition_matrix.npy,
    initial_distribution.npy, log_probs.npy
    fit_summary.yml         everything needed to interpret the above

Run inside the model environment (JAX / dynamax).
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass

import numpy as np

from arhmm import config as cfgmod
from arhmm.core import io, lineage, providers
from arhmm.steps import step_main


@dataclass(frozen=True)
class MaskBundle:
    """The four masks, kept together so they cannot be passed in the wrong order.

    `tarhmm.compute_inputs` takes `(emissions, parent_indices, is_division_mask,
    is_new_root_mask, active_mask)` while `tarhmm.fit_em` takes
    `(..., parent_indices, is_division_mask, active_mask, is_new_root_mask, ...)`.
    Both are positional, and the three masks are interchangeable in shape and
    dtype, so a transposition is silent.  Every call in this module goes through
    the helpers below.
    """

    active: object
    parents: object
    divisions: object
    roots: object

    @classmethod
    def from_dict(cls, masks: dict, cast=lambda x: x) -> "MaskBundle":
        return cls(
            active=cast(masks["active_mask"]),
            parents=cast(masks["parent_indices"]),
            divisions=cast(masks["is_division_mask"]),
            roots=cast(masks["is_new_root_mask"]),
        )

    def compute_inputs(self, model, emissions):
        """`model.compute_inputs` with the arguments in ITS order."""
        return model.compute_inputs(emissions, self.parents, self.divisions, self.roots, self.active)

    def fit_em(self, model, params, props, emissions, inputs, num_iters, verbose=False):
        """`model.fit_em` with the arguments in ITS (different) order."""
        return model.fit_em(
            params, props, emissions, inputs,
            self.parents, self.divisions, self.active, self.roots,
            num_iters=num_iters, verbose=verbose,
        )

    def inference_args(self, model, params, emissions, inputs):
        """`model._inference_args` with the arguments in ITS order."""
        return model._inference_args(
            params, emissions, inputs, self.parents, self.divisions, self.active, self.roots
        )


def _load_inputs(cfg: dict, layout):
    """Assemble the joint emission tensor, diagnostics, masks and cell index.

    Args:
        cfg (dict): resolved configuration.
        layout (Layout): the run's layout.

    Returns:
        tuple: `(emissions, diagnostics, diagnostic_names, masks, cell_index)`.
            `emissions` is `(T, C, D)` in emission-name order; `diagnostics` is
            `(T, C, F)` holding every cached feature, for state profiling.
    """
    emission_blocks, diagnostic_blocks, mask_blocks, index = [], [], [], []
    diagnostic_names: list[str] | None = None

    for crop_id in cfgmod.crop_ids(cfg):
        crop_dir = layout.crop_dir("features", crop_id)
        arrays = io.load_npz(crop_dir / "features.npz")
        meta = io.read_json(crop_dir / "meta.json")

        emissions, _ = providers.load_emissions(cfg, layout, crop_id)
        diagnostics, names = providers.load_diagnostics(cfg, layout, crop_id)
        if diagnostic_names is None:
            diagnostic_names = names
        elif names != diagnostic_names:
            raise ValueError(
                f"{crop_id}: cached feature order {names} differs from {diagnostic_names}; "
                f"the features cache is inconsistent"
            )

        cell_ids = arrays["cell_ids"]
        if not np.array_equal(cell_ids, np.sort(cell_ids)):
            raise ValueError(f"{crop_id}: cached cell_ids are not sorted")
        if emissions.shape[1] != len(cell_ids):
            raise ValueError(
                f"{crop_id}: emissions have {emissions.shape[1]} columns but "
                f"{len(cell_ids)} cell ids; providers are out of step"
            )

        emission_blocks.append(emissions)
        diagnostic_blocks.append(diagnostics)
        mask_blocks.append({k: arrays[k] for k in lineage.MASK_KEYS})
        index.extend(
            {"crop": crop_id, "cell_id": int(cid), "crop_idx": i}
            for i, cid in enumerate(cell_ids)
        )
        print(f"  {crop_id}: {meta['num_cells']} cells, {meta['num_frames']} frames")

    masks = lineage.concatenate_crops(mask_blocks)
    emissions = np.concatenate(emission_blocks, axis=1)
    diagnostics = np.concatenate(diagnostic_blocks, axis=1)
    return emissions, diagnostics, diagnostic_names or [], masks, index


def _canonical_permutation(states: np.ndarray, probs: np.ndarray, emissions: np.ndarray,
                           active: np.ndarray, num_states: int) -> np.ndarray:
    """Order states deterministically so labels mean the same thing across runs.

    EM state indices are arbitrary: two runs that find the same three states can
    number them differently, which makes overlay videos and per-state tables
    incomparable across seeds, reruns and neighbouring `num_states`.  States are
    sorted by their mean value on the **first** emission dimension over inferred
    cell-frames, ties broken by occupancy, then by original index so the result
    is total.

    Args:
        states (np.ndarray): `(T, C)` argmax assignments.
        probs (np.ndarray): `(T, C, K)` smoothed posteriors.
        emissions (np.ndarray): `(T, C, D)` what the model saw.
        active (np.ndarray): `(T, C)` inferred cell-frames.
        num_states (int): `K`.

    Returns:
        np.ndarray: `(K,)` such that `new_state = argsort(perm)[old_state]`.
    """
    keys = []
    for state in range(num_states):
        selected = active & (states == state)
        occupancy = float(selected.sum())
        mean_first = float(emissions[..., 0][selected].mean()) if occupancy else np.inf
        keys.append((mean_first, -occupancy, state))
    return np.array([state for _, _, state in sorted(keys)], dtype=np.int32)


def _fit_once(cfg, emissions_np, masks_np, seed):
    """Run EM once for one seed, returning the fitted params and log probs."""
    import jax.numpy as jnp
    import jax.random as jr

    from models.tarhmm import tARHMM, tree_hmm_two_filter_smoother

    num_states = cfgmod.get_path(cfg, "model.num_states")
    num_lags = cfgmod.get_path(cfg, "model.num_lags")
    stickiness = float(cfgmod.get_path(cfg, "model.init_stickiness", 0.9))
    init_method = cfgmod.get_path(cfg, "model.init_method", "kmeans")
    emission_dim = emissions_np.shape[-1]

    model = tARHMM(num_states, emission_dim, num_lags=num_lags)

    emissions = jnp.asarray(emissions_np)
    bundle = MaskBundle.from_dict(masks_np, cast=jnp.asarray)

    off_diagonal = (1.0 - stickiness) / max(num_states - 1, 1)
    sticky = jnp.eye(num_states) * stickiness + (1.0 - jnp.eye(num_states)) * off_diagonal

    init_kwargs = dict(
        key=jr.PRNGKey(seed),
        method=init_method,
        transition_matrix=sticky,
        division_transition_matrix=sticky,
    )
    if init_method == "kmeans":
        # 2-D, active cell-frames only: see the module docstring.
        init_kwargs["emissions"] = jnp.asarray(emissions_np[masks_np["active_mask"]])

    params, props = model.initialize(**init_kwargs)
    inputs = bundle.compute_inputs(model, emissions)
    params, log_probs = bundle.fit_em(
        model, params, props, emissions, inputs,
        num_iters=cfgmod.get_path(cfg, "model.num_em_iters"),
    )
    posterior = tree_hmm_two_filter_smoother(*bundle.inference_args(model, params, emissions, inputs))
    return model, params, np.asarray(posterior.smoothed_probs), np.asarray(log_probs), inputs


def _run(cfg: dict, layout, args) -> dict:
    # numpy work first; JAX is imported only inside _fit_once so that an
    # import-time GPU grab does not happen while we are still reading files.
    emission_names = cfgmod.emission_names(cfg)
    print(f"emission vector ({len(emission_names)}): {', '.join(emission_names)}")

    emissions, diagnostics, diagnostic_names, masks, index = _load_inputs(cfg, layout)
    raw_shape = emissions.shape

    masks, arrays, keep = lineage.prepare_for_fit(
        masks,
        {"emissions": emissions, "diagnostics": diagnostics},
        min_frames=cfgmod.get_path(cfg, "cells.min_frames"),
        warmup_frames=cfgmod.get_path(cfg, "cells.warmup_frames"),
    )
    emissions, diagnostics = arrays["emissions"], arrays["diagnostics"]
    index = [index[i] for i in keep]
    active = masks["active_mask"]
    print(
        f"cells {raw_shape[1]} -> {emissions.shape[1]} after cells.min_frames; "
        f"{int(active.sum())} inferred cell-frames after warmup"
    )

    # 3. non-finite scrub, reported rather than swallowed.
    non_finite = ~np.isfinite(emissions)
    zeroed_on_active = int((non_finite & active[..., None]).sum())
    if zeroed_on_active:
        per_column = {
            emission_names[d]: int((non_finite[..., d] & active).sum())
            for d in range(emissions.shape[-1])
            if (non_finite[..., d] & active).any()
        }
        print(f"zeroed {zeroed_on_active} non-finite values inside active cell-frames: {per_column}")
    emissions = np.nan_to_num(emissions, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 4. standardize over active cell-frames only.
    standardization = None
    if cfgmod.get_path(cfg, "model.standardize", True):
        flat = emissions[active]
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        emissions = ((emissions - mean) / std).astype(np.float32)
        emissions[~active] = 0.0
        standardization = {"mean": [float(v) for v in mean], "std": [float(v) for v in std]}
        print(f"standardized over {int(active.sum())} active cell-frames")

    # 5. EM once per seed; keep the best final log probability, lowest seed wins ties.
    seeds = list(cfgmod.get_path(cfg, "model.em_seeds"))
    best = None
    per_seed = []
    for seed in seeds:
        model, params, probs, log_probs, _ = _fit_once(cfg, emissions, masks, seed)
        final = float(log_probs[-1])
        per_seed.append({"seed": int(seed), "log_prob_final": final,
                         "log_prob_first": float(log_probs[0])})
        print(f"  seed {seed}: log prob {float(log_probs[0]):.3f} -> {final:.3f}")
        if best is None or final > best["final"]:
            best = {"seed": int(seed), "final": final, "model": model, "params": params,
                    "probs": probs, "log_probs": np.asarray(log_probs)}
    assert best is not None
    print(f"best seed: {best['seed']} (log prob {best['final']:.3f})")

    num_states = cfgmod.get_path(cfg, "model.num_states")
    probs = best["probs"]
    states = probs.argmax(axis=-1).astype(np.int32)

    # Canonical relabelling, applied to every state-indexed array together.
    perm = _canonical_permutation(states, probs, emissions, active, num_states)
    inverse = np.argsort(perm).astype(np.int32)
    probs = probs[..., perm]
    states = inverse[states]

    params = best["params"]
    transition = np.asarray(params.transitions.transition_matrix)[np.ix_(perm, perm)]
    division = np.asarray(params.division_transitions.transition_matrix)[np.ix_(perm, perm)]
    initial = np.asarray(params.initial.probs)[perm]

    out_dir = io.ensure_dir(layout.fit_dir)
    np.save(out_dir / "state_assignments.npy", states)
    np.save(out_dir / "state_probs.npy", probs.astype(np.float32))
    np.save(out_dir / "emissions.npy", emissions)
    np.save(out_dir / "diagnostics.npy", diagnostics.astype(np.float32))
    np.save(out_dir / "active_mask.npy", active)
    np.save(out_dir / "transition_matrix.npy", transition)
    np.save(out_dir / "division_transition_matrix.npy", division)
    np.save(out_dir / "initial_distribution.npy", initial)
    np.save(out_dir / "log_probs.npy", best["log_probs"])
    io.save_npz(out_dir / "masks.npz", **{k: masks[k] for k in lineage.MASK_KEYS})

    with io.atomic_write(out_dir / "cell_index.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["crop", "cell_id", "crop_idx", "column"])
        writer.writeheader()
        for column, entry in enumerate(index):
            writer.writerow({**entry, "column": column})

    occupancy = [float((active & (states == k)).sum()) / max(int(active.sum()), 1)
                 for k in range(num_states)]
    summary = {
        "run_name": cfgmod.get_path(cfg, "run_name"),
        "num_states": num_states,
        "num_lags": cfgmod.get_path(cfg, "model.num_lags"),
        "emission_dim": int(emissions.shape[-1]),
        "emission_names": emission_names,
        "diagnostic_names": diagnostic_names,
        "cells_fit": int(emissions.shape[1]),
        "frames": int(emissions.shape[0]),
        "inferred_cell_frames": int(active.sum()),
        "allow_divisions": False,
        "division_cell_frames": int(masks["is_division_mask"].sum()),
        "best_seed": best["seed"],
        "seeds": per_seed,
        "log_prob_first": float(best["log_probs"][0]),
        "log_prob_final": float(best["log_probs"][-1]),
        "em_iterations": int(len(best["log_probs"])),
        "state_permutation": [int(v) for v in perm],
        "state_occupancy": occupancy,
        "non_finite_zeroed_on_active": zeroed_on_active,
        "standardization": standardization,
    }
    io.write_yaml(out_dir / "fit_summary.yml", summary)
    return {"best_seed": best["seed"], "log_prob_final": best["final"],
            "inferred_cell_frames": int(active.sum())}


def _ensure_repo_on_path(cfg: dict) -> None:
    """Make `models.tarhmm` importable when the step is run by hand."""
    repo_root = str(cfgmod.get_path(cfg, "paths.repo_root"))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _body(cfg, layout, args):
    _ensure_repo_on_path(cfg)
    return _run(cfg, layout, args)


if __name__ == "__main__":
    sys.exit(step_main("fit", "fit the AR-HMM jointly across all crops", _body))
