# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An implementation of a **Tree Autoregressive Hidden Markov Model (Tree AR-HMM)** built on top of [Dynamax](https://github.com/probml/dynamax) (JAX). It is motivated by modeling cell lineages: each cell has a latent discrete state that evolves over time with shared AR(1) Gaussian emissions, but when a cell *divides* its daughters' initial states are drawn from a separate **division transition kernel** conditioned on the parent's state at division. The model also handles spontaneous cell birth ("new roots", e.g. coming into frame) and death.

See `README.md` for the full probabilistic model and `Derivation/` for the forward–backward math.

## Layout

- `models/tarhmm.py` — the entire model. Custom forward–backward inference plus a `tARHMM` class subclassing Dynamax's `LinearAutoregressiveHMM`.
- `utils.py` — `generate_tree_hmm_data()` (synthetic lineage data with strict unique cell-IDs) and `visualize_lineage()`.
- `scripts/cvat_gt_crops/` — the real-data pipeline (T cell microscopy crops); see below.
- `notebooks/` — exploratory runs (arHMM, tarHMM, emission processing). `tree_arhmm.ipynb` at root is the main scratch notebook.
- `analysis/` — committed pipeline outputs, named `cvat_gt_crops_{hmm,arhmm}_k{num_states}_m{model_features}/`.

There is no `__init__.py` and no `pip install` of this repo — code imports it by putting the repo root on `sys.path` and doing `from models.tarhmm import tARHMM`. `pyproject.toml` is inherited from Dynamax and describes the *dynamax* dependency, not this package.

## Conda environments

The pipeline spans two environments and you must use the right one:

- **`treeHMM_env`** — JAX / Dynamax / the model. Use for anything touching `models/tarhmm.py` (`fit_arhmm.py`, the tarHMM notebooks). Build it with `bash scripts/setup_treeHMM_env.sh` (Python 3.11, dynamax + GPU JAX with CUDA 12 wheels). **Gotcha:** `pip install dynamax` pulls *stable* `tensorflow-probability`, which is too old for current JAX (import fails on `jax.interpreters.xla.pytype_aval_mappings`) — the setup script replaces it with `tfp-nightly`, which is what `pyproject.toml` specifies. For CPU-only, drop the `[cuda12]` extra.
- **`occident`** — microscopy I/O, feature extraction, video rendering (`calculate_emissions.py`, `create_overlay_videos.py`). Imports from the external `MarsonImagingPipeline` repo. (The pipeline README calls this `AnalysisEnv`; `run_pipeline.sh` uses `occident` — trust the script.)

## Running the real-data pipeline

```bash
cd scripts/cvat_gt_crops
bash run_pipeline.sh            # all three steps, each in its correct conda env
```

Three ordered steps (run individually with `conda run --no-capture-output -n <env> python <script>`):
1. `calculate_emissions.py` (`occident`) — per-cell features → `.npy` emission arrays.
2. `fit_arhmm.py` (`treeHMM_env`) — fits one joint AR-HMM across all crops, writes state assignments + summary plots.
3. `create_overlay_videos.py` (`occident`) — renders state-colored overlay `.mp4`s.

All shared parameters live in `config.yml` — `num_states`, `num_lags`, `min_t`, `model_features` (subset of `emission_feature_names` actually fed to the model), crop IDs, and absolute input/output paths. Change behavior there, not in the scripts.

There are no automated tests (writing them is an open TODO in the README).

## Model architecture — key concepts to know before editing `models/tarhmm.py`

**Data is a dense `(T, MAX_CELLS, D)` tensor with one fixed column per unique cell.** Columns are never reused: a cell that dies leaves its column inactive forever; a division ends the parent's column and allocates two new columns. Everything is driven by per-`(t, cell)` boolean/index masks that travel together through every function:
- `active_mask` — cell exists/observed at `(t, cell)`.
- `parent_indices` — for each `(t, cell)`, the column index of its parent at `t-1` (self if persisting, the dividing parent if a division child, dummy `0` for new roots).
- `is_division_mask` — `(t, cell)` is a division child (use `P_div`, and AR input is zeroed).
- `is_new_root_mask` — first frame a cell appears spontaneously (reset to the initial distribution; argmax over time gives each root's first frame for the initial-state stats).

**Two transition matrices.** `P_std` (persistence) and `P_div` (division), carried as the tuple `(P_std, P_div)`. Inference picks per-edge via `is_division_mask`. They have parallel parameter/M-step components: `TreeTransitions` + `HMMDivisionTransitions`, with `division_transitions` as a first-class field in `ParamsLinearTreeARHMM` alongside `initial`, `transitions`, `emissions`.

**Custom forward–backward** (not Dynamax's) lives in module-level jitted functions: `tree_hmm_filter`, `tree_hmm_backward_filter`, `tree_hmm_two_filter_smoother`, and `_compute_sum_transition_probs`. The filter/smoother are vmapped over cells per timestep; messages flow along `parent_indices` (children scatter messages up to parents in the backward pass). New roots are re-seeded with the initial distribution; inactive cells are masked to zero. Returns a `TreeHMMPosterior` (adds `division_trans_probs` to the standard fields).

**Emissions support `num_lags=0`** (degenerates to a Gaussian HMM, bias-only — the current default in `config.yml`) and `num_lags=1`. `TreeARHMMEmissions._compute_conditional_logliks` flattens `(T, C, D) → (T·C, D)` to reuse Dynamax's vectorized likelihood. `num_lags > 1` is explicitly **not implemented** (`compute_inputs` raises). The emission M-step adds ridge/jitter regularization and guards dead states (`sum_w` near zero) to avoid NaNs.

**AR inputs are precomputed**, not derived inside the model: call `arhmm.compute_inputs(emissions, parent_indices, is_division_mask, is_new_root_mask, active_mask)` to build the lagged-parent-observation tensor, then pass the result into `fit_em` / the smoother. Division children and inactive cells get zeroed inputs; new roots keep their first valid observation as input.

**EM entry point:** `tARHMM.fit_em(params, props, emissions, inputs, parent_indices, is_division_mask, active_mask, is_new_root_mask, num_iters=...)`. It accepts a single `(T, C, D)` sequence or a batch (leading axis = independent files/crops); the E-step is vmapped over the batch and sufficient stats are summed before one M-step. `fit_em` closes over `props` from the enclosing scope. `sample()` is not implemented.

To get state assignments after fitting, run `tree_hmm_two_filter_smoother(*arhmm._inference_args(...))` and take `argmax(posterior.smoothed_probs, axis=-1)`.

### Gotchas

- Inactive/padded cells carry zeros and NaNs by design; stats use `jnp.nansum` / explicit masking. Preserve this when changing reductions.
- `models/tarhmm.py` sets `config.update("jax_disable_jit", False)` at import and has a commented-out debug toggle — leave jit on unless actively debugging.
- The pipeline reindexes cells (`filter_tracks_by_time`, `min_t` filtering) and **remaps `parent_indices` accordingly**, promoting orphaned cells to new roots. Any code that filters columns must keep `parent_indices`, the masks, and emissions consistent, or inference breaks silently.
