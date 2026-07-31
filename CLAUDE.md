# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An implementation of a **Tree Autoregressive Hidden Markov Model (Tree AR-HMM)** built on top of [Dynamax](https://github.com/probml/dynamax) (JAX). It is motivated by modeling cell lineages: each cell has a latent discrete state that evolves over time with shared AR(1) Gaussian emissions, but when a cell *divides* its daughters' initial states are drawn from a separate **division transition kernel** conditioned on the parent's state at division. The model also handles spontaneous cell birth ("new roots", e.g. coming into frame) and death.

See `README.md` for the full probabilistic model and `Derivation/` for the forward–backward math.

## Input Data
- Well IDs: ['B8', 'B4', 'E4']
- Phase Image Cell Tracks: `/gladstone/engelhardt/lab/MarsonLabIncucyteData/groundTruthTracks/TCR-T/<well_id>/<well_id>_<slice_id>/ALL_tracks.tiff`
    - shape (T, Y, X)
    - includes tracks for both cancer and T cells, cancer cells identified by `/gladstone/engelhardt/lab/MarsonLabIncucyteData/groundTruthTracks/TCR-T/<well_id>/<well_id>_<slice_id>/ALL_cancer_ids.pkl`
- Cancer Nuclei Tracks: `/gladstone/engelhardt/lab/MarsonLabIncucyteData/groundTruthCalibanTracks/<well_id>_<slice_id>.tiff`
    - shape (T, Y, X)
    - includes only cancer cell nuclei
- Raw Phase image: 
    - shape (T, Y, X, 2)
        - in last channel, 0 is RFP intensity, 1 is phase image
- Conditions: 
    - SH: ['B3', 'B4', 'B5', 'B6']
    - RASA2: ['E3', 'E4', 'E5', 'E6']
    - CUL5: ['B7', 'B9', 'B10']

## Layout
- `models/tarhmm.py` — the entire model. Custom forward–backward inference plus a `tARHMM` class subclassing Dynamax's `LinearAutoregressiveHMM`.
- `tree_input.md` — **the binding contract** for what the fit step hands the model: the six arrays, their shapes and dtypes, mask semantics, and the preprocessing order. Read this before touching the fit step.
- `treearhmm/` — the pipeline package. **Read `treearhmm/README.md`** before changing it: it covers the step chain, the config layering, determinism, and the three ways to extend the pipeline (a track feature is one decorated function, a modality is one provider plus a step, an extra is one module).
- `configs/` — run configurations. `site.yml` (paths, envs, crops) ← `default.yml` (experiment defaults) ← one file per run.
- `utils.py` — `generate_tree_hmm_data()` (synthetic lineage data) and `visualize_lineage()`. Used only by notebooks and tests.
- `notebooks/` — exploratory runs. `tree_arhmm.ipynb` at root is the main scratch notebook.
- `analysis/` — **gitignored.** All pipeline output: `analysis/runs/<run_name>/` and the content-addressed `analysis/cache/`. Nothing here is committed.
- `scripts/archive/`, `analysis/archive/` — the eight superseded per-experiment pipelines and their outputs. Reference only; do not extend them.

There is no `pip install` of this repo. `pyproject.toml` is inherited from Dynamax and describes *dynamax*, not this package. The CLI puts the repo root on `PYTHONPATH` when it spawns each step, which is what makes both `treearhmm` and `from models.tarhmm import tARHMM` importable in whichever env that step runs in.

**`.gitignore` note:** lines 17–18 are `/lib/` and `/lib64/`, deliberately anchored. Unanchored `lib/` matches at any depth and once silently swallowed an entire package directory. Do not un-anchor them, and do not name a package directory `lib`.

## Conda environments

The pipeline spans three environments and you must use the right one:

- **`treeHMM_env`** — JAX / Dynamax / the model. Anything touching `models/tarhmm.py`. Python 3.11.15, jax 0.10.1 (sees both A30s), dynamax 1.0.1, numpy 2.4.6. **Gotcha:** `pip install dynamax` pulls *stable* `tensorflow-probability`, too old for current JAX (import fails on `jax.interpreters.xla.pytype_aval_mappings`); it must be replaced with `tfp-nightly`.
- **`OccidentAnalysis`** — microscopy I/O, feature extraction, plotting, video. Python 3.11.9, numpy 1.26.4, skimage 0.23.2, tifffile, pandas 2.1.4, matplotlib 3.8.2, imageio + ffmpeg.
- **`cs229Dino`** — DINOv2 embedding and PCA. torch 2.5.1 (CUDA), transformers 5.2.0, sklearn 1.8.0, numpy 2.4.2.

There is **no env named `occident`** — older docs and scripts say so and are wrong. `AnalysisEnv` exists but is Python 3.14 and unrelated.

**Cross-environment hazard:** the three envs are on numpy 1.26.4 / 2.4.2 / 2.4.6 and pass arrays to each other as `.npz`. Plain numeric and bool arrays round-trip; object arrays and pickled payloads do not. Rule: numeric/bool arrays only, `allow_pickle=False` on every load, all strings in JSON sidecars.

## Running the pipeline

```bash
treearhmm run configs/_smoke.yml      # one crop, no DINO, ~1 minute
treearhmm status configs/_smoke.yml   # which steps are current and why
treearhmm doctor                      # envs, paths, CUDA, npz round-trip
```

One YAML defines one run. Steps run in a fixed chain, each in its own env, and each is independently runnable: `python -m treearhmm.steps.<name> --run-dir <dir>`. Shared steps (`features`, `dino`, `pca`) are content-addressed into `analysis/cache/` and reused by any run with the same inputs; `fit`, `outputs` and `extras` are run-local.

Change behaviour in the config, never in the scripts. Sweeps are several small configs sharing a base via `extends:`.

There are no automated tests yet beyond `tests/` (unit tests for config, lineage, features and the npz round-trip); the smoke config is the end-to-end regression check.

## Model architecture — key concepts to know before editing `models/tarhmm.py`

> **The pipeline does not use the tree.** Divisions are out of scope: `treearhmm`
> builds `is_division_mask` all-False and `parent_indices` always self, so every cell
> is an independent chain, `P_div` is never exercised, and `ALL_graph.pkl` is not read.
> The section below describes the model's full capability, which the pipeline uses only
> the non-division half of. Do not add division handling to the pipeline without
> changing `tree_input.md` first.

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
- The pipeline reindexes cells (`lineage.filter_short_cells`, `cells.min_frames`) and **remaps `parent_indices` accordingly**. Any code that filters columns must keep `parent_indices`, the masks, and emissions consistent, or inference breaks silently rather than loudly. `lineage.assert_consistent()` is called at the end of every column-space operation for exactly this reason — keep it that way.
- `compute_inputs` and `fit_em` take the masks in **different orders** (`is_new_root_mask` before vs after `active_mask`). They are same-shaped booleans, so a swap runs happily and returns nonsense. Never call either positionally: go through the `MaskBundle` wrappers in `treearhmm/steps/fit.py`.
- `initialize(method="kmeans", emissions=...)` wants a **2D array of active cell-frames only**, not the padded `(T, C, D)` tensor. Passing the 3D tensor lets padding zeros dominate the centroids, and once features are z-scored the model's norm-based padding detection stops working.
