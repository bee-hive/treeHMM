# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An implementation of a **Tree Autoregressive Hidden Markov Model (Tree AR-HMM)** built on top of [Dynamax](https://github.com/probml/dynamax) (JAX). It is motivated by modeling cell lineages: each cell has a latent discrete state that evolves over time with shared AR(1) Gaussian emissions, but when a cell *divides* its daughters' initial states are drawn from a separate **division transition kernel** conditioned on the parent's state at division. The model also handles spontaneous cell birth ("new roots", e.g. coming into frame) and death.

See `README.md` for the full probabilistic model and `Derivation/` for the forward–backward math.

## Input Data
- Phase Image Cell Tracks: `/gladstone/engelhardt/lab/MarsonLabIncucyteData/groundTruthTracks/TCR-T/<well_id>/<well_id>_<slice_id>/ALL_tracks.tiff`
    - shape (T, Y, X)
    - includes tracks for both cancer and T cells, cancer cells identified by `/gladstone/engelhardt/lab/MarsonLabIncucyteData/groundTruthTracks/TCR-T/<well_id>/<well_id>_<slice_id>/ALL_cancer_ids.pkl`
- Cancer Nuclei Tracks: `/gladstone/engelhardt/lab/MarsonLabIncucyteData/groundTruthCalibanTracks/<well_id>_<slice_id>.tiff`
    - shape (T, Y, X)
- Raw Phase image: `/gladstone/engelhardt/lab/MarsonLabIncucyteData/TrackingCrops/CarnevaleRepStim/<well_id>/<well_id>_<slice_id>/B4_<slice_id>/crop.tiff`
    - shape (T, Y, X, 2)
        - in last channel, 0 is RFP intensity, 1 is phase image

## The pipeline

**The YAML file defines an experiment run**.

| step | env role | conda env | output |
|---|---|---|---|
| `features` | imaging | `OccidentAnalysis` | per-cell, per-frame track features (**cached**) |
| `dino` | dino | `cs229Dino` | DINOv2 embeddings of centroid patches (**cached**) |
| `pca` | dino | `cs229Dino` | joint PCA → top-k components (**cached**) |
| `fit` | model | `treeHMM_env` | fitted AR-HMM + posteriors (run-local) |
| `outputs` | imaging | `OccidentAnalysis` | the five base outputs (run-local) |
| `extras` | imaging | `OccidentAnalysis` | opt-in extras (run-local, optional) |

`dino`/`pca` are dropped from the chain entirely unless `model.use_dino_pcs: true`; `extras` is
dropped when `outputs.extras` is empty.

Cached steps are content-addressed under `analysis/cache/<step>/<key>/` and shared across
runs, so a sweep over `model.*` recomputes only `fit` onward.

`arhmm/README.md` is the detailed reference (determinism rules, cache keys,
provider abstraction); `env_setup.md` covers building `treeHMM_env`.

## Run Existing Experiment

The package is **not pip-installed**. Run it as a module from the repo root; the CLI itself only
needs PyYAML + numpy, so the conda `base` env is fine — it spawns each step in the
right env itself.

```bash
# inside the repo root

python -m arhmm doctor configs/runs/dino_k5.yml     # envs, paths, CUDA, npz round-trip
python -m arhmm list   configs/_smoke.yml           # run directories under output_root
python -m arhmm show   configs/runs/dino_k5.yml     # fully resolved config
python -m arhmm status configs/runs/dino_k5.yml     # which steps are current, and why
python -m arhmm run --dry-run configs/runs/dino_k5.yml
python -m arhmm run    configs/runs/dino_k5.yml
```

**Every step is skipped when its key and artifacts are still current — run-local ones too.**
So `run` on a finished config is a silent no-op that still exits 0; if you meant to
recompute, you must say so:

| flag | effect |
|---|---|
| `--force` | on its own, recompute **every** step in the chain, cached ones included |
| `--from <step>` | start at `<step>`; combined with `--force`, only `<step>` itself is forced |
| `--force-all` | only meaningful with `--from` — force the later steps too, not just `<step>` |
| `--allow-config-change` | reuse a run directory whose stored config differs (otherwise refused, naming the differing keys) |

`--dry-run` ignores `--force` and reports every step `cached` regardless, so it cannot be
used to preview a forced run.

Results land in `analysis/runs/<run_name>/`:

**Book-keeping outputs:**
```
config.resolved.yml            frozen copy; the sole input to every step
manifest.yml                   commit, host, per-step env / key / timing / status
_stamps/<step>.json            run-local steps only; a cached step stamps its cache dir
logs/<step>.log
features -> <cache_root>/features/<key>     absolute symlink; likewise dino/, pca/
fit/      
  fit_summary.yml, 
  cell_index.csv, 
  and the .npy/.npz arrays
```

**Analytical Outputs**:      
```               
outputs/
  overlays/<crop>_state_overlay.mp4         cancer cells tinted by state
  feature_distributions.png                 every cached feature, per state
  state_feature_summary.csv
  state_age_histogram.png / .csv            state occupancy binned by cell age
  transition_matrix.csv / .png
  initial_distribution.csv
  state_assignments.csv                     one row per inferred cell-frame
  state_assignments_per_cell.csv
  extras/<name>/
```

To debug one step by hand, in its own environment — it reads `config.resolved.yml` and
nothing else, so this is what the driver does. It is stamp-guarded like the driver too,
and takes its own `--force`; without it a current step prints `nothing to do`:

```bash
PYTHONPATH=$PWD conda run --no-capture-output -n treeHMM_env \
  python -m arhmm.steps.fit --run-dir analysis/runs/dino_k5 --force
```

## Create New Features

**A track feature is one decorated function in `arhmm/core/trackfeatures.py`.**
Nothing else changes: config validation, the emission vector, the plots, the CSV
headers and the held-out diagnostics all read `FEATURE_REGISTRY`.

```python
@per_frame("my_ratio", units="", doc="one sentence, shown as the plot subtitle",
           uses=("neighbor_radius_px",), needs_image=False, bounds=(0.0, 1.0))
def _my_ratio(fb):                      # fb: FrameBundle -> (N,) float, NaN where absent
    return fb.prop("area") / fb.prop("area").max()

@temporal("my_delta", units="", depends=("area",), uses=("window_frames",),
          doc="one sentence")
def _my_delta(sb):                      # sb: SeriesBundle -> (T, N) float
    return sb.values["area"] - sb.prev("area")
```

The decorator arguments, none of which are cosmetic:

- `units` — free text, rendered verbatim on the plot axis. Follow the registry: `px`,
  `px^2`, `px/frame`, `count`, `a.u.`, and `""` for anything dimensionless — including
  ratios, log quantities, and deltas of dimensionless quantities.
- `bounds` — the closed range the quantity is **mathematically** confined to, or omit it
  (default `None`) when it has none. It is not a plot range and not a normalization:
  out-of-range values are counted and reported by the `features` step, never clipped,
  because a value that cannot exist means the estimator broke down.
- `uses` — `features.params` keys this feature reads. **Only these enter the features
  cache key**, so a param no computed feature consumes can change without invalidating
  the cache.
- `depends` — other *registered features* this one reads, and only `@temporal` accepts it.
  A regionprops column via `fb.prop("area")` is not a dependency; `sb.values["area"]` is.
- `needs_image` — set when it reads the phase/RFP stack (`fb.image`, `(H, W, 2)`,
  normalized over the whole stack).

Return NaN for both the absent and the undefined case — `np.where(cond, value, np.nan)`
inside `np.errstate(...)`, as `_circularity` and `_win_std_log_area` do. Never `inf`.

Four rules that are load-bearing:

1. **Never loop over cells for regionprops.** A `FrameBundle` already holds one
   `regionprops_table` call for the whole frame; use `fb.prop(...)`.
2. **Temporal features step over a cell's *active* frames.** `SeriesBundle` exposes only
   `prev`, `gap`, `window`, `displacement`, `prev_centroids` — there is deliberately no
   `t - 1` indexing, because a tracking gap would otherwise look like a teleport.
3. **Keep imports light.** This module is imported by `arhmm.config`, hence by every
   step in all three envs — only one of which has scikit-image. Import skimage *inside*
   the function (see `_dilated_t_cell_neighbors`).
4. If you change an **existing** feature's math, hand-bump `version` on the `features`
   `Step` in `arhmm/layout.py`. Adding a new feature needs no bump — it changes
   `computed_features`, so the cache key moves on its own.

Then use it: `features.compute: all` picks it up automatically; add the name to
`model.features` in a run config to feed it to the model. Anything computed but not in
`model.features` is kept as a **held-out diagnostic** and still appears in the
distribution figure — that is what makes a state description checkable.

Verify with `tests/test_features.py` (`python -m unittest discover -s tests -t tests`)
and a smoke run.

Bigger additions, documented in `arhmm/README.md`:
- **A whole modality** (GPU, own env, k anonymous dimensions): a `Provider` in
  `arhmm/core/providers.py` plus a step in `arhmm/steps/` registered in
  `layout.STEPS`. `steps/fit.py` needs no change.
- **A new output**: a module in `arhmm/extras/` exposing `REQUIRES` and
  `run(cfg, layout, out_dir)`. Discovery is by filename; keep module-level imports
  light because config validation imports it just to read `REQUIRES`.

## Create New Experiment Run

Add one small YAML file under `configs/runs/` carrying **only what it does
differently**, then run it.

```yaml
# configs/runs/dino_k5.yml
extends: dino_k5.yml            # resolved relative to THIS file; ultimately -> default.yml -> site.yml
run_name: dino_k5               # the directory under analysis/runs/; ^[A-Za-z0-9_][A-Za-z0-9_.-]*$
description: As dino_k5, five states.
model:
  num_states: 5
```

`extends` is a path relative to the extending file's own directory.

```bash
python -m arhmm run --dry-run configs/runs/dino_k5.yml   # see what is cached vs. RUN
python -m arhmm run           configs/runs/dino_k5.yml
```

Rules worth knowing before you write the file:

- **Merge rule: mappings merge key by key; every other type, including lists, is
  replaced wholesale.** `model: {features: [area]}` yields exactly `[area]`, not the
  inherited list plus `area`. `null` deletes an inherited key. So **check what you are
  inheriting before overriding a list** — `python -m arhmm show <config>` prints the
  resolved document. The usual trap: `_smoke.yml` already sets
  `outputs.extras: [state_timeline, condition_stats]` and `features.compute` to an explicit
  five-feature list, so a child writing `extras: [state_timeline]` silently *drops*
  `condition_stats` rather than adding anything.
- `run_name` must be unique — reusing a run directory with a different config is
  refused unless you pass `--allow-config-change`.
- `configs/site.yml` holds paths, the six crop ids, condition groupings and the conda env
  names. Edit it only to move the repo to another checkout or machine — changing it
  invalidates caches. Nothing in it is an experimental knob.
- `model.em_seeds` is the **only** randomness surface: EM runs once per seed, best final
  log probability wins (ties to the lowest seed), and every seed is recorded in
  `fit/fit_summary.yml`.

## Conda environments

Three environments, one per step role, named in `configs/site.yml` under `envs:`. The
CLI switches between them automatically; you only name one when running a step by hand.

| role | env | used for | key versions |
|---|---|---|---|
| `imaging` | `OccidentAnalysis` | `features`, `outputs`, `extras` | py 3.11.9, numpy 1.26.4, skimage 0.23.2, matplotlib 3.8.2, imageio+ffmpeg, tifffile |
| `dino` | `cs229Dino` | `dino`, `pca` | py 3.11, torch 2.5.1 (CUDA), transformers 5.2.0, sklearn 1.8.0, numpy 2.4.2 |
| `model` | `treeHMM_env` | `fit` | py 3.11.15, jax 0.10.1 (cuda12), dynamax 1.0.1, tfp-nightly, numpy 2.4.6 |

- The CLI driver itself runs from any env with PyYAML + numpy — conda `base` is the usual
  choice. It puts the repo root on `PYTHONPATH` for each child, which is what makes
  `arhmm` and `models.tarhmm` importable without the repo being pip-installed.