# `arhmm`

Deterministic, config-driven runs of the tree AR-HMM.

**One YAML file defines one run.** Every run produces the same five base outputs,
plus whatever extras it asks for. Feature computation is decoupled from model
fitting, so adding a new input is a small, local change.

```bash
python -m arhmm doctor                      # environments, paths, CUDA, npz round-trip
python -m arhmm run    configs/_smoke.yml   # one crop, no DINO, about a minute
python -m arhmm status configs/_smoke.yml   # which steps are current, and why
python -m arhmm show   configs/runs/dino_k3.yml
python -m arhmm list   configs/_smoke.yml
```

The pipeline is not pip-installed. Run it from the repo root; the CLI puts the
repo root on `PYTHONPATH` for each step it spawns, which is what makes both
`arhmm` and `models.tarhmm` importable in whichever conda environment that
step needs.

---

## The step chain

Six steps, fixed order, three conda environments. Each is a separate process
because no single environment has JAX, torch and scikit-image together.

| step | env | output | shared? |
|---|---|---|---|
| `features` | `OccidentAnalysis` | per-cell, per-frame track features | cached |
| `dino` | `cs229Dino` | DINOv2 embeddings of centroid patches | cached |
| `pca` | `cs229Dino` | one joint PCA → top-k components | cached |
| `fit` | `treeHMM_env` | the fitted model and posteriors | run-local |
| `outputs` | `OccidentAnalysis` | the five base outputs | run-local |
| `extras` | `OccidentAnalysis` | opt-in extra outputs | run-local, optional |

`dino` and `pca` are skipped unless `model.use_dino_pcs` is true; `extras` is
skipped unless `outputs.extras` is non-empty.

Every step is independently runnable, which is how you debug one environment at
a time:

```bash
conda run --no-capture-output -n treeHMM_env \
  python -m arhmm.steps.fit --run-dir analysis/runs/dino_k3
```

A step reads `config.resolved.yml` from the run directory and nothing else --
never a parameter passed on the command line. "Rerun this step by hand" and
"the driver ran it" are therefore the same operation.

---

## Configuration

Three layers, deep-merged, `extends` resolved relative to the extending file:

```
configs/site.yml       paths, conda environments, the crop list, conditions
configs/default.yml    experiment-facing defaults
configs/runs/<x>.yml   only what this run does differently
```

**Merge rule: mappings merge key by key; every other type, including lists, is
replaced wholesale.** A run writing `model: {features: [area]}` gets exactly
`[area]`, not the defaults plus `area`. `null` deletes an inherited key.

`cells.source` picks which segmentation defines a cell: `phase` (CVAT whole-body
tracks) or `nuclei` (the cancer-nuclei tracks, `nuclei_tracks.tiff`, which sit in
the crop directory beside `crop.tiff`). It swaps the label space whole --
track features, centroids, DINO patches, the fit and the overlays all follow --
and only `core/cells.py` branches on it. The two ID spaces are never reconciled;
under `nuclei` the T-cell masks still come from the CVAT stack, because the
nucleus segmentation covers the cancer cells only. Both sources hash into the
`features` and `dino` keys, so the caches cannot mix.

The resolved document is frozen into the run directory as
`config.resolved.yml` *before anything executes*, and every step reads that. If
`default.yml` changes tomorrow, a rerun of an existing run directory is still
the run it was.

A sweep is several small files, not a grid inside one file:

```yaml
# configs/runs/dino_k4.yml
extends: dino_k3.yml
run_name: dino_k4
model: {num_states: 4}
```

`python -m arhmm run configs/runs/dino_k{3,4,5,6}.yml` shares one `features` cache and
one `pca` cache across all four, because none of those steps depends on
`model.*`. That is the whole sweep story -- no sweep machinery, just the cache.

---

## Adding things

### Add a track feature — one decorated function

In `core/trackfeatures.py`:

```python
@per_frame("my_feature", units="px", doc="one sentence, shown on the plot panel")
def _my_feature(fb):
    return fb.prop("area") * 2          # fb is a FrameBundle
```

or, for something computed over a cell's history:

```python
@temporal("my_delta", units="", depends=("area",), uses=("window_frames",),
          doc="one sentence")
def _my_delta(sb):
    return sb.values["area"] - sb.prev("area")     # sb is a SeriesBundle
```

Nothing else changes. `features.compute: all` picks it up; config validation
accepts it in `model.features` because the registry is the vocabulary; `depends`
schedules the per-frame features it reads; `uses` folds *only* the parameters it
actually consumes into the features cache key; `units` and `doc` reach the plots
and the CSV headers.

Two rules the bundles enforce structurally rather than by convention:

- A `FrameBundle` already has one `regionprops_table` call done for the whole
  frame. Look values up with `fb.prop(...)`; do not loop over cells.
- A `SeriesBundle` exposes only `prev`, `gap`, `window` and `displacement`.
  There is deliberately no way to index `t - 1`, because a temporal feature must
  step over a cell's **active** frames -- otherwise a tracking gap looks like a
  cell that teleported away and back.

### Add a modality — one provider and one step

DINO components are not registry entries, and that is deliberate: they need a
GPU and a different conda environment, and they produce *k* dimensions with no
individual meaning via a fit that is global across all crops. A modality gets a
`Provider` in `core/providers.py` plus a step to produce its artifact:

```python
MINE = Provider(
    name="mine", step="my_step",
    columns=lambda cfg: [f"mine_{i}" for i in range(cfg["mine"]["k"])],
    load=_load_mine,          # (cfg, layout, crop_id) -> ((T, N, F), names)
    describe=_describe_mine,
)
```

`steps/fit.py` contains no provider-specific code: it asks
`config.emission_names(cfg)` for the ordered column list and each active
provider for its block. Add the step to `layout.STEPS` with its config
dependencies and it joins the cache chain.

### Add an extra — one module

A module in `extras/` exposing `REQUIRES` and `run(cfg, layout, out_dir)`.
Discovery is by filename, so there is no registry to update. `REQUIRES` is
checked at config time, so asking for `dino_confounds` on a run without DINO
columns fails before anything executes. Keep module-level imports light -- config
validation imports the module just to read `REQUIRES`.

---

## Determinism

- The resolved config is frozen into the run directory and is the only input
  every step reads.
- `model.em_seeds` is the **only** randomness surface. EM runs once per seed and
  the best final log probability wins, ties broken by the lowest seed. Every
  seed's result is recorded in `fit_summary.yml`, so a run whose seeds disagree
  wildly is visible rather than lucky.
- Cells are ordered by `np.sort(np.unique(...))` and crops by sorted `crop_ids`,
  so nothing depends on dict or filesystem ordering. Reordering `data.crop_ids`
  in the YAML changes neither a cache key nor a fit.
- States are permuted into a canonical order after fitting -- by mean of the
  first emission dimension over inferred cell-frames, ties by occupancy -- and
  the permutation is applied to the transition matrices, initial distribution,
  posteriors and assignments together. Without this, "state 2" means something
  different on every rerun.
- Cache keys chain: a step's key hashes its declared config subset together with
  the keys of the steps upstream of it. Source code is *not* hashed -- that would
  make every edit invalidate everything and train you to reach for `--force`.
  Each step carries a hand-bumped `version` instead, and `manifest.yml` records
  the git commit so a stale cache is detectable.
- **What "deterministic" means here, precisely.** It means the *structural*
  guarantees above -- same config in, same cache key, same state labelling, same
  cell and crop ordering -- not bit-identical floating point. State assignments
  are stable across reruns; the continuous arrays are not. Measured on `dino_k3`
  over three reruns of identical code and inputs: `state_assignments.npy`,
  `emissions.npy` and `division_transition_matrix.npy` came back bit-identical,
  while `log_probs.npy` moved by up to 1.2e-2, `state_probs.npy` by 2.1e-5 and
  `transition_matrix.npy` by 2.7e-7. XLA autotunes its kernels per process and
  picks different reduction orders, and float32 addition is not associative, so
  this is expected rather than a regression.
- **Do not use `log_prob_final` as a regression baseline.** It moves in its last
  few digits under a plain rerun, so a small change there tells you nothing. To
  check that a code change left the science alone, compare
  `fit/state_assignments.npy` and `best_seed`, which do reproduce. Comparing
  against a run recorded at an older commit is not a controlled comparison at
  all -- `arhmm status` prints a note when a run's cached artifacts predate your
  current code.
- Writes are atomic. A killed step leaves the previous artifact or nothing,
  never a truncated file a later run happily loads.

---

## What a run produces

```
analysis/runs/<run_name>/
  config.resolved.yml     frozen; the sole input to every step
  manifest.yml            commit, host, per-step env/key/timing/status
  _stamps/<step>.json     run-local step completion
  logs/<step>.log
  features -> ../../cache/features/<key>     symlinks to the shared caches
  dino -> ...   pca -> ...
  fit/                    arrays, cell_index.csv, fit_summary.yml
  outputs/
    overlays/<crop>_state_overlay.mp4        cancer cells tinted by state
    feature_distributions.png                every cached feature, per state
    state_feature_summary.csv
    state_age_histogram.png / .csv           state occupancy binned by cell age
    transition_matrix.csv / .png
    initial_distribution.csv
    state_assignments.csv                    one row per inferred cell-frame
    state_assignments_per_cell.csv
    extras/<name>/
```

`features.compute` names everything to cache; `model.features` names the subset
the model sees. Everything computed but unused is kept as a **held-out
diagnostic**, and the distribution figure covers all of it, marking which
features the model was actually fit on. A state characterised only by its own
inputs is a tautology; the held-out features are what make the description
checkable.

`state_age_histogram` bins every inferred cell-frame by **cell age** -- elapsed
frames since the cell first appeared, pooled over all crops. Age is
`t - birth`, not a rank among the frames the cell was seen in, so a cell tracked
at frames 0, 1 and 3 sits at ages 0, 1 and 3: a tracking gap costs it a sample
rather than rewinding its clock. Under the default
`outputs.state_age_histogram.anchor: existence`, age 0 is the cell's first
observed frame, which is read from the **features** cache rather than from
`fit/active_mask.npy` -- the latter has had `cells.warmup_frames` removed. The
axis therefore starts at `cells.warmup_frames`, not at 0: no cell-frame can
carry a state below that age, so those bars would be empty by construction
rather than by measurement. The second panel gives each bin's
composition with the number of cells still alive drawn over it, because a bar
falling off with age means either that the state empties or that hardly any
cells are left that old, and the counts alone cannot tell those apart.

---

## Notes specific to this repo

- **The tree is not used.** Divisions are out of scope, so `is_division_mask` is
  all False, `parent_indices` is always the cell's own column, and `P_div` is
  never exercised. `ALL_graph.pkl` is not read. `tree_input.md` at the repo root
  is the binding contract; `core/lineage.assert_consistent` enforces it.
- **The three environments are on different numpy majors** (1.26 / 2.4 / 2.4) and
  pass arrays as `.npz`. Numeric and bool arrays round-trip; object arrays and
  pickles do not. `io.save_npz` rejects anything else and every string goes in a
  JSON sidecar. `python -m arhmm doctor` checks the round trip.
- **`compute_inputs` and `fit_em` take the masks in different orders.** They are
  same-shaped boolean arrays, so a swap runs happily and returns nonsense. Go
  through `MaskBundle` in `steps/fit.py`; never call them positionally.
- **`.gitignore` line 17 is `/lib/`, anchored on purpose.** Unanchored `lib/`
  matches at any depth and once silently swallowed an entire package directory.
  Do not un-anchor it, and do not name a package directory `lib`.
- **A nuclei run and a phase run are not comparable feature by feature**, and
  nothing in the code compensates for it:
  - `dilated_t_cell_neighbors` dilates the subject mask by `dilate_radius_px`
    and counts the T-cell labels it touches (`core/trackfeatures.py`). A nucleus
    sits well inside the cell body, so at a fixed radius it systematically
    undercounts relative to a phase mask. The same radius means a different
    thing in the two runs.
  - `area`, `circularity`, `solidity` and the rest describe a *nucleus* under
    `cells.source: nuclei`, so a state description does not transfer between the
    two sources. Provenance is recorded as `cell_source` in the features cache's
    `meta.json`.
  - Track counts and lengths differ substantially -- 27 nucleus labels against
    16 CVAT cancer ids in `B4_t50…`, 71 against 51 in `E4_t250…` -- so
    `cells.min_frames` and `cells.warmup_frames` may want revisiting for a
    nuclei run. Their defaults are tuned for `phase`.
- Nothing imports `MarsonImagingPipeline`. The handful of helpers that were used
  from it are reimplemented in `core/viz.py`.

## Tests

```bash
python -m unittest discover -s tests -t tests -v
```

Stdlib `unittest`, not pytest, which is installed in none of the three
environments. The suite passes in all three, which is what keeps the
import-weight discipline honest -- `config.py` and `core/trackfeatures.py` are
imported by every step, so they may not pull in scikit-image, torch or JAX.

`configs/_smoke.yml` is the end-to-end regression check: one crop, no DINO,
about a minute.
