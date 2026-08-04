# Input to the Tree AR-HMM

Reference for exactly what `models/tarhmm.py` consumes, and how the `arhmm/`
pipeline builds it.

---

## 1. The six arrays

Entry point: `tARHMM.fit_em(params, props, emissions, inputs, parent_indices,
is_division_mask, active_mask, is_new_root_mask, num_iters=..., verbose=...)`
(`models/tarhmm.py:808`).

| Argument | Shape (single sequence) | dtype | Meaning |
|---|---|---|---|
| `emissions` | `(T, C, D)` | float32 | observation vector per cell-frame |
| `inputs` | `(T, C, num_lags*D)` | float32 | lagged **parent** observation (AR regressor) |
| `parent_indices` | `(T, C)` | int32 | column of this cell's parent at `t-1` |
| `is_division_mask` | `(T, C)` | bool | this cell-frame is a division child |
| `active_mask` | `(T, C)` | bool | cell exists/observed; states inferred here |
| `is_new_root_mask` | `(T, C)` | bool | first frame of a spontaneous appearance |

**Batched form:** every array gets a leading `B` axis — `(B, T, C, D)` and
`(B, T, C)` — where `B` indexes independent files/crops. `fit_em` auto-promotes
`B = 1` when `emissions.ndim == 3` (`models/tarhmm.py:847`). The E-step is
`vmap`ped over `B` and sufficient statistics are summed before a single M-step.

All six must stay mutually consistent. Any code that filters columns must remap
`parent_indices` alongside the masks and emissions, or inference breaks
**silently** rather than loudly.

---

## 2. Dimensions

### `T` — timesteps
A fixed grid, identical for every crop. `lineage.concatenate_crops` raises if
crops disagree on frame count.

### `C` — cell columns
One fixed column per unique cell, **never reused**:

- a cell that dies leaves its column inactive forever;
- a division ends the parent's column and allocates two new columns.

Crops are concatenated side-by-side into a single column space
(`concatenate_crops` shifts `parent_indices` into the combined space), so
`C = Σ_crops N_crop` after filtering.

### `D` — `emission_dim`
Passed to the constructor: `tARHMM(num_states, emission_dim, num_lags, ...)`.

---

## 3. What `D` contains

From `arhmm/steps/fit.py::_load_inputs` and `arhmm/config.py::emission_names`,
the emission vector is a concatenation **in this order**:

1. **Track features** — `model.features`, a subset of `features.compute`,
   selected out of the cached `features.npz` `values` array `(T, N, F)`.
2. **DINO PCs** — when `model.use_dino_pcs` is set, the top `dino.n_pcs`
   principal components of DINOv2 embeddings of centroid patches, named
   `dino_pc_0 … dino_pc_{k-1}`, from the `pca` step's `pcs.npz`.

```
D = len(model.features) + (dino.n_pcs if model.use_dino_pcs else 0)
```

Config validation rejects `D == 0`.

### Track-feature registry (`arhmm/core/trackfeatures.py`)

**Per-frame** — computed from a single frame's masks and image:

| Feature | Description | Units |
|---|---|---|
| `area` | mask pixel count | px |
| `circularity` | `4π·area / perimeter²` of the largest connected component; 1.0 for a perfect disc | |
| `perimeter` | perimeter of the largest connected component | |
| `eccentricity` | eccentricity of the fitted ellipse; 0 is a circle | |
| `solidity` | area / convex hull area; drops when the cell blebs | |
| `aspect_ratio` | major axis / minor axis of the fitted ellipse | |
| `extent` | area / bounding box area | |
| `t_cell_neighbors` | T cells whose centroid lies within `neighbor_radius_px` | count |
| `dilated_t_cell_neighbors` | T cells touching this cell's mask dilated by `dilate_radius_px` | count |
| `cancer_neighbors` | other cancer cells within `neighbor_radius_px` | count |
| `rfp_mean` | mean RFP intensity inside the mask | a.u. |
| `rfp_total` | summed RFP intensity inside the mask | a.u. |
| `phase_std` | std of phase intensity inside the mask; rises with granularity | a.u. |

**Temporal** — computed over a cell's **active** frames, always against the
previous *active* frame (so a tracking gap gives one wide step, not a spurious
drop to zero and back):

| Feature | Description | Units |
|---|---|---|
| `velocity` | centroid displacement since previous active frame, divided by the gap | px/frame |
| `d_area_frac` | `(area − prev_area) / prev_area`; scale-free | |
| `d_circularity` | `circularity − prev_circularity` | |
| `win_std_log_area` | trailing std of `log(area)` over `window_frames` active frames | |
| `win_std_circularity` | trailing std of circularity | |
| `win_std_displacement` | trailing std of per-frame displacement | |

Because temporal features are undefined on a cell's first frame, config
validation forces `cells.warmup_frames >= 1`.

---

## 4. Mask semantics

Defined in `arhmm/core/lineage.py::build_masks`, consumed throughout
`models/tarhmm.py`.

- **`parent_indices[t, c]`** — the cell **itself** while it persists, the
  **dividing parent's column** on a division frame, and a **dummy `0`** for new
  roots. Inference picks `P_div` vs `P_std` per edge via `is_division_mask`
  (`tree_hmm_filter`, `models/tarhmm.py:188-195`).
- **`is_new_root_mask`** — new roots are re-seeded with the initial
  distribution rather than propagated through a transition matrix
  (`models/tarhmm.py:162-167`); `argmax` over time gives each root's first frame,
  used for the initial-state sufficient statistics (`models/tarhmm.py:307`). The
  backward pass neutralizes messages from new roots into dummy column 0
  (`models/tarhmm.py:263`).
- **`is_division_mask`** — selects `P_div`, and the AR input is zeroed (no
  autoregression from parent to daughter).
- **`active_mask`** — inactive cell-frames are zeroed out of the filtered probs
  and excluded from the log-normalizer. Inactive/padded entries carry zeros and
  NaNs by design; statistics use `jnp.nansum` / explicit masking.

Every cell starts as its own parent and a new root at its first active frame.
When `cells.allow_divisions` is set, each resolvable lineage edge converts the
daughter's first frame from a new root into a division child pointing at the
parent's column.

---

## 5. `inputs` is derived, not supplied

Build it with:

```python
inputs = arhmm.compute_inputs(emissions, parent_indices, is_division_mask,
                              is_new_root_mask, active_mask)
```

**Note the argument order differs from `fit_em`'s.**

It gathers `emissions[t-1][parent_indices[t]]`, then zeroes division children and
inactive cells. New roots are *not* zeroed — their first active frame (marked
inactive for inference) still supplies a valid AR input.

- `num_lags = 0` → returns `(T, C, 0)`; the model degenerates to a bias-only
  Gaussian HMM and `_compute_conditional_logliks` skips the weight matmul.
- `num_lags = 1` → returns `(T, C, D)`.
- `num_lags > 1` → raises `NotImplementedError`.

---

## 6. Pipeline preprocessing before the model sees the arrays

`arhmm/steps/fit.py`, in order:

1. **`lineage.filter_short_cells(min_frames)`** — drops columns with fewer than
   `cells.min_frames` active frames and **remaps `parent_indices`**, promoting
   orphaned cells to new roots. Raises if no cell survives.
2. **`lineage.apply_warmup(warmup_frames)`** — deactivates each cell's leading
   `cells.warmup_frames` frames, in place. Holding the warmup fixed regardless
   of lag order means a lag-0 and a lag-1 run are scored over exactly the same
   cell-frames. A cell shorter than the warmup keeps its final frame.
3. **Non-finite scrub** — `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`;
   the step reports how many zeroed entries fell inside active cell-frames.
4. **Optional standardization** (`model.standardize`) — per-dimension mean/std
   over active cell-frames only; `std < 1e-8` → `1.0`.
5. **Cast to `jnp`**, then run EM once per seed in `model.em_seeds`, keeping the
   fit with the best final log probability (ties broken by lowest seed).

### Upstream cache: `features.npz` (per crop)

```
cell_ids         (N,)       sorted cell IDs; defines column order
centroids        (T, N, 2)  crop-local (y, x), NaN where absent
values           (T, N, F)  the features, NaN where absent or undefined
feature_names    (F,)       column order of `values`
active_mask      (T, N)     bool
parent_indices   (T, N)     int32
is_division_mask (T, N)     bool
is_new_root_mask (T, N)     bool
```

Everything in `features.compute` is cached, not just `model.features` — so state
profiling can read features the model never saw.

---

## 7. Model hyperparameters (not data)

```python
tARHMM(num_states,                        # K
       emission_dim,                      # D
       num_lags=1,                        # must be 0 or 1
       initial_probs_concentration=1.1,
       transition_matrix_concentration=1.1,
       transition_matrix_stickiness=0.0)
```

Parameters are carried as
`ParamsLinearTreeARHMM(initial, transitions, division_transitions, emissions)`,
where the two `K×K` matrices `P_std` (persistence) and `P_div` (division) travel
through inference as the tuple `(P_std, P_div)`.

---

## 8. Minimal usage sketch

```python
from models.tarhmm import tARHMM, tree_hmm_two_filter_smoother

arhmm = tARHMM(num_states=K, emission_dim=D, num_lags=L)
params, props = arhmm.initialize(key=jr.PRNGKey(seed), method="kmeans",
                                 emissions=emissions)

inputs = arhmm.compute_inputs(emissions, parent_indices, is_division_mask,
                              is_new_root_mask, active_mask)

params, log_probs = arhmm.fit_em(params, props, emissions, inputs,
                                 parent_indices, is_division_mask,
                                 active_mask, is_new_root_mask,
                                 num_iters=n)

posterior = tree_hmm_two_filter_smoother(
    *arhmm._inference_args(params, emissions, inputs, parent_indices,
                           is_division_mask, active_mask, is_new_root_mask))
states = posterior.smoothed_probs.argmax(axis=-1)   # (T, C)
```
