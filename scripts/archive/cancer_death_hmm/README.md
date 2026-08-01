# Cancer-cell death-state HMM experiments

Goal: a multi-state (AR-)HMM in which cancer cells undergoing **death
events** concentrate in an identifiable state. Quantifying death per
condition (SH / RASA2 / CUL5) is downstream of this and is not done here.

## Why the phase definition

Cancer cells are defined by their **phase masks** — channel 1 of the
type-separated CVAT tracks — not by Caliban nuclei. The ground-truth death
annotations name CVAT track IDs (verified: B8_t50's phase channel contains
IDs `{1,2,3,4,5,41,…,113}`, which is where cells 1/3/4/41 in the notes live;
the Caliban nuclei for that crop are IDs 1–12, a different space). Using the
phase definition means the labels index the model's columns directly, with no
overlap-mapping step in between.

## The death signature, and what each part needs

From `MarsonImagingPipeline/data/gt-death-events.md`: a sudden single-frame
shape change (**area down, circularity up**), followed by frames of
**elevated position/shape variance**, in cells with **more T-cell
neighbours** than average.

Each part is exposed to the model as an explicit feature, because an HMM
emission sees one frame (plus one predecessor at lag 1) and cannot compute a
window statistic for itself:

| signature | feature |
|---|---|
| acute shape change | `d_area_frac`, `d_circularity` |
| post-death instability | `win_std_log_area`, `win_std_circularity`, `win_std_displacement` |
| T-cell context | `dilated_t_cell_neighbors`, `t_cell_neighbors_20px` |
| appearance | 10 whitened DINOv2 PCs of a 50×50 patch |

All ten scalar features are computed and saved once; each arm selects a
subset. The unselected ones remain available as annotation-independent
diagnostics for state profiling and evaluation.

## Arms

All arms fit the **same cells** with the same `min_t` filter and the same
one-frame AR warmup, so differences are attributable to features and lag
order alone. The tree (division) part is off throughout.

| arm | features | lags | D | emission params/state |
|---|---|---|---|---|
| `delta_lag0` | Δarea, Δcircularity | 0 | 2 | 5 |
| `shape_lag1` | area, circularity | 1 | 2 | 9 |
| `shape_tcell_lag1` | + dilated T-cell contact | 1 | 3 | 18 |
| `shape_window_lag1` | shape + 3 window-instability | 1 | 5 | 45 |
| `dino_lag1` | 10 DINO PCs | 1 | 10 | 165 |
| `dino_window_lag1` | 10 DINO PCs + 3 window | 1 | 13 | 272 |

**The parameter budget is the binding constraint.** There are 5,551 active
cancer cell-frames across the six crops (162 cells). If death is a
single-frame event, the death state holds on the order of 50 of them. At
D=10 with full covariance and AR(1) that is 165 parameters against ~50
observations: the M-step fits nearly perfectly, `Σ` collapses toward the
1e-6 jitter floor, and the "state" becomes a memorized handful of frames at
enormous density. The window arms exist to fix this from the data side — if
the state persists through the ~10–15 unstable post-death frames instead of
one, it gets an order of magnitude more observations and becomes estimable.

## Death-state selection is post-hoc and label-blind

Applied identically to every arm, scoring states on the diagnostic features:

```
transition_score  =  z(mean d_circularity) − z(mean d_area_frac)
instability_score =  mean of z(mean win_std_{log_area, circularity, displacement})
death_score       =  transition_score + instability_score
```

`z(·)` standardizes across the k states of that fit. A single-frame death
state scores mainly on the first term, a durable one mainly on the second,
and a real death state should show both — hence the sum. No arm sees the
annotations during fitting or state selection.

## Evaluation

Four checks, three of them label-free:

1. **Recall** against `death_annotations.yml` — does the cell *enter* the
   death state within ±2 frames of the annotated window? Cells with no
   resolvable frame (already dead at crop start; unresolved timing) are
   scored cell-level instead.
2. **Division specificity** — mitotic rounding has the same shape signature,
   so the annotated cancer divisions are known non-death shape changes and
   any entry landing on one is a confirmed false positive.
3. **T-cell enrichment** — AUROC of `dilated_t_cell_neighbors` separating
   death-state frames from the rest. Independent only for arms that never
   saw a T-cell feature; flagged as circular for the others.
4. **Entry discipline** — a real death state is entered at most once per
   cell. Many entries per cell means a fluctuation state.

Plus Cohen's κ between arms: three unrelated feature sets converging on the
same frames is corroboration; diverging means at most one is finding death.

## Running

```bash
bash run_pipeline.sh              # everything, including the DINO arms
bash run_pipeline.sh --no-dino    # skip embedding; fit the cheap arms only
```

| Step | Script | Env |
|---|---|---|
| 1 | `calculate_cancer_phase_emissions.py` | `OccidentAnalysis` |
| 2 | `compute_dino_embeddings.py` | `cs229Dino` |
| 3 | `reduce_dino_pca.py` | `cs229Dino` |
| 3b | `check_dino_confounds.py` | `OccidentAnalysis` |
| 4 | `fit_death_hmm.py` | `treeHMM_env` |
| 5 | `evaluate_death_states.py` | `OccidentAnalysis` |
| 6 | `create_state_overlay_videos.py` | `OccidentAnalysis` |

### Overlay videos

One mp4 per (fit, crop), written next to the fit as
`fits/{arm}/k{K}/{crop}_state_overlay.mp4`. Presentation matches the earlier
`cancer_dino_hmm` overlays — phase background, T cells as a blue layer,
cancer cells tinted by state, IDs labelled, warmup grey — plus three things
needed to judge a death detector: the death-candidate state is named in the
legend and title, ground-truth cells are labelled in yellow and starred while
inside their annotated window, and the title carries arm / k / lag / features
so videos can't be confused once moved.

`video_fits` in `config.yml` chooses what to render: `auto` (best k per arm
from `evaluation.csv`), `all` (180 videos), or an explicit list. The script
also takes arguments:

```bash
python create_state_overlay_videos.py                  # config setting
python create_state_overlay_videos.py shape_lag1:6     # one fit, all crops
python create_state_overlay_videos.py shape_lag1 delta_lag0:6
```

`fit_death_hmm.py` takes an optional list of arm names to refit just those
(`python fit_death_hmm.py shape_lag1`); `summary.csv` rows for other arms are
preserved.

Step 0 is not repeated — `scripts/cancer_dino_hmm/type_sep_tracks` is reused.

## Notes on the inputs

- **Timing.** The annotation minutes resolve at **5 min/frame**, derived and
  then verified against the masks (see the header of
  `death_annotations.yml`). `experiment.yml` declares `TIME_FACTOR: 4`, which
  is inconsistent; the 5 min/frame value is what the annotations and the
  pixel data agree on. One crop's event (B8_t250, cell 6) does not resolve
  under any convention and is kept as a cell-level label only.
- **Patch size.** Cancer phase masks have equivalent diameter median 13–22 px
  and p90 24–30 px, so the 25 px patches used by the earlier
  `cancer_dino_hmm` pipeline clipped a large fraction of cells and contained
  almost none of the surrounding T cells. 50 px here.
- **Contrast.** Phase is normalized once per crop over the whole stack with
  fixed percentiles. Per-frame min-max (the older pipelines) lets one bright
  artifact rescale a frame and turns the leading PCs into a clock — the
  failure `check_dino_confounds.py` was written to catch.
- **Patch colouring.** Masks are alpha-blended over the phase image rather
  than replacing it, so texture survives; the subject cell gets a colour
  distinct from other cancer cells, because a 50 px patch often contains
  more than one and the embedding must know which cell it describes.
