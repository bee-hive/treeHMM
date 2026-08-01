# Cancer-cell dead-state HMM experiments

Successor to `cancer_death_hmm`. That set targeted the death **event** — a
single-frame transition. This one targets the dead **condition**: a state a
cell enters and does not leave.

## Why the target changed

Counted directly against the B8_t50 annotations:

| target | frames | share of cell-frames |
|---|---|---|
| death **transition** (1 frame/event) | 3 | **0.52%** |
| death **condition** (post-death, still tracked) | 133 | **23.0%** |

Four of twelve cells die and each contributes its whole remaining track to
the dead condition. So the rare-state problem that motivated downsampling is
mostly an artifact of targeting the event; at 23% it is an ordinary split.
**No downsampling or class balancing is done here.**

It also fixes a structural gap. A transition-based detector cannot represent
a cell that is *already dead* when its track begins — there is no transition
to find. That is why the already-dead annotated cell scored 0 in every
`cancer_death_hmm` arm. An absorbing state reaches it through the initial
distribution.

## Three changes

**1. Memory features.** Without them a post-death frame is indistinguishable
from an ordinary small round cell — nothing in pure geometry says a cell has
*already* had its event. Both are monotone within a track:

- `area_over_running_max` — `area_t / max(area so far)`; drops and never recovers
- `running_max_abs_d_circularity` — non-decreasing; once rounded up, stays flagged

**2. Intensity features.** `cancer_death_hmm` was pure mask geometry, with no
pixel intensity anywhere. Apoptotic cells in phase contrast go bright and
refractile before lysing, and RFP nuclei condense and fragment. Both are
persistent, unlike the transient shape jump.

| | |
|---|---|
| `phase_mean` | mean phase inside the mask |
| `phase_std` | within-mask SD — granularity / blebbing |
| `phase_contrast` | mask mean minus a `disk(3)` ring outside it — the refractile halo |
| `rfp_mean` | mean RFP inside the mask |
| `rfp_cv` | within-mask SD / mean — condensation and fragmentation |
| `rfp_concentration` | share of total RFP in the brightest 10% of mask pixels |

**3. An absorbing state.** The last state's transition row is pinned to
one-hot self-transition, so dead → alive is impossible. Implemented in
`absorbing.py` by subclassing the transition component; `models/tarhmm.py`
is shared by every pipeline here and is left untouched.

## Where the images come from

`{tracking_crops_dir}/{well}/{crop}/crop.tiff`, last axis `0 = RFP`,
`1 = phase`.

This matters. Slicing the per-well `registered.tiff` with the coordinates in
the crop id does **not** land on the same region — checked against the masks,
that slice puts as much RFP in T-cell masks as in cancer masks (2.06 vs 2.64
over background), whereas TrackingCrops gives 1.74 vs 3.83. Only cancer cells
carry RFP, so TrackingCrops is the aligned source. `cancer_dino_hmm` and the
first version of `cancer_death_hmm`'s DINO step both used the registered
slice, which is why those DINO PCs correlated with crop identity (η = 0.665)
and not with the shape signature (|r| = 0.017).

Each channel is normalized once per crop over the whole stack with 1st/99th
percentiles — per-frame normalization lets a single bright artifact rescale a
frame and turns leading components into a clock.

## Arms

All fitted over the same cells, same `min_t`, same one-frame warmup, at
`num_lags=1`. Each is fitted **twice**, absorbing and free, so the
constraint's effect is measured rather than assumed.

| arm | features | D |
|---|---|---|
| `memory` | the 2 memory features | 2 |
| `intensity` | 3 phase + 3 RFP | 6 |
| `memory_intensity` | memory + intensity | 8 |
| `shape_memory` | area, circularity + memory | 4 |
| `shape_memory_intensity` | area, circularity + memory + intensity | 10 |

`k ∈ {2,3,4,5}`, 3 seeds each, best final log prob kept — a low-occupancy
state makes the outcome unusually sensitive to which local optimum EM finds.

## Dead-state selection is label-blind

Applied identically to every arm and variant, on the full 18-feature
diagnostic set regardless of which features fed the model:

```
transition_score  = z(mean d_circularity) - z(mean d_area_frac)
persistence_score = mean of z(-mean area_over_running_max),
                            z(+mean running_max_abs_d_circularity)
dead_score        = transition_score + persistence_score
```

The persistence term replaces `cancer_death_hmm`'s instability term: for a
*dead* state the diagnostic is not "this frame is changing" but "this cell
has already shrunk and rounded and not recovered".

The absorbing state is designated **by index**, not by phenotype — EM decides
what lands there. `dead_state_is_absorbing` in each fit records whether the
constrained state is the one that actually carries the death signature.

## Evaluation

1. **Entry recall** against the annotations, with a permutation null that
   relocates each cell's entries among its own active frames (preserving
   entries-per-cell, and so occupancy).
2. **Persistence after entry** — 1.0 by construction for absorbing fits, so
   it only confirms the constraint took; for free fits it is the real
   question.
3. **Already-dead capture** — the check that most directly separates this
   design from `cancer_death_hmm`.
4. **Division specificity** (label-free) — mitotic rounding has the death
   shape signature, so annotated cancer divisions are known non-death changes.
5. **Cell-level death fraction**, overall and per condition — the quantity
   the downstream comparison wants, well defined once the state is absorbing.
6. **T-cell enrichment** (label-free) — independent for every arm here, since
   none uses a T-cell feature.

With 3 frame-level annotated events across 40 fits, `p` is ranking
information, not a significance test. More annotations remain the binding
constraint.

## Running

```bash
bash run_pipeline.sh              # everything
bash run_pipeline.sh --no-video   # skip the slow render
```

| Step | Script | Env |
|---|---|---|
| 1 | `calculate_dead_state_features.py` | `OccidentAnalysis` |
| 2 | `fit_dead_state_hmm.py` | `treeHMM_env` |
| 3 | `evaluate_dead_states.py` | `OccidentAnalysis` |
| 4 | `create_state_overlay_videos.py` | `OccidentAnalysis` |
| 5 | `write_fit_readmes.py` | `OccidentAnalysis` |

`fit_dead_state_hmm.py` takes arm names to refit a subset; the video script
takes `arm`, `arm:variant`, or `arm:variant:k`.

## Outputs

```
analysis/cancer_dead_state_hmm/
├── {crop}/                        per-crop features + centroids
└── fits/
    ├── summary.csv                one row per fit
    ├── evaluation.csv             scored against the annotations
    ├── event_detail.csv           per annotated event, per fit
    ├── condition_death_fractions.csv
    ├── evaluation.png
    ├── cell_index.csv             column -> (crop, CVAT cell id)
    └── {arm}/{absorbing|free}/k{K}/
        ├── README.md              human-readable spec for this fit
        ├── state_assignments.npy
        ├── smoothed_probs.npy
        ├── state_profile.csv
        ├── fit_summary.yml
        ├── states.png
        └── {crop}_state_overlay.mp4
```
