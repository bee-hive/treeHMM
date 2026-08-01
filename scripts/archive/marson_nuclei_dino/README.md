# DINOv2 HMM over Caliban cancer nuclei

Fits a 2-state HMM whose emissions are **DINOv2 embeddings of a 30×30 px patch
centered on each cancer nucleus**, over all 12 wells of
`Finetuned_Analysis_250t_600xy` (250 frames, 600×600).

**Intent:** one state for cancer cells that are left alone, one for cells in
aggregates.

Cancer cells **are** the Caliban nuclei tracks. SAM3 `cancer` tracks are not used
anywhere in this pipeline — the nuclei are the gold standard.

## Quick start

```bash
cd scripts/marson_nuclei_dino
bash run_pipeline.sh
```

| Step | Script | Env | Cost |
|---|---|---|---|
| 1 | `calculate_nuclei_emissions.py` | `OccidentAnalysis` | ~2 min |
| 2 | `compute_dino_embeddings.py` | `cs229Dino` | ~1 h GPU (983k patches) |
| 3 | `reduce_dino_pca.py` | `cs229Dino` | ~2 min |
| 4 | `fit_nuclei_hmm.py` | `treeHMM_env` | ~5 min GPU |
| 5 | `create_nuclei_overlay_videos.py` | `OccidentAnalysis` | ~3 min/well |

All parameters live in `config.yml`. Change behavior there, not in the scripts.

## The validation design

The model sees **only DINO PCs**. Every scalar feature computed in Step 1
(`velocity`, `area`, `circularity`, `nuclei_neighbors_30px`,
`nuclei_neighbors_50px`, `nearest_nucleus_dist`, `t_cell_neighbors_20px`) is
held out and used purely to judge the result. So

> "did the two states separate isolated cells from aggregated ones?"

is answered against data the model never saw. The headline numbers land in
`state_summary.json` (`auroc_neighbors30_predicts_aggregate_state`) and the
picture in `state_vs_aggregation.png`.

Setting `model_features: [nuclei_neighbors_30px]` would of course make the AUROC
meaningless — keep `model_features: []` if you want the validation to mean
anything.

## Know this before reading the results

Across all 982,757 active nucleus-frames in the 12 wells:

| Nuclei within 30 px | Share |
|---|---|
| 0 (truly alone) | **7.6%** |
| 1 | 17.4% |
| 2 | 22.0% |
| 3 | 20.4% |
| ≥4 | 32.6% |

Median distance to the nearest other nucleus is **14.8 px** — *half* the patch
width — and 51.6% of nuclei have their nearest neighbor inside the patch radius.

Two consequences:

1. **"Left alone" is a rare class (~8%), not half the data.** A balanced k=2 split
   cannot be alone-vs-aggregate; if the model finds a 50/50 split it is
   necessarily dividing on something else. Judge the fit by the AUROC and the
   violin overlap, not by whether two states appeared.
2. **A 30×30 patch may be too small to express "aggregate".** At this density the
   patch is nearly always partly filled by a neighbor, so the contrast the model
   can see is more "how crowded is my immediate edge" than "am I inside a clump."
   If the AUROC is weak, raising `dino_patch_px` to 60–80 (enough to contain a
   whole clump) is the first thing to try, and only Steps 2–4 need re-running.

## First result (B3 only, 30 px patches): the split is NOT alone-vs-aggregate

Run on well B3 alone (887 tracks after `min_t=10`, 82,441 active cell-frames):

| Metric | Value |
|---|---|
| AUROC, `nuclei_neighbors_30px` → state | **0.524** (chance) |
| Mean neighbors within 30 px | state 0: 2.71, state 1: 2.54 |
| Occupancy | 91% / 9% |

The two states are near-identical in local crowding. What the minority (9%) state
actually captures, measured against held-out scalars in standard-deviation units:

| Held-out feature | Effect (state 1 − state 0) |
|---|---|
| circularity | **+0.42 σ** |
| nearest-neighbor distance | +0.24 σ |
| area | +0.23 σ |
| velocity | −0.25 σ |
| nuclei within 30 px | −0.09 σ |

So state 1 is **round, large, slow, and slightly more isolated** — the opposite of
an aggregate. The obvious follow-up hypothesis, that it is a pre-mitotic
rounded-up state, is also **refuted**: `P(state 1)` in the 6 frames before a
division runs 0.70–0.95× baseline, i.e. flat-to-depleted, never enriched.

**Diagnosis — the patch is too small.** Mean nucleus area is ~221 µm², which at
`mpp=1.25` is ~141 px², i.e. a nucleus ~13 px across. A 30×30 patch is therefore
barely two nucleus diameters: it is dominated by the cell's *own* morphology and
carries too little surround to express "am I inside a clump." The DINO PCs
confirm this — the split is driven almost entirely by PC0 and PC2 (−0.90 σ and
−0.93 σ), and PC0 is the component that correlates *least* with crowding (|r| =
0.07) while correlating with nothing temporal either.

The crowding signal does exist in the embedding — PC1 reaches |r| = 0.41 against
`nuclei_neighbors_30px`, PC2 0.33, PC6 0.24 — it is simply not what a 2-state
split on 10 equally-weighted PCs latches onto.

**Fix being tried:** `dino_patch_px: 80` (~5 nucleus diameters, enough to contain
a whole clump), in the sibling `scripts/marson_nuclei_dino_80px/`. Only Steps 2–4
need re-running; the Step 1 artifacts are patch-independent and are symlinked.

Confounds are clean either way: no PC correlates with frame index above 0.17, and
`eta` vs well identity is 0.000, so there is no meaningful time drift or batch
effect for the model to latch onto instead.

## Divisions

`allow_divisions: true` wires real lineage from `{well}_div.pkl` into the tree
kernel by direct Caliban ID lookup. The convention was verified empirically
against the data rather than assumed:

- `div.frame` **is** the daughters' first active frame (328/328 in B3, 357/357 in E4)
- the parent's last active frame is `div.frame - 1` in ~99% of cases, occasionally
  1–3 frames earlier — edges where the parent is not active at `div.frame - 1` are
  skipped

This is the first configuration in the repo that actually exercises `P_div`; every
prior run had `is_division_mask` entirely `False`.

## Geometry

Tracking masks and the centroids in `cell_data.parquet` live in the 600×600
**center** crop of `registered.tiff` `(450, 1040, 1408, 2)`, over frames 50:300:

```
y_reg = y_crop + (1040 - 600)//2 = y_crop + 220
x_reg = x_crop + (1408 - 600)//2 = x_crop + 404
t_reg = t_crop + 50
```

Verified: 99.7% of nucleus centroids land above the frame's 90th-percentile RFP
intensity with these offsets, versus 10.7% (chance) without them.

## Two upstream bugs fixed here

Both exist in `scripts/cancer_dino_hmm/` and are worth knowing if you compare results:

- **`get_RGB_image_with_nuclei` overflow.** The upstream version adds the nuclei
  plane into an already-`uint8` array (`rgb_image[..., 0] += ...`), which wraps
  around on overflow; the subsequent `np.clip` cannot undo a wrap, so bright
  nuclei render as *dark* pixels. Here the composite is accumulated in float and
  clipped before the cast.
- **Orphaned division children.** `filter_tracks_by_time` promotes a cell to a new
  root when its parent is filtered out, but upstream leaves `is_division_mask`
  set — so inference applies the division kernel to a self-parent edge. Here the
  division flag is cleared alongside the promotion, and the count is logged.

The AR-warmup step is also off by default (`ar_warmup: false`). It exists only to
hide the artifactual `velocity == 0` on a cell's first frame, which is irrelevant
when emissions are DINO patches — and upstream it silently destroys division
edges by marking each daughter's division frame inactive. When enabled here it
skips division children and reports how many.

## Outputs

Under `analysis/marson_nuclei_dino_k2_30px/`:

| File | What it shows |
|---|---|
| `state_vs_aggregation.png` | **primary validation** — neighbor count and nearest-neighbor distance by state |
| `state_summary.json` | AUROC, occupancy, log-prob, cell/division counts |
| `diagnostic_features_by_state.png` | all 7 held-out scalars by state |
| `dino_pcs_by_state.png` | what the model actually saw |
| `state_assignments.png` | per-nucleus state over time, well colorbar |
| `state_counts.png` | state occupancy over time, by condition |
| `transition_matrices.png` | learned `P_std` and `P_div` |
| `condition_occupancy.png` / `.csv` | occupancy by SH / RASA2 / CUL5, points = wells |
| `{well}/nuclei_state_assignments.npy` | `(T, N_kept)` argmax smoothed states |
| `{well}/nuclei_HMM_overlay_video.mp4` | state-colored markers over the raw video |
