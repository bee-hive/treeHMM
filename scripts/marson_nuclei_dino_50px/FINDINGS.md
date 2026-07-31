# 50 px DINO embeddings, k=2, divisions off: lag0 vs lag1

Wells: B3 (SH), E3 (RASA2), B7 (CUL5). 4,059 nuclei → 2,975 after `min_t=10`.
270,797 patches embedded once into `analysis/marson_nuclei_dino_50px_shared/`
and shared by both fits via symlink. Emissions = 10 whitened DINO PCs only.

## Does autoregression change anything? No.

| | `lag0` | `lag1` |
|---|---|---|
| occupancy | 0.672 / 0.328 | 0.676 / 0.324 |
| AUROC (`nuclei_neighbors_30px`) | 0.545 | 0.550 |
| log prob | −3,700,468 → −3,291,536 | −3,659,501 → **−2,462,276** |

**Assignment agreement 96.4%, Cohen's κ = 0.918.**

AR(1) fits the data dramatically better as a density model — the likelihood gain
is ~3× that of `lag0` — but it partitions the cells essentially identically. The
autoregressive term absorbs the frame-to-frame smoothness of the embeddings
without moving the latent boundary. For this question, `num_lags` is not the
lever.

Held-out separation is likewise near-identical between variants (std units,
state 0 − state 1):

| feature | lag0 | lag1 |
|---|---|---|
| area | −0.382 | −0.390 |
| circularity | +0.230 | +0.229 |
| t_cell_neighbors_20px | +0.212 | +0.218 |
| nuclei_neighbors_30px | +0.181 | +0.199 |
| nearest_nucleus_dist | −0.182 | −0.203 |
| velocity | +0.174 | +0.149 |

State 0 = smaller, rounder, faster, slightly more crowded, more T cells nearby.
State 1 = larger, less round, more isolated. The dominant axis is **nucleus
area**, not crowding.

## The real finding: the signal is there, the readout is wrong

Contrast crowded (≥4 neighbors within 30 px, n=97,604) against isolated
(≤1, n=60,020), dropping the ambiguous middle:

| predictor | AUROC |
|---|---|
| **unsupervised k=2 HMM state** | **0.532** |
| single DINO PC1 | 0.531 |
| single DINO PC6 | 0.654 |
| single DINO PC0 | 0.685 |
| single DINO PC3 | 0.757 |
| single DINO PC2 | **0.823** |
| **LDA on all 10 PCs (linear ceiling)** | **0.953** |
| area alone (held-out scalar) | 0.613 |

A 50 px DINO patch encodes aggregation **almost perfectly** — a linear read-out
of just 10 PCs separates crowded from isolated at AUROC 0.953. (In-sample, but
157k observations against 10 parameters, so overfitting is negligible.)

The unsupervised 2-state HMM recovers essentially none of it (0.532).

**Why:** a 2-state Gaussian mixture splits along the direction of greatest
variance, and in this embedding that direction is nucleus size/appearance, not
aggregation. The aggregation axis is present but subdominant — PC2, at 5.4% of
variance. Whitening equalizes the ten kept PCs but cannot promote the
aggregation direction above the size direction.

So the earlier 30 px conclusion ("patch too small") was **wrong**, or at least
incomplete. Going 30 → 50 px helped the balance a lot (91/9 → 67/33) and the
AUROC barely (0.524 → 0.545). Patch size was never the binding constraint; the
unsupervised objective is.

## What would actually work

1. **Project onto the aggregation axis before fitting.** Feed the HMM the LDA
   direction (or the top 2–3 discriminative directions) instead of the raw PCs.
   The HMM then contributes temporal coherence and transition dynamics on top of
   a signal already aligned with the question. This is supervised by neighbor
   counts, so the AUROC stops being an independent check — report it as
   "temporally-smoothed aggregation state," not as a discovery.
2. **Raise k and merge.** Fit k=6–8 unsupervised, then group states by their
   crowding profile. Keeps the discovery framing; the aggregation axis has a
   chance to claim its own state once the size axis is no longer forced to
   explain everything with two.
3. Feeding `nuclei_neighbors_30px` in as a model feature works but is the least
   interesting option — it hands the model the answer.

Option 2 preserves the unsupervised claim and is cheap (embeddings are already
on disk; only the fit re-runs). Recommended next step.
