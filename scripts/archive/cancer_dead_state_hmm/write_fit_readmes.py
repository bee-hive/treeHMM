"""Write a human-readable README.md into every dead-state fit directory.

Each fit under {output_base_dir}/fits/{arm}/{variant}/k{K}/ gets a
self-contained description: model variant and whether the absorbing
constraint was on, which features fed it and what they mean, how cells were
filtered, how it was initialized and fitted, which state was picked as the
dead candidate and by what rule, what that state looks like, and how it
scored.  A fit directory should be readable on its own, later, without
reconstructing the config.

Everything is read back from what the fit already wrote, so the README
cannot drift from the fit it describes.

Usage:
    conda run -n OccidentAnalysis python write_fit_readmes.py
"""

import os
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

out_base_dir = cfg["output_base_dir"]
fits_dir = os.path.join(out_base_dir, "fits")
crop_ids = cfg["crop_ids"]

cell_index = pd.read_csv(os.path.join(fits_dir, "cell_index.csv"))
active = np.load(os.path.join(fits_dir, "active_mask.npy"))
n_cells, n_cell_frames, T = len(cell_index), int(active.sum()), active.shape[0]

evaluation = None
eval_path = os.path.join(fits_dir, "evaluation.csv")
if os.path.exists(eval_path):
    evaluation = pd.read_csv(eval_path)

FEATURE_BLURBS = {
    "area": "phase-mask pixel count",
    "circularity": "4*pi*area / perimeter^2 of the largest component",
    "velocity": "centroid displacement, px/frame",
    "t_cell_neighbors_20px": "T cells within 20 px of the phase centroid",
    "dilated_t_cell_neighbors": "T cells touching the mask dilated by disk(2)",
    "d_area_frac": "(area_t - area_prev) / area_prev, vs the previous ACTIVE frame",
    "d_circularity": "circularity_t - circularity_prev",
    "win_std_log_area": "trailing 5-frame rolling std of log(area)",
    "win_std_circularity": "trailing 5-frame rolling std of circularity",
    "win_std_displacement": "trailing 5-frame rolling std of per-frame displacement",
    "area_over_running_max":
        "**memory** — area_t / max(area over this cell's active frames so far). "
        "Drops when the cell shrinks and never recovers, so it marks a cell "
        "that has *already* had its event",
    "running_max_abs_d_circularity":
        "**memory** — running max of abs(d_circularity). Non-decreasing along "
        "a track: once the cell has rounded up sharply, it stays flagged",
    "phase_mean":
        "**phase intensity** — mean phase inside the mask; apoptotic cells "
        "round up and go bright/refractile before lysing",
    "phase_std":
        "**phase intensity** — within-mask SD, i.e. granularity / blebbing texture",
    "phase_contrast":
        "**phase intensity** — mask mean minus the mean of a disk(3) ring just "
        "outside it: the refractile halo, background-corrected",
    "rfp_mean":
        "**RFP intensity** — mean RFP inside the mask (only cancer cells carry "
        "RFP nuclei)",
    "rfp_cv":
        "**RFP intensity** — within-mask SD / mean; rises as the nucleus "
        "condenses and fragments into bright puncta on a dark background",
    "rfp_concentration":
        "**RFP intensity** — share of the mask's total RFP held by its "
        "brightest 10% of pixels; a direct condensation measure, scale-free "
        "in absolute intensity",
}


def markdown_table(frame, floatfmt="{:.3f}"):
    """Render a DataFrame as a GitHub markdown table."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + rows)


def write_readme(arm, variant, k):
    """Compose and write one fit's README.md."""
    fit_dir = os.path.join(fits_dir, arm, variant, f"k{k}")
    with open(os.path.join(fit_dir, "fit_summary.yml")) as fh:
        info = yaml.safe_load(fh)
    profile = pd.read_csv(os.path.join(fit_dir, "state_profile.csv"))

    features = info["features"]
    num_lags = info["num_lags"]
    dead_state = info["dead_state"]
    absorbing_state = info["absorbing_state"]
    trans = np.array(info["transition_matrix"])
    initial = np.array(info["initial_probs"])
    d = info["emission_dim"]
    per_state = d + d * (d + 1) // 2 + (d * d if num_lags == 1 else 0)

    L = []
    L.append(f"# Fit: `{arm}` [{variant}] at k={k}")
    L.append("")
    L.append(f"Dead-state HMM over cancer cells, part of the "
             f"`cancer_dead_state_hmm` experiment set. This directory holds "
             f"one fit: arm **{arm}**, **{k} states**, "
             f"**num_lags={num_lags}**, transition constraint "
             f"**{variant}**.")
    L.append("")
    L.append("Unlike `cancer_death_hmm`, which targeted the single-frame "
             "death *event*, this experiment set targets the dead "
             "*condition* — a state a cell enters and does not leave.")
    L.append("")

    L.append("## Model")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append("| implementation | `scripts/cancer_dead_state_hmm/absorbing.py` "
             "-> `AbsorbingTARHMM`, a subclass of `models/tarhmm.py` -> "
             "`tARHMM` (tree AR-HMM on Dynamax/JAX) |")
    L.append(f"| latent states | {k} |")
    L.append(f"| AR lag order | {num_lags} |")
    L.append("| emission model | Gaussian AR(1): `x_t ~ N(A_j x_(t-1) + b_j, "
             "S_j)` given state j, full covariance, shared across cells |"
             if num_lags == 1 else
             "| emission model | Gaussian, bias-only: `x_t ~ N(b_j, S_j)` "
             "given state j, full covariance, shared across cells |")
    L.append(f"| emission dim | {d} |")
    L.append(f"| emission params per state | ~{per_state} (A + b + Sigma) |")
    L.append("| tree / division kernel | **off** — every cell is its own "
             "parent, so `P_div` is unused and this is a per-cell chain |")
    if absorbing_state is None:
        L.append("| transition constraint | **none** (free). This is the "
                 "control arm: any persistence the dead state shows was "
                 "learned from the data, not imposed. |")
    else:
        L.append(f"| transition constraint | **absorbing**. State "
                 f"{absorbing_state} has its transition row pinned to "
                 f"one-hot self-transition, so `P[{absorbing_state}, "
                 f"{absorbing_state}] = 1` and dead -> alive is impossible. "
                 f"Pinned at initialization and re-pinned after every M-step; "
                 f"all other rows are estimated normally. |")
    L.append("")
    if absorbing_state is not None:
        L.append(f"The absorbing state is designated **by index** "
                 f"(state {absorbing_state} = k-1), not by phenotype — EM "
                 f"decides what lands there. Whether it holds the death "
                 f"phenotype is a separate question, answered by the "
                 f"label-blind score below: for this fit the dead candidate "
                 f"is state {dead_state}, so the two "
                 + ("**agree**." if info["dead_state_is_absorbing"]
                    else "**disagree** — the constrained state is not the one "
                         "carrying the death signature, which is itself an "
                         "informative negative result."))
        L.append("")
        L.append("The initial distribution stays free, so cells that are "
                 "already dead when their track begins are representable. A "
                 "purely transition-based detector cannot express that case "
                 "at all — it is why the already-dead annotated cell scored "
                 "zero in every `cancer_death_hmm` arm.")
        L.append("")

    L.append("## Input features")
    L.append("")
    L.append(f"{len(features)} dimensions, all scalar (no image embeddings).")
    L.append("")
    L.append("| feature | definition |")
    L.append("|---|---|")
    for name in features:
        L.append(f"| `{name}` | {FEATURE_BLURBS.get(name, '')} |")
    L.append("")
    L.append("Features were z-scored jointly over all active cell-frames "
             "before fitting. Intensity features come from the TrackingCrops "
             "`crop.tiff` (last axis: 0 = RFP, 1 = phase), normalized once per "
             "crop over the whole stack with 1st/99th percentiles — per-frame "
             "normalization would let illumination wobble become the dominant "
             "axis.")
    L.append("")

    L.append("## Data")
    L.append("")
    L.append("- **Cells:** cancer cells defined by their PHASE masks "
             "(channel 1 of the type-separated CVAT tracks). One CVAT cancer "
             "track = one cell, which is also the ID space the ground-truth "
             "death annotations use.")
    L.append(f"- **Crops:** {len(crop_ids)} (`" + "`, `".join(crop_ids) + "`)")
    L.append(f"- **Cells retained:** {n_cells} after the `min_t="
             f"{cfg['min_t']}` filter")
    L.append(f"- **Inferred cell-frames:** {n_cell_frames} over T={T} frames")
    L.append("- **Warmup:** the first active frame of every cell is masked "
             "out (no predecessor for the AR input; deltas and memory "
             "features are undefined there)")
    L.append("- **No downsampling or class balancing.** Targeting the dead "
             "condition rather than the death event already moves the balance "
             "from ~0.5% to ~23% of cell-frames, measured against the "
             "fully-annotated B8_t50 crop.")
    L.append("")
    L.append("Every arm and variant is fitted over the *same* cells with the "
             "same filter and warmup, so differences are attributable to "
             "features and the constraint alone.")
    L.append("")

    L.append("## Fitting")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| algorithm | EM (`fit_em`), {cfg['num_em_iters']} iterations, "
             f"no early stopping |")
    L.append(f"| initialization | `{cfg['init_method']}` over active "
             f"cell-frames only |")
    L.append(f"| initial transition matrix | sticky, diagonal "
             f"{cfg['init_stickiness']}"
             + (f", with row {absorbing_state} pinned"
                if absorbing_state is not None else "") + " |")
    L.append(f"| seeds tried | {cfg['em_seeds']} |")
    L.append(f"| seed kept | **{info['best_seed']}** (best final log prob) |")
    L.append(f"| log prob | {info['log_prob_first']:.1f} -> "
             f"**{info['log_prob_final']:.1f}** |")
    L.append("")
    L.append("Log probabilities are comparable across `k` **within** an arm "
             "and variant, but not across arms — different feature sets are "
             "different observation spaces.")
    L.append("")

    L.append("## Dead-state selection")
    L.append("")
    L.append("Chosen **after** fitting by a rule that never sees the "
             "ground-truth annotations, applied identically to every arm and "
             "variant:")
    L.append("")
    L.append("```")
    L.append("transition_score  = z(mean d_circularity) - z(mean d_area_frac)")
    L.append("persistence_score = mean of z(-mean area_over_running_max),")
    L.append("                            z(+mean running_max_abs_d_circularity)")
    L.append("dead_score        = transition_score + persistence_score")
    L.append("```")
    L.append("")
    L.append("`z(.)` standardizes across this fit's states. The transition "
             "term is the acute signature from the ground-truth notes (area "
             "drops, circularity rises). The persistence term replaces "
             "`cancer_death_hmm`'s instability term: for a *dead* state the "
             "diagnostic is not \"this frame is changing\" but \"this cell has "
             "already shrunk and rounded and not recovered\". Scores use the "
             "full 18-feature diagnostic set regardless of which features fed "
             "the model.")
    L.append("")
    L.append(f"**Selected: state {dead_state}** — occupancy "
             f"{info['dead_state_occupancy']:.3f}, self-transition "
             f"{info['dead_state_self_transition']:.3f}.")
    L.append("")

    L.append("## State profiles")
    L.append("")
    L.append("Mean of each diagnostic feature within each state "
             "(unstandardized units).")
    L.append("")
    cols = ["state", "n_cell_frames", "occupancy", "area", "circularity",
            "area_over_running_max", "running_max_abs_d_circularity",
            "phase_mean", "rfp_cv", "rfp_concentration",
            "dilated_t_cell_neighbors", "dead_score"]
    cols = [c for c in cols if c in profile.columns]
    L.append(markdown_table(profile[cols]))
    L.append("")

    L.append("## Learned parameters")
    L.append("")
    L.append("Transition matrix:")
    L.append("")
    tdf = pd.DataFrame(trans, columns=[f"-> {j}" for j in range(k)])
    tdf.insert(0, "from", [f"**{i}**" for i in range(k)])
    L.append(markdown_table(tdf))
    L.append("")
    L.append("Initial distribution (what a cell's first inferred frame is "
             "drawn from — the only route into an absorbing state for a cell "
             "already dead when its track begins):")
    L.append("")
    idf = pd.DataFrame([initial], columns=[f"state {j}" for j in range(k)])
    L.append(markdown_table(idf))
    L.append("")

    if evaluation is not None:
        row = evaluation[(evaluation["arm"] == arm)
                         & (evaluation["variant"] == variant)
                         & (evaluation["k"] == k)]
        if len(row):
            r = row.iloc[0]
            L.append("## Evaluation")
            L.append("")
            L.append("| check | result |")
            L.append("|---|---|")
            L.append(f"| annotated deaths entered (frame-level) | "
                     f"**{r['entry_hits']}** |")
            L.append(f"| chance rate (permutation null) | "
                     f"{r['chance_entry_recall']:.3f} "
                     f"(p = {r['entry_recall_p']:.4f}) |")
            L.append(f"| persistence after entry | "
                     f"{r['mean_persistence_after_entry']}"
                     + (" (1.0 by construction for an absorbing fit)"
                        if absorbing_state is not None else
                        " (learned, not imposed)") + " |")
            L.append(f"| already-dead cell captured | "
                     f"{r['already_dead_captured']} |")
            L.append(f"| cell-level events hit | {r['cell_level_hits']} |")
            L.append(f"| cells ever entering the dead state | "
                     f"{r['frac_cells_ever_dead']:.3f} |")
            L.append(f"| entries per entered cell | "
                     f"{r['entries_per_entered_cell']} "
                     f"(1.0 is the ideal for an absorbing state) |")
            L.append(f"| division false positives | "
                     f"{r['division_false_positives']} "
                     f"(chance {r['chance_division_hits']:.2f}) |")
            L.append(f"| T-cell AUROC | {r['tcell_auroc']:.3f} — independent; "
                     f"no arm in this set uses a T-cell feature |")
            L.append("")
            L.append("The permutation null relocates each cell's entries "
                     "uniformly among that cell's own active frames, "
                     "preserving entries-per-cell and therefore occupancy. "
                     "With only 3 frame-level annotated events across many "
                     "fits, **`p` is ranking information, not a significance "
                     "test.**")
            L.append("")
            L.append("Mitotic rounding produces the same shape signature as "
                     "death, so the annotated cancer divisions are known "
                     "non-death changes: entries landing on one are confirmed "
                     "false positives.")
            L.append("")

    L.append("## Files here")
    L.append("")
    L.append("| file | contents |")
    L.append("|---|---|")
    L.append("| `state_assignments.npy` | `(T, n_cells)` argmax of the "
             "smoothed posterior |")
    L.append("| `smoothed_probs.npy` | `(T, n_cells, k)` smoothed posterior |")
    L.append("| `state_profile.csv` | per-state occupancy, diagnostic means, "
             "selection scores |")
    L.append("| `fit_summary.yml` | machine-readable version of the above |")
    L.append("| `states.png` | state heatmap + learned transition matrix |")
    L.append("| `*_state_overlay.mp4` | one per crop, if rendered |")
    L.append("")
    L.append("Column ordering matches `../../../cell_index.csv`, which maps "
             "each column to its `(crop, CVAT cell id)`.")
    L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```bash")
    L.append("cd scripts/cancer_dead_state_hmm")
    L.append(f"conda run -n treeHMM_env python fit_dead_state_hmm.py {arm}")
    L.append("conda run -n OccidentAnalysis python evaluate_dead_states.py")
    L.append("conda run -n OccidentAnalysis python write_fit_readmes.py")
    L.append("```")
    L.append("")
    L.append("Settings come from `scripts/cancer_dead_state_hmm/config.yml`; "
             "`num_states_sweep`, `em_seeds` and `absorbing_variants` there "
             "determine which sub-directories exist.")

    path = os.path.join(fit_dir, "README.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


print("=" * 60)
print("Writing per-fit READMEs")
print("=" * 60)
summary = pd.read_csv(os.path.join(fits_dir, "summary.csv"))
for r in summary.itertuples():
    print(f"  {write_readme(r.arm, r.variant, int(r.k))}")
print("\n" + "=" * 60)
print(f"Wrote {len(summary)} READMEs")
print("=" * 60)
