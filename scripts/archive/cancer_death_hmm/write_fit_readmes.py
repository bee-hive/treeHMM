"""Write a human-readable README.md into every fit directory.

Each fit under {output_base_dir}/fits/{arm}/k{K}/ gets a self-contained
description: which model variant was used, which features fed it, how the
cells were filtered, how it was initialized and fitted, which state was
picked as the death candidate and by what rule, what that state looks like,
and how it scored.  The point is that a fit directory should be readable on
its own, months later, without reconstructing the config.

Everything is read back from what the fit already wrote (fit_summary.yml,
state_profile.csv, evaluation.csv), so the README cannot drift from the fit
it describes.

Usage (any env with numpy/pandas/yaml):
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
n_cells = len(cell_index)
n_cell_frames = int(active.sum())
T = active.shape[0]

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
}


def describe_feature(name, n_dino):
    """One-line description for a model feature."""
    if name.startswith("dino_pc_"):
        return (f"principal component {name.rsplit('_', 1)[1]} of the DINOv2 "
                f"embedding ({n_dino} kept, whitened)")
    return FEATURE_BLURBS.get(name, "")


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


def write_readme(arm, k):
    """Compose and write one fit's README.md."""
    fit_dir = os.path.join(fits_dir, arm, f"k{k}")
    with open(os.path.join(fit_dir, "fit_summary.yml")) as fh:
        info = yaml.safe_load(fh)
    profile = pd.read_csv(os.path.join(fit_dir, "state_profile.csv"))

    features = info["features"]
    dino_cols = [f for f in features if f.startswith("dino_pc_")]
    num_lags = info["num_lags"]
    death_state = info["death_state"]
    trans = np.array(info["transition_matrix"])

    # No raw '|' here -- these strings land inside markdown table cells.
    emission_desc = (
        "Gaussian AR(1): `x_t ~ N(A_j x_(t-1) + b_j, S_j)` given state j, "
        "full covariance, all parameters shared across cells"
        if num_lags == 1 else
        "Gaussian, bias-only (no autoregression): `x_t ~ N(b_j, S_j)` given "
        "state j, full covariance, all parameters shared across cells")

    d = info["emission_dim"]
    per_state = d + d * (d + 1) // 2 + (d * d if num_lags == 1 else 0)

    lines = []
    lines.append(f"# Fit: `{arm}` at k={k}")
    lines.append("")
    lines.append(f"Death-state HMM over cancer cells, part of the "
                 f"`cancer_death_hmm` experiment set. This directory holds one "
                 f"fit: arm **{arm}**, **{k} states**, **num_lags={num_lags}**.")
    lines.append("")

    lines.append("## Model")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append("| implementation | `models/tarhmm.py` -> `tARHMM` "
                 "(tree AR-HMM on Dynamax/JAX) |")
    lines.append(f"| latent states | {k} |")
    lines.append(f"| AR lag order | {num_lags} |")
    lines.append(f"| emission model | {emission_desc} |")
    lines.append(f"| emission dim | {info['emission_dim']} |")
    lines.append(f"| emission params per state | ~{per_state} "
                 f"(A + b + Sigma) |")
    lines.append("| tree / division kernel | **off** — every cell is its own "
                 "parent, `is_division_mask` all False, so `P_div` is unused "
                 "and this is an ordinary per-cell chain |")
    lines.append("| transition matrix | free (unconstrained), one shared "
                 "`P_std` across all cells |")
    lines.append("")

    lines.append("## Input features")
    lines.append("")
    if dino_cols:
        lines.append(f"{len(features)} dimensions: "
                     f"{len(features) - len(dino_cols)} scalar + "
                     f"{len(dino_cols)} DINOv2 PCs.")
    else:
        lines.append(f"{len(features)} dimensions, all scalar "
                     f"(no image embeddings).")
    lines.append("")
    # The DINO PCs collapse into a single row rather than one row per component.
    rows = [(f"`{name}`", describe_feature(name, len(dino_cols)))
            for name in features if not name.startswith("dino_pc_")]
    if dino_cols:
        rows.append((f"`dino_pc_0` … `dino_pc_{len(dino_cols) - 1}`",
                     f"top {len(dino_cols)} whitened principal components of the "
                     f"768-d DINOv2 embedding of a "
                     f"{cfg['dino_patch_px']}x{cfg['dino_patch_px']} px patch "
                     f"centered on the cell"))
    lines.append("| feature | definition |")
    lines.append("|---|---|")
    for label, desc in rows:
        lines.append(f"| {label} | {desc} |")
    lines.append("")
    lines.append("Features were z-scored jointly over all active cell-frames "
                 "before fitting, so no dimension dominates by unit alone.")
    lines.append("")

    lines.append("## Data")
    lines.append("")
    lines.append(f"- **Cells:** cancer cells defined by their PHASE masks "
                 f"(channel 1 of the type-separated CVAT tracks). One CVAT "
                 f"cancer track = one cell.")
    lines.append(f"- **Crops:** {len(crop_ids)} (`" + "`, `".join(crop_ids) + "`)")
    lines.append(f"- **Cells retained:** {n_cells} after the `min_t="
                 f"{cfg['min_t']}` filter")
    lines.append(f"- **Inferred cell-frames:** {n_cell_frames} over T={T} frames")
    lines.append(f"- **Warmup:** the first active frame of every cell is "
                 f"masked out (it has no predecessor for the AR input, and "
                 f"deltas/window statistics are undefined there)")
    lines.append("")
    lines.append("All arms in this experiment set were fitted over the *same* "
                 "cells with the same filter and warmup, so differences "
                 "between arms are attributable to features and lag order "
                 "alone.")
    lines.append("")

    lines.append("## Fitting")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| algorithm | EM (`tARHMM.fit_em`), {cfg['num_em_iters']} "
                 f"iterations, no early stopping |")
    lines.append(f"| initialization | `{cfg['init_method']}` over active "
                 f"cell-frames only |")
    lines.append(f"| initial transition matrix | sticky, diagonal "
                 f"{cfg['init_stickiness']} |")
    lines.append(f"| seeds tried | {cfg['em_seeds']} |")
    lines.append(f"| seed kept | **{info['best_seed']}** (best final log prob) |")
    lines.append(f"| log prob | {info['log_prob_first']:.1f} -> "
                 f"**{info['log_prob_final']:.1f}** |")
    lines.append("")
    lines.append("Multiple seeds are run because the state of interest is "
                 "rare, which makes the outcome unusually sensitive to which "
                 "local optimum EM lands in.")
    lines.append("")

    lines.append("## Death-state selection")
    lines.append("")
    lines.append("Chosen **after** fitting by a rule that never sees the "
                 "ground-truth annotations, applied identically to every arm:")
    lines.append("")
    lines.append("```")
    lines.append("transition_score  =  z(mean d_circularity) - z(mean d_area_frac)")
    lines.append("instability_score =  mean of z(mean win_std_{log_area, circularity, displacement})")
    lines.append("death_score       =  transition_score + instability_score")
    lines.append("```")
    lines.append("")
    lines.append("`z(.)` standardizes across this fit's states. The transition "
                 "term is the acute signature from the ground-truth notes "
                 "(area drops, circularity rises); the instability term is the "
                 "elevated variance that follows. Scores are computed on the "
                 "full 10-feature diagnostic set regardless of which features "
                 "fed the model.")
    lines.append("")
    lines.append(f"**Selected: state {death_state}** — occupancy "
                 f"{info['death_state_occupancy']:.3f}, self-transition "
                 f"{info['death_state_self_transition']:.3f}.")
    lines.append("")

    lines.append("## State profiles")
    lines.append("")
    lines.append("Mean of each diagnostic feature within each state "
                 "(unstandardized units).")
    lines.append("")
    show_cols = ["state", "n_cell_frames", "occupancy", "area", "circularity",
                 "d_area_frac", "d_circularity", "win_std_circularity",
                 "dilated_t_cell_neighbors", "death_score"]
    show_cols = [c for c in show_cols if c in profile.columns]
    lines.append(markdown_table(profile[show_cols]))
    lines.append("")

    lines.append("## Learned transition matrix")
    lines.append("")
    trans_df = pd.DataFrame(trans, columns=[f"-> {j}" for j in range(k)])
    trans_df.insert(0, "from", [f"**{i}**" for i in range(k)])
    lines.append(markdown_table(trans_df))
    lines.append("")

    if evaluation is not None:
        row = evaluation[(evaluation["arm"] == arm) & (evaluation["k"] == k)]
        if len(row):
            r = row.iloc[0]
            lines.append("## Evaluation")
            lines.append("")
            lines.append("| check | result |")
            lines.append("|---|---|")
            lines.append(f"| annotated events hit (frame-level) | "
                         f"**{r['frame_hits']}** |")
            lines.append(f"| chance rate for that (permutation null) | "
                         f"{r['chance_frame_recall']:.3f} (p = "
                         f"{r['frame_recall_p']:.4f}) |")
            lines.append(f"| cell-level events hit | {r['cell_hits']} |")
            lines.append(f"| death-state entries per entered cell | "
                         f"{r['entries_per_entered_cell']:.2f} "
                         f"(1.0 = entered once, as a real death state should) |")
            lines.append(f"| division false positives | "
                         f"{r['division_false_positives']} "
                         f"(chance {r['chance_division_hits']:.2f}) |")
            lines.append(f"| T-cell AUROC | {r['tcell_auroc']:.3f}"
                         + (" — **circular**, this arm uses a T-cell feature"
                            if r["tcell_auroc_is_circular"] else
                            " — independent, this arm has no T-cell feature")
                         + " |")
            lines.append("")
            lines.append("The permutation null relocates each cell's entries "
                         "uniformly among that cell's own active frames, "
                         "preserving entry count per cell and therefore "
                         "occupancy. With only 3 frame-level annotated events "
                         "and 30 (arm, k) combinations scored, **`p` is "
                         "ranking information, not a significance test** — no "
                         "arm can clear a multiplicity-corrected bar at this "
                         "annotation count.")
            lines.append("")
            lines.append("Mitotic rounding produces the same shape signature "
                         "as death, so the annotated cancer divisions serve as "
                         "known non-death shape changes: entries landing on "
                         "one are confirmed false positives.")
            lines.append("")

    lines.append("## Files here")
    lines.append("")
    lines.append("| file | contents |")
    lines.append("|---|---|")
    lines.append("| `state_assignments.npy` | `(T, n_cells)` argmax of the "
                 "smoothed posterior |")
    lines.append("| `smoothed_probs.npy` | `(T, n_cells, k)` smoothed "
                 "posterior |")
    lines.append("| `state_profile.csv` | per-state occupancy and diagnostic "
                 "means, with the selection scores |")
    lines.append("| `fit_summary.yml` | machine-readable version of the above |")
    lines.append("| `states.png` | state heatmap + learned transition matrix |")
    lines.append("| `*_state_overlay.mp4` | one per crop; cells tinted by "
                 "state over the phase image |")
    lines.append("")
    lines.append("Column ordering matches `../../cell_index.csv`, which maps "
                 "each column to its `(crop, CVAT cell id)`.")
    lines.append("")

    lines.append("## Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append("cd scripts/cancer_death_hmm")
    lines.append(f"conda run -n treeHMM_env python fit_death_hmm.py {arm}")
    lines.append("conda run -n OccidentAnalysis python evaluate_death_states.py")
    lines.append(f"conda run -n OccidentAnalysis python "
                 f"create_state_overlay_videos.py {arm}:{k}")
    lines.append("```")
    lines.append("")
    lines.append("Fit settings come from `scripts/cancer_death_hmm/config.yml`; "
                 "the `k` sweep and seed list there determine which "
                 "sub-directories exist.")

    path = os.path.join(fit_dir, "README.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


print("=" * 60)
print("Writing per-fit READMEs")
print("=" * 60)

summary = pd.read_csv(os.path.join(fits_dir, "summary.csv"))
written = 0
for r in summary.itertuples():
    path = write_readme(r.arm, int(r.k))
    print(f"  {path}")
    written += 1

print("\n" + "=" * 60)
print(f"Wrote {written} READMEs")
print("=" * 60)
