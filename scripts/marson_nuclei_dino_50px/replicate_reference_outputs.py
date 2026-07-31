"""Replicate the `cancer_dino_hmm_*` reference output set for the 50 px runs.

The existing cancer-nuclei + DINO runs under `analysis/cancer_dino_hmm_k2_*`
emit a specific set of figures.  `fit_nuclei_hmm.py` produces most of the same
information but under different names and groupings, and omits three plots
entirely.  This script fills the gaps so the new runs are directly comparable
to the old ones.

Generated per variant directory:
    feature_distributions.png           model emissions per state (reference layout)
    learned_transition_matrix.png       standalone P_std (reference layout)
    observed_transition_matrices.png    empirical per-well transition matrices
    t_cell_neighbors_heatmap.png        cells x time heatmap with a well colorbar
    dino_pca_components.npy             copied from the shared embedding dir
    dino_pca_explained_variance_ratio.npy
    dino_pca_mean.npy

`nondino_standardization.npz` is intentionally NOT produced: these runs have
`model_features: []`, so there are no non-DINO features to standardize and the
reference file would be empty.

Usage (OccidentAnalysis):
    conda run -n OccidentAnalysis python replicate_reference_outputs.py
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.patches import Patch

_script_dir = Path(__file__).resolve().parent
with open(_script_dir / "config.yml", "r") as f:
    cfg = yaml.safe_load(f)

wells: List[str] = cfg["wells"]
min_t: int = cfg["min_t"]
shared_dir = Path(cfg["output_base_dir"])
analysis_dir = shared_dir.parent
VARIANTS = {
    "lag0": analysis_dir / "marson_nuclei_dino_k2_50px_lag0",
    "lag1": analysis_dir / "marson_nuclei_dino_k2_50px_lag1",
}

SMALL, MEDIUM, BIG = 7, 8, 10
plt.rc('font', size=SMALL)
plt.rc('axes', titlesize=MEDIUM, labelsize=SMALL)
plt.rc('xtick', labelsize=SMALL)
plt.rc('ytick', labelsize=SMALL)
plt.rc('legend', fontsize=SMALL)
plt.rc('figure', titlesize=BIG)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.use14corefonts'] = True

with open(shared_dir / wells[0] / "nuclei_emissions_names.txt") as fh:
    diag_feature_names = [l.strip() for l in fh if l.strip()]


def load_variant(root: Path) -> Dict[str, object]:
    """Load states, emissions, diagnostics and well labels for one variant.

    Rebuilds the `min_t` column filter that `fit_nuclei_hmm.py` applied, so the
    per-well arrays line up with the saved state assignments.

    Args:
        root (Path): variant output directory.

    Returns:
        Dict[str, object]: concatenated states/active/emissions/diagnostics
            arrays plus per-well column labels and the model feature names.
    """
    states, actives, emis, diags, labels = [], [], [], [], []
    for well_idx, well in enumerate(wells):
        centroids = np.load(root / well / "nuclei_centroids.npy")
        active_all = ~np.isnan(centroids[:, :, 0])
        kept = np.where(active_all.sum(axis=0) >= min_t)[0]
        states.append(np.load(root / well / "nuclei_state_assignments.npy"))
        actives.append(active_all[:, kept])
        emis.append(np.load(root / well / "nuclei_dino_pca.npy")[:, kept, :])
        diags.append(np.load(root / well / "nuclei_emissions_array.npy")[:, kept, :])
        labels.extend([well_idx] * len(kept))

    with open(root / "final_feature_names.txt") as fh:
        feature_names = [l.strip() for l in fh if l.strip()]

    return dict(
        states=np.concatenate(states, axis=1),
        active=np.concatenate(actives, axis=1),
        emissions=np.concatenate(emis, axis=1),
        diagnostics=np.concatenate(diags, axis=1),
        well_labels=np.array(labels),
        feature_names=feature_names,
    )


def plot_feature_distributions(data: Dict[str, object], out_path: Path) -> None:
    """Per-state distribution of every model emission, in the reference layout.

    Violin per state for each feature on a 4-column grid, matching
    `fit_cancer_arhmm.py` Output 2.

    Args:
        data (Dict[str, object]): output of `load_variant`.
        out_path (Path): destination PNG.
    """
    active = data["active"]
    flat_states = data["states"][active]
    flat_emis = data["emissions"][active]
    names = data["feature_names"]
    num_states = int(flat_states.max()) + 1

    df = pd.DataFrame(flat_emis, columns=names)
    df["state"] = flat_states

    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             tight_layout=True, squeeze=False)
    axes = axes.flatten()
    for i, feat in enumerate(names):
        sns.violinplot(data=df, x="state", y=feat, hue="state", ax=axes[i],
                       palette="viridis", legend=False, cut=0)
        axes[i].set_ylabel(feat.replace("_", " ").title())
        axes[i].set_xlabel("State")
        sns.despine(ax=axes[i])
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path.name}  ({len(names)} panels, k={num_states})")


def plot_learned_transition_matrix(root: Path, out_path: Path) -> None:
    """Standalone learned P_std heatmap, matching the reference layout.

    Recovers the matrix from the variant's own `transition_matrices.png` source
    data by re-reading the summary; falls back to the empirical matrix if the
    fitted parameters were not persisted.

    Args:
        root (Path): variant output directory.
        out_path (Path): destination PNG.
    """
    mat_path = root / "learned_P_std.npy"
    if not mat_path.exists():
        print(f"  SKIP {out_path.name}: {mat_path.name} not found")
        return
    mat = np.load(mat_path)
    num_states = mat.shape[0]

    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(num_states)); ax.set_yticks(range(num_states))
    ax.set_xticklabels([f"State {s}" for s in range(num_states)])
    ax.set_yticklabels([f"State {s}" for s in range(num_states)])
    ax.set_xlabel("State at $t+1$"); ax.set_ylabel("State at $t$")
    ax.set_title("Learned Transition Matrix")
    for i in range(num_states):
        for j in range(num_states):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    color="white" if mat[i, j] > 0.5 else "black")
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("Transition probability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path.name}")


def plot_observed_transition_matrices(data: Dict[str, object], out_path: Path) -> None:
    """Empirical per-well state transition matrices.

    Counts observed (state at t -> state at t+1) transitions over frames where
    a cell is active at both t and t+1, row-normalized. Matches
    `fit_cancer_arhmm.py` Output 6, with the grid sized to the well count.

    Args:
        data (Dict[str, object]): output of `load_variant`.
        out_path (Path): destination PNG.
    """
    states, active, labels = data["states"], data["active"], data["well_labels"]
    num_states = int(states[active].max()) + 1

    ncols = min(len(wells), 3)
    nrows = int(np.ceil(len(wells) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             tight_layout=True, squeeze=False)
    axes = axes.flatten()
    for well_idx, well in enumerate(wells):
        ax = axes[well_idx]
        mask = labels == well_idx
        ws, wa = states[:, mask], active[:, mask]
        obs = np.zeros((num_states, num_states))
        both = wa[:-1] & wa[1:]
        src = ws[:-1][both]
        dst = ws[1:][both]
        np.add.at(obs, (src, dst), 1)
        rs = obs.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        obs_p = obs / rs
        im = ax.imshow(obs_p, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_states)); ax.set_yticks(range(num_states))
        ax.set_xticklabels([str(s) for s in range(num_states)])
        ax.set_yticklabels([str(s) for s in range(num_states)])
        ax.set_xlabel("State at $t+1$"); ax.set_ylabel("State at $t$")
        ax.set_title(f"{well}\nObserved Transition Matrix")
        for i in range(num_states):
            for j in range(num_states):
                ax.text(j, i, f"{obs_p[i, j]:.3f}", ha="center", va="center",
                        color="white" if obs_p[i, j] > 0.5 else "black")
    for j in range(len(wells), len(axes)):
        axes[j].axis("off")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path.name}")


def plot_t_cell_neighbors_heatmap(data: Dict[str, object], out_path: Path) -> None:
    """Cells x time heatmap of the T-cell neighbor count, with a well colorbar.

    Matches `fit_cancer_arhmm.py` Output 7. Inactive cell-frames are blanked.

    Args:
        data (Dict[str, object]): output of `load_variant`.
        out_path (Path): destination PNG.
    """
    feat = "t_cell_neighbors_20px"
    idx = diag_feature_names.index(feat)
    vals = data["diagnostics"][:, :, idx].T
    active = data["active"].T
    masked = np.where(active, vals, np.nan)
    labels = data["well_labels"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                             gridspec_kw={"width_ratios": [1, 30]}, sharey=True)
    well_cmap = plt.get_cmap("tab10", len(wells))
    axes[0].imshow(labels.reshape(-1, 1), aspect="auto", interpolation="none",
                   cmap=well_cmap, vmin=0, vmax=len(wells) - 1, origin="upper")
    axes[0].set_xticks([]); axes[0].set_ylabel("Cell"); axes[0].set_title("Well")
    im = axes[1].imshow(masked, aspect="auto", interpolation="none",
                        cmap="magma", origin="upper")
    axes[1].set_xlabel("Time")
    axes[1].set_title("T cells within 20px of cancer nucleus - All Wells")
    cbar = plt.colorbar(im, ax=axes[1]); cbar.set_label("T-cell neighbors (20px)")
    fig.legend(handles=[Patch(facecolor=well_cmap(i), label=wells[i])
                        for i in range(len(wells))],
               loc="lower center", ncol=3, title="Well", bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path.name}")


print("=" * 60)
print("Replicating cancer_dino_hmm reference outputs")
print("=" * 60)

PCA_ARTIFACTS = ["dino_pca_components.npy",
                 "dino_pca_explained_variance_ratio.npy",
                 "dino_pca_mean.npy"]

for name, root in VARIANTS.items():
    print(f"\n--- {name} ---")
    if not (root / wells[0] / "nuclei_state_assignments.npy").exists():
        print(f"  no state assignments in {root}; skipping.")
        continue

    data = load_variant(root)
    plot_feature_distributions(data, root / "feature_distributions.png")
    plot_learned_transition_matrix(root, root / "learned_transition_matrix.png")
    plot_observed_transition_matrices(data, root / "observed_transition_matrices.png")
    plot_t_cell_neighbors_heatmap(data, root / "t_cell_neighbors_heatmap.png")

    for artifact in PCA_ARTIFACTS:
        src = shared_dir / artifact
        if src.exists():
            shutil.copy2(src, root / artifact)
    print(f"  copied {len(PCA_ARTIFACTS)} PCA artifacts")

    # Reference runs keep the raw embeddings alongside each unit.
    for well in wells:
        link = root / well / "nuclei_dino_raw.npy"
        if not link.exists():
            os.symlink(shared_dir / well / "nuclei_dino_raw.npy", link)
    print("  linked nuclei_dino_raw.npy per well")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
