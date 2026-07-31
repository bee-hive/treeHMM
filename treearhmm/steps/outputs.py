"""Step `outputs`: the four base outputs every run produces.

    overlays/{crop}_state_overlay.mp4   each cancer cell tinted by its inferred
                                        state, over the phase image
    feature_distributions.png           distribution of every computed feature
    state_feature_summary.csv           within each state
    transition_matrix.csv / .png        learned state transition probabilities
    initial_distribution.csv
    state_assignments.csv               per cell-frame state and probabilities
    state_assignments_per_cell.csv      one row per cell

Distributions cover every feature in the features cache, not just the ones the
model saw.  A state characterised only by the features it was fit on is a
tautology; the diagnostic features are what make the description checkable, so
the figure marks which is which.

Run inside the imaging environment.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from treearhmm import config as cfgmod  # noqa: E402
from treearhmm.core import cells as cellsmod  # noqa: E402
from treearhmm.core import io, viz  # noqa: E402
from treearhmm.steps import step_main  # noqa: E402


def _load_fit(layout) -> dict:
    """Load everything the fit step wrote."""
    fit_dir = layout.fit_dir
    fit = {path.stem: np.load(path) for path in fit_dir.glob("*.npy")}
    fit["masks"] = io.load_npz(fit_dir / "masks.npz")
    fit["summary"] = io.read_yaml(fit_dir / "fit_summary.yml")
    with open(fit_dir / "cell_index.csv") as handle:
        fit["index"] = list(csv.DictReader(handle))
    return fit


def _state_labels(num_states: int) -> list[str]:
    return [f"state_{k}" for k in range(num_states)]


# --------------------------------------------------------------------------- #
# 1. state assignments
# --------------------------------------------------------------------------- #


def write_state_assignments(fit: dict, out_dir: Path) -> list[Path]:
    """Write one row per inferred cell-frame, plus a per-cell rollup.

    Warmup and inactive cell-frames are **absent** rather than zero-filled: the
    CSV contains only what the model actually inferred.

    Args:
        fit (dict): loaded fit artifacts.
        out_dir (Path): output directory.

    Returns:
        list[Path]: the written CSVs.
    """
    states = fit["state_assignments"]
    probs = fit["state_probs"]
    active = fit["active_mask"]
    roots = fit["masks"]["is_new_root_mask"]
    index = fit["index"]
    num_states = fit["summary"]["num_states"]
    labels = _state_labels(num_states)

    per_frame = out_dir / "state_assignments.csv"
    with io.atomic_write(per_frame, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["crop", "cell_id", "column", "frame", "state", "state_prob", "is_new_root"]
            + [f"prob_{label}" for label in labels]
        )
        for column, entry in enumerate(index):
            for t in np.flatnonzero(active[:, column]):
                state = int(states[t, column])
                writer.writerow(
                    [entry["crop"], entry["cell_id"], column, int(t), state,
                     f"{probs[t, column, state]:.6g}", bool(roots[t, column])]
                    + [f"{probs[t, column, k]:.6g}" for k in range(num_states)]
                )

    per_cell = out_dir / "state_assignments_per_cell.csv"
    with io.atomic_write(per_cell, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["crop", "cell_id", "n_frames", "first_frame", "last_frame", "modal_state",
             "mean_state_prob"] + [f"frac_{label}" for label in labels]
        )
        for column, entry in enumerate(index):
            frames = np.flatnonzero(active[:, column])
            if frames.size == 0:
                continue
            cell_states = states[frames, column]
            counts = np.bincount(cell_states, minlength=num_states)
            confidence = probs[frames, column, cell_states].mean()
            writer.writerow(
                [entry["crop"], entry["cell_id"], int(frames.size), int(frames[0]),
                 int(frames[-1]), int(counts.argmax()), f"{confidence:.6g}"]
                + [f"{c / frames.size:.6g}" for c in counts]
            )
    return [per_frame, per_cell]


# --------------------------------------------------------------------------- #
# 2. transition matrices
# --------------------------------------------------------------------------- #


def write_transition_matrix(fit: dict, out_dir: Path) -> list[Path]:
    """Write the learned transition probabilities as CSV and a figure.

    The division kernel is not emitted: this pipeline does not use the tree, so
    `P_div` is never exercised and reporting it would invite reading meaning
    into an untouched initialization.

    Args:
        fit (dict): loaded fit artifacts.
        out_dir (Path): output directory.

    Returns:
        list[Path]: the written files.
    """
    transition = fit["transition_matrix"]
    initial = fit["initial_distribution"]
    labels = _state_labels(fit["summary"]["num_states"])

    csv_path = out_dir / "transition_matrix.csv"
    with io.atomic_write(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [f"{v:.6g}" for v in transition[i]])

    initial_path = out_dir / "initial_distribution.csv"
    with io.atomic_write(initial_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "initial_prob"])
        for label, value in zip(labels, initial):
            writer.writerow([label, f"{value:.6g}"])

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), width_ratios=[3, 1])
    viz.annotate_matrix(axes[0], transition, labels, "learned transition probabilities")
    axes[1].bar(range(len(labels)), initial, color=viz.state_colours(len(labels)))
    axes[1].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes[1].set_ylabel("initial probability")
    axes[1].set_title("initial distribution")
    figure.tight_layout()
    png_path = out_dir / "transition_matrix.png"
    figure.savefig(png_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [csv_path, initial_path, png_path]


# --------------------------------------------------------------------------- #
# 3. feature distributions
# --------------------------------------------------------------------------- #


def write_feature_distributions(cfg: dict, fit: dict, out_dir: Path) -> list[Path]:
    """Per-state distribution of every cached feature, and a summary table.

    Features the model was fit on are outlined in their own colour; the rest are
    held-out diagnostics.  That distinction is the point of the figure.

    Args:
        cfg (dict): resolved configuration.
        fit (dict): loaded fit artifacts.
        out_dir (Path): output directory.

    Returns:
        list[Path]: the written files.
    """
    summary = fit["summary"]
    names = summary["diagnostic_names"]
    values = fit["diagnostics"]
    states = fit["state_assignments"]
    active = fit["active_mask"]
    num_states = summary["num_states"]
    labels = _state_labels(num_states)
    seen = set(summary["emission_names"])
    kind = cfgmod.get_path(cfg, "outputs.feature_distributions.kind", "violin")
    max_cols = int(cfgmod.get_path(cfg, "outputs.feature_distributions.max_cols", 4))
    colours = viz.state_colours(num_states)

    from treearhmm.core.trackfeatures import FEATURE_REGISTRY

    units = {
        name: FEATURE_REGISTRY[name].units for name in names if name in FEATURE_REGISTRY
    }

    csv_path = out_dir / "state_feature_summary.csv"
    total = max(int(active.sum()), 1)
    with io.atomic_write(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["state", "n_cell_frames", "occupancy"]
        for name in names:
            header += [f"{name}_{stat}" for stat in ("mean", "std", "median", "q25", "q75")]
        writer.writerow(header)
        for k in range(num_states):
            selected = active & (states == k)
            row = [k, int(selected.sum()), f"{selected.sum() / total:.6g}"]
            for index, _ in enumerate(names):
                sample = values[..., index][selected]
                sample = sample[np.isfinite(sample)]
                if sample.size:
                    row += [f"{np.mean(sample):.6g}", f"{np.std(sample):.6g}",
                            f"{np.median(sample):.6g}", f"{np.percentile(sample, 25):.6g}",
                            f"{np.percentile(sample, 75):.6g}"]
                else:
                    row += [""] * 5
            writer.writerow(row)

    viz.apply_style()
    rows, cols = viz.grid_shape(len(names), max_cols)
    figure, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), squeeze=False)
    flat = axes.ravel()

    for panel, name in enumerate(names):
        ax = flat[panel]
        index = names.index(name)
        samples = []
        for k in range(num_states):
            sample = values[..., index][active & (states == k)]
            samples.append(sample[np.isfinite(sample)])

        positions = range(1, num_states + 1)
        if kind == "violin" and all(s.size > 1 for s in samples):
            parts = ax.violinplot(samples, positions=positions, showmedians=True)
            for body, colour in zip(parts["bodies"], colours):
                body.set_facecolor(colour)
                body.set_alpha(0.65)
        else:
            box = ax.boxplot(samples, positions=list(positions), patch_artist=True, widths=0.6)
            for patch, colour in zip(box["boxes"], colours):
                patch.set_facecolor(colour)
                patch.set_alpha(0.65)

        ax.set_xticks(list(positions), [str(k) for k in range(num_states)])
        ax.set_xlabel("state")
        ax.set_title(name, color="black" if name in seen else "0.35")
        # Units, not the description: a truncated sentence on a y-axis reads as
        # a bug.  The full description lives in the features cache's meta.json.
        ax.set_ylabel(units.get(name, ""))
        if name in seen:
            # A visible frame marks the features the model was actually fit on.
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#c44e52")
                spine.set_linewidth(1.6)

    for panel in range(len(names), len(flat)):
        flat[panel].axis("off")

    figure.suptitle(
        f"{summary['run_name']}: feature distributions per state "
        f"(outlined = fed to the model, plain = held out)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    png_path = out_dir / "feature_distributions.png"
    figure.savefig(png_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [csv_path, png_path]


# --------------------------------------------------------------------------- #
# 4. overlay videos
# --------------------------------------------------------------------------- #


def write_overlay_videos(cfg: dict, fit: dict, out_dir: Path) -> list[Path]:
    """Render one state-coloured overlay video per requested crop.

    Args:
        cfg (dict): resolved configuration.
        fit (dict): loaded fit artifacts.
        out_dir (Path): the `overlays/` directory.

    Returns:
        list[Path]: the written mp4s.
    """
    summary = fit["summary"]
    num_states = summary["num_states"]
    cmap, norm = viz.state_cmap_norm(num_states)
    colours = viz.state_colours(num_states)
    fps = int(cfgmod.get_path(cfg, "outputs.video.fps", 2))
    figsize = tuple(cfgmod.get_path(cfg, "outputs.video.figsize", [7, 7]))
    dpi = int(cfgmod.get_path(cfg, "outputs.video.dpi", 160))
    label_ids = bool(cfgmod.get_path(cfg, "outputs.video.label_cell_ids", True))
    feature_note = ", ".join(summary["emission_names"])
    if len(feature_note) > 60:
        feature_note = f"{len(summary['emission_names'])} features"

    import matplotlib.patches as mpatches

    written = []
    for crop_id in cfgmod.video_crops(cfg):
        columns = [i for i, e in enumerate(fit["index"]) if e["crop"] == crop_id]
        if not columns:
            continue
        crop = cellsmod.load_crop(cfg, crop_id, with_image=True)
        cell_ids = np.array([int(fit["index"][c]["cell_id"]) for c in columns], dtype=np.int32)
        states = fit["state_assignments"][:, columns]
        active = fit["active_mask"][:, columns]
        phase = crop.image[..., 1]
        tcells = crop.tcells

        legend = [mpatches.Patch(color=colours[k], label=f"state {k}") for k in range(num_states)]
        legend.append(mpatches.Patch(color=(0.5, 0.5, 0.5), label="no state inferred"))

        # Everything the closure needs is bound as a default argument:
        # frames_to_mp4 calls it lazily, so a closure over `crop_id` would
        # render the last crop into every video.
        def draw(t, _phase=phase, _tcells=tcells, _cancer=crop.cancer, _ids=cell_ids,
                 _states=states, _active=active, _crop=crop_id, _legend=legend):
            ax = plt.gca()
            ax.imshow(_phase[t], cmap="gray", vmin=0.0, vmax=1.0)
            ax.imshow(
                np.ma.masked_where(_tcells[t] == 0, np.ones_like(_tcells[t], dtype=float)),
                cmap="Blues", alpha=0.25, vmin=0.0, vmax=1.0,
            )
            label_image = viz.state_label_image(_cancer[t], _ids, _states[t], _active[t])
            ax.imshow(np.ma.masked_where(label_image == 0, label_image),
                      cmap=cmap, norm=norm, alpha=0.55, interpolation="nearest")
            if label_ids:
                for column, cell_id in enumerate(_ids):
                    pixels = np.argwhere(_cancer[t] == cell_id)
                    if pixels.size:
                        y, x = pixels.mean(axis=0)
                        ax.text(x, y, str(cell_id), color="white", fontsize=6,
                                fontweight="bold", ha="center", va="center")
            ax.set_title(
                f"{summary['run_name']} | {_crop} | frame {t}/{_phase.shape[0] - 1}\n"
                f"k={num_states}  lag={summary['num_lags']}  [{feature_note}]",
                fontsize=8,
            )
            ax.axis("off")
            # Outside the axes: an in-frame legend sits on top of whichever
            # cells happen to be in that corner.
            ax.legend(
                handles=_legend, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                ncol=len(_legend), frameon=False,
            )

        path = viz.frames_to_mp4(
            draw, range(phase.shape[0]), out_dir / f"{crop_id}_state_overlay.mp4",
            fps=fps, figsize=figsize, dpi=dpi,
        )
        print(f"  wrote {path.name}")
        written.append(path)
    return written


def _run(cfg: dict, layout, args) -> dict:
    fit = _load_fit(layout)
    out_dir = io.ensure_dir(layout.outputs_dir)
    viz.apply_style()

    written = []
    written += write_state_assignments(fit, out_dir)
    written += write_transition_matrix(fit, out_dir)
    written += write_feature_distributions(cfg, fit, out_dir)
    written += write_overlay_videos(cfg, fit, io.ensure_dir(layout.overlays_dir))

    for path in written:
        print(f"  {path.relative_to(layout.run_dir)}")
    return {"files": len(written)}


if __name__ == "__main__":
    sys.exit(step_main("outputs", "the four base outputs", _run))
