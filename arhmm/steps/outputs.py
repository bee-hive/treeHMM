"""Step `outputs`: the five base outputs every run produces.

    overlays/{crop}_state_overlay.mp4   each cancer cell tinted by its inferred
                                        state, over the phase image
    feature_distributions.png           distribution of every computed feature
    state_feature_summary.csv           within each state
    state_age_histogram.png             state occupancy against cell age
    state_age_histogram.csv
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

from arhmm import config as cfgmod  # noqa: E402
from arhmm.core import cells as cellsmod  # noqa: E402
from arhmm.core import io, viz  # noqa: E402
from arhmm.steps import step_main  # noqa: E402


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

    from arhmm.core.trackfeatures import FEATURE_REGISTRY

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
# 4. state occupancy against cell age
# --------------------------------------------------------------------------- #


def cell_birth_frames(cfg: dict, layout, index: list[dict], active: np.ndarray) -> np.ndarray:
    """The frame each cell's clock starts on, one per fit column.

    Under the default anchor this is the first frame the cell is **observed at
    all**, which is not its first inferred frame: `lineage.apply_warmup`
    deactivates each cell's leading `cells.warmup_frames` frames, so the fit's
    `active_mask` starts later than the track does.  The raw presence matrix
    survives in the features cache, and `cell_index.csv` carries `crop_idx`
    straight into that crop's column order, so no cell-id matching is needed.

    Args:
        cfg (dict): resolved configuration.
        layout (Layout): the run's layout, for the features cache directories.
        index (list[dict]): `cell_index.csv` rows, in column order.
        active (np.ndarray): `(T, C)` inferred cell-frames.

    Returns:
        np.ndarray: `(C,)` int, the age-zero frame of each column.

    Raises:
        ValueError: if a cell is inferred before the frame it first appears on,
            which means the features cache no longer matches this fit.
    """
    anchor = cfgmod.get_path(cfg, "outputs.state_age_histogram.anchor", "existence")
    first_inferred = np.array(
        [int(np.flatnonzero(active[:, c])[0]) if active[:, c].any() else 0
         for c in range(active.shape[1])],
        dtype=np.int64,
    )
    if anchor == "inferred":
        return first_inferred

    births = np.empty(len(index), dtype=np.int64)
    cached: dict[str, np.ndarray] = {}
    for column, entry in enumerate(index):
        crop_id = entry["crop"]
        if crop_id not in cached:
            arrays = io.load_npz(layout.crop_dir("features", crop_id) / "features.npz")
            cached[crop_id] = arrays["active_mask"]
        present = cached[crop_id][:, int(entry["crop_idx"])]
        frames = np.flatnonzero(present)
        births[column] = int(frames[0]) if frames.size else first_inferred[column]

    ahead = np.flatnonzero(births > first_inferred)
    if ahead.size:
        column = int(ahead[0])
        raise ValueError(
            f"cell {index[column]['crop']}/{index[column]['cell_id']} is inferred at frame "
            f"{first_inferred[column]} but the features cache first observes it at frame "
            f"{births[column]}; the cache does not match this fit"
        )
    return births


def age_histogram(
    states: np.ndarray,
    active: np.ndarray,
    births: np.ndarray,
    num_states: int,
    bin_frames: int,
    max_age: int | None,
    min_age: int = 0,
) -> dict:
    """Bin inferred cell-frames by how long the cell has existed.

    Age is `t - birth`, an elapsed-frame count rather than a rank among the
    frames the cell was seen in: a cell present at frames 0, 1 and 3 is at ages
    0, 1 and 3, so a tracking gap costs the cell a sample rather than rewinding
    its clock.

    Args:
        states (np.ndarray): `(T, C)` inferred state per cell-frame.
        active (np.ndarray): `(T, C)` bool, which of those were inferred.
        births (np.ndarray): `(C,)` age-zero frame per column.
        num_states (int): `K`.
        bin_frames (int): width of one age bin, in frames.
        max_age (int | None): largest age to plot; None takes the largest
            observed.  Older cell-frames are dropped and counted, never folded
            into the last bin, which would put a spike there.
        min_age (int): youngest age to plot, and the origin the bins are laid
            out from.  Ages below it can only be warmup, where the model is
            given no state, so plotting them would open the figure with a run
            of bars that are empty by construction rather than by measurement.

    Returns:
        dict: `counts` `(B, K)` int, `at_risk` `(B,)` int cells still inferred
            at that age, `starts` `(B,)` int inclusive bin start, `stops` `(B,)`
            int inclusive bin end, `dropped` int, `dropped_young` int,
            `max_observed_age` int.

    Raises:
        ValueError: on a negative age, i.e. a cell inferred before it was born.
    """
    ages = np.arange(active.shape[0], dtype=np.int64)[:, None] - np.asarray(births)[None, :]
    if (ages[active] < 0).any():
        raise ValueError("a cell-frame has negative age; births are inconsistent with active_mask")

    flat_ages = ages[active]
    flat_states = states[active].astype(np.int64)
    max_observed = int(flat_ages.max()) if flat_ages.size else 0
    limit = max(max_observed if max_age is None else int(max_age), min_age)

    num_bins = (limit - min_age) // bin_frames + 1
    young = flat_ages < min_age
    keep = (~young) & (flat_ages <= limit)
    dropped = int((flat_ages > limit).sum())

    binned = (flat_ages[keep] - min_age) // bin_frames
    counts = np.bincount(
        binned * num_states + flat_states[keep], minlength=num_bins * num_states
    ).reshape(num_bins, num_states)

    # Cells still contributing at each age, so a bin's shrinking bar can be read
    # against how many cells were left to fill it.  Counted per cell from its
    # last inferred age, since that is the age past which it can contribute
    # nothing whatever the reason -- death, leaving frame, or the end of the movie.
    last_age = np.full(active.shape[1], -1, dtype=np.int64)
    has_any = active.any(axis=0)
    if has_any.any():
        last_frame = active.shape[0] - 1 - np.argmax(active[::-1], axis=0)
        last_age[has_any] = (last_frame - np.asarray(births))[has_any]
    starts = min_age + np.arange(num_bins, dtype=np.int64) * bin_frames
    at_risk = (last_age[None, :] >= starts[:, None]).sum(axis=1).astype(np.int64)

    return {
        "counts": counts,
        "at_risk": at_risk,
        "starts": starts,
        "stops": starts + bin_frames - 1,
        "dropped": dropped,
        "dropped_young": int(young.sum()),
        "max_observed_age": max_observed,
    }


def _draw_age_panel(ax, hist: dict, colours, kind: str, normalize: bool) -> None:
    """Draw one age panel, as counts or as each bin's composition."""
    counts = hist["counts"].astype(float)
    totals = counts.sum(axis=1)
    if normalize:
        # NaN, not 0, for an empty bin: a bin nothing landed in has no
        # composition, and drawing it as all-zero would read as one that does.
        divisor = np.where(totals > 0, totals, 1.0)[:, None]
        values = np.where(totals[:, None] > 0, counts / divisor, np.nan)
    else:
        values = counts

    num_bins, num_states = values.shape
    centres = hist["starts"] + (hist["stops"] - hist["starts"]) / 2.0
    width = float(hist["stops"][0] - hist["starts"][0] + 1)

    if kind == "grouped":
        slot = width / max(num_states, 1)
        for k in range(num_states):
            offset = -width / 2.0 + slot * (k + 0.5)
            ax.bar(centres + offset, np.nan_to_num(values[:, k]), width=slot * 0.9,
                   color=colours[k], label=f"state {k}")
    elif kind == "step":
        edges = np.append(hist["starts"], hist["stops"][-1] + 1)
        for k in range(num_states):
            ax.stairs(np.nan_to_num(values[:, k]), edges, color=colours[k],
                      linewidth=1.4, label=f"state {k}")
    else:  # stacked
        bottom = np.zeros(num_bins)
        for k in range(num_states):
            column = np.nan_to_num(values[:, k])
            ax.bar(centres, column, width=width * 0.95, bottom=bottom,
                   color=colours[k], label=f"state {k}", linewidth=0)
            bottom += column

    if normalize:
        ax.set_ylim(0.0, 1.0)


def write_state_age_histogram(cfg: dict, fit: dict, layout, out_dir: Path) -> list[Path]:
    """Inferred state against cell age, pooled over every crop.

    Two panels sharing an x axis.  The top one is the count of inferred
    cell-frames per state per age bin; the bottom is each bin's composition,
    with the number of cells still alive at that age drawn over it.  The counts
    alone cannot separate "the state empties out" from "there are hardly any
    cells left this old", and that distinction is usually the question being
    asked of this figure.

    Args:
        cfg (dict): resolved configuration.
        fit (dict): loaded fit artifacts.
        layout (Layout): the run's layout.
        out_dir (Path): output directory.

    Returns:
        list[Path]: the written files.
    """
    summary = fit["summary"]
    states = fit["state_assignments"]
    active = fit["active_mask"]
    num_states = summary["num_states"]
    colours = viz.state_colours(num_states)

    settings = cfgmod.get_path(cfg, "outputs.state_age_histogram", {}) or {}
    bin_frames = int(settings.get("bin_frames", 1))
    requested_max = settings.get("max_age", "auto")
    max_age = None if requested_max == "auto" else int(requested_max)
    kind = settings.get("kind", "stacked")
    anchor = settings.get("anchor", "existence")

    # A cell's leading `cells.warmup_frames` frames never carry a state, and the
    # k-th surviving active frame is at least k frames old, so under the
    # existence anchor no cell-frame can land below that age at all.  Starting
    # the axis there drops bars that are empty by construction rather than by
    # measurement -- with the `inferred` anchor there are none, since age zero is
    # by definition the first frame a state exists for.
    min_age = int(cfgmod.get_path(cfg, "cells.warmup_frames", 0)) if anchor == "existence" else 0

    births = cell_birth_frames(cfg, layout, fit["index"], active)
    hist = age_histogram(states, active, births, num_states, bin_frames, max_age, min_age)
    if hist["dropped"]:
        print(
            f"  state_age_histogram: dropped {hist['dropped']} cell-frames older than "
            f"max_age={max_age} (largest observed age {hist['max_observed_age']})"
        )
    if hist["dropped_young"]:
        # Unreachable as the pipeline stands; if it ever fires, the warmup no
        # longer means what this figure assumes and the bars are undercounts.
        print(
            f"  WARNING state_age_histogram: {hist['dropped_young']} inferred cell-frames "
            f"fall below age {min_age} and are not plotted"
        )

    csv_path = out_dir / "state_age_histogram.csv"
    with io.atomic_write(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["age_bin_start", "age_bin_end", "n_cells_at_risk", "total"]
            + [f"count_{label}" for label in _state_labels(num_states)]
        )
        for b in range(hist["counts"].shape[0]):
            row = hist["counts"][b]
            writer.writerow(
                [int(hist["starts"][b]), int(hist["stops"][b]), int(hist["at_risk"][b]),
                 int(row.sum())] + [int(v) for v in row]
            )

    viz.apply_style()
    # Constrained rather than tight: a suptitle plus a figure-level legend plus
    # a twin axis is exactly the combination tight_layout mis-measures.
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                               height_ratios=[2, 1.4], layout="constrained")
    _draw_age_panel(axes[0], hist, colours, kind, normalize=False)
    axes[0].set_ylabel("cell-frames")
    axes[0].set_title("inferred cell-frames per state, by cell age")

    _draw_age_panel(axes[1], hist, colours, kind, normalize=True)
    axes[1].set_ylabel("fraction of the bin")
    axes[1].set_title("composition of each age bin")
    origin = "first appearance" if anchor == "existence" else "first inferred frame"
    axes[1].set_xlabel(f"cell age (frames since {origin})")
    # The composition of a bin filled by three cells is not comparable with one
    # filled by three hundred, so the count of surviving cells is drawn on top
    # of the panel whose y axis has had that information normalized away.
    risk_ax = axes[1].twinx()
    edges = np.append(hist["starts"], hist["stops"][-1] + 1)
    # baseline=None: with a baseline, `stairs` closes the outline down to zero at
    # both ends, and those two vertical drops read as the cell count collapsing.
    risk_ax.stairs(hist["at_risk"], edges, baseline=None, color="0.35",
                   linewidth=1.2, linestyle="--", label="cells at risk")
    risk_ax.set_ylabel("cells at risk", color="0.35")
    risk_ax.tick_params(axis="y", colors="0.35")
    risk_ax.set_ylim(bottom=0)
    risk_ax.spines["right"].set_visible(True)
    risk_ax.spines["right"].set_color("0.35")

    note = f"bin = {bin_frames} frame{'s' if bin_frames != 1 else ''}, anchor = {anchor}"
    if min_age:
        # Otherwise an axis that starts at 1 rather than 0 reads as a bug rather
        # than as the warmup the model was never given a state for.
        note += f", ages below {min_age} omitted (warmup)"
    if hist["dropped"]:
        note += f", {hist['dropped']} cell-frames beyond age {max_age} dropped"
    figure.suptitle(f"{summary['run_name']}: state occupancy over cell age\n{note}", fontsize=10)
    # Outside the axes, at figure level: an in-axes legend covers the bars at
    # whichever ages happen to be tallest, and a per-axes outside legend would
    # narrow the top panel out of alignment with the bottom one.  Below rather
    # than above, because constrained layout puts an outside upper legend and
    # the suptitle in the same place.
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center",
                  ncol=min(num_states, 8), frameon=False)
    png_path = out_dir / "state_age_histogram.png"
    figure.savefig(png_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [csv_path, png_path]


# --------------------------------------------------------------------------- #
# 5. overlay videos
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
    written += write_state_age_histogram(cfg, fit, layout, out_dir)
    written += write_overlay_videos(cfg, fit, io.ensure_dir(layout.overlays_dir))

    for path in written:
        print(f"  {path.relative_to(layout.run_dir)}")
    return {"files": len(written)}


if __name__ == "__main__":
    sys.exit(step_main("outputs", "the five base outputs", _run))
