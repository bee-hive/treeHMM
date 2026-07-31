"""Extra `state_timeline`: every cell's state trajectory as a raster.

One row per cell, one column per frame, coloured by inferred state, with cells
grouped by crop.  The base outputs describe the states; this shows how long
cells stay in them and where they switch, which is the thing a transition matrix
averages away.

Also writes the distribution of run lengths -- consecutive frames a cell spends
in one state -- because a transition matrix with high self-transition
probabilities is consistent with both "cells persist" and "cells flicker at the
boundary", and those are different biological claims.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REQUIRES: tuple[str, ...] = ()


def _run_lengths(states: np.ndarray, active: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive same-state stretches over a cell's active frames.

    Args:
        states (np.ndarray): `(T,)` state per frame for one cell.
        active (np.ndarray): `(T,)` bool.

    Returns:
        list[tuple[int, int]]: `(state, length)`; a gap in `active` ends a run,
            because a cell that vanishes and returns has not demonstrably stayed
            put in between.
    """
    runs: list[tuple[int, int]] = []
    current_state: int | None = None
    length = 0
    previous_frame: int | None = None

    for t in np.flatnonzero(active):
        contiguous = previous_frame is not None and t == previous_frame + 1
        state = int(states[t])
        if contiguous and state == current_state:
            length += 1
        else:
            if current_state is not None:
                runs.append((current_state, length))
            current_state, length = state, 1
        previous_frame = int(t)

    if current_state is not None:
        runs.append((current_state, length))
    return runs


def run(cfg: dict, layout, out_dir: Path) -> None:
    """Write the state raster and the run-length table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from treearhmm.core import io, viz

    fit_dir = layout.fit_dir
    states = np.load(fit_dir / "state_assignments.npy")
    active = np.load(fit_dir / "active_mask.npy")
    summary = io.read_yaml(fit_dir / "fit_summary.yml")
    with open(fit_dir / "cell_index.csv") as handle:
        index = list(csv.DictReader(handle))

    num_states = summary["num_states"]
    cmap, norm = viz.state_cmap_norm(num_states)
    colours = viz.state_colours(num_states)

    # Cells are already grouped by crop (crops are concatenated in sorted
    # order), so the raster only needs the boundaries for its labels.
    crops = [entry["crop"] for entry in index]
    boundaries, seen = [], None
    for column, crop in enumerate(crops):
        if crop != seen:
            boundaries.append((column, crop))
            seen = crop

    raster = np.where(active, states + 1, 0).T.astype(np.int16)
    raster[~active.T] = 0

    viz.apply_style()
    height = max(2.5, 0.16 * len(index))
    figure, ax = plt.subplots(figsize=(10, height))
    ax.imshow(np.ma.masked_where(raster == 0, raster), cmap=cmap, norm=norm,
              aspect="auto", interpolation="nearest")
    ax.set_xlabel("frame")
    ax.set_ylabel("cell")
    ax.set_yticks([c for c, _ in boundaries], [crop for _, crop in boundaries], fontsize=6)
    for column, _ in boundaries[1:]:
        ax.axhline(column - 0.5, color="black", linewidth=0.8)
    ax.set_title(f"{summary['run_name']}: inferred state per cell over time")
    # To the side, not below: this figure's height varies with the cell count,
    # so a legend offset in axes coordinates lands on the x-label at some sizes.
    ax.legend(
        handles=[mpatches.Patch(color=colours[k], label=f"state {k}") for k in range(num_states)],
        loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
    )
    figure.savefig(out_dir / "state_timeline.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    with io.atomic_write(out_dir / "state_run_lengths.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["crop", "cell_id", "state", "run_length"])
        for column, entry in enumerate(index):
            for state, length in _run_lengths(states[:, column], active[:, column]):
                writer.writerow([entry["crop"], entry["cell_id"], state, length])

    per_state: dict[int, list[int]] = {k: [] for k in range(num_states)}
    for column in range(len(index)):
        for state, length in _run_lengths(states[:, column], active[:, column]):
            per_state[state].append(length)
    with io.atomic_write(out_dir / "state_run_length_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "n_runs", "mean_run_length", "median_run_length", "max_run_length"])
        for state, lengths in per_state.items():
            if lengths:
                writer.writerow([state, len(lengths), f"{np.mean(lengths):.4g}",
                                 f"{np.median(lengths):.4g}", max(lengths)])
            else:
                writer.writerow([state, 0, "", "", ""])
