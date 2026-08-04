"""Extra `condition_stats`: state occupancy broken out by experimental condition.

`data.conditions` maps each condition to its crops.  The fit is joint across all
crops and never sees that grouping, so comparing occupancy across conditions
afterwards is a genuine read-out rather than something the model was told.

Occupancy is aggregated per crop first and then per condition, and the per-crop
numbers are kept, because with two crops per condition a condition-level bar is
a mean of two points and should be read as such.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REQUIRES: tuple[str, ...] = ()


def run(cfg: dict, layout, out_dir: Path) -> None:
    """Write per-crop and per-condition occupancy tables and a bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from arhmm import config as cfgmod
    from arhmm.core import io, viz

    fit_dir = layout.fit_dir
    states = np.load(fit_dir / "state_assignments.npy")
    active = np.load(fit_dir / "active_mask.npy")
    summary = io.read_yaml(fit_dir / "fit_summary.yml")
    with open(fit_dir / "cell_index.csv") as handle:
        index = list(csv.DictReader(handle))

    num_states = summary["num_states"]
    colours = viz.state_colours(num_states)
    grouped = cfgmod.conditions(cfg)
    crop_of_column = [entry["crop"] for entry in index]

    def occupancy(columns: list[int]) -> tuple[np.ndarray, int]:
        if not columns:
            return np.full(num_states, np.nan), 0
        selected_active = active[:, columns]
        selected_states = states[:, columns]
        total = int(selected_active.sum())
        if total == 0:
            return np.full(num_states, np.nan), 0
        counts = np.array(
            [int((selected_active & (selected_states == k)).sum()) for k in range(num_states)]
        )
        return counts / total, total

    per_crop: dict[str, tuple[np.ndarray, int]] = {}
    for crop in cfgmod.crop_ids(cfg):
        columns = [i for i, c in enumerate(crop_of_column) if c == crop]
        per_crop[crop] = occupancy(columns)

    with io.atomic_write(out_dir / "occupancy_by_crop.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["crop", "condition", "n_cell_frames"]
                        + [f"frac_state_{k}" for k in range(num_states)])
        for crop, (fractions, total) in per_crop.items():
            condition = next((c for c, members in grouped.items() if crop in members), "")
            writer.writerow([crop, condition, total]
                            + ["" if np.isnan(v) else f"{v:.6g}" for v in fractions])

    per_condition: dict[str, tuple[np.ndarray, int]] = {}
    for condition, members in grouped.items():
        columns = [i for i, c in enumerate(crop_of_column) if c in set(members)]
        per_condition[condition] = occupancy(columns)

    with io.atomic_write(out_dir / "occupancy_by_condition.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "n_crops", "n_cell_frames"]
                        + [f"frac_state_{k}" for k in range(num_states)])
        for condition, (fractions, total) in per_condition.items():
            writer.writerow([condition, len(grouped[condition]), total]
                            + ["" if np.isnan(v) else f"{v:.6g}" for v in fractions])

    if not per_condition:
        print("  no conditions cover this run's crops; wrote the per-crop table only")
        return

    viz.apply_style()
    labels = list(per_condition)
    width = 0.8 / max(num_states, 1)
    figure, ax = plt.subplots(figsize=(max(4.0, 1.6 * len(labels)), 3.6))
    positions = np.arange(len(labels))
    for k in range(num_states):
        heights = [per_condition[c][0][k] for c in labels]
        ax.bar(positions + k * width, heights, width, label=f"state {k}", color=colours[k])
    # Individual crops on top, so a two-crop condition never reads as a
    # tighter estimate than it is.
    for position, condition in enumerate(positions):
        for crop in grouped[labels[position]]:
            fractions, total = per_crop.get(crop, (None, 0))
            if fractions is None or total == 0:
                continue
            for k in range(num_states):
                ax.plot(position + k * width, fractions[k], "o", color="black",
                        markersize=3, alpha=0.7, zorder=3)
    ax.set_xticks(positions + width * (num_states - 1) / 2, labels)
    ax.set_ylabel("fraction of inferred cell-frames")
    ax.set_title(f"{summary['run_name']}: state occupancy by condition\n"
                 f"(points are individual crops)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=num_states, frameon=False)
    figure.savefig(out_dir / "occupancy_by_condition.png", dpi=160,
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)
