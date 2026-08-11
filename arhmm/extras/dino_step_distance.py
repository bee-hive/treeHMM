"""Extra `dino_step_distance`: how far a cell's DINO embedding moves per frame.

One row per cell, one column per frame of age, coloured by the distance between
that frame's embedding and the previous observed frame's.  Rows are ordered
longest-lived at the top, so the raster's staircase edge is the lifespan
distribution and each row is one cell's embedding "speed" over its life.

The question this answers is whether DINO is tracking the cell or the noise.
A raster that is uniformly bright says consecutive frames of the *same* cell are
about as far apart as anything else in the space -- the embedding is dominated
by per-frame nuisance, and no amount of downstream modelling recovers behaviour
from it.  A raster that is mostly dark with bright streaks says the embedding
sits still while the cell does and moves when the cell does, which is what a
behavioural feature has to do.

The scale that makes "bright" mean something is printed on the figure and marked
on the colourbar: the median and mean distance between *different* cells in the
same frame, against the median and mean of the steps in the raster itself.  The
good case is the same-cell pair sitting well below the different-cell pair.  The
mean is shown next to the median because these samples are right-skewed -- a
mean far above its median says a handful of frames carry the average, which is a
different claim from "the embedding is noisy throughout".

Distances are taken against the previous **observed** frame, following the same
rule as the temporal track features -- a tracking gap is not a teleport -- and
the gap length is carried in the CSV so a large step across a gap can be told
apart from a large step across one frame.

The row labels are cell ids in **whichever label space `cells.source` selected**,
and the two spaces are unrelated: a CVAT cancer track id and a nucleus label
that happen to be the same integer are not the same cell, and nothing in
this pipeline maps one onto the other.  So the label space is named in the
output directory, in every filename, and on the y-axis -- a figure that says
only "cell 7" is unreadable the moment two runs are put side by side::

    {run}/outputs/extras/dino_step_distance/
        nucleus_ids/                    <- cells.source: nuclei
            dino_step_distance_{crop}_nucleus_ids.png
            dino_step_distance_nucleus_ids.csv
            dino_step_distance_summary_nucleus_ids.yml

Config, all optional::

    outputs:
      dino_step_distance:
        metric: cosine        # or euclidean
        crops: [B8_...]       # default: every crop in the run
        note: "..."           # one line of provenance, drawn under the title

Reads the raw `dino` embeddings, not the PCs: the PCs are a variance-truncated,
jointly-fitted view, and the question here is about the embedding itself.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REQUIRES: tuple[str, ...] = ("dino",)

METRICS = ("cosine", "euclidean")

#: `cells.source` -> (filename slug, y-axis wording).  Kept here rather than
#: derived from the source name so that "nuclei" cannot be read as CVAT nuclei:
#: these are nucleus-segmentation labels, and `phase` ids are CVAT cancer track
#: ids.
ID_SPACES = {
    "phase": ("cvat_cell_ids", "CVAT cancer track id"),
    "nuclei": ("nucleus_ids", "nucleus id"),
}


def _pairwise_to_previous(vectors: np.ndarray, metric: str) -> np.ndarray:
    """Distance from each row to the row before it.

    Args:
        vectors (np.ndarray): `(m, E)` embeddings, in frame order.
        metric (str): one of `METRICS`.

    Returns:
        np.ndarray: `(m - 1,)` float; empty when fewer than two rows.
    """
    if vectors.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    current, previous = vectors[1:].astype(np.float64), vectors[:-1].astype(np.float64)
    if metric == "euclidean":
        return np.linalg.norm(current - previous, axis=1)
    return _cosine_distance(current, previous)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise `1 - cos(a, b)`; NaN where either vector has no length."""
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.where(norms > 0, np.einsum("ij,ij->i", a, b) / norms, np.nan)
    return 1.0 - similarity


def _centres(values: np.ndarray) -> tuple[float, float]:
    """`(median, mean)` of a distance sample, NaN-safe and NaN when empty.

    Both, because they answer different questions of a right-skewed sample: the
    median is where a typical step sits, the mean is dragged up by the rare
    large ones -- a mean well above the median means the bright cells in the
    raster are carrying the average.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.median(finite)), float(np.mean(finite))


def _between_cell_scale(raw: np.ndarray, valid: np.ndarray, metric: str) -> tuple[float, float]:
    """Distance between two *different* cells observed in the same frame.

    This is the yardstick the per-frame steps are read against: it is how far
    apart the embedding puts two cells that genuinely are different, in the same
    imaging conditions, so it isolates cell identity from acquisition drift.

    Args:
        raw (np.ndarray): `(T, N, E)` embeddings, NaN where absent.
        valid (np.ndarray): `(T, N)` bool.
        metric (str): one of `METRICS`.

    Returns:
        tuple[float, float]: `(median, mean)` over all same-frame cell pairs;
            NaN if there are none.
    """
    distances = []
    for t in np.flatnonzero(valid.any(axis=1)):
        present = np.flatnonzero(valid[t])
        if present.size < 2:
            continue
        vectors = raw[t, present].astype(np.float64)
        rows, cols = np.triu_indices(present.size, k=1)
        if metric == "euclidean":
            distances.append(np.linalg.norm(vectors[rows] - vectors[cols], axis=1))
        else:
            distances.append(_cosine_distance(vectors[rows], vectors[cols]))
    if not distances:
        return float("nan"), float("nan")
    return _centres(np.concatenate(distances))


def _crop_raster(raw: np.ndarray, valid: np.ndarray, cell_ids: np.ndarray, metric: str):
    """Per-cell step distances laid out against age, longest-lived cell first.

    Age is frames since the cell was first observed, so column `j` of a row is
    the step that *landed* on age `j`; age 0 is always blank, having no previous
    frame, and so is any age the cell was not observed at.

    Args:
        raw (np.ndarray): `(T, N, E)` embeddings, NaN where absent.
        valid (np.ndarray): `(T, N)` bool.
        cell_ids (np.ndarray): `(N,)` track ids, in column order.
        metric (str): one of `METRICS`.

    Returns:
        tuple: `raster` `(C, A)` float with NaN where undefined, `order` `(C,)`
            the columns of `raw` it came from, `spans` `(C,)` lifespan in frames,
            and `records` a list of one dict per computed distance.
    """
    lifespans, first_frames = [], []
    for column in range(valid.shape[1]):
        frames = np.flatnonzero(valid[:, column])
        if frames.size == 0:
            lifespans.append(0)
            first_frames.append(-1)
            continue
        lifespans.append(int(frames[-1] - frames[0]) + 1)
        first_frames.append(int(frames[0]))

    lifespans = np.asarray(lifespans, dtype=np.int64)
    observed = np.flatnonzero(lifespans > 0)
    # Longest-lived at the top; ties broken by cell id so the figure is
    # reproducible rather than dependent on argsort's internals.
    order = observed[np.lexsort((cell_ids[observed], -lifespans[observed]))]

    max_age = int(lifespans[order].max()) if order.size else 1
    raster = np.full((order.size, max_age), np.nan, dtype=np.float64)

    records: list[dict] = []
    for row, column in enumerate(order):
        frames = np.flatnonzero(valid[:, column])
        distances = _pairwise_to_previous(raw[frames, column], metric)
        ages = frames[1:] - frames[0]
        raster[row, ages] = distances
        for age, frame, previous, distance in zip(ages, frames[1:], frames[:-1], distances):
            records.append(
                {
                    "cell_id": int(cell_ids[column]),
                    "frame": int(frame),
                    "prev_frame": int(previous),
                    "gap": int(frame - previous),
                    "age": int(age),
                    "distance": float(distance),
                }
            )
    return raster, order, lifespans[order], records


#: Reference lines drawn on the colourbar: key, colour, mark label, full name.
#: Within-cell and between-cell share a hue so each pair reads as one
#: comparison.  Median and mean are told apart by the label beside the mark and
#: not by a dash pattern -- the colourbar is a few tens of pixels wide, and at
#: that width a dashed line and a solid one are the same picture.
REFERENCE_LINES = (
    ("within_median", "#00e5ff", "med", "same cell, median"),
    ("within_mean", "#00e5ff", "mean", "same cell, mean"),
    ("between_median", "#39ff14", "med", "different cells, median"),
    ("between_mean", "#39ff14", "mean", "different cells, mean"),
)


def _draw(raster: np.ndarray, spans: np.ndarray, labels: np.ndarray, crop_id: str,
          metric: str, stats: dict, id_space: str, note: str | None, path: Path) -> None:
    """Save the raster; colour clipped at the 99th percentile of the steps.

    Args:
        raster (np.ndarray): `(C, A)` step distances, NaN where undefined.
        spans (np.ndarray): `(C,)` lifespan in frames, in row order.
        labels (np.ndarray): `(C,)` cell ids, in row order.
        crop_id (str): for the title.
        metric (str): one of `METRICS`.
        stats (dict): the four keys named in `REFERENCE_LINES`.
        id_space (str): what the row labels are, e.g. `"nucleus id"`.
        note (str | None): one line of provenance, drawn under the title.
        path (Path): PNG to write.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as patheffects
    import matplotlib.pyplot as plt

    from arhmm.core import viz

    finite = raster[np.isfinite(raster)]
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    # The reference marks must fit on the bar.  Clipping on the steps alone puts
    # the between-cell marks off the top whenever the embedding is well behaved
    # -- exactly the case the figure is meant to make visible -- so the scale
    # stretches to hold them, at the cost of a little contrast in the raster.
    references = [stats[key] for key, *_ in REFERENCE_LINES if np.isfinite(stats[key])]
    vmax = max([vmax, *references])
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    viz.apply_style()
    height = max(2.6, 0.13 * raster.shape[0])
    figure, ax = plt.subplots(figsize=(9.5, height))
    image = ax.imshow(
        np.ma.masked_invalid(raster),
        cmap="magma", vmin=0.0, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )
    ax.set_xlabel("frames since the cell first appeared")
    # Cell ids only when they will fit; past that the ordering is the message.
    if raster.shape[0] <= 45:
        ax.set_yticks(np.arange(raster.shape[0]),
                      [f"{int(label)} ({int(span)}f)" for label, span in zip(labels, spans)],
                      fontsize=6)
        ax.set_ylabel(f"{id_space} (lifespan), longest-lived first")
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"cell ({raster.shape[0]} {id_space}s, longest-lived first)")

    longest = int(spans.max()) if spans.size else 0

    def _pair_text(median_key: str, mean_key: str) -> str:
        median, mean = stats[median_key], stats[mean_key]
        if not np.isfinite(median):
            return "no pairs"
        return f"median {median:.3f}, mean {mean:.3f}"

    # Bottom-up: the first entry sits closest to the axes, the last under the
    # title.  Offsets are in points, not axes fraction -- this figure's height
    # varies with the cell count, so a fractional offset moves every time.
    header = [
        (f"different cells, same frame: {_pair_text('between_median', 'between_mean')}"
         f"   |   longest life {longest} frames", "#2e7d32"),
        (f"same cell, next observed frame: {_pair_text('within_median', 'within_mean')}", "#00788c"),
        (f"row labels are {id_space}s", "#555555"),
    ]
    if note:
        header.append((note, "#555555"))
    for line, (text, colour) in enumerate(header):
        ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 6 + 11 * line), textcoords="offset points",
                    fontsize=7, color=colour, va="bottom")
    ax.set_title(f"{crop_id}: DINO {metric} distance to the previous observed frame",
                 pad=17 + 11 * len(header))

    bar = figure.colorbar(image, ax=ax, pad=0.02, fraction=0.045, aspect=13, extend="max")
    bar.set_label(f"{metric} distance (colour clipped at {vmax:.2f})")
    stroke = [patheffects.withStroke(linewidth=2.4, foreground="black")]
    for key, colour, mark, _ in REFERENCE_LINES:
        value = stats[key]
        if not np.isfinite(value):
            continue
        bar.ax.axhline(value, color=colour, linewidth=1.4, path_effects=stroke)
        bar.ax.text(0.5, value, mark, transform=bar.ax.get_yaxis_transform(),
                    ha="center", va="bottom", fontsize=5, color=colour,
                    path_effects=stroke)

    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(cfg: dict, layout, out_dir: Path) -> None:
    """Write one raster per crop, plus the long-form distance table."""
    from arhmm import config as cfgmod
    from arhmm.core import io

    params = cfgmod.get_path(cfg, "outputs.dino_step_distance", {}) or {}
    metric = str(params.get("metric", "cosine"))
    if metric not in METRICS:
        raise ValueError(f"outputs.dino_step_distance.metric must be one of {METRICS}, got {metric!r}")
    note = params.get("note") or None
    if note is not None:
        note = str(note)

    known = list(cfgmod.crop_ids(cfg))
    requested = list(params.get("crops") or known)
    unknown = [crop for crop in requested if crop not in known]
    if unknown:
        raise ValueError(f"outputs.dino_step_distance.crops names crops not in this run: {unknown}")

    source = cfgmod.get_path(cfg, "cells.source")
    if source not in ID_SPACES:
        raise ValueError(f"cells.source must be one of {sorted(ID_SPACES)}, got {source!r}")
    slug, id_space = ID_SPACES[source]
    # Everything this extra writes is qualified by the label space, directory
    # included: the ids are meaningless without it and mislead across runs.
    out_dir = io.ensure_dir(out_dir / slug)
    print(f"  cells.source: {source} -> ids are {id_space}s, written to {out_dir.name}/")

    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for crop_id in requested:
        crop_dir = layout.crop_dir("dino", crop_id)
        arrays = io.load_npz(crop_dir / "embeddings.npz")
        raw, valid = arrays["raw"], arrays["valid_mask"]
        cell_ids = np.asarray(io.read_json(crop_dir / "meta.json")["cell_ids"], dtype=np.int64)

        raster, order, spans, records = _crop_raster(raw, valid, cell_ids, metric)
        if order.size == 0:
            print(f"  [{crop_id}] no observed cells; skipped")
            continue

        steps = np.array([record["distance"] for record in records], dtype=np.float64)
        within_median, within_mean = _centres(steps)
        between_median, between_mean = _between_cell_scale(raw, valid, metric)
        stats = {
            "within_median": within_median,
            "within_mean": within_mean,
            "between_median": between_median,
            "between_mean": between_mean,
        }

        _draw(raster, spans, cell_ids[order], crop_id, metric, stats, id_space, note,
              out_dir / f"dino_step_distance_{crop_id}_{slug}.png")

        for record in records:
            rows.append({"crop": crop_id, "cell_id_space": slug, **record})

        summary[crop_id] = {
            "cells": int(order.size),
            "steps": int(steps.size),
            "median_step_distance": within_median,
            "mean_step_distance": within_mean,
            "median_between_cell_distance": between_median,
            "mean_between_cell_distance": between_mean,
            "step_over_between_cell": (
                float(within_median / between_median)
                if np.isfinite(between_median) and between_median > 0 else float("nan")
            ),
            "max_lifespan_frames": int(spans.max()),
        }
        print(f"  [{crop_id}] {order.size} cells, {steps.size} steps; same cell "
              f"median {within_median:.4f} mean {within_mean:.4f}, different cells "
              f"median {between_median:.4f} mean {between_mean:.4f} "
              f"(median ratio {summary[crop_id]['step_over_between_cell']:.3f})")

    with io.atomic_write(out_dir / f"dino_step_distance_{slug}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["crop", "cell_id_space", "cell_id", "frame", "prev_frame",
                        "gap", "age", "distance"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "distance": f"{row['distance']:.6g}"})

    io.write_yaml(
        out_dir / f"dino_step_distance_summary_{slug}.yml",
        {"metric": metric, "cells_source": source, "cell_id_space": slug,
         "note": note, "crops": summary},
    )

    # A ratio near 1 means one frame of the same cell is as far as a different
    # cell entirely, which is the failure mode this extra exists to catch.
    worst = [name for name, stats in summary.items() if stats["step_over_between_cell"] > 0.8]
    if worst:
        print("  WARNING: consecutive frames are nearly as far apart as different cells in "
              f"{sorted(worst)}; the embedding is dominated by per-frame nuisance")
