"""Plotting and video helpers.

Self-contained on purpose: the equivalents in `MarsonImagingPipeline` are fine,
but importing them drags that repo's whole dependency set onto `sys.path` and
ties the pipeline to a sibling checkout.  The pieces that were actually used
amount to well under a hundred lines, so they live here instead.

Runs in the imaging environment only (matplotlib, imageio).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

#: Label value meaning "this cell exists here but no state was inferred for it"
#: -- the warmup frames.  0 is background and 1..K are states, so a negative
#: value cannot collide with either.
WARMUP_LABEL = -1


def apply_style() -> None:
    """Small, consistent, and vector-friendly figure defaults."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Keep text as text when these are dropped into a figure.
            "svg.fonttype": "none",
            "pdf.use14corefonts": True,
        }
    )


def state_colours(num_states: int) -> list[tuple[float, float, float]]:
    """One RGB colour per state, distinguishable at video resolution."""
    import matplotlib.pyplot as plt

    base = plt.get_cmap("tab10")
    return [tuple(base(i % 10)[:3]) for i in range(num_states)]


def state_cmap_norm(num_states: int):
    """A colormap/norm pair for a state label image.

    The label image encodes `0` background, `WARMUP_LABEL` uninferred, and
    `1..K` for states, so one `imshow` draws all three without compositing.

    Args:
        num_states (int): `K`.

    Returns:
        tuple: `(ListedColormap, BoundaryNorm)` covering `[-1, K]`.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colours = [
        (0.5, 0.5, 0.5, 1.0),  # -1 warmup / uninferred
        (0.0, 0.0, 0.0, 0.0),  #  0 background, fully transparent
    ]
    colours += [(*rgb, 1.0) for rgb in state_colours(num_states)]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(-1.5, num_states + 1.5), cmap.N)
    return cmap, norm


def state_label_image(
    labels: np.ndarray,
    cell_ids: np.ndarray,
    states: np.ndarray,
    inferred: np.ndarray,
) -> np.ndarray:
    """Recolour one frame's label image by inferred state, via a lookup table.

    Uses a LUT indexed by track ID rather than a Python loop over cells, which
    matters once a crop has hundreds of them.

    Args:
        labels (np.ndarray): `(H, W)` track label image for this frame.
        cell_ids (np.ndarray): `(N,)` track IDs, in column order.
        states (np.ndarray): `(N,)` inferred state per column for this frame.
        inferred (np.ndarray): `(N,)` bool, whether a state was inferred.

    Returns:
        np.ndarray: `(H, W)` int16 with 0 background, `WARMUP_LABEL` uninferred,
            and `state + 1` elsewhere.
    """
    size = int(max(labels.max(), cell_ids.max() if len(cell_ids) else 0)) + 1
    lut = np.zeros(size, dtype=np.int16)
    lut[cell_ids] = np.where(inferred, states.astype(np.int16) + 1, WARMUP_LABEL)
    return lut[labels]


def frames_to_mp4(
    draw: Callable[[int], None],
    frames: Sequence[int],
    path: str | Path,
    fps: int = 2,
    figsize: tuple[float, float] = (7, 7),
    dpi: int = 160,
) -> Path:
    """Render one figure per frame and mux them into an mp4.

    `draw(t)` is called **lazily**, once per frame, and is expected to draw into
    the current figure.  Anything it needs from an enclosing loop must be bound
    as a default argument -- a closure over the loop variable renders the last
    iteration's data into every frame, which is a mistake that produced
    silently-wrong videos in two of the archived pipelines.

    Args:
        draw (Callable[[int], None]): draws frame `t` into the current figure.
        frames (Sequence[int]): frame indices, in order.
        path (str | Path): destination `.mp4`.
        fps (int): frames per second.
        figsize (tuple[float, float]): figure size in inches.
        dpi (int): render resolution.

    Returns:
        Path: the written file.
    """
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(dir=str(path.parent), prefix=f".{path.stem}."))
    try:
        rendered = []
        for t in frames:
            figure = plt.figure(figsize=figsize, layout="constrained")
            try:
                draw(t)
                png = workdir / f"{t:05d}.png"
                # No bbox_inches="tight": it crops to the drawn content, so a
                # frame where a label sits nearer the edge comes out a different
                # pixel size, and a video needs every frame identical.  A fixed
                # figsize x dpi is exactly constant.
                figure.savefig(png, dpi=dpi, facecolor="white")
                rendered.append(png)
            finally:
                plt.close(figure)

        images = [imageio.imread(png) for png in rendered]
        shapes = {image.shape for image in images}
        if len(shapes) != 1:
            raise ValueError(f"frames differ in size ({sorted(shapes)}); cannot encode a video")

        tmp_out = workdir / "out.mp4"
        with imageio.get_writer(
            str(tmp_out),
            fps=fps,
            codec="libx264",
            quality=8,
            # libx264 with yuv420p needs even dimensions; 2 pads to the nearest
            # even size rather than the default 16, which would add a visible
            # black border.
            macro_block_size=2,
        ) as writer:
            for image in images:
                writer.append_data(image)
        tmp_out.replace(path)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return path


def grid_shape(count: int, max_cols: int) -> tuple[int, int]:
    """Rows and columns for `count` panels at most `max_cols` wide."""
    cols = max(1, min(max_cols, count))
    rows = int(np.ceil(count / cols))
    return rows, cols


def annotate_matrix(ax, matrix: np.ndarray, labels: Sequence[str], title: str) -> None:
    """Draw a probability matrix with its values written into the cells.

    Args:
        ax: matplotlib axes.
        matrix (np.ndarray): `(K, K)` probabilities in [0, 1].
        labels (Sequence[str]): axis tick labels.
        title (str): axes title.
    """
    ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if value > 0.5 else "black",
            )


def blend(image: np.ndarray, mask: np.ndarray, colour: Sequence[float], alpha: float) -> np.ndarray:
    """Alpha-blend a flat colour into an RGB image where `mask` is true.

    Blending rather than flat-filling keeps the underlying texture visible,
    which is the whole point when the texture is what carries the phenotype.

    Args:
        image (np.ndarray): `(H, W, 3)` float in [0, 1].
        mask (np.ndarray): `(H, W)` bool.
        colour (Sequence[float]): RGB in [0, 1].
        alpha (float): weight given to `colour`.

    Returns:
        np.ndarray: a new `(H, W, 3)` image.
    """
    out = image.copy()
    if mask.any():
        out[mask] = (1.0 - alpha) * image[mask] + alpha * np.asarray(colour, dtype=image.dtype)
    return out
