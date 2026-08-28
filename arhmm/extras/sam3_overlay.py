"""Extra `sam3_overlay`: the SAM3 phase masks `cells.extend_nuclei` judged against.

One video per crop, alongside the base `overlays/` but in this extra's own
directory.  It answers the question the base overlay cannot: the extension holds
a nucleus track only while SAM3 still sees a cell body within `evidence_px`, and
nothing else in the run shows what SAM3 actually saw.  A held track whose SAM3
support looks like debris, or a rejection whose support was obviously real, is
visible here and nowhere else.

Three deliberate differences from `outputs/overlays/*_state_overlay.mp4`:

  * **SAM3 is drawn in one flat colour.**  Its labels are a third ID space,
    never reconciled with the nucleus or CVAT ones (see `core.cells`), and the
    extension reads it only as "is any pixel non-zero".  Colouring per label
    would imply an identity correspondence that does not exist.
  * **T cells are not drawn.**  They come from the CVAT stack, play no part in
    any extension criterion, and at this alpha they would be one more
    translucent layer over the region the eye needs to read.
  * Cancer nuclei keep their state colours, so a terminating track can be
    matched to the SAM3 blob that did or did not justify holding it.

Gated on the extension being on: with `cells.extend_nuclei.frames: 0` no SAM3
file is read by the run at all, and a video of masks nothing consulted would
misrepresent what happened.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REQUIRES: tuple[str, ...] = ()

#: RGB for every SAM3 mask, and its alpha over the phase image.  Chosen to sit
#: outside `tab10`, which the state colours come from, so a SAM3 region is never
#: mistakable for a state at video resolution.
SAM3_COLOUR = (1.0, 0.0, 1.0)
SAM3_ALPHA = 0.30


def _read_sam3(cfg: dict, crop_id: str, expected_shape: tuple[int, int, int]) -> np.ndarray:
    """The crop's SAM3 label stack, checked against the phase stack.

    Applies the same trailing-singleton tolerance as `cells._extend_nucleus_tracks`,
    so a `(T, H, W, 1)` export reads here exactly as it does there.

    Args:
        cfg (dict): resolved configuration.
        crop_id (str): the crop being drawn.
        expected_shape (tuple[int, int, int]): `(T, H, W)` of the phase stack.

    Returns:
        np.ndarray: `(T, H, W)` SAM3 labels.

    Raises:
        ValueError: the stack has the wrong rank, or disagrees with the phase
            stack on frame count or frame size -- the same three checks the
            planner makes, because a video drawn from a misaligned stack would
            look plausible and be wrong.
    """
    import tifffile

    from arhmm.core import cells as cellsmod

    path = cellsmod.sam3_tracks_path(cfg, crop_id)
    sam3 = tifffile.imread(str(path))
    if sam3.ndim == 4 and sam3.shape[-1] == 1:
        sam3 = sam3[..., 0]
    if sam3.ndim != 3:
        raise ValueError(f"{crop_id}: expected (T, H, W) SAM3 tracks in {path.name}, "
                         f"got shape {sam3.shape}")
    if sam3.shape[0] != expected_shape[0]:
        raise ValueError(f"{crop_id}: SAM3 {path.name} has {sam3.shape[0]} frames but the "
                         f"images have {expected_shape[0]}; these are different acquisitions")
    if sam3.shape[1:3] != expected_shape[1:3]:
        raise ValueError(f"{crop_id}: SAM3 {path.name} is {sam3.shape[1:3]} but the images "
                         f"are {expected_shape[1:3]}; the masks are not aligned")
    return sam3


def run(cfg: dict, layout, out_dir: Path) -> None:
    """Render one SAM3 overlay video per crop.

    Args:
        cfg (dict): resolved configuration.
        layout: the run's `Layout`.
        out_dir (Path): this extra's directory under `outputs/extras/`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from arhmm import config as cfgmod
    from arhmm.core import cells as cellsmod, io, viz

    params = cfgmod.nucleus_extension(cfg)
    if params is None:
        print("  cells.extend_nuclei is off: this run consults no SAM3 masks, nothing to draw")
        return

    fit_dir = layout.fit_dir
    states_all = np.load(fit_dir / "state_assignments.npy")
    active_all = np.load(fit_dir / "active_mask.npy")
    summary = io.read_yaml(fit_dir / "fit_summary.yml")
    with open(fit_dir / "cell_index.csv") as handle:
        index = list(csv.DictReader(handle))

    num_states = summary["num_states"]
    cmap, norm = viz.state_cmap_norm(num_states)
    colours = viz.state_colours(num_states)
    fps = int(cfgmod.get_path(cfg, "outputs.video.fps", 2))
    figsize = tuple(cfgmod.get_path(cfg, "outputs.video.figsize", [7, 7]))
    dpi = int(cfgmod.get_path(cfg, "outputs.video.dpi", 160))
    label_ids = bool(cfgmod.get_path(cfg, "outputs.video.label_cell_ids", True))

    legend = [mpatches.Patch(color=colours[k], label=f"state {k}") for k in range(num_states)]
    legend.append(mpatches.Patch(color=(0.5, 0.5, 0.5), label="no state inferred"))
    legend.append(mpatches.Patch(color=SAM3_COLOUR, alpha=SAM3_ALPHA, label="SAM3 phase mask"))
    # Same column rule as the base overlay: a legend row grows with k until it
    # is wider than the figure, and constrained layout pays for it by shrinking
    # the image to a thumbnail.
    legend_cols = 1 + (len(legend) - 1) // 20

    for crop_id in cfgmod.video_crops(cfg):
        columns = [i for i, e in enumerate(index) if e["crop"] == crop_id]
        if not columns:
            continue
        crop = cellsmod.load_crop(cfg, crop_id, with_image=True)
        phase = crop.image[..., 1]
        sam3 = _read_sam3(cfg, crop_id, phase.shape)
        cell_ids = np.array([int(index[c]["cell_id"]) for c in columns], dtype=np.int32)
        states = states_all[:, columns]
        active = active_all[:, columns]

        # Every argument the closure needs is bound as a default: frames_to_mp4
        # calls draw lazily, so a closure over `crop_id` would render the last
        # crop into every video.
        def draw(t, _phase=phase, _sam3=sam3, _cancer=crop.cancer, _ids=cell_ids,
                 _states=states, _active=active, _crop=crop_id):
            ax = plt.gca()
            ax.imshow(_phase[t], cmap="gray", vmin=0.0, vmax=1.0)
            # One flat colour for every SAM3 label: an RGBA image rather than a
            # colormap, so no label value can map to a different shade.
            sam3_rgba = np.zeros(_sam3.shape[1:3] + (4,), dtype=float)
            sam3_rgba[_sam3[t] != 0] = (*SAM3_COLOUR, SAM3_ALPHA)
            ax.imshow(sam3_rgba, interpolation="nearest")
            label_image = viz.state_label_image(_cancer[t], _ids, _states[t], _active[t])
            ax.imshow(np.ma.masked_where(label_image == 0, label_image),
                      cmap=cmap, norm=norm, alpha=0.55, interpolation="nearest")
            if label_ids:
                for cell_id in _ids:
                    pixels = np.argwhere(_cancer[t] == cell_id)
                    if pixels.size:
                        y, x = pixels.mean(axis=0)
                        ax.text(x, y, str(cell_id), color="white", fontsize=6,
                                fontweight="bold", ha="center", va="center")
            ax.set_title(
                f"{summary['run_name']} | {_crop} | frame {t}/{_phase.shape[0] - 1}\n"
                f"SAM3 phase masks; extension holds {params['frames']} frames, "
                f"exclusion {params['exclusion_px']} px / evidence {params['evidence_px']} px",
                fontsize=8,
            )
            ax.axis("off")
            ax.legend(
                handles=legend, loc="center left", bbox_to_anchor=(1.01, 0.5),
                ncol=legend_cols, frameon=False, fontsize=8,
                handlelength=1.2, handletextpad=0.6, labelspacing=0.7,
                borderaxespad=0.0,
            )

        path = viz.frames_to_mp4(
            draw, range(phase.shape[0]), out_dir / f"{crop_id}_sam3_overlay.mp4",
            fps=fps, figsize=figsize, dpi=dpi,
        )
        print(f"  wrote {path.name}")
