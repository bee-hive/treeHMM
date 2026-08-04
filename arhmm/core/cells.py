"""Cell-source adapters: what a "cancer cell" is, and where its pixels are.

This is the **only** module in the package that opens a track or image file.
Everything downstream is written against the `CropCells` interface, so nothing
else branches on where the masks came from.

The ground-truth CVAT tracks are one label stack per crop:

    {ground_truth_tracks_dir}/{well}/{crop_id}/ALL_tracks.tiff   (T, H, W) uint16
                                              /ALL_cancer_ids.pkl   list[int]
                                              /ALL_graph.pkl        {child: parent}

`ALL_tracks.tiff` holds both cell types in a single ID space and
`ALL_cancer_ids.pkl` lists the IDs that are cancer -- so T cells are every other
non-zero ID.  `ALL_graph.pkl` is the division lineage, and is deliberately
**not read**: the pipeline does not use the tree half of the model, so every
cell is an independent chain (see `tree_input.md`).

Two definitions of a cancer cell are implemented, chosen by `cells.source`:

    phase   Cancer cells are the ground-truth tracks listed in
            ALL_cancer_ids.pkl.  Masks cover the whole cell body, so shape
            features describe the cell.

    nuclei  Cancer cells are the Caliban nuclei tracks.  Cell identity is
            cleaner, but the masks are nuclei, so shape features describe the
            nucleus, and a fixed dilation radius reaches less far out of it.

The Caliban nuclei live in a **flat** tree -- one file per crop, with no
per-well directory level, unlike either tree above:

    {caliban_tracks_dir}/{crop_id}.tiff   (T, H, W, 1) int64

Caliban ran on the cancer nuclei only.  There is no T-cell equivalent and no
`ALL_cancer_ids.pkl` analogue, so every non-zero label in that file is a cancer
cell, and the T-cell masks still come from the CVAT stack under
`cells.source: nuclei`.  That leaves a crop described by two unrelated ID
spaces, and they are **never reconciled**: nothing here maps a nucleus label
onto a CVAT track ID, by overlap or otherwise.  Nothing needs it.  `cell_ids` is
whatever the chosen source calls its cancer cells, T cells are only ever counted
as neighbours, and switching source swaps the label space whole -- for the
features, the DINO patches, the fit and the overlays alike.

The image comes from a different tree than the tracks:

    {image_crops_dir}/{well}/{crop_id}/crop.tiff   (T, H, W, 2) float32
                                                  channel 0 = RFP, 1 = phase

That file is the only image source spatially aligned with the tracks.  Slicing a
per-well `registered.tiff` by the crop_id's coordinates lands on a different
region -- a real bug in an earlier pipeline, caught by comparing RFP-inside-mask
ratios.  There is one code path here for exactly that reason.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arhmm.core.io import well_of

#: Percentile pair used when no config value is supplied.
DEFAULT_NORM_PERCENTILES = (1.0, 99.0)


@dataclass
class CropCells:
    """One crop's masks, image and column order.

    Attributes:
        crop_id (str): the crop this describes.
        cancer (np.ndarray): `(T, H, W)` int32 label stack, cancer cells only.
        tcells (np.ndarray): `(T, H, W)` int32 label stack, every non-cancer ID.
        image (np.ndarray | None): `(T, H, W, 2)` float32 in [0, 1], channel 0
            RFP and channel 1 phase, normalized over the whole stack.  None when
            no requested feature needs pixel intensities.
        cell_ids (np.ndarray): `(N,)` int32, sorted -- **the column order**, and
            the contract every downstream stage asserts against.
        num_frames (int): `T`.
    """

    crop_id: str
    cancer: np.ndarray
    tcells: np.ndarray
    image: np.ndarray | None
    cell_ids: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.cancer.shape[0])

    @property
    def num_cells(self) -> int:
        return int(len(self.cell_ids))

    def presence(self) -> np.ndarray:
        """`(T, N)` bool: whether each cell has any pixels in each frame."""
        present = np.zeros((self.num_frames, self.num_cells), dtype=bool)
        for t in range(self.num_frames):
            labels = np.unique(self.cancer[t])
            present[t] = np.isin(self.cell_ids, labels[labels > 0])
        return present


def ground_truth_dir(cfg: dict, crop_id: str) -> Path:
    """Directory holding a crop's ground-truth tracks."""
    from arhmm.config import get_path

    root = Path(get_path(cfg, "paths.ground_truth_tracks_dir"))
    return root / well_of(crop_id) / crop_id


def image_path(cfg: dict, crop_id: str) -> Path:
    """The crop's aligned phase/RFP image."""
    from arhmm.config import get_path

    root = Path(get_path(cfg, "paths.image_crops_dir"))
    return root / well_of(crop_id) / crop_id / "crop.tiff"


def nucleus_tracks_path(cfg: dict, crop_id: str) -> Path:
    """The crop's Caliban nucleus tracks.

    Unlike the CVAT tracks and the image crops, these are stored flat: one
    `{crop_id}.tiff` directly under the root, with no per-well directory level.
    """
    from arhmm.config import get_path

    root = Path(get_path(cfg, "paths.caliban_tracks_dir"))
    return root / f"{crop_id}.tiff"


def _read_tiff(path: Path) -> np.ndarray:
    import tifffile

    if not path.is_file():
        raise FileNotFoundError(f"expected image or track stack at {path}")
    return tifffile.imread(str(path))


def _read_pickle(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"expected {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def normalize_stack(stack: np.ndarray, percentiles=DEFAULT_NORM_PERCENTILES) -> np.ndarray:
    """Scale a `(T, H, W)` stack into [0, 1] using **whole-stack** percentiles.

    Normalizing once over the whole stack rather than per frame is deliberate: a
    per-frame min-max lets a single bright artifact rescale that frame, which
    turns intensity features -- and, downstream, the leading DINO principal
    components -- into a clock that tracks acquisition rather than phenotype.

    Args:
        stack (np.ndarray): `(T, H, W)` intensities.
        percentiles (tuple[float, float]): low and high percentile.

    Returns:
        np.ndarray: `(T, H, W)` float32 in [0, 1].
    """
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return np.zeros_like(stack, dtype=np.float32)
    low, high = np.percentile(finite, percentiles)
    if not np.isfinite(high) or high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        return np.zeros_like(stack, dtype=np.float32)
    return np.clip((stack.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def _load_cvat_tracks(cfg: dict, crop_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the CVAT stack and split its single ID space by cell type.

    Both sources come through here: `phase` takes its cancer masks and column
    order from this file, and `nuclei` still takes its T-cell masks from it,
    because Caliban segmented the cancer nuclei only.

    Args:
        cfg (dict): resolved configuration.
        crop_id (str): the crop to read.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: the `(T, H, W)` int32 cancer
            stack, the `(T, H, W)` int32 T-cell stack, and the `(N,)` int32
            sorted cancer IDs.

    Raises:
        ValueError: if the stack is not `(T, H, W)`, or if the cancer ID list
            names IDs absent from the track stack.
    """
    crop_dir = ground_truth_dir(cfg, crop_id)
    tracks = _read_tiff(crop_dir / "ALL_tracks.tiff")
    if tracks.ndim != 3:
        raise ValueError(f"{crop_id}: expected (T, H, W) tracks, got shape {tracks.shape}")
    tracks = tracks.astype(np.int32, copy=False)

    cancer_ids = np.asarray(sorted(set(int(i) for i in _read_pickle(crop_dir / "ALL_cancer_ids.pkl"))),
                            dtype=np.int32)
    present_ids = np.unique(tracks)
    present_ids = present_ids[present_ids > 0]
    missing = np.setdiff1d(cancer_ids, present_ids)
    if missing.size:
        raise ValueError(
            f"{crop_id}: ALL_cancer_ids.pkl names IDs absent from ALL_tracks.tiff: "
            f"{missing.tolist()}"
        )

    # The single ID space is split into two label stacks so that downstream code
    # never has to re-derive "which of these labels is a T cell".  The check
    # above stays live for `nuclei` too: that source ignores the cancer IDs as a
    # column order, but the split below still depends on them being right, and a
    # stale list would corrupt the T-cell masks silently.
    is_cancer = np.isin(tracks, cancer_ids)
    cancer = np.where(is_cancer, tracks, 0).astype(np.int32)
    tcells = np.where(is_cancer | (tracks == 0), 0, tracks).astype(np.int32)
    return cancer, tcells, cancer_ids


def _load_nucleus_tracks(cfg: dict, crop_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the crop's Caliban nucleus tracks and the labels in them.

    Every non-zero label is a cancer nucleus -- Caliban never saw the T cells --
    so unlike the CVAT stack there is no ID list to consult and nothing to
    filter out.

    Args:
        cfg (dict): resolved configuration.
        crop_id (str): the crop to read.

    Returns:
        tuple[np.ndarray, np.ndarray]: the `(T, H, W)` int32 nucleus label stack
            and the `(N,)` int32 sorted non-zero labels -- the column order.

    Raises:
        ValueError: if the stack is neither `(T, H, W)` nor `(T, H, W, 1)`.
    """
    path = nucleus_tracks_path(cfg, crop_id)
    nuclei = _read_tiff(path)
    # Caliban writes a trailing singleton channel axis.  Dropping it here rather
    # than in every consumer is what lets `CropCells.cancer` mean one shape
    # whatever the source.
    if nuclei.ndim == 4 and nuclei.shape[-1] == 1:
        nuclei = nuclei[..., 0]
    if nuclei.ndim != 3:
        raise ValueError(
            f"{crop_id}: expected (T, H, W) or (T, H, W, 1) nucleus tracks in "
            f"{path.name}, got shape {nuclei.shape}"
        )
    # int64 on disk, but the labels number in the tens and every other label
    # stack in this package is int32.
    nuclei = nuclei.astype(np.int32, copy=False)

    labels = np.unique(nuclei)
    return nuclei, labels[labels > 0].astype(np.int32, copy=False)


def _load_image(cfg: dict, crop_id: str, ref_shape: tuple, ref_name: str) -> np.ndarray:
    """Read `crop.tiff` and normalize both channels over the whole stack.

    Args:
        cfg (dict): resolved configuration.
        crop_id (str): the crop to read.
        ref_shape (tuple): `(T, H, W)` of the cancer label stack the image has
            to line up with.
        ref_name (str): that stack's file name, so the error says which of the
            two track sources the image disagrees with.

    Returns:
        np.ndarray: `(T, H, W, 2)` float32 in [0, 1], channel 0 RFP, 1 phase.

    Raises:
        ValueError: if the file is not `(T, H, W, 2)`, or if it disagrees with
            `ref_shape` on frame count or frame size.
    """
    from arhmm.config import get_path

    raw = _read_tiff(image_path(cfg, crop_id))
    if raw.ndim != 4 or raw.shape[-1] < 2:
        raise ValueError(f"{crop_id}: expected (T, H, W, 2) crop.tiff, got {raw.shape}")
    if raw.shape[0] != ref_shape[0]:
        raise ValueError(
            f"{crop_id}: crop.tiff has {raw.shape[0]} frames but {ref_name} has "
            f"{ref_shape[0]}; these are different acquisitions"
        )
    if raw.shape[1:3] != ref_shape[1:3]:
        raise ValueError(
            f"{crop_id}: crop.tiff is {raw.shape[1:3]} but the tracks are "
            f"{ref_shape[1:3]}; the image is not aligned with the tracks"
        )
    percentiles = tuple(get_path(cfg, "features.params.norm_percentiles",
                                 DEFAULT_NORM_PERCENTILES))
    return np.stack(
        [normalize_stack(raw[..., 0], percentiles), normalize_stack(raw[..., 1], percentiles)],
        axis=-1,
    )


def load_crop(cfg: dict, crop_id: str, *, with_image: bool = True) -> CropCells:
    """Load one crop's cancer masks, T-cell masks, image and column order.

    The two sources differ in exactly one thing: where the cancer masks and the
    column order come from.  The T-cell masks are the CVAT non-cancer IDs either
    way, and the image is the same file either way.

    Args:
        cfg (dict): resolved configuration.
        crop_id (str): the crop to load.
        with_image (bool): read and normalize `crop.tiff`.  Skipped when nothing
            requested needs pixel intensities, which saves reading a 10 MB file
            per crop.

    Returns:
        CropCells: the loaded crop.

    Raises:
        ValueError: for an unknown `cells.source`; if the cancer ID list names
            IDs absent from the track stack; or if the nucleus stack, the CVAT
            stack and `crop.tiff` disagree on frame count or frame size.
    """
    from arhmm.config import get_path

    source = get_path(cfg, "cells.source")
    if source not in ("phase", "nuclei"):
        raise ValueError(f"unknown cells.source: {source!r}")

    # Read unconditionally: `phase` gets its cancer masks here, and `nuclei` gets
    # its T cells here, since Caliban has no T-cell equivalent.
    cvat_cancer, tcells, cvat_cancer_ids = _load_cvat_tracks(cfg, crop_id)

    if source == "phase":
        cancer, cell_ids, ref_name = cvat_cancer, cvat_cancer_ids, "ALL_tracks.tiff"
    else:
        cancer, cell_ids = _load_nucleus_tracks(cfg, crop_id)
        ref_name = nucleus_tracks_path(cfg, crop_id).name
        # The nucleus labels and the CVAT IDs are unrelated ID spaces and stay
        # that way.  All that has to hold is that the two files describe the same
        # pixels, or the T-cell neighbour features would be measuring distances
        # in someone else's field of view.
        if cancer.shape[0] != cvat_cancer.shape[0]:
            raise ValueError(
                f"{crop_id}: {ref_name} has {cancer.shape[0]} frames but ALL_tracks.tiff "
                f"has {cvat_cancer.shape[0]}; these are different acquisitions"
            )
        if cancer.shape[1:3] != cvat_cancer.shape[1:3]:
            raise ValueError(
                f"{crop_id}: {ref_name} is {cancer.shape[1:3]} but the CVAT tracks are "
                f"{cvat_cancer.shape[1:3]}; the nucleus masks are not aligned with the "
                f"T-cell masks"
            )

    image = _load_image(cfg, crop_id, cancer.shape, ref_name) if with_image else None
    return CropCells(crop_id=crop_id, cancer=cancer, tcells=tcells, image=image, cell_ids=cell_ids)
