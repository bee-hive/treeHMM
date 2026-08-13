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

    nuclei  Cancer cells are the nuclei tracks.  Cell identity is cleaner, but
            the masks are nuclei, so shape features describe the nucleus, and a
            fixed dilation radius reaches less far out of it.

The nuclei tracks sit in the crop's own directory, beside the image it is
aligned with, so they share a root with it rather than having one of their own:

    {image_crops_dir}/{well}/{crop_id}/nuclei_tracks.tiff   (T, H, W) int32

They cover the cancer nuclei only.  There is no T-cell equivalent and no
`ALL_cancer_ids.pkl` analogue, so every non-zero label in that file is a cancer
cell, and the T-cell masks still come from the CVAT stack under
`cells.source: nuclei`.  That leaves a crop described by two unrelated ID
spaces, and they are **never reconciled**: nothing here maps a nucleus label
onto a CVAT track ID, by overlap or otherwise.  Nothing needs it.  `cell_ids` is
whatever the chosen source calls its cancer cells, T cells are only ever counted
as neighbours, and switching source swaps the label space whole -- for the
features, the DINO patches, the fit and the overlays alike.

`cells.extend_nuclei` optionally holds a nucleus track that simply stops -- the
nucleus signal is lost, possibly at death, while the cell body is still there --
for a few more frames, copying its final mask verbatim.  It reads two more files
from the same crop directory and one from a root of its own:

    {image_crops_dir}/{well}/{crop_id}/nuclei_div.pkl   parent/daughter/frame
    {sam3_tracks_dir}/{crop_id}/tracks.tiff             (T, H, W) uint16

Note the SAM3 layout is **flat** -- one directory per crop id, not nested under
the well like everything else here.  Those tracks are a *third* ID space, and
like the other two they are never reconciled with anything: they are read only
as "is any pixel non-zero", so no SAM3 label ever reaches `CropCells`.
`nuclei_div.pkl` is read only to tell a nucleus that divided from one that
vanished; it does **not** feed the tree half of the model, and no division ever
becomes a lineage flag.  The decision is a pure function of those files, so the
three steps that call `load_crop` cannot disagree about it.

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

#: Why a terminating nucleus track was or was not held for more frames.  Fixed
#: vocabulary so the sidecar CSV can be tallied without parsing prose.
EXTENSION_REASONS = (
    "extended",
    "ends_at_movie_end",
    "divides",
    "neighbour_in_box",
    "no_sam3_evidence",
    "empty_track",
)


@dataclass(frozen=True)
class ExtensionRecord:
    """One terminating nucleus track, and what `cells.extend_nuclei` did with it.

    Every candidate gets a record, rejected ones included, so the sidecar is a
    complete census rather than a list of winners.

    Attributes:
        cell_id (int): the nucleus label.
        final_frame (int): its **last** appearance, so a track with an interior
            gap is anchored at the end of the track and not at the gap.  -1 for
            a label with no pixels anywhere.
        frames_available (int): `min(frames, T - 1 - final_frame)` -- how many
            frames the end of the movie leaves room for.  The criteria are
            judged over exactly these, never over the configured `frames`.
        frames_added (int): `frames_available` or 0.  Never anything between:
            the all-or-nothing rule lives here.
        reason (str): one of `EXTENSION_REASONS`.
        first_failing_frame (int): absolute index of the frame that produced
            `reason`, or -1 when nothing failed or nothing was evaluated.
        divides (bool | None): whether the label is a parent in
            `nuclei_div.pkl`.  None when not evaluated.
        neighbour_in_box (bool | None): whether another nucleus intruded on the
            exclusion box.  None when not evaluated.
        sam3_evidence (bool | None): whether SAM3 saw something in the evidence
            box in **every** frame.  None when not evaluated.
        blocking_labels (str): `;`-joined nucleus labels found in the exclusion
            box, empty when there were none.
    """

    cell_id: int
    final_frame: int
    frames_available: int
    frames_added: int
    reason: str
    first_failing_frame: int
    divides: bool | None
    neighbour_in_box: bool | None
    sam3_evidence: bool | None
    blocking_labels: str


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
            the contract every downstream stage asserts against.  Extending
            tracks never adds or removes an ID, so this is the same either way.
        num_frames (int): `T`.
        extension (tuple[ExtensionRecord, ...] | None): None when
            `cells.extend_nuclei` was off, and a tuple -- possibly empty -- when
            it ran.  That distinction is what lets the `features` step decide
            whether to write the sidecar without re-reading the config.
    """

    crop_id: str
    cancer: np.ndarray
    tcells: np.ndarray
    image: np.ndarray | None
    cell_ids: np.ndarray
    extension: tuple[ExtensionRecord, ...] | None = None

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
    """The crop's nucleus tracks.

    These live in the crop directory alongside `crop.tiff`, so they come off the
    image root rather than a root of their own.
    """
    from arhmm.config import get_path

    root = Path(get_path(cfg, "paths.image_crops_dir"))
    return root / well_of(crop_id) / crop_id / "nuclei_tracks.tiff"


def nucleus_divisions_path(cfg: dict, crop_id: str) -> Path:
    """The crop's nucleus division lineage.

    Sits beside `nuclei_tracks.tiff` in the crop directory, so it comes off the
    image root in the same nested layout.
    """
    from arhmm.config import get_path

    root = Path(get_path(cfg, "paths.image_crops_dir"))
    return root / well_of(crop_id) / crop_id / "nuclei_div.pkl"


def sam3_tracks_path(cfg: dict, crop_id: str) -> Path:
    """The crop's SAM3 phase tracks.

    **Flat layout** -- one directory per crop id, not nested under the well like
    every other root this module reads.  Do not reach for `well_of` here.
    """
    from arhmm.config import get_path

    root = Path(get_path(cfg, "paths.sam3_tracks_dir"))
    return root / crop_id / "tracks.tiff"


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
    """Read the crop's nucleus tracks and the labels in them.

    Every non-zero label is a cancer nucleus -- the segmentation never saw the
    T cells -- so unlike the CVAT stack there is no ID list to consult and
    nothing to filter out.

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
    # Some exports carry a trailing singleton channel axis.  Dropping it here
    # rather than in every consumer is what lets `CropCells.cancer` mean one
    # shape whatever the source.
    if nuclei.ndim == 4 and nuclei.shape[-1] == 1:
        nuclei = nuclei[..., 0]
    if nuclei.ndim != 3:
        raise ValueError(
            f"{crop_id}: expected (T, H, W) or (T, H, W, 1) nucleus tracks in "
            f"{path.name}, got shape {nuclei.shape}"
        )
    # The labels number in the tens whatever the dtype on disk, and every other
    # label stack in this package is int32.
    nuclei = nuclei.astype(np.int32, copy=False)

    labels = np.unique(nuclei)
    return nuclei, labels[labels > 0].astype(np.int32, copy=False)


def _load_nucleus_divisions(cfg: dict, crop_id: str, cell_ids: np.ndarray) -> frozenset[int]:
    """The nucleus labels that divide, read from `nuclei_div.pkl`.

    Division is looked up, never inferred.  A geometric rule -- "two new labels
    appear beside this one in the next frame" -- gets it wrong whenever the
    parent vanishes before its daughters appear, which in this data happens for
    a sixth of the divisions, with gaps of up to four frames.  Extending such a
    parent would paint a phantom nucleus straight through its own division.

    Only the parent column is returned: a daughter is an ordinary track, and
    when it ends it is an ordinary candidate.

    Args:
        cfg (dict): resolved configuration.
        crop_id (str): the crop to read.
        cell_ids (np.ndarray): `(N,)` the crop's nucleus labels, for checking
            the table against the stack it describes.

    Returns:
        frozenset[int]: labels appearing in the `parent` column.

    Raises:
        FileNotFoundError: the file is missing.
        ValueError: it is not a table with the four expected columns, or it
            names a label absent from the track stack.
    """
    # Imported here, not at module scope: this module is imported by
    # `arhmm.config` and so by every step in all three conda environments, and
    # `tifffile` is kept out of the module namespace for the same reason.
    import pandas as pd

    path = nucleus_divisions_path(cfg, crop_id)
    table = _read_pickle(path)
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"{crop_id}: expected a DataFrame in {path.name}, got {type(table)}")
    expected = ("parent", "daughter_1", "daughter_2", "frame")
    missing = [name for name in expected if name not in table.columns]
    if missing:
        raise ValueError(
            f"{crop_id}: {path.name} is missing column(s) {missing}; "
            f"it has {list(table.columns)}"
        )

    known = set(int(label) for label in cell_ids)
    named: set[int] = set()
    for column in ("parent", "daughter_1", "daughter_2"):
        named.update(int(value) for value in table[column])
    unknown = sorted(named - known)
    if unknown:
        raise ValueError(
            f"{crop_id}: {path.name} names labels absent from "
            f"{nucleus_tracks_path(cfg, crop_id).name}: {unknown}"
        )
    # Plain ints, so no pandas object escapes this function into `CropCells` or
    # into the pure planner.
    return frozenset(int(value) for value in table["parent"])


def _box_slices(centroid: tuple[float, float], side: int, shape: tuple[int, int]):
    """The pixels a DINO patch of edge `side` would cover, clipped to the frame.

    Deliberately the same arithmetic as `core.dino.subject_patch`: `half =
    side // 2`, anchored at `int(round(c)) - half`, extent `side`.  That is what
    makes `exclusion_px: 50` mean *exactly* the window a 50 px patch sees, so
    "no neighbour in the box" and "no neighbour in the patch" cannot drift
    apart.  `int(round(...))` is copied verbatim, ties included -- rewriting it
    as `floor(c + 0.5)` shifts the box a pixel at every `.5`.

    Clipped rather than edge-padded, because a pixel outside the frame holds
    neither a nucleus nor a SAM3 label, so it can neither block an extension nor
    support one.

    Args:
        centroid (tuple[float, float]): `(y, x)`.
        side (int): box edge, in pixels.
        shape (tuple[int, int]): `(H, W)` of the frame.

    Returns:
        tuple[slice, slice]: row and column slices into a frame.
    """
    height, width = shape
    half = side // 2
    top = int(round(centroid[0])) - half
    left = int(round(centroid[1])) - half
    return (
        slice(max(0, top), min(height, top + side)),
        slice(max(0, left), min(width, left + side)),
    )


def _centroid_of(frame: np.ndarray, label: int) -> tuple[float, float] | None:
    """Mean `(y, x)` of a label's pixels in one frame, or None when it has none.

    Plain numpy on purpose: this is the same number `regionprops` reports, and
    this module has to stay importable in an environment without scikit-image.
    """
    rows, cols = np.nonzero(frame == label)
    if rows.size == 0:
        return None
    return float(rows.mean()), float(cols.mean())


def plan_nucleus_extension(
    nuclei: np.ndarray,
    cell_ids: np.ndarray,
    sam3: np.ndarray,
    parents: frozenset[int],
    *,
    frames: int,
    exclusion_px: int,
    evidence_px: int,
) -> tuple[ExtensionRecord, ...]:
    """Decide which terminating nucleus tracks to hold, and for how long.

    Pure: no config, no filesystem, no randomness, and no argument is modified.

    **Every criterion is evaluated against `nuclei` exactly as given, never
    against a partially extended copy.** Otherwise whether one track extends
    would depend on whether another was considered first -- a track painted into
    later frames can land in a second track's exclusion box where it originally
    had nothing. Judging against the original makes the plan a function of the
    input alone, which is what lets `features`, `dino` and `outputs` each call
    `load_crop` and get the same answer.

    The end of the movie clips rather than rejects: a track with three frames of
    room is judged over three and, passing, gains three.  Within that budget it
    is all-or-nothing.

    Args:
        nuclei (np.ndarray): `(T, H, W)` nucleus label stack.
        cell_ids (np.ndarray): `(N,)` sorted labels -- the visiting order, so
            the returned tuple's order is fixed.
        sam3 (np.ndarray): `(T, H, W)` SAM3 phase labels on the same grid.  Read
            only as "is any pixel non-zero"; no label of it is ever used.
        parents (frozenset[int]): labels that divide, from `nuclei_div.pkl`.
        frames (int): frames to append at most.
        exclusion_px (int): side of the box no other nucleus may enter.
        evidence_px (int): side of the box SAM3 must fill in every frame.

    Returns:
        tuple[ExtensionRecord, ...]: one record per label, in `cell_ids` order.
    """
    num_frames = int(nuclei.shape[0])
    frame_shape = (int(nuclei.shape[1]), int(nuclei.shape[2]))

    present = np.zeros((num_frames, len(cell_ids)), dtype=bool)
    for t in range(num_frames):
        labels = np.unique(nuclei[t])
        present[t] = np.isin(cell_ids, labels[labels > 0])

    records = []
    for column, raw_label in enumerate(cell_ids):
        label = int(raw_label)
        active = np.flatnonzero(present[:, column])
        if active.size == 0:
            records.append(ExtensionRecord(label, -1, 0, 0, "empty_track", -1,
                                           None, None, None, ""))
            continue

        # The LAST appearance, so a track with an interior gap is anchored at
        # the end of the track rather than at the gap.
        final_frame = int(active[-1])
        available = min(int(frames), num_frames - 1 - final_frame)
        if available <= 0:
            records.append(ExtensionRecord(label, final_frame, 0, 0,
                                           "ends_at_movie_end", -1, None, None, None, ""))
            continue

        centroid = _centroid_of(nuclei[final_frame], label)
        exclusion = _box_slices(centroid, exclusion_px, frame_shape)
        evidence = _box_slices(centroid, evidence_px, frame_shape)

        divides = label in parents

        neighbour_in_box = False
        neighbour_frame = -1
        blocking: list[int] = []
        for step in range(1, available + 1):
            window = nuclei[final_frame + step][exclusion]
            others = np.unique(window)
            others = others[others != 0]
            if label in others:
                raise ValueError(
                    f"nucleus {label} appears at frame {final_frame + step}, after the "
                    f"frame {final_frame} its presence mask calls its last"
                )
            if others.size:
                neighbour_in_box = True
                neighbour_frame = final_frame + step
                blocking = [int(value) for value in others]
                break

        sam3_evidence = True
        sam3_frame = -1
        for step in range(1, available + 1):
            if not sam3[final_frame + step][evidence].any():
                sam3_evidence = False
                sam3_frame = final_frame + step
                break

        # All three are evaluated above rather than short-circuited, so the
        # sidecar supports both tallies: `reason` attributes each track to one
        # cause, while the flags count each criterion independently.  `reason`
        # itself takes the first failure in this fixed order.
        if divides:
            reason, failing = "divides", -1
        elif neighbour_in_box:
            reason, failing = "neighbour_in_box", neighbour_frame
        elif not sam3_evidence:
            reason, failing = "no_sam3_evidence", sam3_frame
        else:
            reason, failing = "extended", -1

        records.append(ExtensionRecord(
            cell_id=label,
            final_frame=final_frame,
            frames_available=available,
            frames_added=available if reason == "extended" else 0,
            reason=reason,
            first_failing_frame=failing,
            divides=divides,
            neighbour_in_box=neighbour_in_box,
            sam3_evidence=sam3_evidence,
            blocking_labels=";".join(str(value) for value in blocking),
        ))
    return tuple(records)


def apply_nucleus_extension(
    nuclei: np.ndarray, records: tuple[ExtensionRecord, ...]
) -> np.ndarray:
    """Paint each held track's final mask into the frames it won.

    Returns a new array; `nuclei` is never modified.  The mask is copied at the
    same pixel positions, so shape, area and centroid are identical by
    construction -- there is no resampling and no re-centring.

    Every painted pixel is checked against the **accumulating** result rather
    than the original, so a collision between two extensions is caught as well
    as a collision with a real track.  It raises rather than warning because the
    planner is supposed to have made it impossible: a nucleus mask that reached
    outside its own exclusion box would mean the box no longer means what the
    criteria assume.

    Args:
        nuclei (np.ndarray): `(T, H, W)` nucleus label stack.
        records (tuple[ExtensionRecord, ...]): the plan.

    Returns:
        np.ndarray: the stack with held tracks painted forward, or `nuclei`
            itself when the plan holds nothing.

    Raises:
        ValueError: a painted pixel was already occupied.
    """
    accepted = [record for record in records if record.frames_added > 0]
    if not accepted:
        return nuclei

    extended = nuclei.copy()
    for record in accepted:
        # From the ORIGINAL stack, so the mask cannot pick up another track's
        # extension -- see the invariant in `plan_nucleus_extension`.
        mask = nuclei[record.final_frame] == record.cell_id
        for step in range(1, record.frames_added + 1):
            target = extended[record.final_frame + step]
            occupied = target[mask]
            clashes = np.unique(occupied[occupied != 0])
            if clashes.size:
                raise ValueError(
                    f"extending nucleus {record.cell_id} into frame "
                    f"{record.final_frame + step} would overwrite label(s) "
                    f"{clashes.tolist()}"
                )
            target[mask] = record.cell_id
    return extended


def _extend_nucleus_tracks(
    cfg: dict, crop_id: str, nuclei: np.ndarray, cell_ids: np.ndarray
) -> tuple[np.ndarray, tuple[ExtensionRecord, ...] | None]:
    """Read what the extension needs, plan it, and paint it.

    Returns `(nuclei, None)` unchanged when this run extends nothing.  The
    decision comes from `config.nucleus_extension`, which is also what the cache
    key hashes, so the key and the behaviour cannot disagree.

    Args:
        cfg (dict): resolved configuration.
        crop_id (str): the crop being loaded.
        nuclei (np.ndarray): `(T, H, W)` nucleus label stack.
        cell_ids (np.ndarray): `(N,)` sorted nucleus labels.

    Returns:
        tuple: the label stack and the per-candidate records, or None records.

    Raises:
        ValueError: the SAM3 stack has the wrong rank or disagrees with the
            nucleus stack on frame count or frame size, or the extension
            changed the set of labels.
    """
    from arhmm import config as cfgmod

    params = cfgmod.nucleus_extension(cfg)
    if params is None:
        return nuclei, None

    sam3 = _read_tiff(sam3_tracks_path(cfg, crop_id))
    # Same trailing-singleton tolerance as the nucleus stack.
    if sam3.ndim == 4 and sam3.shape[-1] == 1:
        sam3 = sam3[..., 0]
    if sam3.ndim != 3:
        raise ValueError(
            f"{crop_id}: expected (T, H, W) SAM3 tracks in "
            f"{sam3_tracks_path(cfg, crop_id).name}, got shape {sam3.shape}"
        )
    if sam3.shape[0] != nuclei.shape[0]:
        raise ValueError(
            f"{crop_id}: SAM3 tracks.tiff has {sam3.shape[0]} frames but the nucleus "
            f"tracks have {nuclei.shape[0]}; these are different acquisitions"
        )
    if sam3.shape[1:3] != nuclei.shape[1:3]:
        raise ValueError(
            f"{crop_id}: SAM3 tracks.tiff is {sam3.shape[1:3]} but the nucleus tracks "
            f"are {nuclei.shape[1:3]}; the phase masks are not aligned with the nuclei"
        )

    parents = _load_nucleus_divisions(cfg, crop_id, cell_ids)
    records = plan_nucleus_extension(
        nuclei,
        cell_ids,
        sam3,
        parents,
        frames=params["frames"],
        exclusion_px=params["exclusion_px"],
        evidence_px=params["evidence_px"],
    )
    try:
        extended = apply_nucleus_extension(nuclei, records)
    except ValueError as error:
        raise ValueError(f"{crop_id}: {error}") from error

    # `cell_ids` is the column order every downstream stage asserts against, and
    # holding a track must not disturb it.
    labels = np.unique(extended)
    labels = labels[labels > 0]
    if not np.array_equal(labels, np.asarray(cell_ids, dtype=labels.dtype)):
        raise ValueError(
            f"{crop_id}: extending nucleus tracks changed the label set; this is a bug"
        )
    return extended, records


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

    extension = None
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

        # After the alignment checks, so a genuinely misaligned crop still fails
        # with the message above rather than with a confusing SAM3 shape error;
        # before `_load_image`, so a missing SAM3 file fails without first
        # reading a 10 MB image.  Writes nothing: this runs in three steps, and
        # a shared cache directory needs exactly one writer.
        cancer, extension = _extend_nucleus_tracks(cfg, crop_id, cancer, cell_ids)

    image = _load_image(cfg, crop_id, cancer.shape, ref_name) if with_image else None
    return CropCells(crop_id=crop_id, cancer=cancer, tcells=tcells, image=image,
                     cell_ids=cell_ids, extension=extension)
