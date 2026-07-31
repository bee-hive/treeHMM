"""Construction of the four per-(frame, cell) masks the model runs on.

`models.tarhmm` drives everything from a dense `(T, C, D)` tensor with one fixed
column per cell, plus four aligned masks:

    active_mask       the cell exists and is observed at (t, cell)
    parent_indices    column of this cell's parent at t-1
    is_division_mask  (t, cell) is a division child
    is_new_root_mask  first frame the cell appears

**This pipeline does not use the tree.**  Divisions are out of scope, so:

    parent_indices    is always the cell's own column where it is active
    is_division_mask  is all False, everywhere, always
    is_new_root_mask  is True at each cell's first active frame

which makes every cell an independent chain and leaves the division kernel
`P_div` unexercised.  See `tree_input.md` section 4.  `assert_consistent` treats
those as invariants rather than conventions, because a filter that drops columns
without remapping `parent_indices` breaks inference **silently** rather than
loudly -- which is exactly the bug that survived in three of the four archived
copies of this logic.

Everything that changes the column space lives here, and every such function
ends by asserting consistency, so it is not possible to update one mask and
forget the other three.
"""

from __future__ import annotations

import numpy as np

MASK_KEYS = ("active_mask", "parent_indices", "is_division_mask", "is_new_root_mask")


class LineageError(AssertionError):
    """The masks are mutually inconsistent -- a bug, not a bad configuration."""


def build_masks(present: np.ndarray) -> dict[str, np.ndarray]:
    """Build the four masks for one crop from a presence matrix.

    Args:
        present (np.ndarray): `(T, N)` bool, whether each cell has pixels in
            each frame.

    Returns:
        dict[str, np.ndarray]: `active_mask` `(T, N)` bool, `parent_indices`
            `(T, N)` int32, `is_division_mask` `(T, N)` bool (all False), and
            `is_new_root_mask` `(T, N)` bool.
    """
    present = np.asarray(present, dtype=bool)
    num_frames, num_cells = present.shape

    active = present.copy()
    # Self-parenting everywhere, including inactive cell-frames: an inactive
    # entry is ignored by inference, and a column index pointing at itself is
    # the only value that can never be mistaken for a real edge.
    parents = np.tile(np.arange(num_cells, dtype=np.int32), (num_frames, 1))
    divisions = np.zeros((num_frames, num_cells), dtype=bool)
    roots = np.zeros((num_frames, num_cells), dtype=bool)

    for col in range(num_cells):
        frames = np.flatnonzero(active[:, col])
        if frames.size:
            roots[frames[0], col] = True

    masks = {
        "active_mask": active,
        "parent_indices": parents,
        "is_division_mask": divisions,
        "is_new_root_mask": roots,
    }
    assert_consistent(masks)
    return masks


def concatenate_crops(per_crop: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Lay several crops side by side in one column space.

    Only `parent_indices` needs shifting, since every other mask is per-column.
    Crops must agree on frame count -- the model's tensor has one time axis.

    Args:
        per_crop (list[dict[str, np.ndarray]]): each crop's masks, in the order
            the columns should appear.

    Returns:
        dict[str, np.ndarray]: the combined masks.

    Raises:
        ValueError: if the crops disagree on frame count.
    """
    if not per_crop:
        raise ValueError("no crops to concatenate")

    frame_counts = {m["active_mask"].shape[0] for m in per_crop}
    if len(frame_counts) != 1:
        raise ValueError(
            f"crops disagree on frame count: {sorted(frame_counts)}; the model's "
            f"tensor has a single time axis, so every crop must have the same T"
        )

    combined: dict[str, np.ndarray] = {}
    for key in MASK_KEYS:
        blocks = []
        offset = 0
        for masks in per_crop:
            block = masks[key]
            blocks.append(block + offset if key == "parent_indices" else block)
            offset += block.shape[1]
        combined[key] = np.concatenate(blocks, axis=1)

    assert_consistent(combined)
    return combined


def filter_short_cells(
    masks: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    min_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Drop columns observed for fewer than `min_frames` frames, remapping indices.

    Any code that filters columns must keep `parent_indices` consistent with the
    new index space.  With self-parenting that is a renumbering rather than a
    lineage repair, but it still has to happen, and getting it wrong corrupts
    inference without raising.

    Args:
        masks (dict[str, np.ndarray]): the four masks, `(T, C)` each.
        arrays (dict[str, np.ndarray]): companion arrays whose axis 1 is the
            column axis (emissions, diagnostics, centroids, ...).
        min_frames (int): minimum active frames for a column to survive.

    Returns:
        tuple:
            masks (dict[str, np.ndarray]): filtered masks.
            arrays (dict[str, np.ndarray]): filtered companions.
            keep (np.ndarray): `(C_kept,)` indices into the original columns.

    Raises:
        ValueError: if no column survives.
    """
    durations = masks["active_mask"].sum(axis=0)
    keep = np.flatnonzero(durations >= min_frames)
    if keep.size == 0:
        raise ValueError(
            f"cells.min_frames = {min_frames} removed every cell; the longest "
            f"track is {int(durations.max()) if durations.size else 0} frames"
        )

    filtered = {key: masks[key][:, keep] for key in MASK_KEYS}

    # Renumber parent columns into the new index space.  -1 marks a parent that
    # did not survive; with self-parenting that can only happen if the cell
    # itself was dropped, so it should never appear on a surviving active entry.
    lookup = np.full(masks["parent_indices"].max() + 2, -1, dtype=np.int32)
    lookup[keep] = np.arange(keep.size, dtype=np.int32)
    filtered["parent_indices"] = lookup[filtered["parent_indices"]]

    orphaned = (filtered["parent_indices"] < 0) & filtered["active_mask"]
    if orphaned.any():
        # Defensive: with a self-parenting lineage this is unreachable, but if a
        # division source is ever added, an orphan must become a root AND lose
        # its division flag AND self-parent -- leaving any one of those undone is
        # the silent-corruption bug this function exists to prevent.
        rows, cols = np.nonzero(orphaned)
        filtered["is_new_root_mask"][rows, cols] = True
        filtered["is_division_mask"][rows, cols] = False
        filtered["parent_indices"][rows, cols] = cols.astype(np.int32)
    # Inactive entries may still carry -1; make them self-parents so the array
    # never holds an index that would be invalid if it were ever read.
    still_negative = filtered["parent_indices"] < 0
    if still_negative.any():
        rows, cols = np.nonzero(still_negative)
        filtered["parent_indices"][rows, cols] = cols.astype(np.int32)

    filtered_arrays = {name: value[:, keep] for name, value in arrays.items()}
    assert_consistent(filtered)
    return filtered, filtered_arrays, keep


def apply_warmup(masks: dict[str, np.ndarray], warmup_frames: int) -> dict[str, np.ndarray]:
    """Deactivate each cell's leading `warmup_frames` frames.

    Holding the warmup fixed regardless of lag order is what makes a lag-0 and a
    lag-1 run comparable: both are scored over exactly the same cell-frames.  It
    also drops the frames on which temporal features are undefined -- a cell's
    first active frame has no previous active frame, so `velocity` and every
    delta are NaN there.

    A cell with no frames left after the warmup keeps its final frame, so
    filtering by `cells.min_frames` remains the only thing that removes a cell.

    Args:
        masks (dict[str, np.ndarray]): the four masks; not modified in place.
        warmup_frames (int): leading active frames to deactivate per cell.

    Returns:
        dict[str, np.ndarray]: masks with the warmup applied and `is_new_root_mask`
            re-seeded at each cell's first surviving frame.
    """
    warmed = {key: masks[key].copy() for key in MASK_KEYS}
    active, roots = warmed["active_mask"], warmed["is_new_root_mask"]
    roots[:] = False

    num_cells = active.shape[1]
    for col in range(num_cells):
        frames = np.flatnonzero(active[:, col])
        if frames.size == 0:
            continue
        drop = frames[:warmup_frames] if frames.size > warmup_frames else frames[:-1]
        active[drop, col] = False
        remaining = np.flatnonzero(active[:, col])
        if remaining.size:
            roots[remaining[0], col] = True

    warmed["is_division_mask"] &= active
    assert_consistent(warmed)
    return warmed


def prepare_for_fit(
    masks: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    min_frames: int,
    warmup_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Apply `tree_input.md` section 6's preprocessing, in its mandated order.

    **Filter first, then warm up.**  The order is load-bearing, not incidental:
    filtering counts a cell's raw observed frames, so a cell sitting just above
    `min_frames` survives and then loses its warmup frames.  Doing it the other
    way round charges the warmup against the threshold and silently drops cells
    -- on the reference crop that is 14 cells / 672 inferred cell-frames instead
    of the correct 16 / 690.

    Exists so the fit step cannot get the order wrong by rearranging two lines.

    Args:
        masks (dict[str, np.ndarray]): the four masks over the combined columns.
        arrays (dict[str, np.ndarray]): companion arrays with the column axis at 1.
        min_frames (int): `cells.min_frames`.
        warmup_frames (int): `cells.warmup_frames`.

    Returns:
        tuple:
            masks (dict[str, np.ndarray]): filtered and warmed-up masks.
            arrays (dict[str, np.ndarray]): filtered companions.
            keep (np.ndarray): surviving column indices into the input space.
    """
    filtered, filtered_arrays, keep = filter_short_cells(masks, arrays, min_frames)
    return apply_warmup(filtered, warmup_frames), filtered_arrays, keep


def assert_consistent(masks: dict[str, np.ndarray]) -> None:
    """Check the invariants every column-space operation must preserve.

    Args:
        masks (dict[str, np.ndarray]): the four masks.

    Raises:
        LineageError: on the first violated invariant.
    """
    missing = [key for key in MASK_KEYS if key not in masks]
    if missing:
        raise LineageError(f"missing masks: {missing}")

    active = masks["active_mask"]
    parents = masks["parent_indices"]
    divisions = masks["is_division_mask"]
    roots = masks["is_new_root_mask"]

    shapes = {key: masks[key].shape for key in MASK_KEYS}
    if len(set(shapes.values())) != 1:
        raise LineageError(f"masks disagree on shape: {shapes}")
    num_frames, num_cells = active.shape

    if parents.min(initial=0) < 0 or parents.max(initial=0) >= max(num_cells, 1):
        raise LineageError(
            f"parent_indices out of range for {num_cells} columns: "
            f"[{parents.min()}, {parents.max()}]"
        )

    if divisions.any():
        raise LineageError(
            "is_division_mask has True entries, but this pipeline does not use "
            "the tree half of the model; every cell must be an independent chain"
        )

    own_column = np.tile(np.arange(num_cells), (num_frames, 1))
    misparented = (parents != own_column) & active
    if misparented.any():
        count = int(misparented.sum())
        raise LineageError(
            f"{count} active cell-frames do not self-parent; with divisions out "
            f"of scope, parent_indices must equal the cell's own column"
        )

    stray_roots = roots & ~active
    if stray_roots.any():
        raise LineageError(f"{int(stray_roots.sum())} new-root flags fall outside active_mask")

    roots_per_cell = roots.sum(axis=0)
    live = active.any(axis=0)
    if not np.array_equal(roots_per_cell[live], np.ones(int(live.sum()), dtype=roots_per_cell.dtype)):
        raise LineageError(
            "every cell with any active frame must have exactly one new-root frame; "
            f"got counts {sorted(set(roots_per_cell[live].tolist()))}"
        )
    if roots_per_cell[~live].any():
        raise LineageError("a cell with no active frames carries a new-root flag")


def summarize(masks: dict[str, np.ndarray]) -> dict[str, int]:
    """Counts worth recording in a step summary."""
    active = masks["active_mask"]
    return {
        "num_frames": int(active.shape[0]),
        "num_cells": int(active.shape[1]),
        "active_cell_frames": int(active.sum()),
        "cells_with_any_frame": int(active.any(axis=0).sum()),
    }
