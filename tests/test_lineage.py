"""Mask construction, filtering and warmup.

The invariants here are the ones whose violation corrupts inference silently
rather than raising, so they are tested directly rather than via a fitted model.

    python -m unittest discover -s tests -t tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

import conftest  # noqa: F401
from treearhmm.core import lineage as LG


def presence(rows: list[str]) -> np.ndarray:
    """Build a `(T, N)` presence matrix from one string per cell.

    Each string is read left to right over frames; `#` means present.

        presence(["###..", ".####"])  ->  two cells, five frames
    """
    width = max(len(r) for r in rows)
    padded = [r.ljust(width, ".") for r in rows]
    return np.array([[c == "#" for c in row] for row in padded], dtype=bool).T


class BuildMasks(unittest.TestCase):
    def test_shapes_and_dtypes(self):
        masks = LG.build_masks(presence(["###", ".##"]))
        self.assertEqual(masks["active_mask"].shape, (3, 2))
        self.assertEqual(masks["parent_indices"].dtype, np.int32)
        self.assertTrue(masks["is_division_mask"].dtype == bool)

    def test_every_cell_self_parents(self):
        masks = LG.build_masks(presence(["###", ".##", "#.#"]))
        expected = np.tile(np.arange(3, dtype=np.int32), (3, 1))
        np.testing.assert_array_equal(masks["parent_indices"], expected)

    def test_no_divisions_ever(self):
        masks = LG.build_masks(presence(["###", "###"]))
        self.assertFalse(masks["is_division_mask"].any())

    def test_new_root_at_each_cells_first_active_frame(self):
        masks = LG.build_masks(presence(["###..", "..###"]))
        roots = masks["is_new_root_mask"]
        self.assertEqual(list(np.flatnonzero(roots[:, 0])), [0])
        self.assertEqual(list(np.flatnonzero(roots[:, 1])), [2])
        self.assertEqual(int(roots.sum()), 2)

    def test_a_gap_does_not_create_a_second_root(self):
        """A cell that disappears and returns is still one chain, not two."""
        masks = LG.build_masks(presence(["##..##"]))
        self.assertEqual(int(masks["is_new_root_mask"].sum()), 1)
        self.assertEqual(list(np.flatnonzero(masks["is_new_root_mask"][:, 0])), [0])

    def test_a_never_present_cell_gets_no_root(self):
        masks = LG.build_masks(presence(["###", "..."]))
        self.assertEqual(int(masks["is_new_root_mask"][:, 1].sum()), 0)


class Concatenate(unittest.TestCase):
    def test_parent_indices_are_shifted_into_the_combined_space(self):
        a = LG.build_masks(presence(["##", "##"]))
        b = LG.build_masks(presence(["##", "##", "##"]))
        combined = LG.concatenate_crops([a, b])
        self.assertEqual(combined["active_mask"].shape, (2, 5))
        np.testing.assert_array_equal(
            combined["parent_indices"], np.tile(np.arange(5, dtype=np.int32), (2, 1))
        )

    def test_root_flags_survive_concatenation(self):
        a = LG.build_masks(presence(["##"]))
        b = LG.build_masks(presence([".#"]))
        combined = LG.concatenate_crops([a, b])
        self.assertEqual(int(combined["is_new_root_mask"].sum()), 2)

    def test_disagreeing_frame_counts_are_rejected(self):
        a = LG.build_masks(presence(["##"]))
        b = LG.build_masks(presence(["###"]))
        with self.assertRaisesRegex(ValueError, "frame count"):
            LG.concatenate_crops([a, b])

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            LG.concatenate_crops([])


class FilterShortCells(unittest.TestCase):
    def setUp(self):
        # three cells: 5 frames, 2 frames, 4 frames
        self.masks = LG.build_masks(presence(["#####", "##...", ".####"]))
        self.arrays = {"values": np.arange(5 * 3 * 2).reshape(5, 3, 2).astype(float)}

    def test_drops_columns_below_the_threshold(self):
        masks, arrays, keep = LG.filter_short_cells(self.masks, self.arrays, min_frames=4)
        self.assertEqual(list(keep), [0, 2])
        self.assertEqual(masks["active_mask"].shape, (5, 2))
        self.assertEqual(arrays["values"].shape, (5, 2, 2))

    def test_companion_arrays_are_filtered_on_the_column_axis(self):
        _, arrays, keep = LG.filter_short_cells(self.masks, self.arrays, min_frames=4)
        np.testing.assert_array_equal(arrays["values"], self.arrays["values"][:, keep])

    def test_parent_indices_are_renumbered_into_the_new_space(self):
        masks, _, _ = LG.filter_short_cells(self.masks, self.arrays, min_frames=4)
        expected = np.tile(np.arange(2, dtype=np.int32), (5, 1))
        np.testing.assert_array_equal(masks["parent_indices"], expected)
        self.assertLess(masks["parent_indices"].max(), masks["active_mask"].shape[1])

    def test_removing_every_cell_raises(self):
        with self.assertRaisesRegex(ValueError, "removed every cell"):
            LG.filter_short_cells(self.masks, self.arrays, min_frames=99)

    def test_keeping_everything_is_a_no_op(self):
        masks, _, keep = LG.filter_short_cells(self.masks, self.arrays, min_frames=1)
        self.assertEqual(list(keep), [0, 1, 2])
        np.testing.assert_array_equal(masks["active_mask"], self.masks["active_mask"])


class ApplyWarmup(unittest.TestCase):
    def test_drops_the_leading_active_frame(self):
        masks = LG.apply_warmup(LG.build_masks(presence(["#####"])), warmup_frames=1)
        self.assertEqual(list(np.flatnonzero(masks["active_mask"][:, 0])), [1, 2, 3, 4])

    def test_reseeds_the_root_at_the_first_surviving_frame(self):
        masks = LG.apply_warmup(LG.build_masks(presence(["#####"])), warmup_frames=2)
        self.assertEqual(list(np.flatnonzero(masks["is_new_root_mask"][:, 0])), [2])

    def test_warmup_counts_active_frames_not_wall_clock(self):
        """A cell absent for a while must lose its first ACTIVE frame, not frame 0."""
        masks = LG.apply_warmup(LG.build_masks(presence(["..###"])), warmup_frames=1)
        self.assertEqual(list(np.flatnonzero(masks["active_mask"][:, 0])), [3, 4])

    def test_a_cell_shorter_than_the_warmup_keeps_its_final_frame(self):
        """Only cells.min_frames removes a cell; the warmup never empties one."""
        masks = LG.apply_warmup(LG.build_masks(presence(["##"])), warmup_frames=5)
        self.assertEqual(int(masks["active_mask"][:, 0].sum()), 1)
        self.assertEqual(int(masks["is_new_root_mask"][:, 0].sum()), 1)

    def test_the_input_is_not_modified_in_place(self):
        original = LG.build_masks(presence(["#####"]))
        before = original["active_mask"].copy()
        LG.apply_warmup(original, warmup_frames=2)
        np.testing.assert_array_equal(original["active_mask"], before)

    def test_warmup_is_independent_of_lag_order(self):
        """The point of a fixed warmup: lag-0 and lag-1 score the same cell-frames."""
        masks = LG.build_masks(presence(["#####", ".####"]))
        lag0 = LG.apply_warmup(masks, warmup_frames=1)
        lag1 = LG.apply_warmup(masks, warmup_frames=1)
        self.assertEqual(int(lag0["active_mask"].sum()), int(lag1["active_mask"].sum()))


class PrepareForFit(unittest.TestCase):
    """The order in tree_input.md section 6 is load-bearing, so it is pinned here."""

    def setUp(self):
        # Cell 1 has exactly min_frames raw frames: it survives filtering, then
        # loses its warmup frame.  Filtering after the warmup would drop it.
        self.masks = LG.build_masks(presence(["#####", "####.", "##..."]))
        self.arrays = {"values": np.zeros((5, 3, 2))}

    def test_filters_on_raw_frames_then_warms_up(self):
        masks, _, keep = LG.prepare_for_fit(self.masks, self.arrays, min_frames=4, warmup_frames=1)
        self.assertEqual(list(keep), [0, 1])
        self.assertEqual(LG.summarize(masks)["active_cell_frames"], (5 - 1) + (4 - 1))

    def test_the_other_order_would_have_dropped_a_cell(self):
        """Documents why prepare_for_fit exists rather than two calls at the call site."""
        warmed = LG.apply_warmup(self.masks, warmup_frames=1)
        _, _, keep_wrong = LG.filter_short_cells(warmed, self.arrays, min_frames=4)
        self.assertEqual(list(keep_wrong), [0])  # cell 1 lost, incorrectly

    def test_companion_arrays_follow_the_same_columns(self):
        _, arrays, keep = LG.prepare_for_fit(self.masks, self.arrays, min_frames=4, warmup_frames=1)
        self.assertEqual(arrays["values"].shape, (5, len(keep), 2))


class AssertConsistent(unittest.TestCase):
    def setUp(self):
        self.masks = LG.build_masks(presence(["####", ".###"]))

    def test_accepts_well_formed_masks(self):
        LG.assert_consistent(self.masks)

    def test_rejects_a_division_flag(self):
        broken = {k: v.copy() for k, v in self.masks.items()}
        broken["is_division_mask"][2, 1] = True
        with self.assertRaisesRegex(LG.LineageError, "does not use the tree"):
            LG.assert_consistent(broken)

    def test_rejects_a_parent_index_that_is_not_the_cells_own_column(self):
        broken = {k: v.copy() for k, v in self.masks.items()}
        broken["parent_indices"][2, 1] = 0
        with self.assertRaisesRegex(LG.LineageError, "self-parent"):
            LG.assert_consistent(broken)

    def test_rejects_an_out_of_range_parent_index(self):
        broken = {k: v.copy() for k, v in self.masks.items()}
        broken["parent_indices"][0, 0] = 99
        with self.assertRaisesRegex(LG.LineageError, "out of range"):
            LG.assert_consistent(broken)

    def test_rejects_a_root_outside_the_active_mask(self):
        broken = {k: v.copy() for k, v in self.masks.items()}
        broken["is_new_root_mask"][0, 1] = True  # cell 1 is inactive at frame 0
        with self.assertRaisesRegex(LG.LineageError, "outside active_mask"):
            LG.assert_consistent(broken)

    def test_rejects_a_cell_with_two_roots(self):
        broken = {k: v.copy() for k, v in self.masks.items()}
        broken["is_new_root_mask"][2, 0] = True
        with self.assertRaisesRegex(LG.LineageError, "exactly one new-root"):
            LG.assert_consistent(broken)

    def test_rejects_a_cell_with_no_root(self):
        broken = {k: v.copy() for k, v in self.masks.items()}
        broken["is_new_root_mask"][:, 0] = False
        with self.assertRaisesRegex(LG.LineageError, "exactly one new-root"):
            LG.assert_consistent(broken)

    def test_rejects_missing_masks(self):
        with self.assertRaisesRegex(LG.LineageError, "missing masks"):
            LG.assert_consistent({"active_mask": self.masks["active_mask"]})


class Summarize(unittest.TestCase):
    def test_counts(self):
        summary = LG.summarize(LG.build_masks(presence(["####", ".##.", "...."])))
        self.assertEqual(summary["num_frames"], 4)
        self.assertEqual(summary["num_cells"], 3)
        self.assertEqual(summary["active_cell_frames"], 6)
        self.assertEqual(summary["cells_with_any_frame"], 2)


if __name__ == "__main__":
    unittest.main()
