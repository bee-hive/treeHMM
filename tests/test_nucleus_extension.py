"""What `cells.extend_nuclei` holds, and what it refuses to.

The planner is pure -- two label stacks, a set of parent labels and three ints
-- so everything here is built in memory and nothing touches a filesystem.  The
loader side is covered in `test_cells.py`.

Two properties get more attention than the rest because the real data cannot
check them: the end-of-movie clip never bites on the six ground-truth crops
(every track that qualifies there has room for the full budget), and the
difference between reading divisions and inferring them only shows up on the
handful of parents that vanish before their daughters appear.

Uses stdlib `unittest` rather than pytest so the suite runs in all three conda
environments without installing anything into them.

    python -m unittest discover -s tests -t tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

import conftest  # noqa: F401  (puts the repo root on sys.path)
from arhmm.core import cells as cellsmod

SIZE = 40
NO_PARENTS: frozenset[int] = frozenset()


def blank(frames: int = 12, size: int = SIZE) -> np.ndarray:
    return np.zeros((frames, size, size), dtype=np.int32)


def put(frame: np.ndarray, label: int, row: int, col: int, half: int = 1) -> None:
    """Stamp a small square of `label` centred on `(row, col)`."""
    frame[row - half:row + half + 1, col - half:col + half + 1] = label


def ids_of(stack: np.ndarray) -> np.ndarray:
    labels = np.unique(stack)
    return labels[labels > 0].astype(np.int32)


def plan(nuclei, sam3=None, parents=NO_PARENTS, *, cell_ids=None, frames=3,
         exclusion_px=10, evidence_px=4):
    """Plan with SAM3 support everywhere unless a stack is supplied."""
    if sam3 is None:
        sam3 = np.ones_like(nuclei)
    return cellsmod.plan_nucleus_extension(
        nuclei,
        ids_of(nuclei) if cell_ids is None else cell_ids,
        sam3,
        parents,
        frames=frames,
        exclusion_px=exclusion_px,
        evidence_px=evidence_px,
    )


def only(records, label):
    return next(r for r in records if r.cell_id == label)


class BoxGeometry(unittest.TestCase):
    """The box must be the window a DINO patch of the same size would see."""

    def test_an_even_side_is_anchored_half_below_the_rounded_centroid(self):
        rows, cols = cellsmod._box_slices((20.0, 20.0), 10, (SIZE, SIZE))
        self.assertEqual((rows.start, rows.stop), (15, 25))
        self.assertEqual((cols.start, cols.stop), (15, 25))

    def test_an_odd_side_keeps_its_full_extent(self):
        rows, _ = cellsmod._box_slices((20.0, 20.0), 11, (SIZE, SIZE))
        self.assertEqual((rows.start, rows.stop), (15, 26))

    def test_it_matches_the_window_dino_would_cut(self):
        """Replicates `dino.subject_patch`'s pad-and-crop on an index grid.

        If these drift, `exclusion_px: 50` stops meaning "the 50 px patch", and
        a neighbour could contaminate a patch the planner called clear.
        """
        grid = np.arange(SIZE * SIZE).reshape(SIZE, SIZE)
        for side in (6, 10, 16, 21):
            for centroid in ((20.0, 20.0), (18.4, 22.6)):
                with self.subTest(side=side, centroid=centroid):
                    half = side // 2
                    padded = np.pad(grid, ((half, half), (half, half)), mode="edge")
                    row = int(round(centroid[0]))
                    col = int(round(centroid[1]))
                    cut = padded[row:row + side, col:col + side]
                    ours = grid[cellsmod._box_slices(centroid, side, (SIZE, SIZE))]
                    np.testing.assert_array_equal(ours, cut)

    def test_a_centroid_near_the_edge_clips_rather_than_padding(self):
        rows, cols = cellsmod._box_slices((1.0, 38.0), 10, (SIZE, SIZE))
        self.assertEqual((rows.start, rows.stop), (0, 6))
        self.assertEqual((cols.start, cols.stop), (33, SIZE))

    def test_the_rounding_tie_goes_the_way_dino_rounds_it(self):
        """`int(round(24.5))` is 24, not 25.

        Rewriting it as `floor(c + 0.5)` would shift the box one pixel at every
        half-integer centroid, which a mask of even width hits constantly.
        """
        self.assertEqual(int(round(24.5)), 24)
        rows, _ = cellsmod._box_slices((24.5, 20.0), 10, (SIZE, SIZE))
        self.assertEqual(rows.start, 19)


class Criteria(unittest.TestCase):
    def test_an_isolated_terminating_track_is_held(self):
        nuclei = blank()
        for t in range(4):
            put(nuclei[t], 1, 20, 20)
        record = only(plan(nuclei), 1)
        self.assertEqual(record.reason, "extended")
        self.assertEqual(record.final_frame, 3)
        self.assertEqual(record.frames_added, 3)
        self.assertEqual(record.first_failing_frame, -1)

    def test_a_track_running_to_the_last_frame_has_nothing_to_do(self):
        nuclei = blank(frames=4)
        for t in range(4):
            put(nuclei[t], 1, 20, 20)
        record = only(plan(nuclei), 1)
        self.assertEqual(record.reason, "ends_at_movie_end")
        self.assertEqual(record.frames_added, 0)
        self.assertIsNone(record.divides)

    def test_a_label_with_no_pixels_is_reported_not_crashed(self):
        nuclei = blank()
        put(nuclei[0], 1, 20, 20)
        record = only(plan(nuclei, cell_ids=np.array([1, 99], np.int32)), 99)
        self.assertEqual(record.reason, "empty_track")
        self.assertEqual(record.final_frame, -1)

    def test_one_pixel_inside_the_exclusion_box_blocks_it(self):
        """And the same pixel one column further out does not.

        The criterion is a strict pixel test, so the boundary is where the whole
        decision turns; both sides of it are pinned here.
        """
        for col, expected in ((24, "neighbour_in_box"), (25, "extended")):
            with self.subTest(col=col):
                nuclei = blank()
                for t in range(4):
                    put(nuclei[t], 1, 20, 20)
                for t in range(4, 8):
                    nuclei[t, 20, col] = 2
                record = only(plan(nuclei), 1)
                self.assertEqual(record.reason, expected)

    def test_a_blocker_in_the_final_appended_frame_costs_every_frame(self):
        """All-or-nothing: two clear frames earn nothing if the third is not."""
        nuclei = blank()
        for t in range(4):
            put(nuclei[t], 1, 20, 20)
        nuclei[6, 20, 22] = 2
        record = only(plan(nuclei), 1)
        self.assertEqual(record.reason, "neighbour_in_box")
        self.assertEqual(record.frames_added, 0)
        self.assertEqual(record.first_failing_frame, 6)

    def test_sam3_must_be_present_in_every_appended_frame(self):
        nuclei = blank()
        for t in range(4):
            put(nuclei[t], 1, 20, 20)
        sam3 = np.ones_like(nuclei)
        sam3[6] = 0
        record = only(plan(nuclei, sam3), 1)
        self.assertEqual(record.reason, "no_sam3_evidence")
        self.assertEqual(record.first_failing_frame, 6)
        self.assertFalse(record.sam3_evidence)

    def test_sam3_support_is_judged_on_the_evidence_box_not_the_exclusion_box(self):
        """A phase mask 4 px away supports nothing if evidence_px is 4."""
        nuclei = blank()
        for t in range(4):
            put(nuclei[t], 1, 20, 20)
        sam3 = np.zeros_like(nuclei)
        sam3[:, 20, 24] = 1          # inside exclusion (15..25), outside evidence (18..22)
        self.assertEqual(only(plan(nuclei, sam3), 1).reason, "no_sam3_evidence")
        sam3[:, 20, 21] = 1          # now inside the evidence box too
        self.assertEqual(only(plan(nuclei, sam3), 1).reason, "extended")

    def test_the_last_appearance_anchors_the_track_not_the_first_gap(self):
        """Interior gaps are common in this data; a gap is not an ending."""
        nuclei = blank()
        for t in (0, 1, 4):
            put(nuclei[t], 1, 20, 20)
        record = only(plan(nuclei), 1)
        self.assertEqual(record.final_frame, 4)
        self.assertEqual(record.reason, "extended")


class DivisionIsALookup(unittest.TestCase):
    """Read `nuclei_div.pkl`; never infer divisions from the label movie."""

    def test_a_parent_is_refused_even_when_its_box_is_clear(self):
        """The gap case: daughters appear later, and far away.

        Nothing geometric can see this -- no new label turns up beside the
        parent when it vanishes -- so a "two daughters born next door" rule
        would hold a nucleus straight through its own division.
        """
        nuclei = blank()
        for t in range(4):
            put(nuclei[t], 1, 20, 20)
        for t in range(8, 12):        # daughters, five frames later, far off
            put(nuclei[t], 2, 5, 5)
            put(nuclei[t], 3, 5, 12)
        record = only(plan(nuclei, parents=frozenset({1})), 1)
        self.assertEqual(record.reason, "divides")
        self.assertEqual(record.frames_added, 0)
        # The box really was clear: the lookup did this on its own.
        self.assertFalse(record.neighbour_in_box)

    def test_two_labels_born_alongside_are_neighbours_not_a_division(self):
        """Absent from the table, so the rejection is attributed to the box."""
        nuclei = blank()
        for t in range(4):
            put(nuclei[t], 1, 20, 20)
        for t in range(4, 8):
            put(nuclei[t], 2, 18, 22, half=0)
            put(nuclei[t], 3, 22, 18, half=0)
        record = only(plan(nuclei), 1)
        self.assertEqual(record.reason, "neighbour_in_box")
        self.assertFalse(record.divides)

    def test_a_daughter_is_an_ordinary_candidate(self):
        nuclei = blank()
        for t in range(2, 6):
            put(nuclei[t], 2, 20, 20)
        record = only(plan(nuclei, parents=frozenset({1})), 2)
        self.assertEqual(record.reason, "extended")


class EndOfMovieClips(unittest.TestCase):
    """Room reduces the budget; it does not reject the track.

    None of the six ground-truth crops exercises this -- every track that
    qualifies there has room for the full budget -- so it is only ever checked
    here.
    """

    def test_the_budget_shrinks_to_what_is_left(self):
        for final, expected in ((8, 3), (9, 2), (10, 1), (11, 0)):
            with self.subTest(final_frame=final):
                nuclei = blank(frames=12)
                for t in range(final + 1):
                    put(nuclei[t], 1, 20, 20)
                record = only(plan(nuclei, frames=3), 1)
                self.assertEqual(record.frames_available, expected)
                self.assertEqual(record.frames_added, expected)
                self.assertEqual(
                    record.reason, "extended" if expected else "ends_at_movie_end"
                )

    def test_nothing_beyond_the_budget_is_looked_at(self):
        """A blocker one frame past the last appended frame is irrelevant."""
        nuclei = blank(frames=12)
        for t in range(5):
            put(nuclei[t], 1, 20, 20)      # final 4, budget 3 -> frames 5, 6, 7
        nuclei[8, 20, 22] = 2              # frame 8 is past the budget
        self.assertEqual(only(plan(nuclei, frames=3), 1).reason, "extended")

        nuclei[8, 20, 22] = 0
        nuclei[7, 20, 22] = 2              # the last appended frame is not
        self.assertEqual(only(plan(nuclei, frames=3), 1).reason, "neighbour_in_box")

    def test_a_clipped_budget_is_still_judged_over_every_frame_it_covers(self):
        nuclei = blank(frames=12)
        for t in range(10):
            put(nuclei[t], 1, 20, 20)      # final 9, budget clipped 5 -> 2
        self.assertEqual(only(plan(nuclei, frames=5), 1).frames_added, 2)
        nuclei[11, 20, 22] = 2             # the second of the two
        record = only(plan(nuclei, frames=5), 1)
        self.assertEqual(record.reason, "neighbour_in_box")
        self.assertEqual(record.frames_added, 0)


class PlanIsAFunctionOfTheInput(unittest.TestCase):
    def test_two_neighbours_ending_together_are_both_held(self):
        """Judged against the original stack, never a partly painted one.

        The two sit inside each other's exclusion box and end on the same frame,
        so the frames they are judged over are empty for both and both qualify.
        Paint either one first, though, and it fills the other's box -- so if the
        planner ever read back its own output, whichever was visited second
        would be refused, and the answer would depend on visiting order.
        """
        nuclei = blank()
        for t in range(6):
            put(nuclei[t], 1, 20, 20)      # A
            put(nuclei[t], 2, 20, 24, half=0)   # B, inside A's box and vice versa
        records = plan(nuclei)
        self.assertEqual(only(records, 1).reason, "extended")
        self.assertEqual(only(records, 2).reason, "extended")
        # And painting the plan really does put each in the other's box.
        painted = cellsmod.apply_nucleus_extension(nuclei, records)
        self.assertTrue((painted[6] == 2).any())
        self.assertTrue((painted[6] == 1).any())

    def test_records_come_back_in_cell_ids_order_however_they_are_passed(self):
        nuclei = blank()
        for t in range(4):
            put(nuclei[t], 1, 10, 10)
            put(nuclei[t], 2, 30, 30)
        forward = plan(nuclei, cell_ids=np.array([1, 2], np.int32))
        reversed_ = plan(nuclei, cell_ids=np.array([2, 1], np.int32))
        self.assertEqual([r.cell_id for r in forward], [1, 2])
        self.assertEqual([r.cell_id for r in reversed_], [2, 1])
        self.assertEqual(
            {r.cell_id: r.reason for r in forward},
            {r.cell_id: r.reason for r in reversed_},
        )


class Painting(unittest.TestCase):
    def test_the_mask_is_copied_pixel_for_pixel(self):
        nuclei = blank()
        for t in range(4):
            put(nuclei[t], 1, 20, 20, half=2)
        records = plan(nuclei)
        painted = cellsmod.apply_nucleus_extension(nuclei, records)
        for step in range(1, 4):
            np.testing.assert_array_equal(painted[3 + step] == 1, nuclei[3] == 1)

    def test_the_input_is_never_modified(self):
        nuclei = blank()
        for t in range(4):
            put(nuclei[t], 1, 20, 20)
        before = nuclei.copy()
        cellsmod.apply_nucleus_extension(nuclei, plan(nuclei))
        np.testing.assert_array_equal(nuclei, before)

    def test_holding_nothing_returns_the_stack_untouched(self):
        nuclei = blank(frames=4)
        for t in range(4):
            put(nuclei[t], 1, 20, 20)
        records = plan(nuclei)
        self.assertIs(cellsmod.apply_nucleus_extension(nuclei, records), nuclei)

    def test_the_label_set_is_unchanged(self):
        """`cell_ids` is the column order, and holding a track must not move it."""
        nuclei = blank()
        for t in range(4):
            put(nuclei[t], 1, 20, 20)
        painted = cellsmod.apply_nucleus_extension(nuclei, plan(nuclei))
        np.testing.assert_array_equal(ids_of(painted), ids_of(nuclei))

    def test_painting_over_an_occupied_pixel_raises(self):
        """Hand-built plan: the planner is supposed to make this impossible."""
        nuclei = blank()
        put(nuclei[0], 1, 20, 20)
        put(nuclei[1], 2, 20, 20)
        record = cellsmod.ExtensionRecord(
            cell_id=1, final_frame=0, frames_available=1, frames_added=1,
            reason="extended", first_failing_frame=-1, divides=False,
            neighbour_in_box=False, sam3_evidence=True, blocking_labels="",
        )
        with self.assertRaisesRegex(ValueError, "would overwrite"):
            cellsmod.apply_nucleus_extension(nuclei, (record,))


if __name__ == "__main__":
    unittest.main()
