"""The cell-age histogram's two numeric pieces: where a cell's clock starts and
how its inferred frames land in bins.

Every case here is one the figure would get silently wrong rather than loudly:
a gap counted as a rewind, a warmup counted as a birth, a per-crop column index
read against the pooled column space.  Nothing is plotted -- `matplotlib` is
imported by the module under test, which is fine in the imaging environment the
`outputs` step itself runs in.
"""

import unittest

import numpy as np

from arhmm.steps.outputs import age_histogram, cell_birth_frames


def _bundle(active, states=None):
    """A `(T, C)` active mask and a matching state assignment."""
    active = np.asarray(active, dtype=bool)
    if states is None:
        states = np.zeros(active.shape, dtype=np.int32)
    return active, np.asarray(states, dtype=np.int32)


class AgeHistogramTest(unittest.TestCase):
    def test_a_gap_costs_a_sample_rather_than_rewinding_the_clock(self):
        # Present at frames 0, 1, 3: frame 3 is the cell's fourth timepoint, so
        # age bin 2 is empty and age bin 3 holds the sample.
        active, states = _bundle([[True], [True], [False], [True]])
        hist = age_histogram(states, active, np.array([0]), 1, bin_frames=1, max_age=None)
        self.assertEqual(list(hist["counts"][:, 0]), [1, 1, 0, 1])
        self.assertEqual(hist["max_observed_age"], 3)

    def test_age_is_measured_from_birth_not_from_the_first_inferred_frame(self):
        # A cell born at frame 2 whose first two frames are warmup contributes
        # nothing to ages 0 and 1 -- but the ages it does contribute to are
        # counted from frame 2, not from the frame inference starts on.
        active, states = _bundle([[False], [False], [False], [False], [True], [True]])
        hist = age_histogram(states, active, np.array([2]), 1, bin_frames=1, max_age=None)
        self.assertEqual(list(hist["counts"][:, 0]), [0, 0, 1, 1])

    def test_states_are_counted_separately_and_pooled_over_columns(self):
        active, states = _bundle(
            [[True, True], [True, True]],
            [[0, 1], [1, 1]],
        )
        hist = age_histogram(states, active, np.array([0, 0]), 2, bin_frames=1, max_age=None)
        self.assertEqual(hist["counts"].tolist(), [[1, 1], [0, 2]])

    def test_bins_wider_than_one_frame_aggregate_whole_bins(self):
        active, states = _bundle([[True]] * 5)
        hist = age_histogram(states, active, np.array([0]), 1, bin_frames=2, max_age=None)
        self.assertEqual(list(hist["counts"][:, 0]), [2, 2, 1])
        self.assertEqual(list(hist["starts"]), [0, 2, 4])
        self.assertEqual(list(hist["stops"]), [1, 3, 5])

    def test_frames_past_max_age_are_dropped_and_counted(self):
        active, states = _bundle([[True]] * 6)
        hist = age_histogram(states, active, np.array([0]), 1, bin_frames=1, max_age=3)
        self.assertEqual(hist["counts"].shape[0], 4)
        self.assertEqual(list(hist["counts"][:, 0]), [1, 1, 1, 1])
        self.assertEqual(hist["dropped"], 2)
        # Not folded into the last bin, which would put a spike there.
        self.assertEqual(int(hist["counts"][-1, 0]), 1)

    def test_at_risk_counts_cells_that_can_still_contribute(self):
        # One cell inferred to age 3, one only to age 1.
        active, states = _bundle(
            [[True, True], [True, True], [True, False], [True, False]]
        )
        hist = age_histogram(states, active, np.array([0, 0]), 1, bin_frames=1, max_age=None)
        self.assertEqual(list(hist["at_risk"]), [2, 2, 1, 1])

    def test_min_age_moves_the_origin_rather_than_slicing_the_result(self):
        # The warmup bins are empty by construction, so the axis starts past
        # them -- and bin 0 is then age `min_age`, not age 0 relabelled.
        active, states = _bundle([[False], [True], [True], [True]])
        hist = age_histogram(states, active, np.array([0]), 1, bin_frames=1,
                             max_age=None, min_age=1)
        self.assertEqual(list(hist["starts"]), [1, 2, 3])
        self.assertEqual(list(hist["counts"][:, 0]), [1, 1, 1])
        self.assertEqual(hist["dropped_young"], 0)

    def test_bins_are_laid_out_from_min_age_not_from_zero(self):
        active, states = _bundle([[False], [True], [True], [True], [True]])
        hist = age_histogram(states, active, np.array([0]), 1, bin_frames=2,
                             max_age=None, min_age=1)
        self.assertEqual(list(hist["starts"]), [1, 3])
        self.assertEqual(list(hist["stops"]), [2, 4])
        self.assertEqual(list(hist["counts"][:, 0]), [2, 2])

    def test_a_cell_frame_below_min_age_is_excluded_and_reported(self):
        active, states = _bundle([[True], [True], [True]])
        hist = age_histogram(states, active, np.array([0]), 1, bin_frames=1,
                             max_age=None, min_age=1)
        self.assertEqual(hist["dropped_young"], 1)
        self.assertEqual(int(hist["counts"].sum()), 2)

    def test_at_risk_is_counted_from_min_age_too(self):
        active, states = _bundle(
            [[False, False], [True, True], [True, False], [True, False]]
        )
        hist = age_histogram(states, active, np.array([0, 0]), 1, bin_frames=1,
                             max_age=None, min_age=1)
        self.assertEqual(list(hist["starts"]), [1, 2, 3])
        self.assertEqual(list(hist["at_risk"]), [2, 1, 1])

    def test_a_cell_inferred_before_it_was_born_is_an_error(self):
        active, states = _bundle([[True], [True]])
        with self.assertRaises(ValueError):
            age_histogram(states, active, np.array([1]), 1, bin_frames=1, max_age=None)


class BirthFrameTest(unittest.TestCase):
    """`cell_birth_frames` against a fake features cache."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _write_cache(self, per_crop):
        """Write one `features.npz` per crop, and a layout that finds them."""
        from pathlib import Path

        from arhmm.core import io

        root = Path(self.root)
        for crop_id, present in per_crop.items():
            crop_dir = io.ensure_dir(root / crop_id)
            io.save_npz(crop_dir / "features.npz",
                        active_mask=np.asarray(present, dtype=bool))

        class _Layout:
            def crop_dir(self, _step, crop_id):
                return root / crop_id

        return _Layout()

    def test_existence_anchor_reads_the_raw_presence_not_the_warmed_up_mask(self):
        # Raw: present from frame 1.  Fit: warmup killed frame 1, so inference
        # starts at 2.  Age zero must be frame 1.
        layout = self._write_cache({"cropA": [[False], [True], [True], [True]]})
        active = np.array([[False], [False], [True], [True]])
        index = [{"crop": "cropA", "cell_id": 7, "crop_idx": "0"}]
        cfg = {"outputs": {"state_age_histogram": {"anchor": "existence"}}}
        self.assertEqual(list(cell_birth_frames(cfg, layout, index, active)), [1])

    def test_inferred_anchor_ignores_the_cache_entirely(self):
        layout = self._write_cache({"cropA": [[False], [True], [True], [True]]})
        active = np.array([[False], [False], [True], [True]])
        index = [{"crop": "cropA", "cell_id": 7, "crop_idx": "0"}]
        cfg = {"outputs": {"state_age_histogram": {"anchor": "inferred"}}}
        self.assertEqual(list(cell_birth_frames(cfg, layout, index, active)), [2])

    def test_crop_idx_is_read_against_its_own_crop_not_the_pooled_columns(self):
        # Two crops of two cells each.  Pooled column 2 is cropB's column 0; a
        # birth of 0 rather than 3 would mean the pooled index leaked through.
        layout = self._write_cache(
            {
                "cropA": [[True, False], [True, True], [True, True], [True, True]],
                "cropB": [[False, True], [False, True], [False, True], [True, True]],
            }
        )
        active = np.ones((4, 4), dtype=bool)
        active[:3, 2] = False
        active[0, 1] = False
        index = [
            {"crop": "cropA", "cell_id": 1, "crop_idx": "0"},
            {"crop": "cropA", "cell_id": 2, "crop_idx": "1"},
            {"crop": "cropB", "cell_id": 1, "crop_idx": "0"},
            {"crop": "cropB", "cell_id": 2, "crop_idx": "1"},
        ]
        cfg = {"outputs": {"state_age_histogram": {"anchor": "existence"}}}
        self.assertEqual(list(cell_birth_frames(cfg, layout, index, active)), [0, 1, 3, 0])

    def test_a_cache_that_starts_after_the_fit_is_an_error(self):
        layout = self._write_cache({"cropA": [[False], [False], [True], [True]]})
        active = np.array([[True], [True], [True], [True]])
        index = [{"crop": "cropA", "cell_id": 7, "crop_idx": "0"}]
        cfg = {"outputs": {"state_age_histogram": {"anchor": "existence"}}}
        with self.assertRaises(ValueError):
            cell_birth_frames(cfg, layout, index, active)


if __name__ == "__main__":
    unittest.main()
