"""The feature registry, and the temporal semantics that are easy to get wrong.

The gap behaviour is the reason `SeriesBundle` exposes only `prev`/`gap`/
`window`: taking a delta against `t - 1` unconditionally makes a tracking gap
look like a cell that teleported and came back.

    python -m unittest discover -s tests -t tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

import conftest  # noqa: F401
from arhmm.core import lineage as LG
from arhmm.core import trackfeatures as tf


def series(active_rows, values_rows=None, centroid_rows=None, params=None):
    """Build a one-cell `SeriesBundle` from per-frame lists."""
    active = np.array(active_rows, dtype=bool).reshape(-1, 1)
    values = {}
    for name, row in (values_rows or {}).items():
        values[name] = np.array(row, dtype=float).reshape(-1, 1)
    if centroid_rows is None:
        centroids = np.full((len(active_rows), 1, 2), np.nan)
    else:
        centroids = np.array(centroid_rows, dtype=float).reshape(-1, 1, 2)
    return tf.SeriesBundle(values=values, centroids=centroids, active=active, params=params or {})


class Registry(unittest.TestCase):
    def test_every_feature_declares_its_metadata(self):
        for name, feature in tf.FEATURE_REGISTRY.items():
            with self.subTest(name):
                self.assertEqual(feature.name, name)
                self.assertIn(feature.stage, (tf.PER_FRAME, tf.TEMPORAL))
                self.assertTrue(feature.doc.strip(), "a feature needs a one-line doc")
                self.assertTrue(callable(feature.fn))

    def test_dependencies_and_params_are_registered_names(self):
        for name, feature in tf.FEATURE_REGISTRY.items():
            with self.subTest(name):
                for dependency in feature.depends:
                    self.assertIn(dependency, tf.FEATURE_REGISTRY)
                    self.assertEqual(tf.FEATURE_REGISTRY[dependency].stage, tf.PER_FRAME)

    def test_only_temporal_features_declare_dependencies(self):
        for name, feature in tf.FEATURE_REGISTRY.items():
            if feature.stage == tf.PER_FRAME:
                self.assertEqual(feature.depends, (), f"{name} is per-frame")

    def test_feature_names_can_be_filtered_by_stage(self):
        per_frame = set(tf.feature_names(tf.PER_FRAME))
        temporal = set(tf.feature_names(tf.TEMPORAL))
        self.assertFalse(per_frame & temporal)
        self.assertEqual(per_frame | temporal, set(tf.feature_names()))

    def test_expand_requested_pulls_in_dependencies_in_registry_order(self):
        expanded = tf.expand_requested(["d_area_frac"])
        self.assertEqual(set(expanded), {"area", "d_area_frac"})
        self.assertEqual(expanded, [n for n in tf.FEATURE_REGISTRY if n in set(expanded)])

    def test_expand_requested_is_idempotent(self):
        once = tf.expand_requested(["d_area_frac", "velocity"])
        self.assertEqual(tf.expand_requested(once), once)

    def test_required_params_is_the_union_of_uses(self):
        self.assertEqual(tf.required_params(["area"]), set())
        self.assertEqual(tf.required_params(["t_cell_neighbors"]), {"neighbor_radius_px"})
        self.assertEqual(
            tf.required_params(["t_cell_neighbors", "win_std_log_area"]),
            {"neighbor_radius_px", "window_frames"},
        )

    def test_needs_image_is_only_true_for_intensity_features(self):
        self.assertFalse(tf.needs_image(["area", "circularity", "velocity"]))
        self.assertTrue(tf.needs_image(["area", "rfp_mean"]))

    def test_an_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            tf.expand_requested(["not_a_feature"])


class PreviousActiveFrame(unittest.TestCase):
    def test_first_active_frame_has_no_predecessor(self):
        sb = series([True, True, True], {"area": [10, 20, 30]})
        self.assertTrue(np.isnan(sb.prev("area")[0, 0]))
        self.assertEqual(sb.prev("area")[1, 0], 10)

    def test_prev_skips_over_an_absence(self):
        sb = series([True, False, True], {"area": [10, np.nan, 30]})
        self.assertEqual(sb.prev("area")[2, 0], 10)

    def test_gap_widens_across_an_absence(self):
        sb = series([True, False, False, True])
        gap = sb.gap()[:, 0]
        self.assertTrue(np.isnan(gap[0]))
        self.assertEqual(gap[3], 3.0)

    def test_a_leading_absence_does_not_shift_the_first_root(self):
        sb = series([False, False, True, True], {"area": [np.nan, np.nan, 5, 7]})
        self.assertTrue(np.isnan(sb.prev("area")[2, 0]))
        self.assertEqual(sb.prev("area")[3, 0], 5)


class Velocity(unittest.TestCase):
    def test_constant_speed_is_reported_the_same_across_a_gap(self):
        """The whole reason temporal features step over ACTIVE frames."""
        sb = series(
            [True, True, False, True, True],
            centroid_rows=[[0, 0], [0, 3], [np.nan, np.nan], [0, 9], [0, 12]],
        )
        velocity = (sb.displacement() / sb.gap())[:, 0]
        self.assertTrue(np.isnan(velocity[0]))
        self.assertAlmostEqual(velocity[1], 3.0)
        self.assertAlmostEqual(velocity[3], 3.0)  # 6 px over a 2-frame gap
        self.assertAlmostEqual(velocity[4], 3.0)

    def test_displacement_is_undivided(self):
        sb = series([True, False, True], centroid_rows=[[0, 0], [np.nan, np.nan], [0, 6]])
        self.assertAlmostEqual(sb.displacement()[2, 0], 6.0)


class Deltas(unittest.TestCase):
    def test_d_area_frac_is_scale_free(self):
        sb = series([True, True], {"area": [100.0, 150.0]})
        self.assertAlmostEqual(tf.FEATURE_REGISTRY["d_area_frac"].fn(sb)[1, 0], 0.5)

    def test_d_area_frac_is_undefined_on_the_first_frame(self):
        sb = series([True, True], {"area": [100.0, 150.0]})
        self.assertTrue(np.isnan(tf.FEATURE_REGISTRY["d_area_frac"].fn(sb)[0, 0]))

    def test_d_area_frac_guards_a_zero_previous_area(self):
        sb = series([True, True], {"area": [0.0, 150.0]})
        self.assertTrue(np.isnan(tf.FEATURE_REGISTRY["d_area_frac"].fn(sb)[1, 0]))

    def test_d_circularity_is_a_plain_difference(self):
        sb = series([True, True], {"circularity": [0.4, 0.9]})
        self.assertAlmostEqual(tf.FEATURE_REGISTRY["d_circularity"].fn(sb)[1, 0], 0.5)


class TrailingWindow(unittest.TestCase):
    def test_window_spans_active_frames_only(self):
        sb = series([True, False, True, True], params={"window_frames": 3})
        values = np.array([[1.0], [np.nan], [2.0], [3.0]])
        window = sb.window(values, 3)
        self.assertEqual(list(window[3, 0]), [1.0, 2.0, 3.0])

    def test_window_is_nan_padded_before_enough_history(self):
        sb = series([True, True], params={"window_frames": 3})
        window = sb.window(np.array([[1.0], [2.0]]), 3)
        self.assertTrue(np.isnan(window[0, 0, 0]) and np.isnan(window[0, 0, 1]))
        self.assertEqual(window[0, 0, 2], 1.0)

    def test_win_std_needs_two_values(self):
        sb = series([True, True, True], {"circularity": [0.2, 0.4, 0.6]},
                    params={"window_frames": 3})
        result = tf.FEATURE_REGISTRY["win_std_circularity"].fn(sb)[:, 0]
        self.assertTrue(np.isnan(result[0]))
        self.assertAlmostEqual(result[1], np.std([0.2, 0.4]))
        self.assertAlmostEqual(result[2], np.std([0.2, 0.4, 0.6]))

    def test_win_std_log_area_guards_nonpositive_area(self):
        sb = series([True, True, True], {"area": [0.0, 10.0, 20.0]}, params={"window_frames": 3})
        result = tf.FEATURE_REGISTRY["win_std_log_area"].fn(sb)[:, 0]
        self.assertAlmostEqual(result[2], np.std([np.log(10.0), np.log(20.0)]))


class RadiusCounts(unittest.TestCase):
    def test_counts_within_the_radius_inclusive(self):
        subjects = np.array([[0.0, 0.0]])
        others = np.array([[0.0, 5.0], [0.0, 10.0], [0.0, 25.0]])
        self.assertEqual(tf._radius_counts(subjects, others, 10.0)[0], 2)

    def test_absent_cells_stay_nan(self):
        subjects = np.array([[np.nan, np.nan]])
        others = np.array([[0.0, 1.0]])
        self.assertTrue(np.isnan(tf._radius_counts(subjects, others, 10.0)[0]))

    def test_no_others_gives_zero_not_nan(self):
        subjects = np.array([[0.0, 0.0]])
        self.assertEqual(tf._radius_counts(subjects, np.zeros((0, 2)), 10.0)[0], 0.0)

    def test_cancer_neighbors_excludes_the_cell_itself(self):
        """The subject matches itself at distance zero, so one is subtracted."""
        centroids = np.array([[0.0, 0.0], [0.0, 1.0]])
        counts = tf._radius_counts(centroids, centroids, 10.0) - 1.0
        self.assertEqual(list(counts), [1.0, 1.0])


class MasksMatchFeatures(unittest.TestCase):
    def test_build_masks_agrees_with_the_presence_a_feature_pass_reports(self):
        present = np.array([[True, False], [True, True], [False, True]])
        masks = LG.build_masks(present)
        np.testing.assert_array_equal(masks["active_mask"], present)


if __name__ == "__main__":
    unittest.main()


class IntensityStats(unittest.TestCase):
    """Regression: the bincount must be sized by the IDs indexed, not those present."""

    def bundle(self, present_ids, all_ids):
        labels = np.zeros((20, 20), np.int32)
        for offset, label in enumerate(present_ids):
            labels[offset * 2 : offset * 2 + 2, 0:2] = label
        image = np.zeros((20, 20, 2), np.float32)
        image[..., 0] = 0.5
        image[..., 1] = 0.25
        return tf.FrameBundle(
            labels=labels,
            other_labels=np.zeros_like(labels),
            image=image,
            cell_ids=np.array(all_ids, np.int32),
            centroids=np.full((len(all_ids), 2), np.nan),
            other_centroids=np.zeros((0, 2)),
            params={},
        )

    def test_a_cell_absent_from_this_frame_does_not_overflow_the_bincount(self):
        """The highest-numbered cell may be missing from any given frame."""
        fb = self.bundle(present_ids=[3], all_ids=[3, 117])
        stats = tf._intensity_stats(fb, channel=0)
        self.assertAlmostEqual(float(stats["mean"][0]), 0.5)
        self.assertTrue(np.isnan(stats["mean"][1]))

    def test_mean_and_total_match_the_pixels(self):
        fb = self.bundle(present_ids=[3], all_ids=[3])
        stats = tf._intensity_stats(fb, channel=1)
        self.assertAlmostEqual(float(stats["mean"][0]), 0.25)
        self.assertAlmostEqual(float(stats["total"][0]), 0.25 * 4)
        self.assertAlmostEqual(float(stats["std"][0]), 0.0, places=6)

    def test_no_image_gives_nan_rather_than_raising(self):
        fb = self.bundle(present_ids=[3], all_ids=[3])
        fb.image = None
        self.assertTrue(np.isnan(tf._intensity_stats(fb, channel=0)["mean"]).all())
