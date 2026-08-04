"""DINO patch composition.

The decisions here are the ones a contact sheet can only hint at: whether the
subject really is centred, whether an overlapping neighbour bleeds into its
colour, and whether blending preserves the phase texture the embedding is
supposed to describe.

Only the numpy half of `core.dino` is exercised, which is why torch is imported
lazily there -- these run in all three conda environments.

    python -m unittest discover -s tests -t tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

import conftest  # noqa: F401
from arhmm.core import dino as D

PARAMS = dict(
    mask_alpha=0.45,
    subject_colour=[1.0, 0.0, 0.0],
    other_cancer_colour=[0.0, 0.0, 1.0],
    tcell_colour=[0.0, 1.0, 0.0],
    include_rfp=False,
    rfp_alpha=0.3,
    patch_px=50,
)


def scene(cancer_boxes=(), tcell_boxes=(), size=150, background=0.5):
    """Build a synthetic frame: uniform phase plus labelled boxes."""
    phase = np.full((size, size), background, np.float32)
    cancer = np.zeros((size, size), np.int32)
    tcells = np.zeros((size, size), np.int32)
    for label, (y0, y1, x0, x1) in cancer_boxes:
        cancer[y0:y1, x0:x1] = label
    for label, (y0, y1, x0, x1) in tcell_boxes:
        tcells[y0:y1, x0:x1] = label
    return phase, cancer, tcells


def redness(patch):
    """Where the patch is dominated by the subject colour."""
    return (patch[..., 0].astype(int) - patch[..., 2].astype(int)) > 30


class Centring(unittest.TestCase):
    def test_subject_lands_at_the_patch_centre(self):
        phase, cancer, tcells = scene([(7, (60, 70, 100, 110))])
        shared, all_cancer = D.compose_frame(phase, cancer, tcells, None, PARAMS)
        patch = D.subject_patch(shared, all_cancer, cancer == 7, (64.5, 104.5),
                                50, PARAMS["subject_colour"], PARAMS["mask_alpha"])
        ys, xs = np.nonzero(redness(patch))
        self.assertEqual(patch.shape, (50, 50, 3))
        self.assertAlmostEqual(ys.mean(), 24.5, delta=1.0)
        self.assertAlmostEqual(xs.mean(), 24.5, delta=1.0)

    def test_a_cell_at_the_frame_edge_is_still_centred_and_whole(self):
        """Edge padding must widen the view, not clip or smear the subject."""
        phase, cancer, tcells = scene([(7, (60, 70, 2, 12))])
        shared, all_cancer = D.compose_frame(phase, cancer, tcells, None, PARAMS)
        patch = D.subject_patch(shared, all_cancer, cancer == 7, (64.5, 6.5),
                                50, PARAMS["subject_colour"], PARAMS["mask_alpha"])
        red = redness(patch)
        ys, xs = np.nonzero(red)
        self.assertEqual(int(red.sum()), 100, "the subject was clipped or smeared")
        self.assertAlmostEqual(ys.mean(), 24.5, delta=1.0)
        self.assertAlmostEqual(xs.mean(), 24.5, delta=1.0)

    def test_patch_size_is_fixed_wherever_the_cell_is(self):
        phase, cancer, tcells = scene([(7, (0, 6, 0, 6))])
        shared, all_cancer = D.compose_frame(phase, cancer, tcells, None, PARAMS)
        for centroid in [(2.5, 2.5), (75.0, 75.0), (147.0, 147.0)]:
            with self.subTest(centroid):
                patch = D.subject_patch(shared, all_cancer, cancer == 7, centroid, 50,
                                        PARAMS["subject_colour"], PARAMS["mask_alpha"])
                self.assertEqual(patch.shape, (50, 50, 3))


class Composition(unittest.TestCase):
    def test_cell_types_get_their_own_hues(self):
        phase, cancer, tcells = scene(
            [(7, (60, 70, 60, 70)), (8, (60, 70, 90, 100))],
            [(1, (100, 110, 60, 70))],
        )
        shared, all_cancer = D.compose_frame(phase, cancer, tcells, None, PARAMS)
        patch = D.subject_patch(shared, all_cancer, cancer == 7, (64.5, 64.5), 100,
                                PARAMS["subject_colour"], PARAMS["mask_alpha"])
        centre = patch[50, 50]                       # the subject
        other = patch[50, 50 + 30]                   # cancer cell 8
        tcell = patch[50 + 40, 50]                   # the T cell
        self.assertGreater(int(centre[0]), int(centre[2]), "subject should be red")
        self.assertGreater(int(other[2]), int(other[0]), "other cancer should be blue")
        self.assertGreater(int(tcell[1]), int(tcell[0]), "T cells should be green")

    def test_subject_is_repainted_from_the_pre_blue_image(self):
        """An overlapping neighbour must not turn the subject purple.

        If the subject were composited over the image that already carries the
        blue cancer layer, its colour would encode overlap rather than identity.
        """
        phase, cancer, tcells = scene([(8, (60, 70, 60, 70))])
        overlapping = cancer == 8                     # subject shares these pixels
        shared, all_cancer = D.compose_frame(phase, cancer, tcells, None, PARAMS)
        patch = D.subject_patch(shared, all_cancer, overlapping, (64.5, 64.5), 50,
                                PARAMS["subject_colour"], PARAMS["mask_alpha"])
        centre = patch[25, 25]
        self.assertGreater(int(centre[0]), int(centre[2]) + 30,
                           "subject came out red-over-blue")

    def test_blending_preserves_texture_rather_than_flat_filling(self):
        """Two different phase values under the mask must stay different."""
        size = 150
        phase = np.full((size, size), 0.2, np.float32)
        phase[60:65, 60:70] = 0.9
        cancer = np.zeros((size, size), np.int32)
        cancer[60:70, 60:70] = 7
        tcells = np.zeros((size, size), np.int32)
        shared, all_cancer = D.compose_frame(phase, cancer, tcells, None, PARAMS)
        patch = D.subject_patch(shared, all_cancer, cancer == 7, (64.5, 64.5), 50,
                                PARAMS["subject_colour"], PARAMS["mask_alpha"])
        bright = int(patch[22, 25, 1])   # green channel where phase was 0.9
        dark = int(patch[28, 25, 1])     # where phase was 0.2
        self.assertGreater(bright, dark + 20, "the mask was flat-filled, losing texture")

    def test_rfp_goes_into_luminance_not_a_hue(self):
        phase, cancer, tcells = scene([(7, (60, 70, 60, 70))])
        rfp = np.zeros_like(phase)
        rfp[60:70, 60:70] = 1.0
        params = dict(PARAMS, include_rfp=True, rfp_alpha=0.3)
        shared_off, _ = D.compose_frame(phase, cancer, tcells, None, PARAMS)
        shared_on, _ = D.compose_frame(phase, cancer, tcells, rfp, params)
        delta = shared_on[65, 65] - shared_off[65, 65]
        self.assertTrue(np.allclose(delta, delta[0], atol=1e-6),
                        "RFP shifted the hue instead of the luminance")
        self.assertGreater(float(delta[0]), 0.0)

    def test_background_is_left_alone(self):
        phase, cancer, tcells = scene([(7, (60, 70, 60, 70))])
        shared, all_cancer = D.compose_frame(phase, cancer, tcells, None, PARAMS)
        np.testing.assert_allclose(all_cancer[10, 10], [0.5, 0.5, 0.5], atol=1e-6)


class ModelSourceResolution(unittest.TestCase):
    def test_falls_back_to_the_model_id_without_a_path(self):
        self.assertEqual(D.resolve_model_src("facebook/dinov2-base", None),
                         "facebook/dinov2-base")

    def test_prefers_the_newest_snapshot_directory(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "snapshots"
            (snapshots / "old").mkdir(parents=True)
            (snapshots / "new").mkdir(parents=True)
            import os
            os.utime(snapshots / "old", (1, 1))
            os.utime(snapshots / "new", (10_000_000, 10_000_000))
            self.assertEqual(D.resolve_model_src("x", root), str(snapshots / "new"))


if __name__ == "__main__":
    unittest.main()
