"""What `load_crop` promises, for both cell sources.

`core/cells.py` is the one module that opens files, so unlike the rest of the
suite these tests need a filesystem -- but they build a synthetic three-frame
crop in a temporary directory rather than reading the real data, so they say
nothing about the network filesystem being mounted.

The point of the two sources is that they produce the *same* dataclass over a
*different* label space: `nuclei` swaps the cancer masks and the column order for
the nucleus tracks and keeps the CVAT T cells, and the two ID spaces are never
reconciled.

Uses stdlib `unittest` rather than pytest so the suite runs in all three conda
environments without installing anything into them.

    python -m unittest discover -s tests -t tests -v
"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

import conftest  # noqa: F401  (puts the repo root on sys.path)
from arhmm.core import cells as cellsmod

try:
    import tifffile
except ImportError:  # pragma: no cover - only the imaging and dino envs have it
    tifffile = None

CROP = "B4_t0t3y0y6x0x6"

#: CVAT ids: 1 and 2 are cancer, 5 and 6 are T cells.
CANCER_IDS = [1, 2]
#: Nucleus labels, deliberately overlapping neither the cancer ids nor the
#: T-cell ids in meaning -- they are a different ID space that is never mapped
#: onto the CVAT one.
NUCLEUS_LABELS = [3, 7]


def _write_tiff(path: Path, stack: np.ndarray) -> None:
    """Write a label or image stack, unpaged.

    `photometric="minisblack"` is not cosmetic: without it tifffile reads a
    three-frame 6x6 stack back as an RGB image with separate component planes.
    """
    tifffile.imwrite(path, stack, photometric="minisblack")


def _cvat_frame() -> np.ndarray:
    frame = np.zeros((6, 6), dtype=np.uint16)
    frame[0:2, 0:2] = 1     # cancer
    frame[0:2, 4:6] = 2     # cancer
    frame[4:6, 0:2] = 5     # T cell
    frame[4:6, 4:6] = 6     # T cell
    return frame


def _nucleus_frame() -> np.ndarray:
    frame = np.zeros((6, 6), dtype=np.int64)
    frame[0:1, 0:1] = 7     # inside cancer 1, but labelled independently
    frame[1:2, 4:5] = 3     # inside cancer 2
    return frame


@unittest.skipIf(tifffile is None, "tifffile is not installed in this environment")
class LoadCrop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        self.gt_root = root / "gt"
        self.img_root = root / "img"
        self.crop_images = self.img_root / "B4" / CROP
        crop_dir = self.gt_root / "B4" / CROP
        crop_dir.mkdir(parents=True)
        self.crop_images.mkdir(parents=True)

        self.cvat = np.stack([_cvat_frame()] * 3)
        self.nuclei = np.stack([_nucleus_frame()] * 3)[..., None]
        # Two channels with different ranges, so a swap would show up.
        self.image = np.stack(
            [np.full((3, 6, 6), 10.0, np.float32), np.full((3, 6, 6), 200.0, np.float32)],
            axis=-1,
        )
        self.image[:, 0, 0, :] = 0.0

        _write_tiff(crop_dir / "ALL_tracks.tiff", self.cvat)
        with open(crop_dir / "ALL_cancer_ids.pkl", "wb") as handle:
            pickle.dump(CANCER_IDS, handle)
        _write_tiff(self.crop_images / "crop.tiff", self.image)
        self.write_nuclei(self.nuclei)

    def write_nuclei(self, stack: np.ndarray) -> None:
        """The nucleus tracks live in the crop directory, beside crop.tiff."""
        _write_tiff(self.crop_images / "nuclei_tracks.tiff", stack)

    def cfg(self, source: str) -> dict:
        return {
            "paths": {
                "ground_truth_tracks_dir": str(self.gt_root),
                "image_crops_dir": str(self.img_root),
            },
            "cells": {"source": source},
        }

    # ---- phase, which the refactor must leave alone ---------------------- #

    def test_phase_takes_its_masks_and_column_order_from_cvat(self):
        crop = cellsmod.load_crop(self.cfg("phase"), CROP)
        np.testing.assert_array_equal(np.unique(crop.cancer), [0, 1, 2])
        np.testing.assert_array_equal(crop.cell_ids, CANCER_IDS)
        np.testing.assert_array_equal(np.unique(crop.tcells), [0, 5, 6])
        self.assertEqual(crop.num_frames, 3)

    # ---- nuclei ---------------------------------------------------------- #

    def test_nuclei_cancer_masks_are_the_squeezed_nucleus_stack(self):
        crop = cellsmod.load_crop(self.cfg("nuclei"), CROP)
        np.testing.assert_array_equal(crop.cancer, self.nuclei[..., 0])
        self.assertEqual(crop.cancer.dtype, np.int32)

    def test_nuclei_column_order_is_the_sorted_nonzero_labels(self):
        crop = cellsmod.load_crop(self.cfg("nuclei"), CROP)
        np.testing.assert_array_equal(crop.cell_ids, sorted(NUCLEUS_LABELS))
        self.assertEqual(crop.cell_ids.dtype, np.int32)
        self.assertEqual(crop.num_cells, 2)

    def test_nuclei_t_cells_still_come_from_cvat(self):
        """The nucleus tracks cover cancer only, so the T cells are borrowed."""
        crop = cellsmod.load_crop(self.cfg("nuclei"), CROP)
        np.testing.assert_array_equal(np.unique(crop.tcells), [0, 5, 6])
        np.testing.assert_array_equal(
            crop.tcells, cellsmod.load_crop(self.cfg("phase"), CROP).tcells
        )

    def test_a_missing_trailing_axis_is_also_accepted(self):
        self.write_nuclei(self.nuclei[..., 0])
        crop = cellsmod.load_crop(self.cfg("nuclei"), CROP)
        np.testing.assert_array_equal(crop.cancer, self.nuclei[..., 0])

    def test_a_two_channel_nucleus_stack_is_rejected(self):
        self.write_nuclei(np.repeat(self.nuclei, 2, axis=-1))
        with self.assertRaisesRegex(ValueError, CROP):
            cellsmod.load_crop(self.cfg("nuclei"), CROP)

    # ---- the two label spaces have to describe the same pixels ----------- #

    def test_a_frame_count_mismatch_names_the_crop(self):
        self.write_nuclei(self.nuclei[:2])
        with self.assertRaisesRegex(ValueError, f"{CROP}.*different acquisitions"):
            cellsmod.load_crop(self.cfg("nuclei"), CROP)

    def test_a_frame_size_mismatch_names_the_crop(self):
        self.write_nuclei(self.nuclei[:, :4, :4])
        with self.assertRaisesRegex(ValueError, f"{CROP}.*not aligned"):
            cellsmod.load_crop(self.cfg("nuclei"), CROP)

    # ---- the image is the same file, and the same normalization ---------- #

    def test_the_image_is_normalized_over_the_whole_stack_for_both_sources(self):
        for source in ("phase", "nuclei"):
            with self.subTest(source):
                crop = cellsmod.load_crop(self.cfg(source), CROP)
                self.assertEqual(crop.image.shape, (3, 6, 6, 2))
                self.assertTrue(np.all((crop.image >= 0.0) & (crop.image <= 1.0)))
                # Channel 0 is RFP and channel 1 phase, each scaled on its own.
                self.assertGreater(crop.image[0, 1, 1, 0], crop.image[0, 0, 0, 0])

    def test_the_image_is_skipped_on_request(self):
        for source in ("phase", "nuclei"):
            with self.subTest(source):
                self.assertIsNone(cellsmod.load_crop(self.cfg(source), CROP, with_image=False).image)

    def test_the_nucleus_tracks_sit_beside_the_image(self):
        """Both come off `image_crops_dir`, in the crop's own directory."""
        cfg = self.cfg("nuclei")
        self.assertEqual(
            cellsmod.nucleus_tracks_path(cfg, CROP).parent,
            cellsmod.image_path(cfg, CROP).parent,
        )
        self.assertEqual(cellsmod.nucleus_tracks_path(cfg, CROP).name, "nuclei_tracks.tiff")

    def test_an_unknown_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown cells.source"):
            cellsmod.load_crop(self.cfg("wat"), CROP)


if __name__ == "__main__":
    unittest.main()
