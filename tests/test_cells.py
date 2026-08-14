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


def _extension_nucleus_stack() -> np.ndarray:
    """A stack with one track that stops early, for `cells.extend_nuclei`.

    Label 7 appears in frame 0 only, so it is the sole candidate; 3 and 9 run to
    the last frame and are both well outside 7's exclusion box at this size.
    """
    frame = np.zeros((6, 6), dtype=np.int64)
    frame[1:2, 4:5] = 3
    frame[4:5, 4:5] = 9
    stack = np.stack([frame] * 3)
    stack[0, 0, 0] = 7
    return stack[..., None]


@unittest.skipIf(tifffile is None, "tifffile is not installed in this environment")
class LoadCrop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        self.gt_root = root / "gt"
        self.img_root = root / "img"
        self.sam3_root = root / "sam3"
        self.crop_images = self.img_root / "B4" / CROP
        crop_dir = self.gt_root / "B4" / CROP
        crop_dir.mkdir(parents=True)
        self.crop_images.mkdir(parents=True)
        # Flat: one directory per crop id, with no well level.
        (self.sam3_root / CROP).mkdir(parents=True)

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

    def write_sam3(self, stack: np.ndarray) -> None:
        """SAM3 phase tracks, in their own FLAT root -- no well level."""
        _write_tiff(self.sam3_root / CROP / "tracks.tiff", stack)

    def write_divisions(self, rows: list[tuple[int, int, int, int]]) -> None:
        """`nuclei_div.pkl`, a real pickled DataFrame beside the nucleus tracks.

        Pickled for real rather than stubbed, so the test covers the unpickle
        path that has to work across the three environments' pandas versions.
        """
        import pandas as pd

        table = pd.DataFrame(rows, columns=["parent", "daughter_1", "daughter_2", "frame"])
        with open(self.crop_images / "nuclei_div.pkl", "wb") as handle:
            pickle.dump(table, handle)

    def setup_extension_fixture(self, divisions=()) -> np.ndarray:
        """Nucleus tracks with an early-ending label, SAM3 support, a division
        table.  Returns the squeezed nucleus stack."""
        stack = _extension_nucleus_stack()
        self.write_nuclei(stack)
        self.write_sam3(np.ones((3, 6, 6), dtype=np.uint16))
        self.write_divisions(list(divisions))
        return stack[..., 0]

    def cfg(self, source: str, extend: dict | None = None) -> dict:
        return {
            "paths": {
                "ground_truth_tracks_dir": str(self.gt_root),
                "image_crops_dir": str(self.img_root),
                "sam3_tracks_dir": str(self.sam3_root),
            },
            # The 6x6 fixture cannot host a 50 px box, so the sides are given
            # explicitly; `auto` resolution is covered in `test_config.py`,
            # where it needs no filesystem.
            "cells": {"source": source, "extend_nuclei": extend or {"frames": 0}},
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

    # ---- cells.extend_nuclei --------------------------------------------- #

    ON = {"frames": 1, "exclusion_px": 3, "evidence_px": 1}

    def test_the_sam3_tracks_do_not_hang_off_the_well(self):
        """Flat, unlike every other root -- exactly what a tidy-up would break."""
        path = cellsmod.sam3_tracks_path(self.cfg("nuclei"), CROP)
        self.assertEqual(path.parent.name, CROP)
        self.assertEqual(path.parent.parent, self.sam3_root)
        self.assertNotIn("B4", path.relative_to(self.sam3_root).parts)

    def test_it_is_inert_by_default_for_both_sources(self):
        for source in ("phase", "nuclei"):
            with self.subTest(source):
                crop = cellsmod.load_crop(self.cfg(source), CROP, with_image=False)
                self.assertIsNone(crop.extension)

    def test_a_nuclei_run_with_it_off_is_byte_identical(self):
        crop = cellsmod.load_crop(self.cfg("nuclei"), CROP, with_image=False)
        np.testing.assert_array_equal(crop.cancer, self.nuclei[..., 0])

    def test_a_phase_run_asking_for_it_is_still_inert(self):
        """`load_crop` guards on the source itself, not on `validate` having run."""
        crop = cellsmod.load_crop(self.cfg("phase", self.ON), CROP, with_image=False)
        self.assertIsNone(crop.extension)
        np.testing.assert_array_equal(np.unique(crop.cancer), [0, 1, 2])

    def test_a_terminating_track_gains_exactly_its_frames(self):
        stack = self.setup_extension_fixture()
        crop = cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)

        held = [r for r in crop.extension if r.frames_added]
        self.assertEqual([(r.cell_id, r.frames_added) for r in held], [(7, 1)])
        # Same pixels, so same shape and same centroid.
        np.testing.assert_array_equal(crop.cancer[1] == 7, stack[0] == 7)
        # Frame 2 is untouched: the budget was one frame.
        self.assertFalse((crop.cancer[2] == 7).any())

    def test_holding_a_track_moves_neither_the_column_order_nor_the_t_cells(self):
        self.setup_extension_fixture()
        crop = cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)
        np.testing.assert_array_equal(crop.cell_ids, [3, 7, 9])
        np.testing.assert_array_equal(np.unique(crop.tcells), [0, 5, 6])
        # Exactly one cell-frame appears that was not observed.
        self.assertEqual(int(crop.presence().sum()), 3 + 3 + 1 + 1)

    def test_every_candidate_is_recorded_not_just_the_held_ones(self):
        self.setup_extension_fixture()
        crop = cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)
        self.assertEqual({r.cell_id for r in crop.extension}, {3, 7, 9})
        self.assertEqual(
            {r.cell_id: r.reason for r in crop.extension},
            {3: "ends_at_movie_end", 7: "extended", 9: "ends_at_movie_end"},
        )

    def test_a_parent_named_in_the_table_is_not_held(self):
        self.setup_extension_fixture(divisions=[(7, 3, 9, 1)])
        crop = cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)
        record = next(r for r in crop.extension if r.cell_id == 7)
        self.assertEqual(record.reason, "divides")
        self.assertEqual(record.frames_added, 0)
        self.assertFalse((crop.cancer[1] == 7).any())

    def test_a_crop_with_no_divisions_is_fine(self):
        self.setup_extension_fixture(divisions=[])
        crop = cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)
        self.assertTrue(any(r.frames_added for r in crop.extension))

    def test_a_division_naming_an_unknown_label_names_the_crop(self):
        self.setup_extension_fixture(divisions=[(7, 3, 404, 1)])
        with self.assertRaisesRegex(ValueError, f"{CROP}.*404"):
            cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)

    def test_a_missing_division_table_names_the_path(self):
        self.setup_extension_fixture()
        (self.crop_images / "nuclei_div.pkl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "nuclei_div.pkl"):
            cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)

    def test_a_missing_sam3_stack_names_the_path(self):
        self.setup_extension_fixture()
        (self.sam3_root / CROP / "tracks.tiff").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "tracks.tiff"):
            cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)

    def test_a_sam3_frame_count_mismatch_names_the_crop(self):
        self.setup_extension_fixture()
        self.write_sam3(np.ones((2, 6, 6), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, f"{CROP}.*different acquisitions"):
            cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)

    def test_a_sam3_frame_size_mismatch_names_the_crop(self):
        self.setup_extension_fixture()
        self.write_sam3(np.ones((3, 4, 4), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, f"{CROP}.*not aligned"):
            cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)

    def test_no_sam3_evidence_leaves_the_track_alone(self):
        self.setup_extension_fixture()
        self.write_sam3(np.zeros((3, 6, 6), dtype=np.uint16))
        crop = cellsmod.load_crop(self.cfg("nuclei", self.ON), CROP, with_image=False)
        record = next(r for r in crop.extension if r.cell_id == 7)
        self.assertEqual(record.reason, "no_sam3_evidence")
        self.assertFalse((crop.cancer[1] == 7).any())


if __name__ == "__main__":
    unittest.main()
