"""`feature_distributions_dino.png`: the values it plots, and when it exists.

The two things that would go wrong silently are plotting z-scores as if they
were PC scores -- the figure looks fine either way, only the axis lies -- and
picking the emission columns by position rather than by name, which puts a track
feature under a `dino_pc_*` title as soon as `model.features` is non-empty.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from arhmm.steps.outputs import dino_component_values, write_dino_distributions


def _fit(emission_names, emissions, standardization=None, num_states=2, states=None):
    """A fit bundle holding just what the DINO figure reads."""
    emissions = np.asarray(emissions, dtype=np.float32)
    shape = emissions.shape[:2]
    if states is None:
        states = np.indices(shape)[1] % num_states
    return {
        "emissions": emissions,
        "state_assignments": np.asarray(states, dtype=np.int32),
        "active_mask": np.ones(shape, dtype=bool),
        "summary": {
            "run_name": "test_run",
            "num_states": num_states,
            "emission_names": list(emission_names),
            "standardization": standardization,
        },
    }


class _Layout:
    """A layout whose PCA cache does not exist, so the ratios drop out."""

    def step_dir(self, _name):
        return Path("/nonexistent")


class DinoComponentValuesTest(unittest.TestCase):
    def test_standardization_is_inverted_back_to_pc_scores(self):
        # The fit stored z = (x - 3) / 2, so x must come back.
        fit = _fit(
            ["dino_pc_0"],
            [[[0.0]], [[1.0]], [[-1.0]]],
            standardization={"mean": [3.0], "std": [2.0]},
        )
        names, values = dino_component_values(fit)
        self.assertEqual(names, ["dino_pc_0"])
        self.assertEqual(list(values[:, 0, 0]), [3.0, 5.0, 1.0])

    def test_an_unstandardized_fit_is_passed_through_untouched(self):
        fit = _fit(["dino_pc_0"], [[[7.0]], [[8.0]]], standardization=None)
        _, values = dino_component_values(fit)
        self.assertEqual(list(values[:, 0, 0]), [7.0, 8.0])

    def test_columns_are_picked_by_name_not_by_position(self):
        # area sits first in the emission vector; the components are 1 and 2,
        # and their standardization must be read at those same offsets.
        fit = _fit(
            ["area", "dino_pc_0", "dino_pc_1"],
            [[[10.0, 0.0, 0.0]]],
            standardization={"mean": [100.0, 3.0, -3.0], "std": [1.0, 2.0, 4.0]},
        )
        names, values = dino_component_values(fit)
        self.assertEqual(names, ["dino_pc_0", "dino_pc_1"])
        self.assertEqual(list(values[0, 0]), [3.0, -3.0])

    def test_a_run_without_components_yields_nothing(self):
        fit = _fit(["area"], [[[10.0]]])
        names, values = dino_component_values(fit)
        self.assertEqual(names, [])
        self.assertEqual(values.shape[-1], 0)


class WriteDinoDistributionsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _cfg(self):
        return {"outputs": {"feature_distributions": {"kind": "violin", "max_cols": 4}},
                "dino": {"model_id": "dinov2-base", "patch_px": [30, 50], "whiten": True}}

    def test_a_dino_run_gets_the_figure(self):
        rng = np.random.default_rng(0)
        emissions = rng.normal(size=(20, 8, 3)).astype(np.float32)
        fit = _fit(["area", "dino_pc_0", "dino_pc_1"], emissions,
                   standardization={"mean": [0.0] * 3, "std": [1.0] * 3})
        written = write_dino_distributions(self._cfg(), fit, _Layout(), self.out_dir)
        self.assertEqual([p.name for p in written], ["feature_distributions_dino.png"])
        self.assertTrue(written[0].is_file() and written[0].stat().st_size > 0)

    def test_a_run_without_components_writes_nothing(self):
        fit = _fit(["area"], np.ones((5, 4, 1), dtype=np.float32))
        self.assertEqual(write_dino_distributions(self._cfg(), fit, _Layout(), self.out_dir), [])
        self.assertEqual(list(self.out_dir.iterdir()), [])

    def test_a_state_too_sparse_for_a_violin_still_renders(self):
        # One state holds a single cell-frame, which has no density to estimate;
        # the panel must fall back to a boxplot rather than raising.
        emissions = np.arange(12, dtype=np.float32).reshape(4, 3, 1)
        states = np.zeros((4, 3), dtype=np.int32)
        states[0, 0] = 1
        fit = _fit(["dino_pc_0"], emissions, num_states=2, states=states)
        written = write_dino_distributions(self._cfg(), fit, _Layout(), self.out_dir)
        self.assertTrue(written[0].is_file())


if __name__ == "__main__":
    unittest.main()
