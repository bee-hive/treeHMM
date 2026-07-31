"""Run directory layout, stamps, and what `status` reports.

`status` and `run` must agree about which steps are current. They disagreed
once: `status` cascaded staleness downstream, so a step whose upstream had
merely lost its stamp (a killed process) was reported stale even though its own
key was unchanged and `run` then correctly skipped it.

    python -m unittest discover -s tests -t tests -v
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401
from treearhmm import config as C
from treearhmm import layout as L

SMOKE = "configs/_smoke.yml"


class LayoutPaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.cfg = C.load_config(SMOKE)
        self.cfg["paths"]["output_root"] = str(root / "runs")
        self.cfg["paths"]["cache_root"] = str(root / "cache")
        self.layout = L.Layout(self.cfg)

    def test_shared_steps_live_in_the_cache_and_local_steps_in_the_run(self):
        cache_root = Path(self.cfg["paths"]["cache_root"])
        run_dir = self.layout.run_dir
        self.assertTrue(str(self.layout.step_dir("features")).startswith(str(cache_root)))
        self.assertTrue(str(self.layout.step_dir("fit")).startswith(str(run_dir)))

    def test_a_shared_step_stamps_its_cache_not_the_run(self):
        """Otherwise a second run with identical inputs recomputes a full cache entry."""
        self.assertTrue(str(self.layout.stamp_path("features")).startswith(
            str(self.layout.step_dir("features"))))
        self.assertTrue(str(self.layout.stamp_path("fit")).startswith(
            str(self.layout.stamp_dir)))

    def test_the_cache_directory_is_named_by_the_step_key(self):
        key = L.step_key(self.cfg, "features")
        self.assertEqual(self.layout.step_dir("features").name, key)

    def test_two_runs_with_the_same_inputs_share_a_cache_directory(self):
        other = copy.deepcopy(self.cfg)
        other["run_name"] = "another"
        other["model"]["num_states"] = 5     # changes fit, not features
        self.assertEqual(L.Layout(other).step_dir("features"),
                         self.layout.step_dir("features"))
        self.assertNotEqual(L.Layout(other).step_dir("fit"), self.layout.fit_dir)


class Currency(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.cfg = C.load_config(SMOKE)
        self.cfg["paths"]["output_root"] = str(root / "runs")
        self.cfg["paths"]["cache_root"] = str(root / "cache")
        self.layout = L.Layout(self.cfg)
        self.layout.ensure_run_dir()

    def complete(self, name):
        """Pretend a step finished: create its directory and stamp it."""
        directory = self.layout.step_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        for produced in L.declared_outputs(name, self.cfg):
            (directory / produced).mkdir(parents=True, exist_ok=True)
        self.layout.write_stamp(name)

    def reasons(self):
        return {row["step"]: row["reason"] for row in self.layout.status()}

    def test_a_step_that_never_ran_is_not_current(self):
        self.assertFalse(self.layout.is_current("features"))
        self.assertEqual(self.reasons()["features"], "never run")

    def test_a_completed_step_is_current(self):
        self.complete("features")
        self.assertTrue(self.layout.is_current("features"))
        self.assertEqual(self.reasons()["features"], "cached")

    def test_a_stamp_alone_is_not_trusted(self):
        """A hand-deleted cache directory must be noticed."""
        self.complete("features")
        stamp = self.layout.stamp_path("features")
        directory = self.layout.step_dir("features")
        for child in sorted(directory.iterdir()):
            if child != stamp:
                child.rmdir()
        self.assertFalse(self.layout.is_current("features"))
        self.assertEqual(self.reasons()["features"], "output missing")

    def test_status_does_not_cascade_over_a_merely_missing_stamp(self):
        """The regression: a killed upstream step must not mark a valid one stale."""
        self.complete("features")
        self.complete("outputs")          # fit was killed, so it has no stamp
        reasons = self.reasons()
        self.assertEqual(reasons["fit"], "never run")
        self.assertEqual(reasons["outputs"], "cached")
        self.assertTrue(self.layout.is_current("outputs"))

    def test_a_real_input_change_still_invalidates_downstream(self):
        for step in L.active_steps(self.cfg):
            self.complete(step.name)
        self.assertTrue(all(row["current"] for row in self.layout.status()))

        changed = copy.deepcopy(self.cfg)
        changed["features"]["params"]["neighbor_radius_px"] = 25
        after = {row["step"]: row["reason"] for row in L.Layout(changed).status()}
        self.assertEqual(after["features"], "never run")     # a different cache key
        self.assertEqual(after["fit"], "inputs changed")
        self.assertEqual(after["outputs"], "inputs changed")

    def test_stamp_records_the_key_it_finished_for(self):
        self.complete("features")
        self.assertEqual(self.layout.read_stamp("features")["key"],
                         L.step_key(self.cfg, "features"))


class StepGraph(unittest.TestCase):
    def test_every_step_declares_an_env_that_the_config_maps(self):
        cfg = C.load_config(SMOKE)
        for step in L.STEPS:
            with self.subTest(step.name):
                self.assertIn(step.env, cfg["envs"])

    def test_upstream_names_are_real_steps(self):
        for step in L.STEPS:
            for upstream in step.upstream:
                self.assertIn(upstream, L.STEPS_BY_NAME)

    def test_steps_from_starts_at_the_named_step(self):
        cfg = C.load_config(SMOKE)
        names = [s.name for s in L.steps_from(cfg, "fit")]
        self.assertEqual(names[0], "fit")
        self.assertNotIn("features", names)

    def test_steps_from_rejects_an_inactive_step(self):
        cfg = C.load_config(SMOKE)          # no DINO
        with self.assertRaises(KeyError):
            L.steps_from(cfg, "dino")


if __name__ == "__main__":
    unittest.main()
