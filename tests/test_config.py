"""Config merge semantics, validation rules, and cache-key behaviour.

These are the rules the pipeline's reproducibility rests on, and all of them are
cheap to check because `validate` deliberately touches no filesystem.

Uses stdlib `unittest` rather than pytest so the suite runs in all three conda
environments without installing anything into them.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

import conftest  # noqa: F401  (puts the repo root on sys.path)
from treearhmm import config as C
from treearhmm import layout as L

SMOKE = "configs/_smoke.yml"


def _write(directory: Path, name: str, payload: dict) -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(payload))
    return path


class DeepMerge(unittest.TestCase):
    def test_mappings_merge_key_by_key(self):
        base = {"model": {"num_states": 3, "num_lags": 1}}
        merged = C.deep_merge(base, {"model": {"num_states": 4}})
        self.assertEqual(merged, {"model": {"num_states": 4, "num_lags": 1}})

    def test_lists_are_replaced_wholesale(self):
        """A run narrowing a list must get exactly what it wrote, not an append."""
        base = {"model": {"features": ["area", "circularity", "velocity"]}}
        merged = C.deep_merge(base, {"model": {"features": ["area"]}})
        self.assertEqual(merged["model"]["features"], ["area"])

    def test_null_deletes_an_inherited_key(self):
        base = {"outputs": {"extras": ["state_timeline"], "video": {"fps": 2}}}
        merged = C.deep_merge(base, {"outputs": {"extras": None}})
        self.assertEqual(merged, {"outputs": {"video": {"fps": 2}}})

    def test_arguments_are_not_mutated(self):
        base, override = {"a": {"b": 1}}, {"a": {"c": 2}}
        C.deep_merge(base, override)
        self.assertEqual(base, {"a": {"b": 1}})
        self.assertEqual(override, {"a": {"c": 2}})


class ExtendsChain(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_resolves_transitively(self):
        _write(self.dir, "a.yml", {"model": {"num_states": 2, "num_lags": 1}, "kept": True})
        _write(self.dir, "b.yml", {"extends": "a.yml", "model": {"num_states": 3}})
        leaf = _write(self.dir, "c.yml", {"extends": "b.yml", "model": {"num_lags": 0}})

        cfg = C.load_config(leaf, validate_config=False)
        self.assertEqual(cfg["model"], {"num_states": 3, "num_lags": 0})
        self.assertTrue(cfg["kept"])
        self.assertNotIn("extends", cfg)

    def test_cycle_is_reported(self):
        _write(self.dir, "a.yml", {"extends": "b.yml"})
        _write(self.dir, "b.yml", {"extends": "a.yml"})
        with self.assertRaisesRegex(C.ConfigError, "cycle"):
            C.load_config(self.dir / "a.yml", validate_config=False)

    def test_missing_extends_target_is_reported(self):
        leaf = _write(self.dir, "a.yml", {"extends": "nope.yml"})
        with self.assertRaisesRegex(C.ConfigError, "not found"):
            C.load_config(leaf, validate_config=False)

    def test_config_without_extends_layers_over_sibling_default(self):
        _write(self.dir, "default.yml", {"model": {"num_states": 3}, "from_default": 1})
        leaf = _write(self.dir, "run.yml", {"model": {"num_states": 5}})
        cfg = C.load_config(leaf, validate_config=False)
        self.assertEqual(cfg["model"]["num_states"], 5)
        self.assertEqual(cfg["from_default"], 1)


class Hashing(unittest.TestCase):
    def test_subset_is_keyed_by_dotted_path(self):
        """Renesting the source config must not silently change a hash."""
        self.assertEqual(C.subset({"model": {"num_states": 3}}, ["model.num_states"]),
                         {"model.num_states": 3})

    def test_absent_keys_are_recorded_as_none(self):
        """Adding a key with a value must differ from never having had it."""
        self.assertEqual(C.subset({}, ["model.nope"]), {"model.nope": None})

    def test_canonical_json_ignores_key_order(self):
        self.assertEqual(C.canonical_json({"a": 1, "b": 2}), C.canonical_json({"b": 2, "a": 1}))

    def test_hash_is_stable_and_sized(self):
        payload = {"x": [1, 2, 3]}
        self.assertEqual(C.hash_obj(payload, 12), C.hash_obj(payload, 12))
        self.assertEqual(len(C.hash_obj(payload, 12)), 12)


class Validation(unittest.TestCase):
    def setUp(self):
        self.cfg = C.load_config(SMOKE)

    def broken(self, mutate):
        cfg = copy.deepcopy(self.cfg)
        mutate(cfg)
        return cfg

    def test_smoke_config_is_valid(self):
        self.assertEqual(self.cfg["run_name"], "_smoke")

    def test_rejects(self):
        cases = [
            ("run_name with a slash", lambda c: c.update(run_name="a/b"), "single directory name"),
            ("run_name climbing out", lambda c: c.update(run_name="../esc"), "single directory name"),
            ("empty crop_ids", lambda c: c["data"].update(crop_ids=[]), "non-empty"),
            ("duplicate crops", lambda c: c["data"].update(crop_ids=["x", "x"]), "duplicates"),
            ("unknown cell source", lambda c: c["cells"].update(source="wat"), "cells.source"),
            ("zero warmup", lambda c: c["cells"].update(warmup_frames=0), "warmup_frames"),
            ("min_frames below warmup", lambda c: c["cells"].update(min_frames=1), "min_frames"),
            ("unknown computed feature",
             lambda c: c["features"].update(compute=["not_a_feature"]), "unknown feature"),
            ("unknown model feature",
             lambda c: c["model"].update(features=["not_a_feature"]), "unknown feature"),
            ("num_lags above 1", lambda c: c["model"].update(num_lags=2), "num_lags"),
            ("one state", lambda c: c["model"].update(num_states=1), "num_states"),
            ("no seeds", lambda c: c["model"].update(em_seeds=[]), "em_seeds"),
            ("duplicate seeds", lambda c: c["model"].update(em_seeds=[0, 0]), "duplicates"),
            ("no model inputs",
             lambda c: c["model"].update(features=[], use_dino_pcs=False), "no inputs"),
            ("unknown extra", lambda c: c["outputs"].update(extras=["nope"]), "unknown extra"),
        ]
        for label, mutate, message in cases:
            with self.subTest(label):
                with self.assertRaisesRegex(C.ConfigError, message):
                    C.validate(self.broken(mutate))

    def test_model_features_must_be_computed(self):
        def mutate(c):
            c["features"]["compute"] = ["area"]
            c["model"]["features"] = ["area", "solidity"]

        with self.assertRaisesRegex(C.ConfigError, "features step will not"):
            C.validate(self.broken(mutate))

    def test_dino_columns_require_a_model_path(self):
        def mutate(c):
            c["model"]["use_dino_pcs"] = True
            c["dino"]["model_path"] = ""

        with self.assertRaisesRegex(C.ConfigError, "model_path"):
            C.validate(self.broken(mutate))

    def test_nuclei_source_rejects_whole_cell_shape_features(self):
        def mutate(c):
            c["cells"]["source"] = "nuclei"
            c["features"]["compute"] = ["area", "solidity"]

        with self.assertRaisesRegex(C.ConfigError, "nucleus rather than a cell"):
            C.validate(self.broken(mutate))

    def test_a_crop_cannot_be_in_two_conditions(self):
        crop = self.cfg["data"]["crop_ids"][0]
        with self.assertRaisesRegex(C.ConfigError, "two conditions"):
            C.validate(self.broken(lambda c: c["data"].update(conditions={"SH": [crop],
                                                                          "CUL5": [crop]})))

    def test_conditions_may_name_crops_this_run_does_not_use(self):
        """site.yml describes the dataset; a run narrowing crop_ids still validates."""
        self.assertEqual(len(self.cfg["data"]["crop_ids"]), 1)
        self.assertEqual(len(self.cfg["data"]["conditions"]), 3)
        C.validate(self.cfg)

    def test_unknown_feature_message_suggests_a_near_miss(self):
        with self.assertRaisesRegex(C.ConfigError, "did you mean 'circularity'"):
            C.validate(self.broken(lambda c: c["features"].update(compute=["circularty"])))


class DerivedViews(unittest.TestCase):
    def setUp(self):
        self.cfg = C.load_config(SMOKE)

    def test_computed_features_pulls_in_temporal_dependencies(self):
        """d_area_frac reads area, so area is computed even when not requested."""
        cfg = copy.deepcopy(self.cfg)
        cfg["features"]["compute"] = ["d_area_frac"]
        self.assertEqual(set(C.computed_features(cfg)), {"area", "d_area_frac"})

    def test_feature_params_covers_only_params_actually_used(self):
        """This is what stops an unused knob from invalidating the features cache."""
        used = C.feature_params(self.cfg)
        self.assertIn("neighbor_radius_px", used)
        self.assertNotIn("dilate_radius_px", used)

    def test_emission_names_is_features_then_dino_pcs(self):
        self.assertEqual(C.emission_names(self.cfg), ["area", "circularity", "velocity"])
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["use_dino_pcs"] = True
        cfg["dino"]["n_pcs"] = 2
        self.assertEqual(
            C.emission_names(cfg),
            ["area", "circularity", "velocity", "dino_pc_0", "dino_pc_1"],
        )

    def test_crop_ids_are_sorted(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["data"]["crop_ids"] = ["E4_x", "B4_x"]
        self.assertEqual(C.crop_ids(cfg), ["B4_x", "E4_x"])

    def test_conditions_are_restricted_to_the_runs_crops(self):
        grouped = C.conditions(self.cfg)
        self.assertEqual(grouped, {"SH": ["B4_t50t100y200y350x750x900"]})


class CacheKeys(unittest.TestCase):
    def setUp(self):
        # Extras are pinned off rather than inherited, so these expectations do
        # not drift when configs/_smoke.yml changes what it asks for.
        self.cfg = C.load_config(SMOKE)
        self.cfg["outputs"]["extras"] = []

    @staticmethod
    def keys(cfg):
        return {s.name: L.step_key(cfg, s.name) for s in L.active_steps(cfg)}

    def changed(self, mutate, base=None):
        base = base or self.cfg
        before = self.keys(base)
        mutated = copy.deepcopy(base)
        mutate(mutated)
        after = self.keys(mutated)
        return [n for n in before if after[n] != before[n]]

    def test_keys_are_stable_across_invocations(self):
        self.assertEqual(self.keys(C.load_config(SMOKE)), self.keys(C.load_config(SMOKE)))

    def test_requesting_extras_does_not_change_any_upstream_key(self):
        without = self.keys(self.cfg)
        with_extras = copy.deepcopy(self.cfg)
        with_extras["outputs"]["extras"] = ["state_timeline", "condition_stats"]
        after = self.keys(with_extras)
        for step in without:
            self.assertEqual(after[step], without[step], step)

    def test_dino_steps_active_only_when_the_model_uses_them(self):
        self.assertEqual([s.name for s in L.active_steps(self.cfg)],
                         ["features", "fit", "outputs"])
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["use_dino_pcs"] = True
        self.assertEqual([s.name for s in L.active_steps(cfg)],
                         ["features", "dino", "pca", "fit", "outputs"])

    def test_extras_step_active_only_when_requested(self):
        self.assertNotIn("extras", [s.name for s in L.active_steps(self.cfg)])
        with_extras = copy.deepcopy(self.cfg)
        with_extras["outputs"]["extras"] = ["state_timeline"]
        self.assertEqual([s.name for s in L.active_steps(with_extras)],
                         ["features", "fit", "outputs", "extras"])

    def test_extras_are_last_in_the_key_chain(self):
        """Changing what extras run must not invalidate anything upstream."""
        with_extras = copy.deepcopy(self.cfg)
        with_extras["outputs"]["extras"] = ["state_timeline"]
        self.assertEqual(
            self.changed(lambda c: c["outputs"].update(extras=["condition_stats"]),
                         base=with_extras),
            ["extras"],
        )

    def test_the_shipped_smoke_config_resolves_its_extras(self):
        """The regression config must name extras that exist and are satisfiable."""
        shipped = C.load_config(SMOKE)
        self.assertEqual(shipped["outputs"]["extras"], ["state_timeline", "condition_stats"])

    def test_invalidation_is_scoped(self):
        cases = [
            ("reordering crops changes nothing", lambda c: c["data"]["crop_ids"].reverse(), []),
            ("an unused feature param changes nothing",
             lambda c: c["features"]["params"].update(dilate_radius_px=9), []),
            ("dino settings do not matter when dino is off",
             lambda c: c["dino"].update(patch_px=80), []),
            ("video settings touch only outputs",
             lambda c: c["outputs"]["video"].update(fps=5), ["outputs"]),
            ("num_states touches fit downward",
             lambda c: c["model"].update(num_states=4), ["fit", "outputs"]),
            ("seeds touch fit downward",
             lambda c: c["model"].update(em_seeds=[0, 1]), ["fit", "outputs"]),
            ("a used feature param invalidates features downward",
             lambda c: c["features"]["params"].update(neighbor_radius_px=25),
             ["features", "fit", "outputs"]),
        ]
        for label, mutate, expected in cases:
            with self.subTest(label):
                self.assertEqual(self.changed(mutate), expected)

    def test_dino_key_chain_propagates_downstream(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["use_dino_pcs"] = True
        self.assertEqual(
            self.changed(lambda c: c["dino"].update(patch_px=80), base=cfg),
            ["dino", "pca", "fit", "outputs"],
        )

    def test_batch_size_is_throughput_only(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["use_dino_pcs"] = True
        self.assertEqual(self.changed(lambda c: c["dino"].update(batch_size=8), base=cfg), [])


if __name__ == "__main__":
    unittest.main()
