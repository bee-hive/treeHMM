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
from arhmm import config as C
from arhmm import layout as L

SMOKE = "configs/_smoke.yml"
RUNS = "configs/runs"


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
            ("min_frames below warmup", lambda c: c["cells"].update(min_frames=0), "min_frames"),
            ("boolean min_frames", lambda c: c["cells"].update(min_frames=True), "min_frames"),
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
            ("zero-width age bins",
             lambda c: c["outputs"]["state_age_histogram"].update(bin_frames=0), "bin_frames"),
            ("a max_age that is neither auto nor a count",
             lambda c: c["outputs"]["state_age_histogram"].update(max_age="lots"), "max_age"),
            ("unknown age histogram style",
             lambda c: c["outputs"]["state_age_histogram"].update(kind="violin"), "kind"),
            ("unknown age anchor",
             lambda c: c["outputs"]["state_age_histogram"].update(anchor="division"), "anchor"),
            # `_smoke.yml` resolves to cells.source: phase, so this fires.
            ("extension on a phase run",
             lambda c: c["cells"]["extend_nuclei"].update(frames=5), "cells.source"),
            ("negative extension",
             lambda c: c["cells"]["extend_nuclei"].update(frames=-1), "frames"),
            ("a zero exclusion box",
             lambda c: c["cells"]["extend_nuclei"].update(exclusion_px=0), "exclusion_px"),
            ("a named evidence box",
             lambda c: c["cells"]["extend_nuclei"].update(evidence_px="wide"), "evidence_px"),
            ("an evidence box wider than the exclusion box",
             lambda c: (c["cells"].update(source="nuclei"),
                        c["cells"]["extend_nuclei"].update(
                            frames=5, exclusion_px=30, evidence_px=50)), "wider than"),
        ]
        for label, mutate, message in cases:
            with self.subTest(label):
                with self.assertRaisesRegex(C.ConfigError, message):
                    C.validate(self.broken(mutate))

    def test_min_frames_may_equal_warmup(self):
        """`apply_warmup` keeps a one-frame cell's only frame, so this is legal.

        The `>` rule that used to reject it made `min_frames: 1` unreachable --
        `warmup_frames` is itself forced to >= 1 -- which put single-frame tracks
        permanently out of reach of every run.
        """
        cfg = self.broken(lambda c: c["cells"].update(min_frames=1, warmup_frames=1))
        C.validate(cfg)

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

    def test_nuclei_source_accepts_whole_cell_shape_features(self):
        """`solidity` and friends are well-defined on a nucleus mask.

        They simply describe a nucleus, which is what `cell_source` in the
        features cache's `meta.json` records.  Rejecting them would make every
        nuclei run under the default `features.compute: all` a config error.
        """
        for compute in (["area", "solidity"], "all"):
            with self.subTest(compute=compute):
                def mutate(c, compute=compute):
                    c["cells"]["source"] = "nuclei"
                    c["features"]["compute"] = compute
                    c["model"]["features"] = ["area"]

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


class NucleusExtension(unittest.TestCase):
    """The derived record that decides both the behaviour and the cache key."""

    def setUp(self):
        self.cfg = C.load_config(SMOKE)

    def on(self, **extend):
        cfg = copy.deepcopy(self.cfg)
        cfg["cells"]["source"] = "nuclei"
        cfg["cells"]["extend_nuclei"].update(extend)
        return cfg

    def test_it_is_none_whenever_the_run_extends_nothing(self):
        cases = {
            "phase, off": self.cfg,
            "phase, asking for it": self.broken_phase(),
            "nuclei, off": self.on(frames=0),
        }
        for label, cfg in cases.items():
            with self.subTest(label):
                self.assertIsNone(C.nucleus_extension(cfg))

    def broken_phase(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["cells"]["extend_nuclei"]["frames"] = 5
        return cfg

    def test_it_carries_the_resolved_boxes_and_the_sam3_root(self):
        record = C.nucleus_extension(self.on(frames=5))
        self.assertEqual(record["frames"], 5)
        self.assertEqual((record["exclusion_px"], record["evidence_px"]),
                         C.EXTEND_NUCLEI_DEFAULT_PX)
        self.assertEqual(record["sam3_tracks_dir"],
                         C.get_path(self.cfg, "paths.sam3_tracks_dir"))

    def test_boxes_are_read_verbatim(self):
        cfg = self.on(frames=5, exclusion_px=64, evidence_px=16)
        self.assertEqual(C.nucleus_extension_boxes(cfg), (64, 16))

    def test_boxes_ignore_dino_entirely(self):
        """The old `auto` took them from `dino.patch_px`; nothing does now.

        Retuning the patch sizes must not silently move which tracks are held:
        the extension reads the nucleus stack, `nuclei_div.pkl` and the SAM3
        masks, and DINO supplies none of the three.
        """
        for uses_dino, patch_px in ((False, 80), (True, 50), (True, [70, 30, 50])):
            with self.subTest(patch_px=patch_px, dino=uses_dino):
                cfg = self.on(frames=5)
                cfg["model"]["use_dino_pcs"] = uses_dino
                cfg["dino"]["patch_px"] = patch_px
                self.assertEqual(C.nucleus_extension_boxes(cfg),
                                 C.EXTEND_NUCLEI_DEFAULT_PX)

    def test_the_extension_needs_nuclei_but_not_dino(self):
        """`cells.source: nuclei` is the whole requirement."""
        cfg = self.on(frames=5)
        cfg["model"]["use_dino_pcs"] = False
        C.validate(cfg)
        self.assertIsNotNone(C.nucleus_extension(cfg))

    def test_auto_is_refused_with_a_migration_message(self):
        for key in ("exclusion_px", "evidence_px"):
            with self.subTest(key):
                cfg = self.on(frames=5, **{key: "auto"})
                with self.assertRaisesRegex(C.ConfigError, "was removed"):
                    C.nucleus_extension_boxes(cfg)


class GoldenKeys(unittest.TestCase):
    """Cache keys that name directories already on disk under `analysis/cache`.

    These are not arbitrary regression values: a change here orphans real
    cached artifacts and silently recomputes hours of work.  `nuclei_k5.yml` and
    `nuclei_dino_k5.yml` are reconstructed rather than loaded because they are
    untracked in some checkouts; `run_name` is in no step's `depends`, so the
    reconstruction hashes identically to the file.
    """

    @staticmethod
    def nuclei(path):
        cfg = C.load_config(path)
        cfg["cells"]["source"] = "nuclei"
        return cfg

    def test_the_live_cache_keys_have_not_moved(self):
        cases = [
            ("_smoke", C.load_config(SMOKE), {"features": "b01a3d265189"}),
            ("base_k5", C.load_config(f"{RUNS}/base_k5.yml"), {"features": "f640ca30feac"}),
            ("dino_k5", C.load_config(f"{RUNS}/dino_k5.yml"),
             {"features": "f640ca30feac", "dino": "e6f5d31e5a1a"}),
            ("nuclei_k5", self.nuclei(f"{RUNS}/base_k5.yml"), {"features": "db4409ef2642"}),
            ("nuclei_dino_k5", self.nuclei(f"{RUNS}/dino_k5.yml"),
             {"features": "db4409ef2642", "dino": "114473663754"}),
        ]
        for label, cfg, expected in cases:
            with self.subTest(label):
                active = {s.name for s in L.active_steps(cfg)}
                got = {n: L.step_key(cfg, n) for n in expected if n in active}
                self.assertEqual(got, expected)


class CacheKeys(unittest.TestCase):
    def setUp(self):
        # Extras are pinned off rather than inherited, so these expectations do
        # not drift when configs/_smoke.yml changes what it asks for.  The
        # auto-included DINO extras are pinned off for the same reason: without
        # this, every test here that switches DINO on would also be asserting
        # about the extras step.  `AutoDinoExtras` below covers that on its own.
        self.cfg = C.load_config(SMOKE)
        self.cfg["outputs"]["extras"] = []
        self.cfg["outputs"]["auto_dino_extras"] = False
        # Likewise for the extension's automatic extra: without this, every test
        # here that switches `extend_nuclei` on would silently also be asserting
        # that the extras step went from inactive to active.  `ResolvedExtras`
        # covers that behaviour directly.
        self.cfg["outputs"]["auto_extension_extras"] = False

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
            ("cell-age binning touches only outputs",
             lambda c: c["outputs"]["state_age_histogram"].update(bin_frames=5), ["outputs"]),
            ("num_states touches fit downward",
             lambda c: c["model"].update(num_states=4), ["fit", "outputs"]),
            ("seeds touch fit downward",
             lambda c: c["model"].update(em_seeds=[0, 1]), ["fit", "outputs"]),
            ("a used feature param invalidates features downward",
             lambda c: c["features"]["params"].update(neighbor_radius_px=25),
             ["features", "fit", "outputs"]),
            ("the image root invalidates features downward",
             lambda c: c["paths"].update(image_crops_dir="/elsewhere"),
             ["features", "fit", "outputs"]),
            ("the cell source invalidates features downward",
             lambda c: c["cells"].update(source="nuclei"),
             ["features", "fit", "outputs"]),
            # Nothing about a feature this run does not use may move a key --
            # otherwise adding the block to default.yml orphans every cache.
            ("the extension block itself changes nothing while it is off",
             lambda c: c["cells"]["extend_nuclei"].update(frames=0), []),
            ("the sam3 root does not matter while the extension is off",
             lambda c: c["paths"].update(sam3_tracks_dir="/elsewhere"), []),
            ("the extension boxes do not matter while it is off",
             lambda c: c["cells"]["extend_nuclei"].update(exclusion_px=11, evidence_px=7), []),
        ]
        for label, mutate, expected in cases:
            with self.subTest(label):
                self.assertEqual(self.changed(mutate), expected)

    def test_turning_the_extension_on_invalidates_features_downward(self):
        nuclei = copy.deepcopy(self.cfg)
        nuclei["cells"]["source"] = "nuclei"
        self.assertEqual(
            self.changed(lambda c: c["cells"]["extend_nuclei"].update(frames=5), base=nuclei),
            ["features", "fit", "outputs"],
        )

    def test_the_sam3_root_matters_once_the_extension_is_on(self):
        """It travels inside the derived record rather than as a `paths.*` key,
        so this is the only thing proving it is hashed at all."""
        base = copy.deepcopy(self.cfg)
        base["cells"]["source"] = "nuclei"
        base["cells"]["extend_nuclei"]["frames"] = 5
        self.assertEqual(
            self.changed(lambda c: c["paths"].update(sam3_tracks_dir="/elsewhere"), base=base),
            ["features", "fit", "outputs"],
        )

    def test_patch_sizes_never_reach_features(self):
        """The `auto` boxes used to make them, which was the wrong coupling.

        `features` reads no DINO input, so retuning `dino.patch_px` must
        recompute embeddings and nothing else -- with the extension on or off.
        """
        base = copy.deepcopy(self.cfg)
        base["cells"]["source"] = "nuclei"
        base["model"]["use_dino_pcs"] = True
        resize = lambda c: c["dino"].update(patch_px=[30, 80])  # noqa: E731

        self.assertNotIn("features", self.changed(resize, base=base))
        base["cells"]["extend_nuclei"]["frames"] = 5
        self.assertNotIn("features", self.changed(resize, base=base))

    def test_the_boxes_invalidate_features_once_the_extension_is_on(self):
        """They are now the only thing that sizes the criteria, so they must."""
        base = copy.deepcopy(self.cfg)
        base["cells"]["source"] = "nuclei"
        base["cells"]["extend_nuclei"]["frames"] = 5
        self.assertIn(
            "features",
            self.changed(lambda c: c["cells"]["extend_nuclei"].update(exclusion_px=64),
                         base=base),
        )

    def test_the_image_root_invalidates_both_sources(self):
        """It carries the nucleus tracks as well as the image, for either source.

        There is no separate nuclei root to repoint: `nuclei_tracks.tiff` sits in
        the crop directory beside `crop.tiff`, so moving that root moves the
        masks of a nuclei run and the pixels of both.
        """
        repoint = lambda c: c["paths"].update(image_crops_dir="/elsewhere")  # noqa: E731
        for source in ("phase", "nuclei"):
            with self.subTest(source):
                cfg = copy.deepcopy(self.cfg)
                cfg["cells"]["source"] = source
                cfg["model"]["use_dino_pcs"] = True
                self.assertEqual(self.changed(repoint, base=cfg),
                                 ["features", "dino", "pca", "fit", "outputs"])

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

    def test_a_single_patch_size_hashes_the_same_scalar_or_list(self):
        """The guarantee that multi-scale support orphans no existing cache.

        `dino.patch_px` is normalized through `_DERIVED` under its own name, so
        a one-size run must hash exactly as it did when the key could only be a
        scalar.  If this breaks, every cached DINO embedding is recomputed.
        """
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["use_dino_pcs"] = True
        self.assertEqual(self.changed(lambda c: c["dino"].update(patch_px=[50]), base=cfg), [])

    def test_patch_size_order_and_repeats_do_not_move_the_key(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["use_dino_pcs"] = True
        cfg["dino"]["patch_px"] = [30, 50, 70]
        for equivalent in ([70, 50, 30], [50, 30, 70, 50]):
            with self.subTest(equivalent=equivalent):
                self.assertEqual(
                    self.changed(lambda c: c["dino"].update(patch_px=equivalent), base=cfg), []
                )

    def test_adding_a_patch_size_invalidates_dino_downward(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["use_dino_pcs"] = True
        self.assertEqual(
            self.changed(lambda c: c["dino"].update(patch_px=[30, 50]), base=cfg),
            ["dino", "pca", "fit", "outputs"],
        )

    def test_unusable_patch_sizes_are_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["use_dino_pcs"] = True
        for bad in ([], 0, -5, [30, 0], ["50"], [30.0], True):
            with self.subTest(bad=bad):
                broken = copy.deepcopy(cfg)
                broken["dino"]["patch_px"] = bad
                with self.assertRaises(C.ConfigError):
                    C.validate(broken)

    def test_patch_sizes_are_not_checked_when_dino_is_off(self):
        """Consistent with the rest of the `dino` block, which is only read when used."""
        broken = copy.deepcopy(self.cfg)
        broken["dino"]["patch_px"] = []
        C.validate(broken)


class AutoDinoExtras(unittest.TestCase):
    """A DINO run produces the DINO extras whether or not it names them."""

    def setUp(self):
        self.cfg = C.load_config(SMOKE)
        self.cfg["outputs"]["extras"] = []

    def dino(self, **outputs):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["use_dino_pcs"] = True
        cfg["outputs"].update(outputs)
        return cfg

    def test_a_dino_run_gets_them_with_no_extras_named(self):
        self.assertEqual(C.resolved_extras(self.dino()), list(C.DINO_DEFAULT_EXTRAS))

    def test_a_non_dino_run_does_not(self):
        self.assertEqual(C.resolved_extras(self.cfg), [])
        named = copy.deepcopy(self.cfg)
        named["outputs"]["extras"] = ["state_timeline"]
        self.assertEqual(C.resolved_extras(named), ["state_timeline"])

    def test_named_extras_keep_their_order_and_come_first(self):
        cfg = self.dino(extras=["condition_stats", "state_timeline"])
        self.assertEqual(C.resolved_extras(cfg),
                         ["condition_stats", "state_timeline", *C.DINO_DEFAULT_EXTRAS])

    def test_naming_one_explicitly_does_not_duplicate_it(self):
        cfg = self.dino(extras=[C.DINO_DEFAULT_EXTRAS[0]])
        self.assertEqual(C.resolved_extras(cfg), [C.DINO_DEFAULT_EXTRAS[0]])

    def test_the_opt_out_leaves_only_what_was_named(self):
        cfg = self.dino(extras=["state_timeline"], auto_dino_extras=False)
        self.assertEqual(C.resolved_extras(cfg), ["state_timeline"])

    # ---- the extension's automatic extra ---------------------------------- #

    def extending(self, **outputs):
        cfg = copy.deepcopy(self.cfg)
        cfg["cells"]["source"] = "nuclei"
        cfg["cells"]["extend_nuclei"]["frames"] = 3
        cfg["outputs"].update(outputs)
        return cfg

    def test_an_extending_run_gets_the_sam3_overlay(self):
        self.assertEqual(C.resolved_extras(self.extending()),
                         list(C.EXTENSION_DEFAULT_EXTRAS))

    def test_a_run_that_extends_nothing_does_not(self):
        """`frames: 0` opens no SAM3 file, so there is nothing to draw."""
        nuclei = copy.deepcopy(self.cfg)
        nuclei["cells"]["source"] = "nuclei"
        self.assertEqual(C.resolved_extras(nuclei), [])
        # ... and neither does a phase run that asks for it, since the record is
        # None there too.
        self.assertEqual(C.resolved_extras(self.cfg), [])

    def test_the_extension_opt_out_leaves_only_what_was_named(self):
        cfg = self.extending(extras=["state_timeline"], auto_extension_extras=False)
        self.assertEqual(C.resolved_extras(cfg), ["state_timeline"])

    def test_naming_the_sam3_overlay_explicitly_does_not_duplicate_it(self):
        cfg = self.extending(extras=[C.EXTENSION_DEFAULT_EXTRAS[0]])
        self.assertEqual(C.resolved_extras(cfg), [C.EXTENSION_DEFAULT_EXTRAS[0]])

    def test_both_automatic_groups_appear_in_a_fixed_order(self):
        """The `extras` step key hashes this list, so its order must be stable."""
        cfg = self.extending(extras=["state_timeline"])
        cfg["model"]["use_dino_pcs"] = True
        self.assertEqual(
            C.resolved_extras(cfg),
            ["state_timeline", *C.DINO_DEFAULT_EXTRAS, *C.EXTENSION_DEFAULT_EXTRAS],
        )

    def test_they_activate_the_extras_step_on_their_own(self):
        """`outputs.extras: []` no longer means "no extras step" for a DINO run."""
        self.assertNotIn("extras", [s.name for s in L.active_steps(self.cfg)])
        self.assertIn("extras", [s.name for s in L.active_steps(self.dino())])
        self.assertNotIn(
            "extras",
            [s.name for s in L.active_steps(self.dino(auto_dino_extras=False))],
        )

    def test_the_step_key_covers_what_actually_runs(self):
        """Two runs differing only in the opt-out must not share an extras key."""
        on = self.dino(extras=["state_timeline"])
        off = self.dino(extras=["state_timeline"], auto_dino_extras=False)
        self.assertNotEqual(L.step_key(on, "extras"), L.step_key(off, "extras"))

    def test_auto_included_extras_are_validated_like_named_ones(self):
        """Every default must be a real extra whose requirements a DINO run meets."""
        from arhmm.extras import EXTRA_NAMES, requirements_for

        for name in C.DINO_DEFAULT_EXTRAS:
            with self.subTest(name=name):
                self.assertIn(name, EXTRA_NAMES)
                self.assertIn("dino", requirements_for(name))
        C.validate(self.dino())          # must not raise
        C.validate(self.cfg)             # nor the non-DINO case


if __name__ == "__main__":
    unittest.main()
