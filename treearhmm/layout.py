"""Pipeline step definitions, cache keys, and the run directory layout.

The pipeline is a fixed linear chain.  Each step declares three things: the
conda environment it needs, the config keys it depends on, and whether its
output is *shared* (content-addressed in the cache, reusable by any run with the
same inputs) or *run-local*.

    features  imaging   per-cell, per-frame track features        [cached]
    dino      dino      DINOv2 embeddings of centroid patches     [cached]
    pca       dino      joint PCA -> top-k principal components   [cached]
    fit       model     fit the AR-HMM                            [run-local]
    outputs   imaging   the four base outputs                     [run-local]
    extras    imaging   any additional outputs the run asked for  [run-local]

Cache keys chain: a step's key hashes its own config subset together with the
key of the step before it, so changing `dino.patch_px` invalidates `dino`, `pca`
and `fit` but leaves `features` alone.

Source code is deliberately **not** hashed.  Hashing the tree would make every
edit invalidate everything, which trains people to reach for `--force` reflexively
and defeats the point.  Instead each step carries a hand-bumped `version`, and
the manifest records the git commit so a stale cache is at least detectable
after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from treearhmm import config as cfgmod
from treearhmm.core import io


@dataclass(frozen=True)
class Step:
    """One stage of the pipeline.

    Attributes:
        name (str): step name; also the module under `treearhmm.steps`.
        env (str): key into the config's `envs` mapping.
        upstream (tuple[str, ...]): steps whose keys feed this one's key.
        depends (tuple[str, ...]): dotted config paths this step's output
            depends on.  Anything not listed here can change without
            invalidating the step, so the list is a claim worth getting right.
        shared (bool): output is content-addressed into the cache and reusable
            across runs, rather than written into the run directory.
        optional (bool): the run continues if this step fails.
        version (int): bump by hand when the step's numerical output changes for
            reasons the config does not capture.
        summary (str): one line, shown by `treearhmm status`.
    """

    name: str
    env: str
    upstream: tuple[str, ...]
    depends: tuple[str, ...]
    shared: bool
    summary: str
    optional: bool = False
    version: int = 1

    @property
    def module(self) -> str:
        return f"treearhmm.steps.{self.name}"


#: Config keys derived rather than read verbatim.  `step_key` resolves these
#: through `config` helpers so that, for example, `features.compute: all` and the
#: explicit equivalent list hash identically.
_DERIVED = {
    "_derived.crop_ids": cfgmod.crop_ids,
    "_derived.computed_features": cfgmod.computed_features,
    "_derived.feature_params": cfgmod.feature_params,
    "_derived.emission_names": cfgmod.emission_names,
}


STEPS: tuple[Step, ...] = (
    Step(
        name="features",
        env="imaging",
        upstream=(),
        depends=(
            "paths.ground_truth_tracks_dir",
            "paths.image_crops_dir",
            "cells.source",
            "_derived.crop_ids",
            "_derived.computed_features",
            "_derived.feature_params",
        ),
        shared=True,
        summary="per-cell, per-frame features computed from the tracks",
    ),
    Step(
        name="dino",
        env="dino",
        upstream=("features",),
        depends=(
            "paths.image_crops_dir",
            "cells.source",
            "_derived.crop_ids",
            "dino.model_id",
            "dino.patch_px",
            "dino.mask_alpha",
            "dino.subject_colour",
            "dino.other_cancer_colour",
            "dino.tcell_colour",
            "dino.include_rfp",
            "dino.rfp_alpha",
            "dino.norm_percentiles",
        ),
        shared=True,
        summary="DINOv2 embeddings of patches centred on each cell",
    ),
    Step(
        name="pca",
        env="dino",
        upstream=("dino",),
        depends=("dino.n_pcs", "dino.whiten"),
        shared=True,
        summary="joint PCA reducing the embeddings to the top-k components",
    ),
    Step(
        name="fit",
        env="model",
        upstream=("features", "pca"),
        depends=(
            "cells.min_frames",
            "cells.warmup_frames",
            "model.features",
            "model.use_dino_pcs",
            "model.num_states",
            "model.num_lags",
            "model.standardize",
            "model.init_method",
            "model.init_stickiness",
            "model.num_em_iters",
            "model.em_seeds",
            "_derived.crop_ids",
            "_derived.emission_names",
        ),
        shared=False,
        summary="fit the AR-HMM jointly across all crops",
    ),
    Step(
        name="outputs",
        env="imaging",
        upstream=("fit",),
        depends=("outputs.video", "outputs.feature_distributions"),
        shared=False,
        summary="overlay videos, feature distributions, transitions, assignments",
    ),
    Step(
        name="extras",
        env="imaging",
        upstream=("outputs",),
        depends=("outputs.extras",),
        shared=False,
        optional=True,
        summary="additional run-specific outputs",
    ),
)

STEPS_BY_NAME: dict[str, Step] = {s.name: s for s in STEPS}


def active_steps(cfg: dict) -> list[Step]:
    """The steps this config actually runs, in order.

    `dino` and `pca` are dropped unless the run feeds DINO components to the
    model, and `extras` is dropped when nothing was requested.  Deriving this
    from `model.use_dino_pcs` rather than a separate `enabled` flag removes the
    class of bug where the two disagree.

    Args:
        cfg (dict): resolved configuration.

    Returns:
        list[Step]: steps to execute, in dependency order.
    """
    use_dino = cfgmod.uses_dino(cfg)
    wants_extras = bool(cfgmod.get_path(cfg, "outputs.extras", []) or [])
    steps = []
    for step in STEPS:
        if step.name in ("dino", "pca") and not use_dino:
            continue
        if step.name == "extras" and not wants_extras:
            continue
        steps.append(step)
    return steps


def _depends_payload(cfg: dict, step: Step) -> dict:
    """The config values a step depends on, with derived keys resolved."""
    plain = [k for k in step.depends if k not in _DERIVED]
    payload = cfgmod.subset(cfg, plain)
    for key in step.depends:
        if key in _DERIVED:
            payload[key] = _DERIVED[key](cfg)
    return payload


def step_key(cfg: dict, name: str) -> str:
    """Content hash identifying a step's inputs.

    Args:
        cfg (dict): resolved configuration.
        name (str): step name.

    Returns:
        str: short hex digest covering this step's config subset, its version,
            and the key of every step upstream of it.
    """
    step = STEPS_BY_NAME[name]
    active = {s.name for s in active_steps(cfg)}
    payload = {
        "step": step.name,
        "version": step.version,
        "depends": _depends_payload(cfg, step),
        # An upstream step that this config does not run contributes nothing,
        # so a no-DINO run's `fit` key does not depend on DINO settings.
        "upstream": {u: step_key(cfg, u) for u in step.upstream if u in active},
    }
    return cfgmod.hash_obj(payload, length=12)


def env_for(cfg: dict, step: Step) -> str:
    """Conda environment name for a step."""
    return cfgmod.get_path(cfg, f"envs.{step.env}")


class Layout:
    """Where everything for one run lives on disk.

    Shared steps resolve to `{cache_root}/{step}/{key}/`; run-local steps to a
    subdirectory of the run.  The run directory carries a symlink to each shared
    step's cache directory so that looking at the run shows the artifacts its
    current config resolves to.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.run_name = cfgmod.get_path(cfg, "run_name")
        self.output_root = Path(cfgmod.get_path(cfg, "paths.output_root"))
        self.cache_root = Path(cfgmod.get_path(cfg, "paths.cache_root"))
        self.run_dir = self.output_root / self.run_name

    # ---- fixed files ---------------------------------------------------- #

    @property
    def resolved_config_path(self) -> Path:
        return self.run_dir / "config.resolved.yml"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.yml"

    @property
    def stamp_dir(self) -> Path:
        return self.run_dir / "_stamps"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def fit_dir(self) -> Path:
        return self.run_dir / "fit"

    @property
    def outputs_dir(self) -> Path:
        return self.run_dir / "outputs"

    @property
    def extras_dir(self) -> Path:
        return self.outputs_dir / "extras"

    @property
    def overlays_dir(self) -> Path:
        return self.outputs_dir / "overlays"

    # ---- step directories ----------------------------------------------- #

    def step_dir(self, name: str) -> Path:
        """Directory a step writes into.

        Args:
            name (str): step name.

        Returns:
            Path: the cache directory for shared steps, a run subdirectory
                otherwise.
        """
        step = STEPS_BY_NAME[name]
        if step.shared:
            return self.cache_root / name / step_key(self.cfg, name)
        if name == "extras":
            return self.extras_dir
        return self.run_dir / name

    def crop_dir(self, name: str, crop_id: str) -> Path:
        """Per-crop subdirectory of a step's output directory."""
        return self.step_dir(name) / crop_id

    def stamp_path(self, name: str) -> Path:
        """Where a step records that it completed, and with which key."""
        return self.stamp_dir / f"{name}.json"

    def log_path(self, name: str) -> Path:
        return self.log_dir / f"{name}.log"

    # ---- lifecycle ------------------------------------------------------ #

    def ensure_run_dir(self) -> None:
        """Create the run directory skeleton."""
        for path in (self.run_dir, self.stamp_dir, self.log_dir):
            io.ensure_dir(path)

    def link_shared(self) -> None:
        """Point `{run_dir}/{step}` at each shared step's cache directory."""
        for step in active_steps(self.cfg):
            if step.shared:
                io.relink(self.run_dir / step.name, self.step_dir(step.name))

    def write_stamp(self, name: str, extra: dict | None = None) -> None:
        """Record that a step finished, with the key it finished for."""
        import datetime as _dt

        payload = {
            "step": name,
            "key": step_key(self.cfg, name),
            "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        payload.update(extra or {})
        io.write_json(self.stamp_path(name), payload)

    def read_stamp(self, name: str) -> dict | None:
        """Read a step's completion stamp, or None if it never completed."""
        path = self.stamp_path(name)
        if not path.is_file():
            return None
        try:
            return io.read_json(path)
        except (ValueError, OSError):
            return None

    def is_current(self, name: str) -> bool:
        """Whether a step's recorded output matches the current config.

        A stamp alone is not trusted: the step's output directory must still be
        present, so a hand-deleted cache entry is noticed rather than skipped.

        Args:
            name (str): step name.

        Returns:
            bool: True when the step completed for this exact key and its output
                is still on disk.
        """
        stamp = self.read_stamp(name)
        if stamp is None or stamp.get("key") != step_key(self.cfg, name):
            return False
        directory = self.step_dir(name)
        if not directory.is_dir():
            return False
        for produced in stamp.get("produced", []):
            if not (directory / produced).exists():
                return False
        return True

    def status(self) -> list[dict]:
        """One row per active step describing whether it would run, and why."""
        rows = []
        stale_upstream = False
        for step in active_steps(self.cfg):
            current = self.is_current(step.name) and not stale_upstream
            if current:
                reason = "cached"
            elif stale_upstream:
                reason = "upstream step will rerun"
            elif self.read_stamp(step.name) is None:
                reason = "never run"
            else:
                reason = "config changed"
            rows.append(
                {
                    "step": step.name,
                    "env": env_for(self.cfg, step),
                    "key": step_key(self.cfg, step.name),
                    "dir": str(self.step_dir(step.name)),
                    "current": current,
                    "reason": reason,
                }
            )
            stale_upstream = stale_upstream or not current
        return rows


def steps_from(cfg: dict, first: str | None) -> list[Step]:
    """Active steps starting at `first`, or all of them when `first` is None.

    Raises:
        KeyError: if `first` is not an active step for this config.
    """
    active = active_steps(cfg)
    if first is None:
        return active
    names = [s.name for s in active]
    if first not in names:
        raise KeyError(f"{first!r} is not an active step for this config; active: {names}")
    return active[names.index(first) :]


def declared_outputs(name: str, cfg: dict) -> Iterable[str]:
    """Paths, relative to the step directory, that must exist for it to count as done."""
    if name in ("features", "dino", "pca"):
        return [crop for crop in cfgmod.crop_ids(cfg)]
    if name == "fit":
        return ["fit_summary.yml", "state_assignments.npy"]
    if name == "outputs":
        return ["state_assignments.csv", "transition_matrix.csv"]
    return []
