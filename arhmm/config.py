"""Loading, merging, validating and hashing run configurations.

A run is defined by exactly one YAML file, layered over two shared ones:

    configs/site.yml      paths, conda environments, the crop list -- the things
                          that change with the machine or the dataset
    configs/default.yml   the experiment-facing defaults, extending site.yml
    <run>.yml             only what this particular run does differently

The run's file is deep-merged over that chain, validated, and written into the
run directory as `config.resolved.yml`.  Everything downstream reads the
resolved config, never the user's file, so a run stays reproducible even if the
defaults change afterwards.

Determinism comes from hashing.  Each pipeline step declares which parts of the
config it depends on; `layout.step_key` hashes that subset together with the
keys of the steps upstream of it.  A step reruns when and only when something it
actually depends on changed, and two runs that agree on a step's inputs share
its cached output.

Validation is split in two on purpose:

    validate(cfg)         pure; touches no filesystem, so a config can be checked
                          anywhere and the rules are unit-testable
    validate_inputs(cfg)  the filesystem and conda checks, run by `arhmm run`
                          and `arhmm doctor`

**Import weight matters here.**  This module is imported by every step, and the
steps run in three different conda environments.  It may import only `yaml`,
`numpy` and the standard library -- never skimage, torch or jax.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from arhmm.core import trackfeatures

CONFIG_DIR_NAME = "configs"
DEFAULT_CONFIG_NAME = "default.yml"

# Sentinel distinguishing "no default supplied" from "the default is None".
_MISSING = object()

# A single directory name.  A leading `_` is allowed (it marks a non-scientific
# run such as `_smoke` and sorts it to the top); a leading `.` is not, so no
# run_name can climb out of output_root.
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")

CELL_SOURCES = ("phase", "nuclei")
INIT_METHODS = ("kmeans", "prior", "random")
#: What is drawn over the phase patch handed to DINOv2.  At most one of them:
#: `rfp` can use the red channel only because `masks` is not.
PATCH_OVERLAYS = ("rfp", "masks", "none")
#: Bar styles for the cell-age histogram, and what its age zero means.
AGE_HISTOGRAM_KINDS = ("stacked", "grouped", "step")
AGE_ANCHORS = ("existence", "inferred")


class ConfigError(Exception):
    """A configuration is malformed, contradictory, or names something unknown."""


# --------------------------------------------------------------------------- #
# loading and merging
# --------------------------------------------------------------------------- #


def _read_yaml(path: Path) -> dict:
    """Read a YAML file, returning `{}` for an empty one.

    Args:
        path (Path): file to read.

    Returns:
        dict: parsed mapping.

    Raises:
        ConfigError: if the file is missing or does not hold a mapping.
    """
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    return loaded


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict.

    Mappings merge key-by-key; **every other type, including lists, is replaced
    wholesale**.  Lists are replaced rather than concatenated so that a run
    config narrowing `data.crop_ids` or `model.features` gets exactly what it
    wrote, not the defaults plus its own.  A value of `None` in `override`
    deletes the key, which is the only way to unset something inherited.

    Args:
        base (dict): lower-priority mapping.
        override (dict): higher-priority mapping.

    Returns:
        dict: a new merged mapping; neither argument is modified.
    """
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_chain(path: Path, seen: list[Path] | None = None) -> dict:
    """Load `path` merged over everything it extends, recursively.

    `extends` is resolved relative to the extending file's own directory, so a
    config can be moved between directories as long as it moves with what it
    extends.

    Args:
        path (Path): config file to load.
        seen (list[Path] | None): files already on the chain, for cycle detection.

    Returns:
        dict: the merged mapping, with `extends` stripped.

    Raises:
        ConfigError: on a missing file or an `extends` cycle.
    """
    path = path.resolve()
    seen = list(seen or [])
    if path in seen:
        loop = " -> ".join(p.name for p in [*seen, path])
        raise ConfigError(f"`extends` cycle: {loop}")
    seen.append(path)

    raw = _read_yaml(path)
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw

    if not isinstance(parent_ref, str):
        raise ConfigError(f"{path}: `extends` must be a string, got {type(parent_ref).__name__}")
    parent_path = (path.parent / parent_ref).resolve()
    return deep_merge(_resolve_chain(parent_path, seen), raw)


def load_config(path: str | Path, *, validate_config: bool = True) -> dict:
    """Load a run config, merged over everything it extends, and validate it.

    A config that declares no `extends` is layered over `configs/default.yml`
    beside it, so a run file only ever has to carry its own deltas.

    Args:
        path (str | Path): path to the run's YAML file.
        validate_config (bool): run `validate` before returning.

    Returns:
        dict: the fully resolved configuration, with `_source_config` recording
            where it came from.

    Raises:
        ConfigError: if the config is invalid.
    """
    path = Path(path).resolve()
    raw = _read_yaml(path)

    if "extends" in raw:
        cfg = _resolve_chain(path)
    else:
        default_path = path.parent / DEFAULT_CONFIG_NAME
        if default_path.resolve() == path:
            cfg = raw
        elif default_path.is_file():
            cfg = deep_merge(_resolve_chain(default_path), raw)
        else:
            cfg = raw

    cfg["_source_config"] = str(path)
    if validate_config:
        validate(cfg)
    return cfg


def load_resolved(run_dir: str | Path) -> dict:
    """Load the `config.resolved.yml` frozen into a run directory.

    This is what every step reads.  It is deliberately *not* re-validated and
    *not* re-merged: the run is whatever it was created as.

    Args:
        run_dir (str | Path): the run directory.

    Returns:
        dict: the resolved configuration exactly as the run was created with.
    """
    return _read_yaml(Path(run_dir) / "config.resolved.yml")


def dump_config(cfg: dict, path: str | Path) -> None:
    """Write a config to `path` with stable key order.

    Args:
        cfg (dict): configuration to write.
        path (str | Path): destination file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))


# --------------------------------------------------------------------------- #
# nested access, subsetting, hashing
# --------------------------------------------------------------------------- #


def get_path(cfg: dict, dotted: str, default: Any = _MISSING) -> Any:
    """Read a nested value by dotted path, e.g. `"model.num_states"`.

    Args:
        cfg (dict): configuration mapping.
        dotted (str): dot-separated key path.
        default (Any): value to return when the path is absent.  If left at the
            sentinel, a missing path raises instead.

    Returns:
        Any: the value at that path.

    Raises:
        ConfigError: when the path is absent and no default was given.
    """
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise ConfigError(f"missing required config key: {dotted}")
            return default
        node = node[part]
    return node


def subset(cfg: dict, dotted_keys: Iterable[str]) -> dict:
    """Extract a flat mapping containing only `dotted_keys`.

    The result is keyed by the dotted paths *themselves*, not renested, so that
    reordering or renesting the source config cannot silently change the hash of
    an unchanged value.  Absent keys are recorded as `None` rather than skipped,
    so that adding a key with a value is distinguishable from never having it.

    Args:
        cfg (dict): configuration mapping.
        dotted_keys (Iterable[str]): dot-separated paths to keep.

    Returns:
        dict: `{dotted_key: value}`.
    """
    return {key: get_path(cfg, key, None) for key in sorted(dotted_keys)}


def canonical_json(obj: Any) -> str:
    """Serialize `obj` so that equal configs always produce equal strings.

    Args:
        obj (Any): any JSON-serializable structure.

    Returns:
        str: canonical JSON with sorted keys and no insignificant whitespace.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_obj(obj: Any, length: int = 12) -> str:
    """Short, stable hex digest of a JSON-serializable structure.

    Args:
        obj (Any): structure to hash.
        length (int): digest length in hex characters.

    Returns:
        str: the truncated SHA-256 digest.
    """
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest[:length]


# --------------------------------------------------------------------------- #
# derived views the whole pipeline agrees on
# --------------------------------------------------------------------------- #


def crop_ids(cfg: dict) -> list[str]:
    """The run's crops, **sorted**.

    Sorting here is what makes reordering the YAML list a no-op: it fixes both
    the cache key and the column order that crops are concatenated in.

    Args:
        cfg (dict): resolved configuration.

    Returns:
        list[str]: sorted crop ids.
    """
    return sorted(get_path(cfg, "data.crop_ids"))


def conditions(cfg: dict) -> dict[str, list[str]]:
    """Condition -> crops, restricted to the crops this run actually uses.

    `data.conditions` in `site.yml` describes the whole dataset; a run that
    narrows `data.crop_ids` gets the corresponding narrowed grouping here, with
    now-empty conditions dropped.

    Args:
        cfg (dict): resolved configuration.

    Returns:
        dict[str, list[str]]: condition name -> its crops in sorted order.
    """
    active = set(crop_ids(cfg))
    grouped = {}
    for condition, members in (get_path(cfg, "data.conditions", {}) or {}).items():
        present = sorted(c for c in members if c in active)
        if present:
            grouped[condition] = present
    return grouped


def computed_features(cfg: dict) -> list[str]:
    """Track features the `features` step will compute and cache.

    Expands the sentinel `all` to the whole registry, and pulls in the per-frame
    features that a requested temporal feature reads even when the config did
    not name them.

    Args:
        cfg (dict): resolved configuration.

    Returns:
        list[str]: feature names in registry order.
    """
    requested = get_path(cfg, "features.compute", "all")
    if requested == "all":
        requested = trackfeatures.feature_names()
    return trackfeatures.expand_requested(requested)


def feature_params(cfg: dict) -> dict:
    """The `features.params` entries actually consumed by the computed features.

    Restricting this to what is used is what stops a change to
    `dilate_radius_px` from invalidating a cache whose features never read it.

    Args:
        cfg (dict): resolved configuration.

    Returns:
        dict: `{param_name: value}` for the params the computed features declare.
    """
    params = get_path(cfg, "features.params", {}) or {}
    used = trackfeatures.required_params(computed_features(cfg))
    return {name: params.get(name) for name in sorted(used)}


def caliban_dir_if_read(cfg: dict) -> str | None:
    """`paths.caliban_tracks_dir`, but only when the run actually opens it.

    Returning None under `cells.source: phase` is what keeps the Caliban root out
    of a phase run's cache key: `layout` drops a derived key that resolves to
    None, so repointing a directory nothing reads invalidates nothing.

    Args:
        cfg (dict): resolved configuration.

    Returns:
        str | None: the directory under `cells.source: nuclei`, else None.
    """
    if get_path(cfg, "cells.source", None) != "nuclei":
        return None
    return get_path(cfg, "paths.caliban_tracks_dir", None)


def uses_dino(cfg: dict) -> bool:
    """Whether this run feeds DINO principal components to the model."""
    return bool(get_path(cfg, "model.use_dino_pcs", False))


def dino_column_names(cfg: dict) -> list[str]:
    """Names of the DINO principal component columns, in order."""
    if not uses_dino(cfg):
        return []
    return [f"dino_pc_{i}" for i in range(int(get_path(cfg, "dino.n_pcs")))]


def emission_names(cfg: dict) -> list[str]:
    """Names of the emission dimensions, in the order the model sees them.

    Track features in `model.features` order, then the DINO PCs -- the order
    fixed by `tree_input.md` section 3.  This one list is what the fit step
    builds the `(T, C, D)` tensor from, what `fit_summary.yml` records, and what
    every plot and CSV downstream reads.

    Args:
        cfg (dict): resolved configuration.

    Returns:
        list[str]: emission column names.
    """
    return list(get_path(cfg, "model.features", [])) + dino_column_names(cfg)


def video_crops(cfg: dict) -> list[str]:
    """Resolve `outputs.video.crops` to an explicit list.

    Args:
        cfg (dict): resolved configuration.

    Returns:
        list[str]: crop ids to render, in `crop_ids` order.
    """
    requested = get_path(cfg, "outputs.video.crops", "all")
    available = crop_ids(cfg)
    if requested == "all":
        return available
    if requested == "none":
        return []
    return [crop for crop in available if crop in set(requested)]


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def _did_you_mean(name: str, known: Iterable[str]) -> str:
    """A short ` (did you mean X?)` suffix, or empty when nothing is close."""
    import difflib

    close = difflib.get_close_matches(name, list(known), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def validate(cfg: dict) -> None:
    """Check a resolved config for the errors that would otherwise surface late.

    Everything checked here is cheap and local to the config.  Existence of
    input files and conda environments is deliberately *not* checked, so that a
    config can be validated without touching the filesystem the data lives on --
    see `validate_inputs`.

    Args:
        cfg (dict): the resolved configuration.

    Raises:
        ConfigError: on the first problem found, naming the offending key.
    """
    run_name = get_path(cfg, "run_name")
    if not isinstance(run_name, str) or not _RUN_NAME_RE.match(run_name):
        raise ConfigError(
            f"run_name must be a single directory name matching {_RUN_NAME_RE.pattern}, "
            f"got {run_name!r}"
        )

    # ---- data ----------------------------------------------------------- #
    crops = get_path(cfg, "data.crop_ids")
    if not isinstance(crops, list) or not crops:
        raise ConfigError("data.crop_ids must be a non-empty list")
    duplicates = sorted({c for c in crops if crops.count(c) > 1})
    if duplicates:
        raise ConfigError(f"data.crop_ids contains duplicates: {duplicates}")

    known_crops = set(crops)
    # `data.conditions` is a dataset-level mapping, so it may name crops this
    # particular run does not use -- a run narrowing `crop_ids` must not have to
    # restate it.  What would be a real error is one crop in two conditions.
    seen_in: dict[str, str] = {}
    for condition, members in (get_path(cfg, "data.conditions", {}) or {}).items():
        if not isinstance(members, list):
            raise ConfigError(f"data.conditions.{condition} must be a list")
        for crop in members:
            if crop in seen_in and seen_in[crop] != condition:
                raise ConfigError(
                    f"crop {crop!r} is listed under two conditions: "
                    f"{seen_in[crop]!r} and {condition!r}"
                )
            seen_in[crop] = condition

    uncovered = sorted(c for c in crops if c not in seen_in)
    if seen_in and uncovered:
        raise ConfigError(
            f"data.conditions covers some crops but not these: {uncovered}; "
            f"add them or drop the conditions mapping entirely"
        )

    # ---- cells ---------------------------------------------------------- #
    source = get_path(cfg, "cells.source")
    if source not in CELL_SOURCES:
        raise ConfigError(f"cells.source must be one of {CELL_SOURCES}, got {source!r}")

    warmup = get_path(cfg, "cells.warmup_frames")
    if not isinstance(warmup, int) or warmup < 1:
        raise ConfigError(
            "cells.warmup_frames must be an integer >= 1: temporal features are "
            f"undefined on a cell's first active frame (got {warmup!r})"
        )
    min_frames = get_path(cfg, "cells.min_frames")
    if not isinstance(min_frames, int) or min_frames <= warmup:
        raise ConfigError(
            f"cells.min_frames ({min_frames!r}) must be an integer greater than "
            f"cells.warmup_frames ({warmup}), or every cell is filtered away"
        )

    # ---- features ------------------------------------------------------- #
    requested = get_path(cfg, "features.compute", "all")
    if requested != "all":
        if not isinstance(requested, list) or not requested:
            raise ConfigError("features.compute must be 'all' or a non-empty list")
        known = trackfeatures.feature_names()
        for name in requested:
            if name not in known:
                raise ConfigError(
                    f"features.compute names unknown feature {name!r}"
                    f"{_did_you_mean(name, known)}"
                )
        dupes = sorted({n for n in requested if requested.count(n) > 1})
        if dupes:
            raise ConfigError(f"features.compute contains duplicates: {dupes}")

    computed = computed_features(cfg)

    # ---- model ---------------------------------------------------------- #
    model_features = get_path(cfg, "model.features", [])
    if not isinstance(model_features, list):
        raise ConfigError("model.features must be a list")
    known = trackfeatures.feature_names()
    for name in model_features:
        if name not in known:
            raise ConfigError(
                f"model.features names unknown feature {name!r}{_did_you_mean(name, known)}"
            )
    dupes = sorted({n for n in model_features if model_features.count(n) > 1})
    if dupes:
        raise ConfigError(f"model.features contains duplicates: {dupes}")
    missing = [n for n in model_features if n not in computed]
    if missing:
        raise ConfigError(
            f"model.features names {missing} which the features step will not "
            f"compute; add them to features.compute (or set it to 'all')"
        )

    if uses_dino(cfg):
        n_pcs = get_path(cfg, "dino.n_pcs", 0)
        if not isinstance(n_pcs, int) or n_pcs < 1:
            raise ConfigError(f"model.use_dino_pcs is set, so dino.n_pcs must be >= 1 (got {n_pcs!r})")
        if not get_path(cfg, "dino.model_path", None):
            raise ConfigError("model.use_dino_pcs is set, so dino.model_path must be given")
        overlay = get_path(cfg, "dino.patch_overlay", "none")
        if overlay not in PATCH_OVERLAYS:
            raise ConfigError(
                f"dino.patch_overlay must be one of {PATCH_OVERLAYS}, got {overlay!r}"
            )

    dim = len(emission_names(cfg))
    if dim == 0:
        raise ConfigError(
            "the model has no inputs: set model.features, model.use_dino_pcs, or both"
        )

    num_lags = get_path(cfg, "model.num_lags")
    if num_lags not in (0, 1):
        raise ConfigError(
            f"model.num_lags must be 0 or 1 -- tarhmm.compute_inputs raises above "
            f"that (got {num_lags!r})"
        )
    num_states = get_path(cfg, "model.num_states")
    if not isinstance(num_states, int) or num_states < 2:
        raise ConfigError(f"model.num_states must be an integer >= 2, got {num_states!r}")

    init_method = get_path(cfg, "model.init_method", "kmeans")
    if init_method not in INIT_METHODS:
        raise ConfigError(f"model.init_method must be one of {INIT_METHODS}, got {init_method!r}")

    seeds = get_path(cfg, "model.em_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("model.em_seeds must be a non-empty list")
    if not all(isinstance(s, int) for s in seeds):
        raise ConfigError(f"model.em_seeds must be integers, got {seeds!r}")
    if len(set(seeds)) != len(seeds):
        raise ConfigError(f"model.em_seeds contains duplicates: {seeds!r}")

    iters = get_path(cfg, "model.num_em_iters")
    if not isinstance(iters, int) or iters < 1:
        raise ConfigError(f"model.num_em_iters must be an integer >= 1, got {iters!r}")

    # ---- outputs -------------------------------------------------------- #
    requested_video = get_path(cfg, "outputs.video.crops", "all")
    if requested_video not in ("all", "none"):
        if not isinstance(requested_video, list):
            raise ConfigError("outputs.video.crops must be 'all', 'none', or a list")
        unknown = [c for c in requested_video if c not in known_crops]
        if unknown:
            raise ConfigError(f"outputs.video.crops names crops not in data.crop_ids: {unknown}")

    histogram = get_path(cfg, "outputs.state_age_histogram", {}) or {}
    if not isinstance(histogram, dict):
        raise ConfigError("outputs.state_age_histogram must be a mapping")
    bin_frames = histogram.get("bin_frames", 1)
    if not isinstance(bin_frames, int) or isinstance(bin_frames, bool) or bin_frames < 1:
        raise ConfigError(
            f"outputs.state_age_histogram.bin_frames must be an integer >= 1, got {bin_frames!r}"
        )
    max_age = histogram.get("max_age", "auto")
    if max_age != "auto":
        if not isinstance(max_age, int) or isinstance(max_age, bool) or max_age < 1:
            raise ConfigError(
                f"outputs.state_age_histogram.max_age must be 'auto' or an integer >= 1, "
                f"got {max_age!r}"
            )
    kind = histogram.get("kind", "stacked")
    if kind not in AGE_HISTOGRAM_KINDS:
        raise ConfigError(
            f"outputs.state_age_histogram.kind must be one of {AGE_HISTOGRAM_KINDS}, got {kind!r}"
        )
    anchor = histogram.get("anchor", "existence")
    if anchor not in AGE_ANCHORS:
        raise ConfigError(
            f"outputs.state_age_histogram.anchor must be one of {AGE_ANCHORS}, got {anchor!r}"
        )

    extras = get_path(cfg, "outputs.extras", []) or []
    if not isinstance(extras, list):
        raise ConfigError("outputs.extras must be a list")
    # Imported lazily: `extras/__init__.py` only needs to be importable in the
    # environment that actually runs the extras step.
    from arhmm.extras import EXTRA_NAMES, requirements_for

    for name in extras:
        if name not in EXTRA_NAMES:
            raise ConfigError(
                f"outputs.extras names unknown extra {name!r}"
                f"{_did_you_mean(name, EXTRA_NAMES)}"
            )
        for requirement in requirements_for(name):
            if requirement == "dino" and not uses_dino(cfg):
                raise ConfigError(
                    f"extra {name!r} requires DINO columns, but model.use_dino_pcs is false"
                )


def validate_inputs(cfg: dict) -> list[str]:
    """Check the things `validate` deliberately does not: files and environments.

    Args:
        cfg (dict): the resolved configuration.

    Returns:
        list[str]: human-readable problems; empty when everything resolves.
    """
    problems: list[str] = []

    # The CVAT tracks are read whatever the source: `nuclei` still takes its
    # T cells from them.
    roots = ["ground_truth_tracks_dir", "image_crops_dir"]
    reads_nuclei = get_path(cfg, "cells.source", None) == "nuclei"
    if reads_nuclei:
        roots.append("caliban_tracks_dir")
    for key in roots:
        directory = Path(get_path(cfg, f"paths.{key}", "") or "")
        if not directory.is_dir():
            problems.append(f"paths.{key} is not a directory: {directory}")

    for key in ("output_root", "cache_root"):
        root = Path(get_path(cfg, f"paths.{key}", "") or "")
        problem = _writability_problem(root)
        if problem:
            problems.append(f"paths.{key} {problem}: {root}")

    repo_root = Path(get_path(cfg, "paths.repo_root", "") or "")
    if not (repo_root / "models" / "tarhmm.py").is_file():
        problems.append(f"paths.repo_root does not contain models/tarhmm.py: {repo_root}")

    # Crops: one directory per crop, holding the tracks and the aligned image.
    # The Caliban nuclei are the exception -- flat, one file per crop.
    gt_root = Path(get_path(cfg, "paths.ground_truth_tracks_dir", "") or "")
    img_root = Path(get_path(cfg, "paths.image_crops_dir", "") or "")
    caliban_root = Path(get_path(cfg, "paths.caliban_tracks_dir", "") or "")
    for crop in crop_ids(cfg):
        well = crop.split("_", 1)[0]
        candidates = [
            gt_root / well / crop / "ALL_tracks.tiff",
            gt_root / well / crop / "ALL_cancer_ids.pkl",
            img_root / well / crop / "crop.tiff",
        ]
        if reads_nuclei:
            candidates.append(caliban_root / f"{crop}.tiff")
        for candidate in candidates:
            if not candidate.is_file():
                problems.append(f"crop {crop}: missing {candidate}")

    if uses_dino(cfg):
        model_path = Path(get_path(cfg, "dino.model_path", "") or "")
        if not model_path.is_dir():
            problems.append(f"dino.model_path is not a directory: {model_path}")

    problems.extend(_missing_envs(cfg))
    return problems


def _writability_problem(root: Path) -> str | None:
    """Whether a directory can be created and written in; None when it can.

    Probes by actually creating and removing a file rather than calling
    `os.access`, which is unreliable on the network filesystem this data lives
    on: it reports `W_OK` false for directories the user demonstrably owns and
    can write to.

    Args:
        root (Path): the directory the pipeline wants to write into.

    Returns:
        str | None: a short description of the problem, or None if all is well.
    """
    import tempfile

    if not str(root):
        return "is not set"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"could not be created ({exc.strerror})"
    try:
        with tempfile.NamedTemporaryFile(dir=str(root), prefix=".arhmm-probe-"):
            pass
    except OSError as exc:
        return f"is not writable ({exc.strerror})"
    return None


def _missing_envs(cfg: dict) -> list[str]:
    """Names in `envs` that conda does not know about."""
    import subprocess

    try:
        completed = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"could not run `conda env list`: {exc}"]
    if completed.returncode != 0:
        return [f"`conda env list` failed: {completed.stderr.strip()}"]

    known = set()
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            known.add(line.split()[0])

    envs = get_path(cfg, "envs", {}) or {}
    return [
        f"envs.{role} names conda environment {name!r}, which does not exist"
        for role, name in sorted(envs.items())
        if name not in known
    ]
