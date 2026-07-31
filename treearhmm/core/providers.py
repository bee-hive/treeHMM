"""Where emission columns come from.

Two genuinely different kinds of thing feed the model, and the pipeline keeps
them separate where they differ and unifies them where they do not.

**Track features** are cheap, CPU, computed in one pass with everything else,
and each has a name, a unit and a physical meaning.  Adding one is a single
decorated function in `trackfeatures.py`.

**Modalities** -- DINO principal components today, some other embedding
tomorrow -- are expensive, need a GPU and a different conda environment, and
produce *k* dimensions with no individual meaning, via a fit (the joint PCA)
that is global across all crops.  A modality cannot honour the track-feature
contract ("call this function on one frame's masks"), so forcing both into one
registry would mean every entry carrying flags that only one kind ever uses.

They are unified at **assembly**, not at computation: both answer the same three
questions -- which columns do you offer, where do I load them from, and what
does this column mean -- so `steps/fit.py` contains no DINO-specific code.  It
asks `config.emission_names(cfg)` for the ordered column list and then asks each
active provider to load its block.

Adding a third modality is therefore a new `Provider` here plus a step to
produce its artifact.  Nothing in the fit step changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from treearhmm import config as cfgmod
from treearhmm.core import io


@dataclass(frozen=True)
class Provider:
    """A source of emission columns.

    Attributes:
        name (str): short identifier, used by extras' `REQUIRES`.
        step (str): the step whose output directory holds this provider's arrays.
        columns (Callable[[dict], list[str]]): every column it offers, in order.
        load (Callable): `(cfg, layout, crop_id) -> ((T, N, F) float32, names)`.
        describe (Callable[[str, dict], str]): one line explaining a column,
            used as a plot subtitle and in generated documentation.
    """

    name: str
    step: str
    columns: Callable[[dict], list[str]]
    load: Callable[..., tuple[np.ndarray, list[str]]]
    describe: Callable[[str, dict], str]


# --------------------------------------------------------------------------- #
# track features
# --------------------------------------------------------------------------- #


def _tracks_columns(cfg: dict) -> list[str]:
    return cfgmod.computed_features(cfg)


def _tracks_load(cfg: dict, layout, crop_id: str) -> tuple[np.ndarray, list[str]]:
    """Load one crop's cached track features.

    Returns:
        tuple: `(T, N, F)` values and the `F` feature names in column order.
    """
    crop_dir = layout.crop_dir("features", crop_id)
    arrays = io.load_npz(crop_dir / "features.npz")
    meta = io.read_json(crop_dir / "meta.json")
    return arrays["values"], list(meta["feature_names"])


def _tracks_describe(column: str, cfg: dict) -> str:
    from treearhmm.core.trackfeatures import FEATURE_REGISTRY

    feature = FEATURE_REGISTRY[column]
    unit = f" [{feature.units}]" if feature.units else ""
    return f"{feature.doc}{unit}"


TRACKS = Provider(
    name="tracks",
    step="features",
    columns=_tracks_columns,
    load=_tracks_load,
    describe=_tracks_describe,
)


# --------------------------------------------------------------------------- #
# DINO principal components
# --------------------------------------------------------------------------- #


def _dino_columns(cfg: dict) -> list[str]:
    return cfgmod.dino_column_names(cfg)


def _dino_load(cfg: dict, layout, crop_id: str) -> tuple[np.ndarray, list[str]]:
    """Load one crop's DINO principal components.

    Invalid cell-frames are stored as zeros rather than NaN, matching how the
    model pads inactive cells.
    """
    arrays = io.load_npz(layout.crop_dir("pca", crop_id) / "dino_pcs.npz")
    values = arrays["pcs"]
    return values, [f"dino_pc_{i}" for i in range(values.shape[-1])]


def _dino_describe(column: str, cfg: dict) -> str:
    index = int(column.rsplit("_", 1)[-1])
    model_id = cfgmod.get_path(cfg, "dino.model_id", "dinov2")
    patch = cfgmod.get_path(cfg, "dino.patch_px", "?")
    whitened = "whitened " if cfgmod.get_path(cfg, "dino.whiten", True) else ""
    return f"{whitened}principal component {index} of {model_id} on {patch}px centroid patches"


DINO = Provider(
    name="dino",
    step="pca",
    columns=_dino_columns,
    load=_dino_load,
    describe=_dino_describe,
)


PROVIDERS: tuple[Provider, ...] = (TRACKS, DINO)
PROVIDERS_BY_NAME: dict[str, Provider] = {p.name: p for p in PROVIDERS}


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def provider_of(column: str, cfg: dict) -> Provider:
    """Which provider offers a given emission column.

    Raises:
        KeyError: if no provider claims it.
    """
    for provider in PROVIDERS:
        if column in provider.columns(cfg):
            return provider
    raise KeyError(f"no provider offers emission column {column!r}")


def active_providers(cfg: dict) -> list[Provider]:
    """Providers this run's emission vector actually draws on."""
    names = set()
    for column in cfgmod.emission_names(cfg):
        names.add(provider_of(column, cfg).name)
    return [p for p in PROVIDERS if p.name in names]


def describe(column: str, cfg: dict) -> str:
    """One line explaining an emission column, for plots and documentation."""
    try:
        return provider_of(column, cfg).describe(column, cfg)
    except KeyError:
        return column


def load_emissions(cfg: dict, layout, crop_id: str) -> tuple[np.ndarray, list[str]]:
    """Assemble one crop's emission block in `config.emission_names` order.

    Args:
        cfg (dict): resolved configuration.
        layout (Layout): the run's layout.
        crop_id (str): the crop to load.

    Returns:
        tuple: `(T, N, D)` float32 emissions and the `D` column names, which
            equal `config.emission_names(cfg)`.

    Raises:
        ValueError: if two providers disagree on the crop's shape.
    """
    wanted = cfgmod.emission_names(cfg)
    blocks: dict[str, np.ndarray] = {}
    shape: tuple[int, int] | None = None

    for provider in active_providers(cfg):
        values, names = provider.load(cfg, layout, crop_id)
        if shape is None:
            shape = values.shape[:2]
        elif values.shape[:2] != shape:
            raise ValueError(
                f"{crop_id}: provider {provider.name!r} has shape {values.shape[:2]} "
                f"but an earlier provider had {shape}; the caches are out of step"
            )
        for index, name in enumerate(names):
            blocks[name] = values[..., index]

    missing = [name for name in wanted if name not in blocks]
    if missing:
        raise ValueError(f"{crop_id}: no cached values for emission columns {missing}")

    stacked = np.stack([blocks[name] for name in wanted], axis=-1).astype(np.float32)
    return stacked, wanted


def load_diagnostics(cfg: dict, layout, crop_id: str) -> tuple[np.ndarray, list[str]]:
    """Every cached track feature for a crop, whether or not the model saw it.

    These are what make a state description checkable: a state characterised
    only by the features it was fit on is a tautology.

    Returns:
        tuple: `(T, N, F)` float32 values and the `F` feature names.
    """
    values, names = TRACKS.load(cfg, layout, crop_id)
    return values.astype(np.float32), names
