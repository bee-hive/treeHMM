"""Optional, run-specific outputs.

Every run produces the five base outputs.  Anything beyond that is an extra:
named in `outputs.extras`, produced into `{run}/outputs/extras/{name}/`, and
free to fail without taking the run's real outputs down with it.

An extra is a module in this package exposing:

    REQUIRES: tuple[str, ...]                      providers it needs, e.g. ("dino",)
    def run(cfg: dict, layout: Layout, out_dir: Path) -> None

**Keep module-level imports light.**  Config validation imports every named
extra just to read `REQUIRES`, and that happens in whichever environment the
user invoked the CLI from.  Do the matplotlib/pandas imports inside `run`.

Adding an extra is one new module here; discovery is by filename, so nothing
else needs editing.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Callable

_PACKAGE_DIR = Path(__file__).parent


def _discover() -> tuple[str, ...]:
    """Extra names available in this package, sorted."""
    return tuple(
        sorted(
            module.name
            for module in pkgutil.iter_modules([str(_PACKAGE_DIR)])
            if not module.name.startswith("_")
        )
    )


#: Names usable in `outputs.extras`.
EXTRA_NAMES: tuple[str, ...] = _discover()


def load(name: str):
    """Import an extra module by name.

    Raises:
        KeyError: if the name is not a known extra.
    """
    if name not in EXTRA_NAMES:
        raise KeyError(f"unknown extra {name!r}; known: {list(EXTRA_NAMES)}")
    return importlib.import_module(f"{__name__}.{name}")


def requirements_for(name: str) -> tuple[str, ...]:
    """Provider names an extra needs, e.g. `("dino",)`.

    Returns an empty tuple when the module cannot be imported cheaply, so that a
    missing optional dependency surfaces when the extra actually runs rather
    than blocking config validation.
    """
    try:
        return tuple(getattr(load(name), "REQUIRES", ()))
    except ImportError:
        return ()


def runner(name: str) -> Callable:
    """The `run(cfg, layout, out_dir)` callable for an extra."""
    module = load(name)
    if not hasattr(module, "run"):
        raise AttributeError(f"extra {name!r} does not define run(cfg, layout, out_dir)")
    return module.run
