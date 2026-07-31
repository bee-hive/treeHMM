"""Filesystem helpers shared by the pipeline steps.

Every write goes through `atomic_write` / `save_npz` / `write_json`, which write
to a temporary file in the destination directory and rename into place.  A step
killed mid-write therefore leaves either the previous artifact or nothing --
never a truncated file that a later run happily loads and trusts.

**The npz rule.** The three conda environments are on different numpy majors
(imaging 1.26, dino 2.4, model 2.4) and hand arrays to each other as `.npz`.
Plain numeric and bool arrays round-trip between them; object arrays and pickled
payloads do not.  So `save_npz` rejects anything that is not a numeric or bool
array, `load_npz` always passes `allow_pickle=False`, and every string -- feature
names, cell sources, column orders -- goes in a JSON sidecar instead.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

#: dtype kinds that survive a numpy-1 <-> numpy-2 npz round trip.
_SAFE_KINDS = frozenset("biufc")  # bool, int, uint, float, complex


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and its parents if needed, returning it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def atomic_write(path: str | Path, mode: str = "w", **kwargs) -> Iterator[Any]:
    """Open `path` for writing such that it appears complete or not at all.

    Args:
        path (str | Path): destination file.
        mode (str): file mode, `"w"` or `"wb"`.
        **kwargs: passed through to `open`.

    Yields:
        A writable file object for a temporary file in the destination directory.
    """
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, mode, **kwargs) as handle:
            yield handle
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    """Atomically save numeric/bool arrays to a compressed `.npz`.

    Args:
        path (str | Path): destination `.npz`.
        **arrays: the arrays to store.

    Returns:
        Path: the written file.

    Raises:
        TypeError: if any array has a dtype that would not survive the
            cross-environment round trip (strings, objects, datetimes).
    """
    for name, array in arrays.items():
        array = np.asarray(array)
        if array.dtype.kind not in _SAFE_KINDS:
            raise TypeError(
                f"{path}: array {name!r} has dtype {array.dtype!r}, which does not "
                f"round-trip between this repo's numpy versions. Numeric and bool "
                f"arrays only -- put strings in a JSON sidecar."
            )
    with atomic_write(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    return Path(path)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an `.npz` written by `save_npz`, materializing every array.

    Args:
        path (str | Path): the `.npz` to read.

    Returns:
        dict[str, np.ndarray]: every stored array, eagerly loaded so the file
            handle does not outlive the call.
    """
    with np.load(str(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a JSON sidecar with stable key order."""
    with atomic_write(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return Path(path)


def read_json(path: str | Path) -> dict:
    """Read a JSON sidecar."""
    return json.loads(Path(path).read_text())


def write_yaml(path: str | Path, payload: Any) -> Path:
    """Atomically write YAML, preserving insertion order."""
    import yaml

    with atomic_write(path, "w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    return Path(path)


def read_yaml(path: str | Path) -> Any:
    """Read a YAML file."""
    import yaml

    return yaml.safe_load(Path(path).read_text())


def well_of(crop_id: str) -> str:
    """The well a crop belongs to, e.g. `B4_t50t100y200y350x750x900` -> `B4`."""
    return crop_id.split("_", 1)[0]


def banner(text: str, width: int = 78) -> str:
    """A one-line section header for step logs."""
    return f"\n{'=' * width}\n{text}\n{'=' * width}"


def relink(link_path: str | Path, target: str | Path) -> None:
    """Point `link_path` at `target`, replacing any existing link.

    Used to make a run directory show the shared cache directories its config
    currently resolves to.  Links are refreshed rather than left stale, so a run
    directory never advertises artifacts from a previous configuration.
    """
    link_path = Path(link_path)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            # A real directory where a link belongs: leave it alone rather than
            # recursively deleting something the user may have put there.
            return
    ensure_dir(link_path.parent)
    link_path.symlink_to(Path(target).resolve(), target_is_directory=True)
