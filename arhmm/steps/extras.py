"""Step `extras`: run whatever additional outputs the config asked for.

The base outputs have already been written by the time this runs, so an extra
that fails takes only itself down.  Each one gets its own directory under
`{run}/outputs/extras/{name}/`, and its exception -- if any -- is recorded in
the step's stamp and printed with a traceback rather than swallowed.

The step exits non-zero when any extra failed, so `arhmm run` reports the
problem; but because the step is marked optional in `layout.STEPS`, the run as a
whole still counts the base outputs as delivered.

Run inside the imaging environment.
"""

from __future__ import annotations

import sys
import traceback

from arhmm import config as cfgmod
from arhmm import extras as extras_pkg
from arhmm.core import io
from arhmm.steps import step_main


def _run(cfg: dict, layout, args) -> dict:
    requested = list(cfgmod.get_path(cfg, "outputs.extras", []) or [])
    if not requested:
        print("no extras requested")
        return {"requested": [], "failed": {}}

    failed: dict[str, str] = {}
    for name in requested:
        out_dir = io.ensure_dir(layout.extras_dir / name)
        print(io.banner(f"extra: {name}"))
        try:
            extras_pkg.runner(name)(cfg, layout, out_dir)
            print(f"  ok -> {out_dir}")
        except Exception as exc:  # noqa: BLE001 - an extra must not sink the run
            failed[name] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {failed[name]}")
            traceback.print_exc()

    if failed:
        print(f"\n{len(failed)} of {len(requested)} extras failed: {sorted(failed)}")
        print("the base outputs are unaffected")

    summary = {"requested": requested, "failed": failed}
    if failed:
        # Written by hand: step_main only stamps on a clean return, and a
        # partial result should still be inspectable.
        layout.write_stamp("extras", summary)
        sys.exit(1)
    return summary


if __name__ == "__main__":
    sys.exit(step_main("extras", "additional run-specific outputs", _run))
