"""Pipeline steps.

Each step module is independently runnable inside its own conda environment:

    python -m treearhmm.steps.<name> --run-dir <run directory>

It reads `config.resolved.yml` from that directory and **nothing else** -- never
a parameter passed on the command line.  That is what makes "rerun this step by
hand" and "the driver ran it" the same operation, which matters a great deal
when the only way to debug the model step is to run it alone in `treeHMM_env`.

`step_main` handles the shared scaffolding: argument parsing, loading the
resolved config, building the `Layout`, timing, and writing the completion stamp
that `treearhmm status` reads.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

from treearhmm import config as cfgmod
from treearhmm import layout as layoutmod
from treearhmm.core import io


def parse_step_args(name: str, description: str, argv: list[str] | None = None):
    """Parse the argv every step accepts.

    Args:
        name (str): step name, for the program name in `--help`.
        description (str): shown in `--help`.
        argv (list[str] | None): defaults to `sys.argv[1:]`.

    Returns:
        argparse.Namespace: with `run_dir` and `force`.
    """
    parser = argparse.ArgumentParser(prog=f"treearhmm.steps.{name}", description=description)
    parser.add_argument("--run-dir", required=True, help="run directory holding config.resolved.yml")
    parser.add_argument(
        "--force", action="store_true", help="recompute even if the output is already current"
    )
    return parser.parse_args(argv)


def step_main(name: str, description: str, body: Callable, argv: list[str] | None = None) -> int:
    """Run one step: load its config, execute `body`, stamp the result.

    Args:
        name (str): step name.
        description (str): shown in `--help`.
        body (Callable): `(cfg, layout, args) -> dict | None`; anything it
            returns is merged into the completion stamp.
        argv (list[str] | None): defaults to `sys.argv[1:]`.

    Returns:
        int: process exit code.
    """
    args = parse_step_args(name, description, argv)
    cfg = cfgmod.load_resolved(args.run_dir)
    layout = layoutmod.Layout(cfg)

    if not args.force and layout.is_current(name):
        print(f"[{name}] already current for key {layoutmod.step_key(cfg, name)}; nothing to do")
        return 0

    print(io.banner(f"[{name}] {description}"))
    started = time.time()
    summary = body(cfg, layout, args) or {}
    elapsed = time.time() - started

    stamp = {"seconds": round(elapsed, 2), "produced": list(layoutmod.declared_outputs(name, cfg))}
    stamp.update(summary)
    layout.write_stamp(name, stamp)
    print(f"[{name}] done in {elapsed:.1f}s -> {layout.step_dir(name)}")
    return 0
