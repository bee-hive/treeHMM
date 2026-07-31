"""The `treearhmm` command line: create, run and inspect runs.

    treearhmm run     configs/my_run.yml    create the run and execute it
    treearhmm status  configs/my_run.yml    what is current, what would rerun
    treearhmm show    configs/my_run.yml    the fully resolved configuration
    treearhmm list                          runs that exist under output_root
    treearhmm doctor  [configs/my_run.yml]  environments, paths, CUDA, npz round trip

Steps run as subprocesses under `conda run -n <env>`, because the chain spans
three environments.  `PYTHONPATH` is set to the repo root for each child, which
is what makes both `treearhmm` and `models.tarhmm` importable without this repo
being pip-installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import subprocess
import sys
from pathlib import Path

from treearhmm import config as cfgmod
from treearhmm import layout as layoutmod
from treearhmm.core import io

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# run preparation
# --------------------------------------------------------------------------- #


def _config_identity(cfg: dict) -> str:
    """Hash of everything in the config except where it came from."""
    return cfgmod.hash_obj({k: v for k, v in cfg.items() if k != "_source_config"}, length=32)


def _git_commit() -> tuple[str, bool]:
    """The repo's HEAD commit and whether the working tree is dirty."""
    def _git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = _git("rev-parse", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain"))
    return commit, dirty


def prepare_run(config_path: str | Path, allow_config_change: bool = False) -> tuple[dict, layoutmod.Layout]:
    """Resolve a config, create the run directory, and freeze the config into it.

    Refuses to reuse a run directory whose stored config differs from the newly
    resolved one, naming the keys that differ -- the guard against silently
    overwriting yesterday's run with a different one under the same name.

    Args:
        config_path (str | Path): the run's YAML file.
        allow_config_change (bool): overwrite a differing stored config.

    Returns:
        tuple[dict, Layout]: the resolved config and the run's layout.

    Raises:
        cfgmod.ConfigError: on a conflicting stored config.
    """
    cfg = cfgmod.load_config(config_path)
    layout = layoutmod.Layout(cfg)

    if layout.resolved_config_path.is_file():
        stored = cfgmod.load_resolved(layout.run_dir)
        if _config_identity(stored) != _config_identity(cfg) and not allow_config_change:
            differing = _differing_keys(stored, cfg)
            raise cfgmod.ConfigError(
                f"run directory {layout.run_dir} was created with a different config.\n"
                f"  differing keys: {differing}\n"
                f"  pass --allow-config-change to overwrite it, or pick another run_name."
            )

    layout.ensure_run_dir()
    cfgmod.dump_config(cfg, layout.resolved_config_path)
    layout.link_shared()
    return cfg, layout


def _differing_keys(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ between two configs."""
    differing: list[str] = []
    for key in sorted(set(old) | set(new)):
        if key == "_source_config":
            continue
        dotted = f"{prefix}{key}"
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            differing.extend(_differing_keys(a, b, f"{dotted}."))
        elif a != b:
            differing.append(dotted)
    return differing


# --------------------------------------------------------------------------- #
# step execution
# --------------------------------------------------------------------------- #


def run_step(cfg: dict, layout: layoutmod.Layout, step: layoutmod.Step, force: bool) -> dict:
    """Execute one step in its conda environment, teeing output to its log.

    Args:
        cfg (dict): resolved configuration.
        layout (Layout): the run's layout.
        step (Step): the step to run.
        force (bool): recompute even when the output is current.

    Returns:
        dict: an execution record for the manifest.
    """
    env_name = layoutmod.env_for(cfg, step)
    log_path = layout.log_path(step.name)
    io.ensure_dir(log_path.parent)

    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([child_env["PYTHONPATH"]] if child_env.get("PYTHONPATH") else [])
    )
    devices = cfgmod.get_path(cfg, "hardware.cuda_visible_devices", None)
    if devices is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = str(devices)
    if step.env == "model":
        # Keep JAX from grabbing the whole card, so a stray notebook on the same
        # GPU does not turn a fit into an OOM.
        child_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    command = [
        "conda", "run", "--no-capture-output", "-n", env_name,
        "python", "-m", step.module, "--run-dir", str(layout.run_dir),
    ]
    if force:
        command.append("--force")

    print(io.banner(f">>> {step.name}  [{env_name}]  {step.summary}"))
    started = dt.datetime.now()
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command, env=child_env, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = process.wait()
    seconds = (dt.datetime.now() - started).total_seconds()

    record = {
        "step": step.name,
        "env": env_name,
        "seconds": round(seconds, 1),
        "returncode": returncode,
        "status": "ok" if returncode == 0 else "failed",
        "log": str(log_path),
    }
    if returncode != 0:
        print(f"[{step.name}] FAILED (exit {returncode}); see {log_path}")
    return record


def write_manifest(cfg: dict, layout: layoutmod.Layout, execution: list[dict]) -> None:
    """Record what ran, where, with which commit and environments."""
    commit, dirty = _git_commit()
    manifest = {
        "run_name": cfgmod.get_path(cfg, "run_name"),
        "description": cfgmod.get_path(cfg, "description", ""),
        "source_config": cfg.get("_source_config"),
        "config_identity": _config_identity(cfg),
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "git_commit": commit,
        "git_dirty": dirty,
        "emission_names": cfgmod.emission_names(cfg),
        "steps": [
            {
                "name": s.name,
                "env": layoutmod.env_for(cfg, s),
                "key": layoutmod.step_key(cfg, s.name),
                "output_dir": str(layout.step_dir(s.name)),
            }
            for s in layoutmod.active_steps(cfg)
        ],
        "execution": execution,
    }
    io.write_yaml(layout.manifest_path, manifest)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_run(args) -> int:
    cfg, layout = prepare_run(args.config, allow_config_change=args.allow_config_change)

    problems = cfgmod.validate_inputs(cfg)
    if problems:
        print("configuration does not resolve against this machine:")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    try:
        steps = layoutmod.steps_from(cfg, args.from_step)
    except KeyError as exc:
        print(exc)
        return 2

    if args.dry_run:
        for row in layout.status():
            marker = "cached " if row["current"] else "RUN    "
            print(f"{marker} {row['step']:<9} {row['key']}  {row['reason']}")
        return 0

    execution: list[dict] = []
    exit_code = 0
    for index, step in enumerate(steps):
        force = args.force and (args.from_step is None or index == 0 or args.force_all)
        if not force and layout.is_current(step.name):
            print(f"[{step.name}] cached ({layoutmod.step_key(cfg, step.name)})")
            execution.append({"step": step.name, "status": "cached"})
            continue
        record = run_step(cfg, layout, step, force=force)
        execution.append(record)
        if record["returncode"] != 0:
            if step.optional:
                print(f"[{step.name}] optional step failed; continuing")
                exit_code = max(exit_code, 1)
                continue
            exit_code = record["returncode"]
            break

    layout.link_shared()
    write_manifest(cfg, layout, execution)
    print(f"\nrun directory: {layout.run_dir}")
    return exit_code


def cmd_status(args) -> int:
    cfg = cfgmod.load_config(args.config)
    layout = layoutmod.Layout(cfg)
    print(f"run: {cfgmod.get_path(cfg, 'run_name')}  ->  {layout.run_dir}")
    for row in layout.status():
        marker = "current" if row["current"] else "STALE  "
        print(f"  {marker}  {row['step']:<9} {row['key']}  [{row['env']}]  {row['reason']}")
        print(f"           {row['dir']}")

    commit, _ = _git_commit()
    manifest_path = layout.manifest_path
    if manifest_path.is_file():
        recorded = (io.read_yaml(manifest_path) or {}).get("git_commit")
        if recorded and recorded != commit:
            print(
                f"\n  note: this run was produced at commit {recorded[:12]}, "
                f"HEAD is now {commit[:12]}; cached artifacts predate your current code."
            )
    return 0


def cmd_show(args) -> int:
    cfg = cfgmod.load_config(args.config, validate_config=not args.no_validate)
    import yaml

    sys.stdout.write(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    return 0


def cmd_list(args) -> int:
    site = cfgmod.load_config(args.config, validate_config=False) if args.config else None
    root = Path(args.output_root) if args.output_root else Path(cfgmod.get_path(site or {}, "paths.output_root"))
    if not root.is_dir():
        print(f"no runs: {root} does not exist")
        return 0
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = run_dir / "manifest.yml"
        description = ""
        if manifest.is_file():
            data = io.read_yaml(manifest) or {}
            description = data.get("description", "") or ""
        print(f"{run_dir.name:<32} {description}")
    return 0


def cmd_doctor(args) -> int:
    """Check that the machine can actually run the pipeline."""
    config_path = args.config or (REPO_ROOT / "configs" / "_smoke.yml")
    problems: list[str] = []
    print(io.banner("treearhmm doctor"))

    try:
        cfg = cfgmod.load_config(config_path)
        print(f"  ok    config resolves and validates: {config_path}")
    except cfgmod.ConfigError as exc:
        print(f"  FAIL  config: {exc}")
        return 2

    for problem in cfgmod.validate_inputs(cfg):
        problems.append(problem)
        print(f"  FAIL  {problem}")
    if not problems:
        print("  ok    every input path, crop and conda environment resolves")

    # Each environment must import what its steps need.
    checks = {
        "imaging": "import numpy, skimage, tifffile, pandas, matplotlib, imageio; "
                   "from skimage.measure import regionprops_table; print(numpy.__version__)",
        "model": "import numpy, jax, dynamax; print(numpy.__version__, jax.__version__, "
                 "[d.platform for d in jax.devices()])",
    }
    if cfgmod.uses_dino(cfg):
        checks["dino"] = ("import numpy, torch, transformers, sklearn; "
                          "print(numpy.__version__, torch.__version__, torch.cuda.is_available())")

    for role, snippet in checks.items():
        env_name = cfgmod.get_path(cfg, f"envs.{role}")
        done = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", snippet],
            capture_output=True, text=True, timeout=600, check=False,
        )
        if done.returncode == 0:
            print(f"  ok    {role} env {env_name}: {done.stdout.strip().splitlines()[-1]}")
        else:
            problems.append(f"{role} env {env_name} cannot import its dependencies")
            print(f"  FAIL  {role} env {env_name}: {done.stderr.strip().splitlines()[-1:]}")

    # The cross-numpy npz contract: written by the model env, read by imaging.
    if _npz_roundtrip(cfg):
        print("  ok    npz round-trips between the model and imaging environments")
    else:
        problems.append("npz written in the model env could not be read in the imaging env")
        print("  FAIL  npz round trip")

    ignored = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "treearhmm/core/config.py"],
        capture_output=True, text=True, check=False,
    )
    if ignored.returncode == 0:
        problems.append("treearhmm/core is gitignored -- check .gitignore for an unanchored lib/")
        print("  FAIL  treearhmm/core/config.py is gitignored")
    else:
        print("  ok    package source is not gitignored")

    print(f"\n{'all checks passed' if not problems else f'{len(problems)} problem(s)'}")
    return 0 if not problems else 1


def _npz_roundtrip(cfg: dict) -> bool:
    """Write an npz under the model env and read it back under imaging."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "probe.npz"
        write = (
            "import numpy as np;"
            f"np.savez_compressed(r'{target}', a=np.arange(6, dtype=np.float32).reshape(2,3),"
            " m=np.array([[True,False,True],[False,True,False]]))"
        )
        read = (
            "import numpy as np;"
            f"d=np.load(r'{target}', allow_pickle=False);"
            "assert d['a'].shape==(2,3) and d['m'].sum()==3; print('ok')"
        )
        for role, snippet in (("model", write), ("imaging", read)):
            env_name = cfgmod.get_path(cfg, f"envs.{role}")
            done = subprocess.run(
                ["conda", "run", "-n", env_name, "python", "-c", snippet],
                capture_output=True, text=True, timeout=600, check=False,
            )
            if done.returncode != 0:
                return False
    return True


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="treearhmm", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="create the run and execute its steps")
    run.add_argument("config")
    run.add_argument("--from", dest="from_step", default=None, help="start at this step")
    run.add_argument("--force", action="store_true", help="recompute rather than reuse cache")
    run.add_argument("--force-all", action="store_true", help="with --from, force every later step too")
    run.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    run.add_argument("--allow-config-change", action="store_true",
                     help="reuse a run directory created with a different config")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="which steps are current, and why")
    status.add_argument("config")
    status.set_defaults(func=cmd_status)

    show = sub.add_parser("show", help="print the fully resolved config")
    show.add_argument("config")
    show.add_argument("--no-validate", action="store_true")
    show.set_defaults(func=cmd_show)

    listing = sub.add_parser("list", help="runs under output_root")
    listing.add_argument("config", nargs="?", default=None)
    listing.add_argument("--output-root", default=None)
    listing.set_defaults(func=cmd_list)

    doctor = sub.add_parser("doctor", help="check environments, paths, CUDA and the npz contract")
    doctor.add_argument("config", nargs="?", default=None)
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except cfgmod.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
