"""Allow `python -m arhmm ...` as an alias for the `arhmm` command."""

from arhmm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
