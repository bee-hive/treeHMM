"""Allow `python -m treearhmm ...` as an alias for the `treearhmm` command."""

from treearhmm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
