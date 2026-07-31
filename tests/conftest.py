"""Put the repo root on `sys.path` so tests import the package without installing it.

Mirrors what `cli.run_step` does for the real steps via `PYTHONPATH`.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
