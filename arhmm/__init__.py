"""Deterministic, config-driven runs of the tree AR-HMM.

One YAML file defines one run.  It is layered over `configs/site.yml` (paths,
conda environments, the crop list) and `configs/default.yml` (the
experiment-facing defaults), resolved once, and frozen into the run directory so
the run stays reproducible even if the defaults change afterwards.

    arhmm run configs/_smoke.yml

Every run produces the same four base outputs -- state-coloured overlay videos,
per-state feature distributions, the learned transition matrix, and per-cell
state assignments with their probabilities -- plus whatever extras it names in
`outputs.extras`.

`tree_input.md` at the repo root is the binding contract for what the fit step
hands `models/tarhmm.py`.  **The pipeline does not use the tree**: divisions are
out of scope, so `is_division_mask` is all False, `parent_indices` is always
self, and the division kernel is never exercised.

To add a track feature, add one decorated function to
`arhmm/core/trackfeatures.py`.  To add a whole modality, add a provider to
`arhmm/core/providers.py` and a step to produce it.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
