"""Shared library code used by more than one pipeline step.

Modules here are import-light on purpose: a step running in the DINO or model
environment must be able to import `arhmm.config` and `arhmm.layout`
without scikit-image, matplotlib or torch being installed.  Heavy dependencies
are imported inside the functions that need them.

`core/dino.py` is the one exception -- it imports torch at module level, so it
may be imported only by `arhmm.steps.dino`, which runs in the DINO
environment.  Nothing else may import it, and it is deliberately not imported
here.
"""
