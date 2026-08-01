#!/usr/bin/env bash
# ============================================================
# setup_treeHMM_env.sh
#
# Creates the `treeHMM_env` conda environment used to run the
# tARHMM model (models/tarhmm.py) and the fit_arhmm.py pipeline step.
#
# GPU build: installs JAX with CUDA 12 wheels (needs NVIDIA driver >= 525;
# the A30 nodes here run 570 / CUDA 12.8). For CPU-only, drop the
# "[cuda12]" extra from the jax install below.
#
# Usage:
#   bash scripts/setup_treeHMM_env.sh
# ============================================================
set -euo pipefail

ENV_NAME=treeHMM_env

# 1. Fresh env
mamba create -y -n "$ENV_NAME" python=3.11 pip

# 2. dynamax + GPU JAX + the plotting/IO deps fit_arhmm.py needs.
#    Resolved together so jax satisfies both dynamax's pin and the cuda12 extra.
conda run --no-capture-output -n "$ENV_NAME" pip install \
  dynamax "jax[cuda12]" \
  matplotlib seaborn pandas tifffile pyyaml fastprogress

# 3. CRITICAL: `pip install dynamax` pulls *stable* tensorflow-probability,
#    which is too old for current JAX (errors on
#    jax.interpreters.xla.pytype_aval_mappings at import time).
#    Replace it with tfp-nightly, exactly as pyproject.toml specifies.
conda run --no-capture-output -n "$ENV_NAME" pip uninstall -y tensorflow-probability || true
conda run --no-capture-output -n "$ENV_NAME" pip install tfp-nightly

# 4. Verify: GPU visible + model imports.
conda run --no-capture-output -n "$ENV_NAME" python -c "
import jax
from tensorflow_probability.substrates import jax as tfp
print('jax', jax.__version__, '| devices:', jax.devices())
print('treeHMM_env OK')
"
