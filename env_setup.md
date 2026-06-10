# Environment Setup

This project spans **two conda environments**. Use the right one for each task:

| Env | Used for | Scripts |
|-----|----------|---------|
| `treeHMM_env` | JAX / Dynamax / the tARHMM model | `fit_arhmm.py`, the tarHMM notebooks |
| `occident` | microscopy I/O, feature extraction, video rendering | `calculate_emissions.py`, `create_overlay_videos.py` |

This document covers building **`treeHMM_env`** (the model environment). `occident` is a pre-existing lab environment with the `MarsonImagingPipeline` dependencies and is not built here.

---

## Quick start

```bash
bash scripts/setup_treeHMM_env.sh
```

This creates `treeHMM_env`, installs everything, applies the TFP fix below, and verifies the GPU is visible. The rest of this document explains what it does and why.

---

## Requirements

- **conda / mamba** (this repo uses miniforge at `/gladstone/engelhardt/lab/jadjasu/miniforge3`).
- **For the GPU build:** an NVIDIA GPU with driver **≥ 525** (CUDA 12). The A30 nodes here run driver 570 / CUDA 12.8. The CUDA libraries themselves are installed as pip wheels — you do **not** need a system CUDA toolkit, only a compatible driver.
- For CPU-only, no GPU/driver is needed (see [CPU-only](#cpu-only-build) below).

---

## Manual installation (GPU)

### 1. Create the environment

```bash
mamba create -y -n treeHMM_env python=3.11 pip
```

### 2. Install dynamax + GPU JAX + the pipeline's plotting/IO deps

Install them in a single `pip` resolve so JAX satisfies both dynamax's pin and the CUDA 12 extra:

```bash
conda run --no-capture-output -n treeHMM_env pip install \
  dynamax "jax[cuda12]" \
  matplotlib seaborn pandas tifffile pyyaml fastprogress
```

`dynamax` brings in jax, jaxlib, optax, scikit-learn, jaxtyping, and numpy. The remaining packages are what `scripts/cvat_gt_crops/fit_arhmm.py` needs for its plots and data loading.

### 3. ⚠️ Critical: replace TensorFlow Probability with `tfp-nightly`

`pip install dynamax` pulls **stable** `tensorflow-probability`, which is **too old for current JAX**. The model imports fail at:

```
AttributeError: module 'jax.interpreters.xla' has no attribute 'pytype_aval_mappings'
```

Fix it by swapping in `tfp-nightly` (which is what the repo's `pyproject.toml` actually specifies):

```bash
conda run --no-capture-output -n treeHMM_env pip uninstall -y tensorflow-probability
conda run --no-capture-output -n treeHMM_env pip install tfp-nightly
```

### 4. Verify

```bash
conda run --no-capture-output -n treeHMM_env python -c "
import jax
from tensorflow_probability.substrates import jax as tfp
print('jax', jax.__version__, '| devices:', jax.devices())
print('treeHMM_env OK')
"
```

Expected output includes `CudaDevice(id=0)` (and any other GPUs). If you see `CpuDevice`, the GPU wheels didn't take — recheck your driver version and that `jax[cuda12]` (not plain `jax`) was installed.

---

## CPU-only build

If you don't need the GPU, drop the `[cuda12]` extra in step 2:

```bash
conda run --no-capture-output -n treeHMM_env pip install \
  dynamax jax \
  matplotlib seaborn pandas tifffile pyyaml fastprogress
```

Then apply the same `tfp-nightly` fix (step 3) and verify (step 4 — devices will show `CpuDevice`). The tARHMM here is small (≈50 timesteps × ≤50 cells), so EM is fast on CPU.

---

## Verified working versions

The setup above was verified end-to-end (model imports, 10 iterations of EM on synthetic data, log-prob increasing) with:

| Package | Version |
|---------|---------|
| python | 3.11 |
| dynamax | 1.0.1 |
| jax / jaxlib | 0.10.1 (+ jax-cuda12 plugins 0.10.1) |
| tfp-nightly | 0.26.0.dev (replaces stable tensorflow-probability) |
| numpy | 2.x |

---

## Running the model

```bash
# Fit the AR-HMM (uses treeHMM_env)
conda run --no-capture-output -n treeHMM_env python scripts/cvat_gt_crops/fit_arhmm.py
```

The full three-step pipeline (which switches between `occident` and `treeHMM_env` automatically) is:

```bash
cd scripts/cvat_gt_crops && bash run_pipeline.sh
```

Before running, update the absolute paths in `scripts/cvat_gt_crops/config.yml` (`treehmm_dir`, `output_base_dir`, input dirs) to your checkout.

---

## GPU usage notes

- The A30s here are **not** MIG-partitioned. JAX grabs **GPU 0 by default** and pre-allocates most of its memory.
- To pick a specific GPU (e.g. if GPU 0 is busy), set `CUDA_VISIBLE_DEVICES`:
  ```bash
  CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n treeHMM_env python scripts/cvat_gt_crops/fit_arhmm.py
  ```
- To cap JAX's memory pre-allocation, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` (or `XLA_PYTHON_CLIENT_PREALLOCATE=false`).
