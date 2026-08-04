# Environment Setup

This project spans **three conda environments**. Use the right one for each task:

| Env | Used for | Pipeline steps | Key versions |
|-----|----------|----------------|--------------|
| `treeHMM_env` | JAX / Dynamax / the tARHMM model | `fit` | py 3.11.15, jax 0.10.1, dynamax 1.0.1, numpy 2.4.6 |
| `OccidentAnalysis` | microscopy I/O, feature extraction, plotting, video | `features`, `outputs`, `extras` | py 3.11.9, numpy 1.26.4, skimage 0.23.2, matplotlib 3.8.2, imageio + ffmpeg |
| `cs229Dino` | DINOv2 embedding and PCA | `dino`, `pca` | torch 2.5.1 (CUDA), transformers 5.2.0, sklearn 1.8.0, numpy 2.4.2 |

> There is **no env named `occident`** — earlier revisions of this file and of the
> archived scripts said so and were wrong. `AnalysisEnv` also exists but is Python 3.14
> and unrelated to this project. `python -m arhmm doctor` checks that all three names in
> `configs/site.yml` resolve.

> **Cross-environment hazard.** The three envs are on numpy 1.26.4 / 2.4.2 / 2.4.6 and
> hand arrays to each other as `.npz`. Numeric and bool arrays round-trip; object
> arrays and pickled payloads do not. The pipeline's rule is numeric/bool arrays only,
> `allow_pickle=False` on every load, and all strings in JSON sidecars.

This document covers building **`treeHMM_env`** (the model environment).
`OccidentAnalysis` and `cs229Dino` are pre-existing lab environments and are not built
here.

---

## Quick start

```bash
bash scripts/archive/setup_treeHMM_env.sh
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

`dynamax` brings in jax, jaxlib, optax, scikit-learn, jaxtyping, and numpy. The remaining packages are what `arhmm/steps/fit.py` needs for data loading and its summary output.

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

The pipeline switches environments per step automatically. The package is not
pip-installed, so run the CLI as a module from the repo root, in conda `base`:

```bash
python -m arhmm doctor                      # check all three envs, paths, CUDA
python -m arhmm run configs/_smoke.yml      # one crop, no DINO, ~1 minute
```

To run one step by hand in its own environment (it is stamp-guarded like the
driver, so pass `--force` to redo a step that is already current):

```bash
PYTHONPATH=$PWD conda run --no-capture-output -n treeHMM_env \
  python -m arhmm.steps.fit --run-dir analysis/runs/_smoke --force
```

Before the first run, set the absolute paths in `configs/site.yml` (`repo_root`,
`output_root`, `cache_root`, and the three input directories) to your checkout.

---

## GPU usage notes

- The A30s here are **not** MIG-partitioned. JAX grabs **GPU 0 by default** and pre-allocates most of its memory.
- To pick a specific GPU (e.g. if GPU 0 is busy), set `CUDA_VISIBLE_DEVICES`:
  ```bash
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD conda run --no-capture-output -n treeHMM_env \
    python -m arhmm.steps.fit --run-dir analysis/runs/_smoke --force
  ```
  For a normal run, set `hardware.cuda_visible_devices` in `configs/site.yml`
  instead — the CLI applies it to every step it spawns.
- To cap JAX's memory pre-allocation, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` (or `XLA_PYTHON_CLIENT_PREALLOCATE=false`).
