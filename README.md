# Tree AR-HMM for Cell Lineages
<p align="center">
  <img src="figs/fig2.png" alt="Tree AR-HMM probabilistic model" width="70%">
</p>

This repository implements inference and parameter learning for a **Tree Autoregressive Hidden Markov Model (Tree AR-HMM)** using **Dynamax** (Jax). This was inspired by the problem of modeling cell lineages, where we would like to learn latent states of cells, but naively applying an AR-HMM treats daughter cells independently of their parent.  Division events create a branching tree structure, and we assume that the latent state of a daughter cell depends on its parent's latent state at division. The implementation also accommodates spontaneous cell birth (e.g. coming into frame at time $t$) as well as cell death.

In summary:
- Each cell (or agent, more generally) has a **latent discrete state** that evolves over time.  
- Emissions are **shared AR(1) dynamics** conditioned on the latent state.  
- When a cell divides, its daughters’ initial latent states are drawn from a **division-specific transition kernel** that depends on the parent’s latent state at division.

  
(Figure courtesy of Nano Banana Pro)
---

## Probabilistic Model

<p align="center">
  <img src="figs/fig1.png" alt="Tree AR-HMM probabilistic model" width="100%">
</p>

We index cells by $i$ (e.g. root $r$, daughters $n,m,\dots$) and time by $t$.  
Each cell has a lifetime interval over which it exists and can persist or divide.

### Latent States for Root Cells

For each root cell $r$, alive from $t = 1$ until it divides at time $\tilde t$:

- **Initial state**

  $$z_{r,1} \sim \mathrm{Cat}(\pi_0)$$

- **Within-cell transitions**

  $$z_{r,t+1} \mid z_{r,t} = k \sim \mathrm{Cat}(\pi_k), \quad t = 1,\dots,\tilde t - 1$$

Here, $\{\pi_k\}$ define the standard Markov transition matrix over discrete states.

### Auto-regressive Emissions 

For any cell $i$ and any time $t$ during its lifetime, we observe a vector $x_{i,t}$.  
Conditional on the latent state, we use a shared AR(1) emission model:

$$p\bigl(x_{i,t} \mid z_{i,t} = k,\; x_{i,t-1}\bigr) = \ell_t^{(i)}(k)$$

For example, a Gaussian AR(1):

$$x_{i,t} \mid z_{i,t} = k,\; x_{i,t-1} \sim \mathcal{N}(A_k x_{i,t-1} + b_k,\; \Sigma_k)$$

All emission parameters $\{A_k, b_k, \Sigma_k\}$ are **shared across all cells** (parents and daughters).

### Division Events and Daughters’ Initial States

Suppose a parent cell $r$ divides at time $\tilde t$, producing daughters $n$ and $m$.  
Each daughter’s initial latent state at time $\tilde t + 1$ depends on the parent’s latent state at division:

$$z_{i,\tilde t + 1} \mid z_{r,\tilde t} = k \sim \mathrm{Cat}(\tilde \pi_k), \quad i \in \{n,m\}$$

The collection $\{\tilde \pi_k\}$ defines a **division transition kernel** that maps the parent’s state at division to the daughters’ starting states.

After birth, each daughter follows the same within-cell transition dynamics:

$$z_{i,t+1} \mid z_{i,t} = k \sim \mathrm{Cat}(\pi_k), \quad t \ge \tilde t+1$$

again with AR(1) emissions as above.

### Full probabilistic model 

Let $R$ be the number of roots, and for each root $r$, let $C_{r,t}$ be the set of cells alive at time $t$ in that lineage. For each parent–child pair $(n, c_n)$ at times $(t, t+1)$, the joint likelihood factors as

$$p(x, z) = \prod_{r=1}^R \left[ p(z_{r,1}) \prod_{t} \prod_{n \in C_{r,t}} p\bigl(x_{n,t} \mid z_{n,t}, x_{n,t-1}\bigr) \prod_{c_n} p\bigl(z_{c_n, t+1} \mid z_{n,t}\bigr) \right]$$

where the transition term $p(z_{c_n, t+1} \mid z_{n,t})$ is given by:
- the standard transition matrix $P$ for **non-division** (self) transitions;
- the division transition matrix $\tilde P$ for **division** events.

## Fitting
Fit using EM-- see the **Derivation** folder for forward–backward details. 

---

## Running it on real data

The model above is driven by the `treearhmm` pipeline: one YAML file defines one
run, every run produces the same four base outputs, and feature computation is
decoupled from model fitting.

The package is **not pip-installed**, so there is no `treearhmm` command — run it
as a module from the repo root, in conda `base`:

```bash
cd /gladstone/engelhardt/lab/jadjasu/LiveCellUmbrella/treeHMM

python -m treearhmm doctor                    # environments, paths, CUDA, npz round-trip
python -m treearhmm run configs/_smoke.yml    # one crop, no DINO, under a minute warm
```

Results land in `analysis/runs/_smoke/outputs/`: a state-coloured overlay video,
per-state feature distributions, the learned transition matrix, and per-cell
state assignments.

See **`treearhmm/README.md`** for the step chain, the config layering, and how to
add a feature, a modality or an extra. See **`tree_input.md`** for the exact
contract between the pipeline and `models/tarhmm.py` -- the six arrays, their
shapes and dtypes, and the preprocessing order.

Note that the pipeline **does not use the tree**: divisions are out of scope, so
every cell is an independent chain and the division kernel is never exercised.
The model retains the capability; the pipeline does not use it.

## Environments

No single environment has JAX, torch and scikit-image together, so the pipeline
spans three. **You do not activate any of them.** Run the CLI from conda `base`
(it needs only PyYAML + numpy); it spawns each step under `conda run -n <env>`
itself, using the names in `configs/site.yml` under `envs:`.

| env | role in `site.yml` | what it is for |
|-----|-----|------|
| `OccidentAnalysis` | `imaging` | microscopy I/O, features, plotting, video |
| `cs229Dino` | `dino` | DINOv2 embedding and PCA |
| `treeHMM_env` | `model` | JAX / Dynamax / this model |

Specs live in `envs/`. Create them with:

```bash
conda env create -f envs/OccidentAnalysis.yml
conda env create -f envs/cs229Dino.yml
conda env create -f envs/treeHMM_env.yml
```

`envs/AnalysisEnv.yml` is a separate lab environment and is **not** used by this
pipeline. See `env_setup.md` for how `treeHMM_env` is built by hand and for the
`tensorflow-probability` gotcha — `pip install dynamax` pulls a stable
`tensorflow-probability` that is too old for current JAX, and it has to be
replaced with `tfp-nightly`.

Check that all three resolve, along with paths, CUDA and the cross-environment
`.npz` contract:

```bash
python -m treearhmm doctor configs/runs/dino_k3.yml
```

### Which code runs in which environment

Each step is a separate process in its own environment, so an import that is
fine in one is unavailable in the other two.

| code | environment | third-party imports it needs |
|---|---|---|
| `treearhmm/cli.py` — the driver | conda `base` (any env with the two) | `pyyaml`, `numpy` |
| `treearhmm/steps/features.py` | `OccidentAnalysis` | `numpy`, `tifffile`, `scikit-image` |
| `treearhmm/steps/outputs.py`, `steps/extras.py`, `treearhmm/extras/*` | `OccidentAnalysis` | `numpy`, `matplotlib`, `imageio` + `imageio-ffmpeg` |
| `treearhmm/steps/dino.py` | `cs229Dino` | `numpy`, `torch`, `transformers`, `tifffile`, `matplotlib` |
| `treearhmm/steps/pca.py` | `cs229Dino` | `numpy`, `scikit-learn` |
| `treearhmm/steps/fit.py` → `models/tarhmm.py` | `treeHMM_env` | `numpy`, `jax`, `dynamax`, `tfp-nightly`, `fastprogress`, `jaxtyping` |

Three modules are imported by **every** step and therefore have to import
cleanly in all three environments: `treearhmm/config.py`, `treearhmm/core/io.py`
and `treearhmm/core/trackfeatures.py`. They are restricted to `numpy`, `pyyaml`
and the standard library — which is why `trackfeatures.py` imports scikit-image
*inside* the functions that use it rather than at module level. Keep it that way.

The three environments sit on different numpy majors (1.26.4 / 2.4.2 / 2.4.6)
and hand arrays to each other as `.npz`. Numeric and bool arrays round-trip;
object arrays and pickles do not, so the pipeline writes numeric/bool arrays
only, loads with `allow_pickle=False`, and puts every string in a JSON sidecar.
`doctor` checks this round trip.

## TODOs

- Optionally, division events themselves can be modeled by appending a division
  indicator to the observations.
- Improve sampling and the demonstration notebook.
- `sample()` is not implemented on `tARHMM`.
