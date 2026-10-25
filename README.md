# MNESIS — Working Memory in Recurrent Spiking Neural Networks With Heterogeneous Synaptic Delays

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ICANN%202026-orange.svg)](https://laurentperrinet.github.io/publication/perrinet-26-icann/)

> **MNESIS** — *a Memory Network where Every Spike Is Sacred*

This repository contains the full implementation, experiments, and paper source for **MNESIS**, a recurrent spiking neural network (SNN) with heterogeneous synaptic delays that stores and recalls arbitrary spike patterns as sequential chains of overlapping Spiking Motifs.

---

## Overview

Working memory in biological neural circuits relies on precise spike timing rather than sustained firing rates. MNESIS models this by equipping every synapse with $D$ learnable delays, parameterised as a single weight tensor $\mathbf{W} \in \mathbb{R}^{N \times N \times D}$. Each stored pattern is encoded as a chain of overlapping **Spiking Motifs**: contiguous context windows of length $D$ that uniquely predict the next time step of activity. A closed-form initialisation derived by deconvolving the LIF membrane response reaches near-perfect recall before any gradient step, while surrogate-gradient BPTT then adds robustness to noise.

The same `HD_SNN` framework is applied to sparse synthetic motifs and to periodic motifs. The experiments below build on one another: each numbered notebook imports the shared infrastructure and loads the cached artifacts produced by the previous one.

**Paper:** Laurent U. Perrinet (2026). *Working Memory in Recurrent Spiking Neural Networks With Heterogeneous Synaptic Delays*. ICANN 2026.
[`tex/Perrinet26mnesis.pdf`](tex/Perrinet26mnesis.pdf) · [`https://laurentperrinet.github.io/publication/perrinet-26-icann/`](https://laurentperrinet.github.io/publication/perrinet-26-icann/)

---

## Repository structure

```
MNESIS/
├── src/                            # Source code and Jupyter notebooks
│   ├── mnesis_boilerplate.py        # Imports, device setup, Params dataclass, utilities
│   ├── mnesis_chains.py             # SpikingPattern generators, HD_SNN class, analytical init
│   └── notebooks (numbered pipeline, run sequentially):
│        ├── 10_MNESIS_generative-model.ipynb       # Generative model for synthetic patterns
│        ├── 11_MNESIS_learn-synthetic.ipynb        # Training on synthetic patterns
│        ├── 13_MNESIS_testing-inference.ipynb      # Sequential retrieval of M patterns
│        ├── 14_MNESIS_testing-noise.ipynb          # Robustness to bit-flip noise
│        ├── 15_MNESIS_testing-trigger-duration.ipynb    # Effect of trigger-window length
│        ├── 16_MNESIS_testing-trigger-fraction.ipynb    # Effect of partial neuron coverage
│        ├── 18_MNESIS_learn-periodic.ipynb    # Learning and retrieval of periodic memories
│        ├── 20_MNESIS_scanning-parameters.ipynb # Parameter scans (D, T, p_A, ...)
│        ├── 25_MNESIS_optuna.ipynb            # Hyperparameter optimisation (Optuna)
│        ├── 99_MNESIS_run-all.ipynb           # End-to-end notebook orchestrator
│        └── requirements.txt                  # Python dependencies for notebook runs
├── figures/                       # Generated figures (PDF/PNG)
├── tex/                           # Paper source
│    ├── Perrinet26mnesis.tex      # Main LaTeX source
│    ├── Perrinet26mnesis.pdf      # Compiled paper
│    └── mnesis.bib                # Bibliography
├── cached_data/                   # Cached weights and scan results (git-ignored)
├── LICENSE
└── README.md
```

---

## Quickstart

### Install dependencies

```bash
pip install -r src/requirements.txt
```

Core dependencies: `torch`, `snntorch`, `numpy`, `scipy`, `matplotlib`, `jupyter`. The optimisation/scan notebooks additionally use `optuna` (bundled in `requirements.txt`).

### Run the notebooks in order

The notebooks are numbered and designed to be run sequentially. They import shared infrastructure from the Python modules `mnesis_boilerplate.py` (imports, device detection, `Params` dataclass, utilities) and `mnesis_chains.py` (`HD_SNN` class, pattern generators, analytical initialisation). Each notebook saves its outputs (model weights, scan results) to `cached_data/` so that downstream notebooks can load them without recomputation. Notebook 99 can orchestrate a full multi-notebook run in one place.

```bash
cd src
jupyter notebook
```

Each notebook opens with its own `opt_dict = dict(...)` that overrides the global `Params`
defaults for that experiment (a different `N_neuron`, `num_delay`, `N_pretime`, threshold, …).
The global defaults in `Params` are the canonical reference; per-notebook `opt_dict`s tune the
run for the pattern family being studied.

| # | Notebook / Module | Purpose |
|---|----------|---------|
| — | `mnesis_boilerplate.py` | Shared imports, device detection (MPS / CUDA / CPU), random-seed utilities, `Params` dataclass, and helper functions reused by all downstream notebooks. |
| — | `mnesis_chains.py` | Defines `SpikingPattern`, `StochasticSpikingPattern`, and the `HD_SNN` class with analytical weight initialisation (a pseudo-inverse or Hebbian cross-correlation, optionally deconvolving the LIF membrane). Corresponds to the Methods section of the paper. |
| 10 | `10_MNESIS_generative-model.ipynb` | Implements the generative model for synthetic sparse patterns: draws Gaussian logit maps $\ell \sim \mathcal{N}(0, E_\mathrm{SM})$, thresholds to keep the top $p_\mathrm{SM}$ fraction, convolves with the biphasic spike shape, and samples Bernoulli spike trains at rate $p_A$. Visualises the resulting patterns. |
| 11 | `11_MNESIS_learn-synthetic.ipynb` | Trains the network on $M = 16$ synthetic sparse patterns. Demonstrates that the analytical init alone reaches high $F_1$; gradient training with a cosine schedule then improves noise robustness. |
| 13 | `13_MNESIS_testing-inference.ipynb` | Concatenates all $M = 16$ patterns in sequence with $N_\mathrm{pretime} = 50$ steps of spontaneous inter-trial activity; evaluates sliding-window $F_1$ to confirm selective, cross-interference-free retrieval. |
| 14 | `14_MNESIS_testing-noise.ipynb` | Bit-flip noise on the trigger window ($p_\mathrm{flip} \in [0, 1]$). Quantifies attractor-like robustness; $F_1 \approx 0.63$ at $p_\mathrm{flip} = 0.25$. |
| 15 | `15_MNESIS_testing-trigger-duration.ipynb` | Truncated trigger window (0 to $D-1$ steps). Finds the minimum cue length for reliable recall; $F_1 \approx 0.42$ at 75% of $D$. |
| 16 | `16_MNESIS_testing-trigger-fraction.ipynb` | Partial neuron coverage (0 to $N$ neurons silenced in trigger). $F_1 \approx 0.60$ with 87.5% of neurons active. |
| 18 | `18_MNESIS_learn-periodic.ipynb` | Builds periodic targets, trains periodic memories, and evaluates retrieval robustness under increasing input noise. |
| 20 | `20_MNESIS_scanning-parameters.ipynb` | Systematic one-at-a-time scans over $D$, $T$, $p_A$, $N$, $E_\mathrm{SM}$, $p_\mathrm{SM}$, and the new `do_pinv` / `do_deconv` flags, with $N_\mathrm{cv} = 10$ seeds. Produces the parameter-scan figures of the paper. |
| 25 | `25_MNESIS_optuna.ipynb` | Automated hyperparameter search with [Optuna](https://optuna.org/) over learning dynamics, thresholds, and regularisation. |
| 99 | `99_MNESIS_run-all.ipynb` | Scripted orchestrator to run the full notebook pipeline with progress timestamps. |

### Cached data

Results are saved to `cached_data/` (excluded from git via `.gitignore`):

| File pattern | Content |
|---|---|
| `*_init.pth` | Analytically initialised weights (pseudo-inverse or Hebbian, deconvolved or not) |
| `*.pth` | Trained model weights after gradient steps |
| `*_scan_*.json` | Parameter sweep results (loss, precision, recall per condition) |
| `*_periodic-with-noise.npz` | Periodic-memory robustness curves across noise levels and time chunks |
| `*_optuna.sqlite3` | Optuna studies (synthetic motifs) |

Delete a cached `.pth`, `.json`, `.npz`, or `.sqlite3` file to force recomputation; set `RECOMPUTE = True` at the top of any notebook to invalidate the full cache for that notebook. The scan/optimisation notebooks additionally guard each run with a `.lock` sentinel file to allow safe incremental resumption.

### Build and publish the documentation

The site is built and deployed by the GitHub Actions workflow
`.github/workflows/static.yml` on every push to `main`; no generated HTML is
committed (`docs/` only holds the sources). To preview it locally:

```bash
pip install -r docs/requirements.txt && make -C docs html   # output in docs/_build/html
```

The build symlinks the `src/` notebooks into `docs/notebooks/` and renders them
with their stored outputs. The Pages source must be set to "GitHub Actions"
(repository Settings > Pages); the site is served at
[https://laurentperrinet.github.io/MNESIS/](https://laurentperrinet.github.io/MNESIS/).

---

## Model

### Membrane dynamics

Each neuron $j$ evolves as:

$$u_j(t) = \beta \cdot u_j(t-1) \cdot (1 - s_j(t-1)) + \sum_{i=1}^{N} \sum_{d=1}^{D} W_{j,i,d} \cdot s_i(t-d)$$

The global `Params` defaults in the current `datetag` use $\beta = 0.7$ and threshold
$\vartheta = 0.75$; individual experiments override these in their local `opt_dict`
(e.g. a different $\beta$ / $\vartheta$ pair). `Params`
is the single source of hyperparameter truth; see the "Parameter drift" note below for
why the prose, the paper, and the code need not always agree.

### Variables

Defaults are those of the `Params` dataclass in `src/mnesis_boilerplate.py`; one time bin is 1 ms.

| Symbol | Code name | Default | Role |
|---|---|---|---|
| $N$ | `N_neuron` | 1024 | number of neurons |
| $D$ | `num_delay` | 41 time bins | number of delays per synapse, i.e. length of a spiking motif |
| $M$ | `N_pattern` | 16 | number of stored motifs |
| $T$ | `N_time` | 1000 time bins | duration of each motif |
| $N_\mathrm{pretime}$ | `N_pretime` | 50 time bins | spontaneous activity before and after the stimulus |
| $p_A$ | `p_A` | 0.00016 per time bin | prior probability of firing |
| $p_\mathrm{flip}$ | `p_flip` | 0.01 | probability of flipping a bit in the stochastic pattern generator |
| — | `seed` | 2026 | seed of the random number generators |
| $\beta$ | `lif_beta` | 0.70 | membrane decay per time bin |
| $\vartheta$, $\vartheta_0$ | `lif_threshold` | 0.75 (membrane potential, a.u.) | firing threshold |
| — | `learn_beta`, `learn_threshold` | `False` | make $\beta$ and $\vartheta$ learnable |
| — | `do_pinv`, `do_deconv` | `True` | pseudo-inverse solution and membrane deconvolution of the initialisation |
| — | `num_epochs` | 256 | number of training epochs |
| — | `num_warmup_epochs` | 16 | epochs of the learning-rate warmup ramp |
| — | `base_lr`, `final_lr` | 0.04, 0.0004 | learning rates at the peak and at the end of the cosine schedule |
| $\delta_1$, $\delta_2$ | `delta1`, `delta2` | 0.003, 0.00001 | complements to 1 of the optimiser coefficients (momentum, decay) |
| $p$ | `dropout` | 0.60 | dropout probability |
| $\alpha$ | `alpha_surrogate` | 5.0 | sharpness of the surrogate gradient (`FastSigmoid`) |
| — | `N_scan`, `N_cv` | 14, 11 | number of values per scan, number of cross-validation seeds |

### Analytical initialisation

The LIF membrane is a causal IIR lowpass. Two orthogonal flags control the closed-form
initialisation in `HD_SNN.get_W_init()`:

- **`do_deconv`** (default `True`) — deconvolve the membrane so the input current that
  drives neuron $j$ to a target spike at time $t$ is

  $$I^*_j(t) = \vartheta_0 \bigl(s^*_j(t) - \beta \cdot s^*_j(t-1)\bigr).$$

  Here $\vartheta_0$ is the firing threshold (`lif_threshold`). With `do_deconv = False` the raw target $s^*_j(t)$ is used directly.

- **`do_pinv`** (default `True`) — solve for $\mathbf{W}$ via the (numerically stable,
  computed on CPU) full pseudo-inverse $\mathbf{W} = \mathrm{pinv}(\mathbf{C})\,\mathbf{T}$.
  With `do_pinv = False` a Hebbian cross-correlation rule is used instead, relying on the
  Gram-matrix approximation $\mathbf{C}\mathbf{C}^\top \approx N D p_A \mathbf{I}$:

  $$w_{i,j,d} = \frac{1}{N \cdot D \cdot p_A \cdot M} \sum_{\mu,\,t} s_i^{*(\mu)}(t-d) \cdot \bigl(s_j^{*(\mu)}(t) - \beta \cdot s_j^{*(\mu)}(t-1)\bigr).$$

  (When `do_pinv = False` this is the form quoted in the paper; the Gram-matrix
  approximation makes the cross-correlation rule exact.) The default `do_pinv = True`
  path is the most accurate and is what the camera-ready runs use.

### Training

- **Loss**: $\mathcal{L} = 1 - F_1$ (harmonic mean of precision and recall, evaluated after the trigger window) — `SpikeF1scoreLoss`
- **Optimiser**: Adadelta by default (`adadelta`), configurable via `Params.optimizer` (also supports `sgd`, `adam`, `adamw`, `rmsprop`)
- **Schedule**: cosine decay with a `num_warmup_epochs` warmup ramp (`final_lr` → `base_lr`)
- **Surrogate**: fast sigmoid by default, sharpness $\alpha = 5.0$ — configurable via `Params.surrogate_name` / `Params.alpha_surrogate`
- **Regularisation**: dropout $p = 0.6$, weight decay $\lambda = 0$
- **LIF dynamics**: $\beta = 0.7$, $\vartheta = 0.75$, zero-reset (`zero`)
- **Hardware**: Apple M3 Ultra (MPS) or NVIDIA GPU (CUDA / Jean Zay GENCI)

> **Parameter drift.** As of the reconciliation run `2026-09-07` (the current `datetag`),
> the `Params` defaults ($\beta = 0.7$, $\vartheta = 0.75$, $\alpha = 5$, `adadelta`,
> dropout $0.6$) match the methods text of the paper. Earlier revisions of this README and
> the camera-ready runs cite other values ($\beta = 0.8$, $\alpha = 12$, SGD, dropout
> $0.10$). Each notebook also sets its own `opt_dict`. The code is authoritative for *what
> the current default run does*; consult the specific `opt_dict` and
> `tex/Perrinet26mnesis.pdf` for the exact value a given paper figure used.

> An AI assistant was used to improve the readability and structure of this codebase — not to create it.

---

## Results summary

| Experiment | Key result |
|---|---|
| Training (NB 11) | High $F_1$ with analytical init alone; gradient training adds noise robustness |
| Sequential retrieval (NB 13) | All 16 patterns retrieved without cross-interference |
| Noise robustness (NB 14) | $F_1 \approx 0.63$ at $p_\mathrm{flip} = 0.25$; chance only near $p_\mathrm{flip} = 0.5$ |
| Trigger duration (NB 15) | $F_1 \approx 0.42$ at 75% of $D$; rises steeply above $D/2$ |
| Neuron coverage (NB 16) | $F_1 \approx 0.60$ with 87.5% of neurons active in trigger |
| Delay scan (NB 20) | $\mathcal{L} \approx 0.85$ at $D=3$; $\mathcal{L} \to 0$ at $D=127$ |
| Duration scan (NB 20) | $\mathcal{L} \approx 0.004$ at $T=64$; $\mathcal{L} \approx 0.08$ at $T=2048$ |
| Rate scan (NB 20) | Optimal at $p_A \in [10^{-4}, 10^{-3}]$; degrades for $p_A \geq 2\times10^{-3}$ |
| Periodic memory (NB 18) | Stable periodic retrieval with dedicated robustness evaluation under progressive bit-flip noise |
| Full orchestrator (NB 99) | Provides a single notebook entry point for sequential multi-notebook execution |

---

## Citation

```bibtex
@inproceedings{Perrinet2026MNESIS,
  author       = {Perrinet, Laurent U.},
  title        = {Working Memory in Recurrent Spiking Neural Networks
                With Heterogeneous Synaptic Delays},
  booktitle    = {ICANN 2026},
  year         = {2026},
  url          = {https://laurentperrinet.github.io/publication/perrinet-26-icann/}
}
```

---

## License

GPL-3.0 — see [LICENSE](LICENSE).

---

*Institut de Neurosciences de la Timone (UMR 7289), Aix Marseille Université / CNRS, Marseille, France.*
*Supported by GENCI-IDRIS (Grant 2025–AD010314955R2).*
