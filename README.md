# MNESIS — Working Memory in a Recurrent Spiking Neural Network with Heterogeneous Synaptic Delays

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-AIROV%202026-orange.svg)](tex/Perrinet26mnesis.pdf)

> **MNESIS** — *a Memory Network where Every Spike Is Sacred*

This repository contains the full implementation, experiments, and paper source for **MNESIS**, a recurrent spiking neural network (SNN) with heterogeneous synaptic delays that stores and recalls arbitrary spike patterns as sequential chains of overlapping Spiking Motifs.

---

## Overview

Working memory in biological neural circuits relies on precise spike timing rather than sustained firing rates. MNESIS models this by equipping every synapse with $D$ learnable delays, parameterised as a single weight tensor $\mathbf{W} \in \mathbb{R}^{N \times N \times D}$. Each stored pattern is encoded as a chain of overlapping **Spiking Motifs**: contiguous context windows of length $D$ that uniquely predict the next time step of activity. A closed-form initialisation derived by deconvolving the LIF membrane response reaches near-perfect recall before any gradient step, while surrogate-gradient BPTT then adds robustness to noise.

The same `HD_SNN` framework is applied across a growing zoo of pattern families: sparse synthetic motifs, periodic motifs, structured travelling waves, chaotic (Lorenz) trajectories, a tokenised text corpus, and real spiking data (Spiking Heidelberg Digits). The experiments below build on one another: each numbered notebook imports the shared infrastructure and loads the cached artifacts produced by the previous one.

**Paper:** Laurent U. Perrinet (2026). *Working Memory in a Recurrent Spiking Neural Network with Heterogeneous Synaptic Delays*. AIROV 2026.
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
│        ├── 32_MNESIS_learn-travelling-waves.ipynb # Structured spatiotemporal travelling waves
│        ├── 34_MNESIS_learn-Lorenz-attractor.ipynb # Chaotic trajectory encoding and recall
│        ├── 36_MNESIS_learn-text.ipynb        # Token-text corpus → spike codebook → recall
│        ├── 38_MNESIS_learn-SHD.ipynb         # Spiking Heidelberg Digits experiments
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

Core dependencies: `torch`, `snntorch`, `numpy`, `scipy`, `matplotlib`, `jupyter`. The text experiment additionally uses the `datasets` and `tiktoken` packages, and the optimisation/scan notebooks use `optuna` (all bundled in `requirements.txt`).

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
| 14 | `14_MNESIS_testing-noise.ipynb` | Bit-flip noise on the trigger window ($p_\mathrm{flip} \in [0, 1]$). Quantifies attractor-like robustness; $F_1 = 0.967$ at $p_\mathrm{flip} = 0.25$. |
| 15 | `15_MNESIS_testing-trigger-duration.ipynb` | Truncated trigger window (0 to $D-1$ steps). Finds the minimum cue length for reliable recall; $F_1 = 0.862$ at 75% of $D$. |
| 16 | `16_MNESIS_testing-trigger-fraction.ipynb` | Partial neuron coverage (0 to $N$ neurons silenced in trigger). Perfect recall maintained with 87.5% of neurons active. |
| 18 | `18_MNESIS_learn-periodic.ipynb` | Builds periodic targets, trains periodic memories, and evaluates retrieval robustness under increasing input noise. |
| 20 | `20_MNESIS_scanning-parameters.ipynb` | Systematic one-at-a-time scans over $D$, $T$, $p_A$, $N$, $E_\mathrm{SM}$, $p_\mathrm{SM}$, and the new `do_pinv` / `do_deconv` flags, with $N_\mathrm{cv} = 10$ seeds. Produces the parameter-scan figures of the paper. |
| 25 | `25_MNESIS_optuna.ipynb` | Automated hyperparameter search with [Optuna](https://optuna.org/) over learning dynamics, thresholds, and regularisation. |
| 32 | `32_MNESIS_learn-travelling-waves.ipynb` | Introduces MotionClouds-based travelling-wave patterns and benchmarks retrieval with structured spatiotemporal motifs. |
| 34 | `34_MNESIS_learn-Lorenz-attractor.ipynb` | Encodes Lorenz chaotic trajectories into spike codes and evaluates memory recall on non-periodic continuous dynamics. |
| 36 | `36_MNESIS_learn-text.ipynb` | Tokenises a Wikipedia corpus (`tiktoken`, `cl100k_base`) into token IDs, builds a frequency-ordered **spike codebook** (most-frequent tokens get the sparsest neuron rows, minimising the Hamming distance), wraps it in a `TextSpikingPattern` generator, learns and recalls the resulting spike patterns, then decodes the output spikes back to text via the inverse codebook + tokenizer. Includes Optuna tuning and parameter scans. |
| 38 | `38_MNESIS_learn-SHD.ipynb` | Integrates Spiking Heidelberg Digits data loading/preprocessing for dataset-grounded experiments. |
| 99 | `99_MNESIS_run-all.ipynb` | Scripted orchestrator to run the full notebook pipeline with progress timestamps. |

### Cached data

Results are saved to `cached_data/` (excluded from git via `.gitignore`):

| File pattern | Content |
|---|---|
| `*_init.pth` | Analytically initialised weights (pseudo-inverse or Hebbian, deconvolved or not) |
| `*.pth` | Trained model weights after gradient steps |
| `*_scan_*.json` | Parameter sweep results (loss, precision, recall per condition) |
| `*_periodic-with-noise.npz` | Periodic-memory robustness curves across noise levels and time chunks |
| `*_TW_*.json` | Travelling-wave parameter scans |
| `*_lorenz_chaotic_*.json` | Lorenz-attractor scan and optimisation outputs |
| `wikipedia_dataset.parquet` | Downloaded Wikipedia corpus (text experiment) |
| `*_wikipedia_fr_token_ids.npy` | Token-ID array of the tokenised corpus |
| `*_codebook.pt` | Frequency-ordered token→neuron spike codebook |
| `*_optuna.sqlite3` | Optuna studies (synthetic, travelling-wave, Lorenz, text) |

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

The global `Params` defaults in the current `datetag` use $\beta = 0.8$ and threshold
$\vartheta = 0.72$; individual experiments override these in their local `opt_dict`
(e.g. the text and travelling-wave runs use other $\beta$ / $\vartheta$ pairs). `Params`
is the single source of hyperparameter truth; see the "Parameter drift" note below for
why the prose, the paper, and the code need not always agree.

### Analytical initialisation

The LIF membrane is a causal IIR lowpass. Two orthogonal flags control the closed-form
initialisation in `HD_SNN.get_W_init()`:

- **`do_deconv`** (default `True`) — deconvolve the membrane so the input current that
  drives neuron $j$ to a target spike at time $t$ is

  $$I^*_j(t) = \vartheta_0 \bigl(s^*_j(t) - \beta \cdot s^*_j(t-1)\bigr).$$

  With `do_deconv = False` the raw target $s^*_j(t)$ is used directly.

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
- **Optimiser**: SGD by default (`sgd`), configurable via `Params.optimizer` (also supports `adam`, `adamw`, `rmsprop`, `adadelta`)
- **Schedule**: cosine decay with a `num_warmup_epochs` warmup ramp (`final_lr` → `base_lr`)
- **Surrogate**: fast sigmoid by default, sharpness $\alpha = 5.0$ — configurable via `Params.surrogate_name` / `Params.alpha_surrogate`
- **Regularisation**: dropout $p = 0.25$, weight decay $\lambda = 0$
- **LIF dynamics**: $\beta = 0.8$, $\vartheta = 0.72$, zero-reset (`subtract`)
- **Hardware**: Apple M3 Ultra (MPS) or NVIDIA GPU (CUDA / Jean Zay GENCI)

> **Parameter drift.** The `Params` defaults are the *latest camera-ready run* (the current
> `datetag`, `2026-08-06`) and have moved on from the values in the paper text and earlier
> revisions of this README (notably $\beta$: `0.8` vs `0.7`; $\vartheta$: `0.72` vs `1.0`;
> $\alpha$: `5.0` vs `15`; dropout `0.25` vs `0.37`). Each notebook also sets its own
> `opt_dict`. The code (and this table) is authoritative for *what the current default run
> does*; consult the specific `opt_dict` and `tex/Perrinet26mnesis.pdf` for the exact value
> a given paper figure used.

> An AI assistant was used to improve the readability and structure of this codebase — not to create it.

---

## Results summary

| Experiment | Key result |
|---|---|
| Training (NB 11) | High $F_1$ with analytical init alone; gradient training adds noise robustness |
| Sequential retrieval (NB 13) | All 16 patterns retrieved without cross-interference |
| Noise robustness (NB 14) | $F_1 = 0.967$ at $p_\mathrm{flip} = 0.25$; chance only near $p_\mathrm{flip} = 0.5$ |
| Trigger duration (NB 15) | $F_1 = 0.862$ at 75% of $D$; reliable above $D/2$ |
| Neuron coverage (NB 16) | $F_1 = 1.0$ with 87.5% of neurons active in trigger |
| Delay scan (NB 20) | $\mathcal{L} \approx 0.85$ at $D=3$; $\mathcal{L} \to 0$ at $D=127$ |
| Duration scan (NB 20) | $\mathcal{L} \approx 0.004$ at $T=64$; $\mathcal{L} \approx 0.08$ at $T=2048$ |
| Rate scan (NB 20) | Optimal at $p_A \in [10^{-4}, 10^{-3}]$; degrades for $p_A \geq 2\times10^{-3}$ |
| Periodic memory (NB 30) | Stable periodic retrieval with dedicated robustness evaluation under progressive bit-flip noise |
| Travelling waves (NB 32) | MotionClouds-derived structured motifs can be stored/recalled with the same HD-SNN framework |
| Lorenz attractor (NB 34) | Extends retrieval tests to non-periodic chaotic trajectories encoded as spike events |
| Text (NB 36) | A tokenised text corpus is encoded as a frequency-ordered spike codebook, learned and recalled by `HD_SNN`, then decoded back to text |
| SHD integration (NB 40) | Adds real-event dataset loading and preprocessing for external benchmark experiments |
| Full orchestrator (NB 99) | Provides a single notebook entry point for sequential multi-notebook execution |

---

## Citation

```bibtex
@inproceedings{Perrinet2026MNESIS,
  author       = {Perrinet, Laurent U.},
  title        = {Working Memory in a Recurrent Spiking Neural Network
                with Heterogeneous Synaptic Delays},
  booktitle    = {AIROV 2026},
  year         = {2026},
  url          = {https://laurentperrinet.github.io/publication/perrinet-26-icann/}
}
```

---

## TODO list

- [ ] https://laurentperrinet.github.io/MNESIS/ : Build docs from the code ipynb and  (and tex?) / publish to GitHub Pages


---

## License

GPL-3.0 — see [LICENSE](LICENSE).

---

*Institut de Neurosciences de la Timone (UMR 7289), Aix Marseille Université / CNRS, Marseille, France.*
*Supported by GENCI-IDRIS (Grant 2025–AD010314955R2).*
