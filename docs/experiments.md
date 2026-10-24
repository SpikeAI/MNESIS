(experiments-reference)=
# Experiments & Notebooks

The experimental pipeline is a sequence of numbered Jupyter notebooks, running
simple to complex: each one imports the shared infrastructure
({doc}`api`) and loads the cached artifacts produced by the previous one. The
pages below are rendered from `src/*.ipynb` (notebooks live in `src/` and are
symlinked into `docs/notebooks/` at build time by `make -C docs notebooks`;
they are rendered with their stored outputs, never re-executed by the docs
build — run them yourself as described in {doc}`installation`).

## Module-level infrastructure

| Module | Purpose |
|---|---|
| {mod}`mnesis_boilerplate` | Shared imports, device detection (MPS / CUDA / CPU), random-seed utilities, {class}`~mnesis_boilerplate.Params` dataclass, and helper functions reused by all downstream notebooks. |
| {mod}`mnesis_chains` | Defines {class}`~mnesis_chains.SpikingPattern`, {class}`~mnesis_chains.StochasticSpikingPattern`, and the {class}`~mnesis_chains.HD_SNN` class with analytical weight initialisation (pseudo-inverse or Hebbian cross-correlation, optionally deconvolving the LIF membrane). Corresponds to the Methods section of the paper. |

## Generative model

```{toctree}
:maxdepth: 1

notebooks/10_MNESIS_generative-model
```

- {doc}`notebooks/10_MNESIS_generative-model` — implements the generative model
  for synthetic sparse patterns: draws Gaussian logit maps
  $\ell \sim \mathcal{N}(0, E_\mathrm{SM})$, thresholds to keep the top
  $p_\mathrm{SM}$ fraction, convolves with the biphasic spike shape, and samples
  Bernoulli spike trains at rate $p_A$. Produces the `frozen` and `two_ssp`
  figures and the cached artifacts used by all downstream notebooks.

## Learning and retrieval on synthetic motifs

```{toctree}
:maxdepth: 1

notebooks/11_MNESIS_learn-synthetic
notebooks/13_MNESIS_testing-inference
```

- {doc}`notebooks/11_MNESIS_learn-synthetic` — trains the network on
  $M = 16$ synthetic sparse patterns. Demonstrates that the analytical init
  alone reaches high $F_1$; gradient training with a cosine schedule then
  improves noise robustness. Figures: `pattern`, `target_init`, `target`.
- {doc}`notebooks/13_MNESIS_testing-inference` — concatenates all $M = 16$
  patterns in sequence with $N_\mathrm{pretime} = 50$ steps of spontaneous
  inter-trial activity; evaluates sliding-window $F_1$ to confirm selective,
  cross-interference-free retrieval. Figure: `retrieval` (plus `.mp4`).

## Robustness tests

```{toctree}
:maxdepth: 1

notebooks/14_MNESIS_testing-noise
notebooks/15_MNESIS_testing-trigger-duration
notebooks/16_MNESIS_testing-trigger-fraction
```

- {doc}`notebooks/14_MNESIS_testing-noise` — bit-flip noise on the trigger
  window ($p_\mathrm{flip} \in [0, 1]$). Quantifies attractor-like robustness:
  $F_1 = 0.967$ at $p_\mathrm{flip} = 0.25$, chance only near $0.5$.
- {doc}`notebooks/15_MNESIS_testing-trigger-duration` — truncated trigger
  window (0 to $D-1$ steps); finds the minimum cue length for reliable recall:
  $F_1 = 0.862$ at 75% of $D$.
- {doc}`notebooks/16_MNESIS_testing-trigger-fraction` — partial neuron coverage
  (0 to $N$ neurons silenced in the trigger); perfect recall is maintained with
  87.5% of neurons active.

## Parameter scans & optimisation

```{toctree}
:maxdepth: 1

notebooks/20_MNESIS_scanning-parameters
notebooks/25_MNESIS_optuna
```

- {doc}`notebooks/20_MNESIS_scanning-parameters` — systematic one-at-a-time
  scans over $D$, $T$, $p_A$, $N$, $E_\mathrm{SM}$, $p_\mathrm{SM}$, and the
  `do_pinv` / `do_deconv` flags, with $N_\mathrm{cv} = 10$ seeds; produces the
  parameter-scan figures of the paper.
- {doc}`notebooks/25_MNESIS_optuna` — automated hyperparameter search with
  [Optuna](https://optuna.org/) over learning dynamics, thresholds, and
  regularisation.

```{note}
The scan/optimisation notebooks use file-based locking: a `.lock` sentinel
guards against concurrent runs of the same scan, computed parameter values are
appended incrementally to the JSON, and runs can be safely resumed. See
{ref}`installation <cached-data>`.
```

## Pattern zoo

```{toctree}
:maxdepth: 1

notebooks/18_MNESIS_learn-periodic
notebooks/32_MNESIS_learn-travelling-waves
notebooks/34_MNESIS_learn-Lorenz-attractor
notebooks/36_MNESIS_learn-text
notebooks/36_MNESIS_learn-text-alice
notebooks/36_MNESIS_learn-text-alice-onehot
notebooks/38_MNESIS_learn-SHD
```

- {doc}`notebooks/18_MNESIS_learn-periodic` — builds periodic targets, trains
  periodic memories, and evaluates retrieval robustness under increasing input
  noise.
- {doc}`notebooks/32_MNESIS_learn-travelling-waves` — MotionClouds-based
  travelling-wave patterns; benchmarks retrieval with structured spatiotemporal
  motifs.
- {doc}`notebooks/34_MNESIS_learn-Lorenz-attractor` — encodes Lorenz chaotic
  trajectories into spike codes and evaluates memory recall on non-periodic
  continuous dynamics.
- {doc}`notebooks/36_MNESIS_learn-text` — tokenises a Wikipedia corpus
  (`tiktoken`, `cl100k_base`) into token IDs, builds a frequency-ordered
  **spike codebook** (most-frequent tokens get the sparsest neuron rows,
  minimising the Hamming distance), wraps it in a `TextSpikingPattern`
  generator, learns and recalls the resulting spike patterns, then decodes the
  output spikes back to text via the inverse codebook + tokenizer. Includes
  Optuna tuning and parameter scans.
- {doc}`notebooks/36_MNESIS_learn-text-alice` /
  {doc}`notebooks/36_MNESIS_learn-text-alice-onehot` — variants of the text
  experiment on a fixed Alice-in-Wonderland-style corpus (the one-hot version
  encodes tokens as one-hot spike patterns).
- {doc}`notebooks/38_MNESIS_learn-SHD` — Spiking Heidelberg Digits data
  loading/preprocessing for dataset-grounded experiments with real spiking
  recordings.

## Orchestrator

```{toctree}
:maxdepth: 1

notebooks/99_MNESIS_run-all
```

- {doc}`notebooks/99_MNESIS_run-all` — scripted orchestrator that runs the
  full notebook pipeline with progress timestamps (the SHD notebook is
  currently commented out there).

---

All figures cited above are generated by the notebooks into `figures/`
(git-ignored directory populated by the runs); the paper captions in
`tex/figure_*.tex` use the same colour convention as the raster legend — see
{doc}`installation` to reproduce them.
