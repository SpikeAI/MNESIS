# MNESIS — Working Memory with Polychronous Chains

```{eval-rst}
.. image:: https://img.shields.io/badge/Python-3.10%2B-blue.svg
   :target: https://www.python.org/downloads/
.. image:: https://img.shields.io/badge/License-GPL--3.0-green.svg
   :target: https://github.com/SpikeAI/MNESIS/blob/main/LICENSE
.. image:: https://img.shields.io/badge/Paper-preprint-orange.svg
   :target: https://laurentperrinet.github.io/publication/perrinet-26/
```

> **MNESIS** — *a Memory Network where Every Spike Is Sacred*

This documentation covers the full implementation and experiments of **MNESIS**, a
recurrent spiking neural network (SNN) with heterogeneous synaptic delays that stores
and recalls arbitrary motifs as sequential chains of overlapping Spiking Motifs.

## Overview

Working memory in biological neural circuits relies on precise spike timing rather than
sustained firing rates. MNESIS models this by equipping every synapse with $D$ learnable
delays, parameterised as a single weight tensor $\mathbf{W} \in \mathbb{R}^{N \times N \times D}$.
Each stored motif is encoded as a chain of overlapping **Spiking Motifs**: contiguous
context windows of length $D$ that uniquely predict the next time step of activity.
A closed-form initialisation derived by deconvolving the LIF membrane response reaches
near-perfect recall before any gradient step, while surrogate-gradient BPTT then adds
robustness to noise.

The same `HD_SNN` framework is applied to sparse synthetic motifs and to periodic
motifs. The experiments build on one another: each numbered notebook imports the
shared infrastructure and loads the cached artifacts produced by the previous one.

**Paper:** Laurent U. Perrinet (2026). *Working Memory with Polychronous Chains*.
Submitted preprint.
[`https://laurentperrinet.github.io/publication/perrinet-26/`](https://laurentperrinet.github.io/publication/perrinet-26/)

### Interactive raster

An interactive spike raster (space: pause, click a spike while paused to highlight the
spikes of the preceding context window, `s`: shade that window).

```{raw} html
<p><a href="izhikevich.html" target="_blank" rel="noopener">Open it in its own page</a>.</p>
<iframe src="izhikevich.html" title="Interactive spike raster"
        style="width:100%; height:32em; border:1px solid #888; border-radius:6px;"
        loading="lazy" allowfullscreen></iframe>
```

## Model

### Membrane dynamics

Each neuron $j$ evolves as:

$$u_j(t) = \beta \cdot u_j(t-1) \cdot (1 - s_j(t-1)) + \sum_{i=1}^{N} \sum_{d=1}^{D} W_{j,i,d} \cdot s_i(t-d)$$

where $u_j(t)$ is the membrane potential of neuron $j$ at time bin $t$, $s_j(t) \in \{0, 1\}$
its spike (emitted when $u_j(t)$ crosses the threshold $\vartheta$, after which the membrane is
reset to zero), $\beta$ the membrane decay per time bin, and $W_{j,i,d}$ the weight from neuron
$i$ to neuron $j$ at delay $d$. In the code, $\mathbf{W}$ is the `lin` layer of
{class}`~mnesis_chains.HD_SNN`, of shape `(N_neuron, N_neuron * num_delay)`. During training,
dropout of probability $p$ is applied to the post-synaptic current.

The motifs are drawn by {class}`~mnesis_chains.SpikingPattern` as a frozen Bernoulli raster of
rate $p_A$; {class}`~mnesis_chains.StochasticSpikingPattern` returns realisations of it in which
each bit is redrawn with probability $p_\mathrm{flip}$ (balanced bit flips that preserve the
marginal rate).

### Variables

Defaults are those of the {class}`~mnesis_boilerplate.Params` dataclass; one time bin is 1 ms.

| Symbol | Code name | Default | Role |
|---|---|---|---|
| $N$ | `N_neuron` | 1024 | number of neurons |
| $D$ | `num_delay` | 41 time bins | number of delays per synapse, i.e. length of a spiking motif |
| $M$ | `N_pattern` | 16 | number of stored motifs |
| $T$ | `N_time` | 1000 time bins | duration of each motif |
| $N_\mathrm{pretime}$ | `N_pretime` | 50 time bins | spontaneous activity before the stimulus |
| $p_A$ | `p_A` | 0.00016 per time bin | prior probability of firing |
| $p_\mathrm{flip}$ | `p_flip` | 0.01 | probability of flipping a bit in the stochastic pattern generator |
| — | `seed` | 2026 | seed of the random number generators |
| $\beta$ | `lif_beta` | 0.70 | membrane decay per time bin |
| $\vartheta$ | `lif_threshold` | 0.75 (membrane potential, a.u.) | firing threshold |
| — | `learn_beta`, `learn_threshold` | `False` | make $\beta$ and $\vartheta$ learnable |
| — | `do_pinv`, `do_deconv` | `True` | pseudo-inverse solution and membrane deconvolution of the initialisation |
| — | `num_epochs` | 256 | number of training epochs |
| — | `num_warmup_epochs` | 16 | epochs of the learning-rate warmup ramp |
| — | `base_lr`, `final_lr` | 0.04, 0.0004 | learning rates at the peak and at the end of the cosine schedule |
| $\delta_1$, $\delta_2$ | `delta1`, `delta2` | 0.003, 0.00001 | complements to 1 of the optimiser coefficients (momentum, decay) |
| $p$ | `dropout` | 0.60 | dropout probability |
| $\alpha$ | `alpha_surrogate` | 5.0 | sharpness of the surrogate gradient (`FastSigmoid`) |
| — | `N_scan`, `N_cv` | 14, 11 | number of values per scan, number of seeded repetitions per value: 11 in notebooks 14–16 (the default); notebook 20 sets `N_cv = 10` locally |

### Analytical initialisation

The LIF membrane is a causal IIR lowpass. Two orthogonal flags control the closed-form
initialisation in {meth}`mnesis_chains.HD_SNN.get_W_init`:

- **`do_deconv`** (default `True`) — deconvolve the membrane so the input current that
  drives neuron $j$ to a target spike at time $t$ is

  $$I^*_j(t) = s^*_j(t) - \beta \cdot s^*_j(t-1).$$

  A perfect context then drives the membrane to $1$, a margin $\delta = 1 - \vartheta = 0.25$
  above the firing threshold $\vartheta$ (`lif_threshold`). With `do_deconv = False` the raw target $s^*_j(t)$ is used directly.

- **`do_pinv`** (default `True`) — solve for $\mathbf{W}$ via the (numerically stable,
  computed on CPU) full pseudo-inverse $\mathbf{W} = \mathrm{pinv}(\mathbf{C})\,\mathbf{T}$.
  With `do_pinv = False` a Hebbian cross-correlation rule is used instead, relying on the
  Gram-matrix approximation $\mathbf{C}^\top\mathbf{C} \approx N D p_A \mathbf{I}$
  ($\mathbf{C} \in \{0,1\}^{ND \times K}$ stacks the $K$ contexts as columns):

  $$w_{i,j,d} = \frac{1}{N \cdot D \cdot p_A} \sum_{\mu,\,t} s_i^{*(\mu)}(t-d) \cdot \bigl(s_j^{*(\mu)}(t) - \beta \cdot s_j^{*(\mu)}(t-1)\bigr).$$

### Training

- **Loss**: $\mathcal{L} = 1 - F_1$ (harmonic mean of precision and recall, evaluated
  after the trigger window) — {class}`mnesis_boilerplate.SpikeF1scoreLoss`
- **Optimiser**: Adadelta by default, configurable via
  {attr}`mnesis_boilerplate.Params.optimizer` (also `sgd`, `adam`, `adamw`, `rmsprop`)
- **Schedule**: cosine decay with a `num_warmup_epochs` warmup ramp (`final_lr` → `base_lr`)
- **Surrogate**: fast sigmoid by default, sharpness $\alpha = 5.0$ — configurable via
  `Params.surrogate_name` / `Params.alpha_surrogate`
- **Regularisation**: dropout $p = 0.6$
- **LIF dynamics**: $\beta = 0.7$, $\vartheta = 0.75$, zero-reset (`zero`)
- **Hardware**: Apple silicon (MPS) or NVIDIA GPU (CUDA / Jean Zay GENCI)

```{note}
**Parameter drift.** As of the reconciliation run `2026-09-07` (the current `datetag`),
the `Params` defaults ($\beta = 0.7$, $\vartheta = 0.75$, $\alpha = 5$, `adadelta`,
dropout $0.6$) match the methods text of the paper. Earlier documents and the
camera-ready runs cite other values ($\beta = 0.8$, $\alpha = 12$, SGD, dropout $0.10$).
Each notebook also sets its own `opt_dict`. The code is authoritative for *what the
current default run does*; consult the specific `opt_dict` and the paper PDF for the
exact value a given figure used.
```

## Results summary

The values below are those quoted in the paper. The scans of notebook 20 are
complete in the caches of the current `datetag` (940 runs: 10 per value, over the
grids $D \in [1, 60]$, $T \in [100, 1558]$, $N \in [50, 2000]$, $M \in [1, 26]$ and
$p_A \in [2 \times 10^{-4}, 5 \times 10^{-2}]$), and the means of these runs are the
scores quoted.

| Experiment | Key result (paper) |
|---|---|
| Training ({doc}`NB 11 <notebooks/11_MNESIS_learn-synthetic>`) | High $F_1$ with analytical init alone; gradient training adds noise robustness |
| Sequential retrieval ({doc}`NB 13 <notebooks/13_MNESIS_testing-inference>`) | 9 of the 16 motifs replayed exactly ($F_1 = 1$), 4 stop at a gap of $D$ silent steps, 3 are over-active; mean $F_1$ 0.79–0.84 over realisations (0.96 without the 4 motifs with a gap) |
| Noise robustness ({doc}`NB 14 <notebooks/14_MNESIS_testing-noise>`) | $F_1 \approx 0.53$ at $p_\mathrm{flip} = 0.25$; below 0.1 beyond $p_\mathrm{flip} \approx 0.6$; follows a motif-wise ignition model of the replay with no fitted parameter |
| Trigger duration ({doc}`NB 15 <notebooks/15_MNESIS_testing-trigger-duration>`) | $F_1 \approx 0.79$ at 75% of $D$; rises steeply above $D/2$; follows the cascade model of the replay with no fitted parameter |
| Neuron coverage ({doc}`NB 16 <notebooks/16_MNESIS_testing-trigger-fraction>`) | $F_1 \approx 0.76$ with 87.5% of neurons active in trigger |
| Delay scan ({doc}`NB 20 <notebooks/20_MNESIS_scanning-parameters>`) | $F_1 \approx 0$ for $D \le 15$; $F_1 \approx 0.86$ at $D=40$; $F_1 > 0.95$ from $D=50$ |
| Size scan ({doc}`NB 20 <notebooks/20_MNESIS_scanning-parameters>`) | $F_1 \approx 0$ up to $N \approx 430$; $F_1 \approx 0.93$ at $N=1081$; $F_1 \approx 0.97$ from $N \approx 1500$; $D$ and $N$ act through $p_A N D$ |
| Duration scan ({doc}`NB 20 <notebooks/20_MNESIS_scanning-parameters>`) | $F_1 \approx 0.97$ at $T=100$; $F_1 \approx 0.88$ at $T=945$; $F_1 \approx 0.78$ at $T=1558$ |
| Motif-number scan ({doc}`NB 20 <notebooks/20_MNESIS_scanning-parameters>`) | $F_1$ between 0.84 and 0.94, with no trend, from $M=3$ to $M=26$ |
| Rate scan ({doc}`NB 20 <notebooks/20_MNESIS_scanning-parameters>`) | $F_1 \approx 0.98$ for $p_A \in [3\times10^{-4}, 3\times10^{-3}]$; $F_1 \approx 0.94$ at $2\times10^{-2}$; $F_1 \approx 0.69$ at $p_A \approx 3\times10^{-2}$ |
| Periodic memory ({doc}`NB 18 <notebooks/18_MNESIS_learn-periodic>`) | Replay sustained for 30 periods without failures ($F_1 \approx 0.83$ from the second period on; 12 of 16 motifs re-trigger themselves); with transmission failures $p_\mathrm{drop}$, the lifetime falls from $\approx 280$ periods at $10^{-4}$ to $\approx 2$ at $10^{-2}$ |

## Citation

```bibtex
@misc{Perrinet2026MNESIS,
  author        = {Perrinet, Laurent U.},
  title         = {Working Memory with Polychronous Chains},
  year          = {2026},
  eprint        = {2604.14096},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2604.14096},
  url           = {https://laurentperrinet.github.io/publication/perrinet-26/}
}
```

Machine-readable citation metadata is in
[`CITATION.cff`](https://github.com/SpikeAI/MNESIS/blob/main/CITATION.cff).

```{toctree}
:maxdepth: 2
:caption: Contents

installation
api
experiments
```

```{toctree}
:maxdepth: 1
:caption: Notebooks

NB 10: Generative model <notebooks/10_MNESIS_generative-model>
NB 11: Learning synthetic motifs <notebooks/11_MNESIS_learn-synthetic>
NB 13: Sequential retrieval <notebooks/13_MNESIS_testing-inference>
NB 14: Noise robustness <notebooks/14_MNESIS_testing-noise>
NB 15: Trigger duration <notebooks/15_MNESIS_testing-trigger-duration>
NB 16: Neuron coverage <notebooks/16_MNESIS_testing-trigger-fraction>
NB 18: Periodic motifs <notebooks/18_MNESIS_learn-periodic>
NB 20: Parameter scans <notebooks/20_MNESIS_scanning-parameters>
NB 25: Optuna optimisation <notebooks/25_MNESIS_optuna>
NB 99: Run all <notebooks/99_MNESIS_run-all>
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
