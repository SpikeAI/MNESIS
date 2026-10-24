# MNESIS — Working Memory in a Recurrent Spiking Neural Network with Heterogeneous Synaptic Delays

```{eval-rst}
.. image:: https://img.shields.io/badge/Python-3.10%2B-blue.svg
   :target: https://www.python.org/downloads/
.. image:: https://img.shields.io/badge/License-GPL--3.0-green.svg
   :target: https://github.com/laurentperrinet/MNESIS/blob/main/LICENSE
.. image:: https://img.shields.io/badge/Paper-ICANN%202026-orange.svg
   :target: https://laurentperrinet.github.io/publication/perrinet-26-icann/
```

> **MNESIS** — *a Memory Network where Every Spike Is Sacred*

This documentation covers the full implementation and experiments of **MNESIS**, a
recurrent spiking neural network (SNN) with heterogeneous synaptic delays that stores
and recalls arbitrary spike patterns as sequential chains of overlapping Spiking Motifs.

## Overview

Working memory in biological neural circuits relies on precise spike timing rather than
sustained firing rates. MNESIS models this by equipping every synapse with $D$ learnable
delays, parameterised as a single weight tensor $\mathbf{W} \in \mathbb{R}^{N \times N \times D}$.
Each stored pattern is encoded as a chain of overlapping **Spiking Motifs**: contiguous
context windows of length $D$ that uniquely predict the next time step of activity.
A closed-form initialisation derived by deconvolving the LIF membrane response reaches
near-perfect recall before any gradient step, while surrogate-gradient BPTT then adds
robustness to noise.

The same `HD_SNN` framework is applied to sparse synthetic motifs and to periodic
motifs. The experiments build on one another: each numbered notebook imports the
shared infrastructure and loads the cached artifacts produced by the previous one.

**Paper:** Laurent U. Perrinet (2026). *Working Memory in a Recurrent Spiking Neural
Network with Heterogeneous Synaptic Delays*. ICANN 2026.
[`https://laurentperrinet.github.io/publication/perrinet-26-icann/`](https://laurentperrinet.github.io/publication/perrinet-26-icann/)

## Model

### Membrane dynamics

Each neuron $j$ evolves as:

$$u_j(t) = \beta \cdot u_j(t-1) \cdot (1 - s_j(t-1)) + \sum_{i=1}^{N} \sum_{d=1}^{D} W_{j,i,d} \cdot s_i(t-d)$$

### Analytical initialisation

The LIF membrane is a causal IIR lowpass. Two orthogonal flags control the closed-form
initialisation in {meth}`mnesis_chains.HD_SNN.get_W_init`:

- **`do_deconv`** (default `True`) — deconvolve the membrane so the input current that
  drives neuron $j$ to a target spike at time $t$ is

  $$I^*_j(t) = \vartheta_0 \bigl(s^*_j(t) - \beta \cdot s^*_j(t-1)\bigr).$$

  Here $\vartheta_0$ is the firing threshold (`lif_threshold`). With `do_deconv = False` the raw target $s^*_j(t)$ is used directly.

- **`do_pinv`** (default `True`) — solve for $\mathbf{W}$ via the (numerically stable,
  computed on CPU) full pseudo-inverse $\mathbf{W} = \mathrm{pinv}(\mathbf{C})\,\mathbf{T}$.
  With `do_pinv = False` a Hebbian cross-correlation rule is used instead, relying on the
  Gram-matrix approximation $\mathbf{C}\mathbf{C}^\top \approx N D p_A \mathbf{I}$:

  $$w_{i,j,d} = \frac{1}{N \cdot D \cdot p_A \cdot M} \sum_{\mu,\,t} s_i^{*(\mu)}(t-d) \cdot \bigl(s_j^{*(\mu)}(t) - \beta \cdot s_j^{*(\mu)}(t-1)\bigr).$$

### Training

- **Loss**: $\mathcal{L} = 1 - F_1$ (harmonic mean of precision and recall, evaluated
  after the trigger window) — {class}`mnesis_boilerplate.SpikeF1scoreLoss`
- **Optimiser**: configurable via {attr}`mnesis_boilerplate.Params.optimizer`
  (`adam`, `adamw`, `sgd`, `rmsprop`, `adadelta`, …)
- **Schedule**: cosine decay with a `num_warmup_epochs` warmup ramp
- **Surrogate**: fast sigmoid by default, sharpness $\alpha$ configurable in `Params`
- **Hardware**: Apple silicon (MPS) or NVIDIA GPU (CUDA / Jean Zay GENCI)

```{note}
**Parameter drift.** The `Params` defaults are the *latest camera-ready run* (see the
`datetag`) and have moved on from the values in the paper text and earlier revisions.
Each notebook also sets its own `opt_dict`. The code is authoritative for *what the
current default run does*; consult the specific `opt_dict` and the paper PDF for the
exact value a given figure used.
```

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
| Periodic memory (NB 30) | Stable periodic retrieval under progressive bit-flip noise |

## Citation

```bibtex
@inproceedings{Perrinet2026MNESIS,
  author        = {Perrinet, Laurent U.},
  title         = {Working Memory in a Recurrent Spiking Neural Network
                  with Heterogeneous Synaptic Delays},
  booktitle     = {ICANN 2026},
  year          = {2026},
  url           = {https://laurentperrinet.github.io/publication/perrinet-26-icann/}
}
```

```{toctree}
:maxdepth: 2
:caption: Contents

installation
api
experiments
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
