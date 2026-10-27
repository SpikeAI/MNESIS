# MNESIS — Working Memory with Polychronous Chains

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Licence: GPL-3.0](https://img.shields.io/badge/Licence-GPL--3.0-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ICANN%202026-orange.svg)](https://laurentperrinet.github.io/publication/perrinet-26/)
[![Docs](https://img.shields.io/badge/Docs-spikeai.github.io%2FMNESIS-purple.svg)](https://spikeai.github.io/MNESIS/)

> **MNESIS** — *a Memory Network where Every Spike Is Sacred*

MNESIS is a recurrent spiking neural network (SNN) with heterogeneous synaptic delays that stores and recalls arbitrary motifs as sequential chains of overlapping Spiking Motifs. Every synapse carries $D$ learnable delays; a closed-form initialisation, obtained by deconvolving the LIF membrane response, reaches near-perfect recall before any gradient step, and surrogate-gradient training then adds robustness to noise.

This repository holds the implementation, the experiment notebooks and the LaTeX source of the paper.

**Paper:** Laurent U. Perrinet (2026). *Working Memory with Polychronous Chains*. ICANN 2026 · [publication page](https://laurentperrinet.github.io/publication/perrinet-26/) · arXiv [`2604.14096`](https://arxiv.org/abs/2604.14096).

---

## Documentation

**[https://spikeai.github.io/MNESIS/](https://spikeai.github.io/MNESIS/)**

| Page | Content |
|---|---|
| [Home](https://spikeai.github.io/MNESIS/) | Model equations, table of variables and defaults, analytical initialisation, training, results summary |
| [Installation & Quickstart](https://spikeai.github.io/MNESIS/installation.html) | Dependencies, running the notebooks, debug mode, cached data, building the paper and the docs |
| [Experiments & Notebooks](https://spikeai.github.io/MNESIS/experiments.html) | The annotated notebook pipeline and the figures each notebook produces |
| [API Reference](https://spikeai.github.io/MNESIS/api.html) | `mnesis_chains` (`HD_SNN`, pattern generators) and `mnesis_boilerplate` (`Params`, utilities) |

## Model

Each neuron $j$ is a leaky integrate-and-fire unit driven by the delayed spikes of all neurons:

$$u_j(t) = \beta \cdot u_j(t-1) \cdot (1 - s_j(t-1)) + \sum_{i=1}^{N} \sum_{d=1}^{D} W_{j,i,d} \cdot s_i(t-d)$$

where $u_j(t)$ is the membrane potential at time bin $t$, $s_j(t) \in \{0, 1\}$ the spike emitted when $u_j$ crosses the threshold $\vartheta$, $\beta$ the membrane decay per time bin, $N$ the number of neurons and $W_{j,i,d}$ the weight from neuron $i$ to neuron $j$ at delay $d \in \{1, \dots, D\}$. The variables, their defaults and the initialisation rules are described on the [documentation home page](https://spikeai.github.io/MNESIS/).

## Notebooks

The numbered notebooks in `src/` run from simple to complex; each one loads the cached artifacts of the previous one. The links below open the rendered version, with its outputs.

| # | Notebook | Experiment |
|---|---|---|
| 10 | [Generative model](https://spikeai.github.io/MNESIS/notebooks/10_MNESIS_generative-model.html) | Frozen and stochastic spiking motifs |
| 11 | [Learning synthetic motifs](https://spikeai.github.io/MNESIS/notebooks/11_MNESIS_learn-synthetic.html) | Analytical initialisation and training on $M = 16$ motifs |
| 13 | [Sequential retrieval](https://spikeai.github.io/MNESIS/notebooks/13_MNESIS_testing-inference.html) | Retrieval of the $M$ motifs in sequence |
| 14 | [Noise robustness](https://spikeai.github.io/MNESIS/notebooks/14_MNESIS_testing-noise.html) | Bit-flip noise in the trigger window, against the $F_1$ expected from theory |
| 15 | [Trigger duration](https://spikeai.github.io/MNESIS/notebooks/15_MNESIS_testing-trigger-duration.html) | Truncated trigger window |
| 16 | [Neuron coverage](https://spikeai.github.io/MNESIS/notebooks/16_MNESIS_testing-trigger-fraction.html) | Neurons silenced in the trigger |
| 18 | [Periodic motifs](https://spikeai.github.io/MNESIS/notebooks/18_MNESIS_learn-periodic.html) | Learning and retrieval of periodic memories |
| 20 | [Parameter scans](https://spikeai.github.io/MNESIS/notebooks/20_MNESIS_scanning-parameters.html) | One-at-a-time scans of the parameters |
| 25 | [Optuna optimisation](https://spikeai.github.io/MNESIS/notebooks/25_MNESIS_optuna.html) | Hyperparameter search |
| 99 | [Run all](https://spikeai.github.io/MNESIS/notebooks/99_MNESIS_run-all.html) | Orchestrator of the full pipeline |

## Quickstart

```bash
pip install -r src/requirements.txt   # Python 3.10+
cd src && jupyter notebook            # notebooks must be run from inside src/
```

See [Installation & Quickstart](https://spikeai.github.io/MNESIS/installation.html) for the debug mode, the cache and how to build the paper.

## Repository layout

| Path | Content |
|---|---|
| `src/` | Python modules (`mnesis_boilerplate.py`, `mnesis_chains.py`) and the numbered notebooks |
| `figures/` | Figures written by the notebooks and read by the paper |
| `tex/` | LaTeX source of the paper (build with `lualatex`) |
| `docs/` | Sphinx sources of the documentation site |
| `AGENTS.md` | Rules, commands and gotchas for developers and AI assistants |

---

## Citation

```bibtex
@inproceedings{Perrinet2026MNESIS,
  author       = {Perrinet, Laurent U.},
  title        = {Working Memory with Polychronous Chains},
  booktitle    = {ICANN 2026},
  year         = {2026},
  eprint       = {2604.14096},
  archivePrefix = {arXiv},
  doi          = {10.48550/arXiv.2604.14096},
  url          = {https://laurentperrinet.github.io/publication/perrinet-26/}
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## Licence

GPL-3.0 — see [LICENSE](LICENSE).

---

*An AI assistant was used to improve the readability and structure of this codebase — not to create it.*
*Institut de Neurosciences de la Timone (UMR 7289), Aix Marseille Université / CNRS, Marseille, France.*
*Supported by GENCI-IDRIS (Grant 2025–AD010314955R2).*
