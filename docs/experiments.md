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

- {doc}`NB 10: Generative model <notebooks/10_MNESIS_generative-model>` — defines the motifs every
  downstream notebook memorises: {class}`~mnesis_chains.SpikingPattern` draws a
  frozen Bernoulli raster of rate $p_A$ (seeded with `seed`), and
  {class}`~mnesis_chains.StochasticSpikingPattern` returns realisations of it with
  balanced bit flips of probability $p_\mathrm{flip}$. Figures:
  `ssp_frozen_pattern`, `ssp_two_frozen_pattern`.

## Learning and retrieval on synthetic motifs

- {doc}`NB 11: Learning synthetic motifs <notebooks/11_MNESIS_learn-synthetic>` — trains the network on
  $M = 16$ synthetic sparse patterns. Demonstrates that the analytical init
  alone reaches high $F_1$; gradient training with a cosine schedule then
  improves noise robustness. Figures: `pattern`, `target_init`, `target`,
  `histogram_log`.
- {doc}`NB 13: Sequential retrieval <notebooks/13_MNESIS_testing-inference>` — concatenates all $M = 16$
  patterns in sequence, separated by silent inter-trial intervals
  ($4 N_\mathrm{pretime} = 200$ steps on each side, no spontaneous activity); evaluates sliding-window $F_1$ to confirm selective,
  cross-interference-free retrieval. Figure: `retrieval` (plus `.mp4`).

## Robustness tests

- {doc}`NB 14: Noise robustness <notebooks/14_MNESIS_testing-noise>` — bit-flip noise on the trigger
  window ($p_\mathrm{flip} \in [0, 1]$), with fixed weights and, as a control,
  with the network fine-tuned at each noise level. The paper reports
  $F_1 \approx 0.63$ at $p_\mathrm{flip} = 0.25$, chance only near $0.5$. Both
  score figures overlay the $F_1$ expected from theory for a motif-wise
  ignition of the replay, which starts only if a fraction
  $\vartheta$ of the $\approx p_A N D$ trigger spikes supporting it survive
  (`get_f1_ignition`, with no fitted parameter).
  Figures: `p_flip_target`, `p_flip_score`, `p_flip_retrain_score`.
- {doc}`NB 15: Trigger duration <notebooks/15_MNESIS_testing-trigger-duration>` — truncated trigger
  window; finds the minimum cue length for reliable recall. The paper reports
  $F_1 \approx 0.42$ at 75% of $D$. Figures: `trigger_time`,
  `trigger_time_score`.
- {doc}`NB 16: Neuron coverage <notebooks/16_MNESIS_testing-trigger-fraction>` — partial neuron coverage
  (0 to $N$ neurons silenced in the trigger). The paper reports
  $F_1 \approx 0.60$ with 87.5% of neurons active. Figures: `fraction_target`,
  `fraction_target_score`.

## Parameter scans & optimisation

- {doc}`NB 20: Parameter scans <notebooks/20_MNESIS_scanning-parameters>` — systematic one-at-a-time
  scans over `N_pattern` ($M$), `N_time` ($T$), `num_delay` ($D$), `N_neuron`
  ($N$), `p_A`, `lif_threshold` ($\vartheta$), `p_flip`, and the `do_deconv` /
  `do_pinv` flags, with $N_\mathrm{cv} = 10$ seeds per value; produces the
  `scan_{key}` figures, of which `scan_num_delay`, `scan_N_time` and
  `scan_N_neuron` appear in the paper.
- {doc}`NB 25: Optuna optimisation <notebooks/25_MNESIS_optuna>` — automated hyperparameter search with
  [Optuna](https://optuna.org/) (multivariate TPE, 200 trials of 128 epochs)
  over the initialisation flags, the optimiser, the LIF parameters ($\beta$,
  $\vartheta$, reset mechanism, learnt or not), the surrogate sharpness, the
  learning rates and the dropout rate. Its figure is displayed, not saved.

```{note}
The scan/optimisation notebooks use file-based locking: a `.lock` sentinel
guards against concurrent runs of the same scan, computed parameter values are
appended incrementally to the JSON, and runs can be safely resumed. See
{ref}`installation <cached-data>`.
```

## Periodic motifs

- {doc}`NB 18: Periodic motifs <notebooks/18_MNESIS_learn-periodic>` — builds periodic targets, trains
  periodic memories, and evaluates retrieval robustness under increasing input
  noise. Only `periodic_scores` ($F_1$ per period of free replay, one curve per
  $p_\mathrm{flip}$) is saved, for the paper; the other `periodic_*` figures are
  displayed, not saved.

## Orchestrator

- {doc}`NB 99: Run all <notebooks/99_MNESIS_run-all>` — scripted orchestrator: purges the
  placeholder rows of interrupted scans
  ({func}`~mnesis_boilerplate.purge_unfinished_scans`), then runs the full
  notebook pipeline (10 → 25) with progress timestamps.

---

All saved figures cited above are written by the notebooks into `figures/` (in
pdf, png and svg, versioned in the repository) through
{func}`~mnesis_boilerplate.printfig`, which overwrites existing files by default
at the exact size the paper prints them. The paper reads them from there; its captions in `tex/figure_*.tex`
use the same colour convention as the raster legend — see {doc}`installation` to
reproduce them.
