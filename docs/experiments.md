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
  $M = 16$ synthetic sparse motifs. Demonstrates that the analytical init
  alone reaches high $F_1$; gradient training with a cosine schedule then
  improves noise robustness. Figures: `pattern`, `target_init`, `target`,
  `histogram_log`.
- {doc}`NB 13: Sequential retrieval <notebooks/13_MNESIS_testing-inference>` — concatenates all $M = 16$
  motifs in sequence, separated by silent inter-trial intervals
  ($4 N_\mathrm{pretime} = 200$ steps on each side, no spontaneous activity); evaluates the $F_1$ score in consecutive
  60-step windows against the frozen motifs: 9 of the 16 motifs are replayed exactly ($F_1 = 1$), 4 stop at a gap of $D$
  silent steps and 3 are over-active (mean $F_1$ 0.79–0.84 over realisations, 0.96 without the 4 motifs with a gap). Figure: `retrieval` (plus `.mp4`).

## Robustness tests

In the three score figures, thin lines show the $F_1$ score of each motif
(averaged over the 11 seeded realisations of each value): under the deterministic perturbations
(truncation, lesion) each motif switches between zero and its intact value at
its own threshold, the all-or-nothing ignition of the replay.

- {doc}`NB 14: Noise robustness <notebooks/14_MNESIS_testing-noise>` — bit-flip noise on the trigger
  window ($p_\mathrm{flip} \in [0, 1]$; 14 values, 11 seeded realisations each), with
  fixed weights. The fine-tuned control (`noise_retrain`) has no saved result yet:
  only its lock file exists. The paper reports
  $F_1 \approx 0.53$ at $p_\mathrm{flip} = 0.25$, below 0.1 beyond $0.6$. Both
  score figures overlay the $F_1$ expected from theory for a motif-wise
  ignition of the replay. The replay is a cascade, a positive feedback loop
  decided by its first evoked spike, which fires only if a fraction
  $\vartheta$ of the $\approx p_A N D$ trigger spikes supporting it survive
  (`get_f1_ignition`, with no fitted parameter); a simulation of the whole
  cascade (`get_f1_cascade`) gives the same curve. The recall of each motif
  against time shows the bistability (ignited or silent), and an open-loop
  control (`forward_pass(..., feedback=False)`) shows that the replay is
  sustained by the recurrence. Figures: `p_flip_target`, `p_flip_score`,
  `p_flip_retrain_score`, `p_flip_cascade`.
- {doc}`NB 15: Trigger duration <notebooks/15_MNESIS_testing-trigger-duration>` — truncated trigger
  window (41 values of `N_trigger_time`, 11 seeded realisations each); finds the
  minimum trigger-window length for reliable recall. The paper reports
  $F_1 \approx 0.79$ at 75% of $D$. The same cascade model, with the last
  `N_trigger_time` bins of the window shown, predicts the curve: the support of
  the first evoked spike is more recent than a whole window, and so better
  shown than the fraction `N_trigger_time / num_delay`. Figures: `trigger_time`,
  `trigger_time_score`, `trigger_time_cascade`.
- {doc}`NB 16: Neuron coverage <notebooks/16_MNESIS_testing-trigger-fraction>` — partial neuron coverage
  (0 to $N$ neurons silenced in the trigger; 14 values, 11 seeded realisations
  each). The paper reports
  $F_1 \approx 0.76$ with 87.5% of neurons active. The same cascade model, with
  a fraction `1 - n / N` of the neurons shown, predicts the curve. Figures:
  `fraction_target`, `fraction_target_score`, `fraction_target_cascade`.

## Parameter scans & optimisation

- {doc}`NB 20: Parameter scans <notebooks/20_MNESIS_scanning-parameters>` — systematic one-at-a-time
  scans over `N_pattern` ($M$), `N_time` ($T$), `num_delay` ($D$), `N_neuron`
  ($N$), `p_A`, `lif_threshold` ($\vartheta$), `p_flip`, and the `do_deconv` /
  `do_pinv` flags, with $10$ independent runs (random seeds) per value; produces the
  `scan_{key}` figures, of which `scan_num_delay`, `scan_N_neuron`, `scan_N_time`
  and `scan_N_pattern` form the figure of the paper (the $p_A$ scan is reported in
  its text only). All the scans are complete: 940 runs, in the files
  `{datetag}_scan_{key}.json`.
- {doc}`NB 25: Optuna optimisation <notebooks/25_MNESIS_optuna>` — automated hyperparameter search with
  [Optuna](https://optuna.org/) (multivariate TPE, 200 trials of 128 epochs launched: 197 completed, 3 still marked
  as running in the study database)
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
  noise. Its transmission-failure sweep covers 5 values of $p_\mathrm{drop}$, with 5
  seeded realisations each, over 100 periods. Only `periodic_scores` ($F_1$ per period of free replay against time,
  one period of 959 steps lasting 0.959 s; one curve per $p_\mathrm{flip}$) is
  saved, for the paper; the other `periodic_*` figures are
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
