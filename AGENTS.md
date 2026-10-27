# AGENTS.md — Developer & AI-Assistant Reference

## Project Summary

MNESIS is a recurrent spiking neural network (SNN) with heterogeneous synaptic
delays. The repository has two independent, buildable halves:

- **`src/`** — Python source and Jupyter experiment notebooks. Core code lives in
  two modules (`mnesis_boilerplate.py`, `mnesis_chains.py`); the numbered
  notebooks import from them and are run sequentially.
- **`tex/`** — LaTeX source of the paper (`Perrinet26mnesis.tex`, KOMA-Script
  `scrartcl` class), a small set of standalone TikZ figures, and a bibliography.

Around them:

- **`figures/`** — figures written by the notebooks (pdf/png/svg, versioned) and
  read by the paper.
- **`docs/`** — Sphinx sources of the documentation site (Markdown pages,
  `conf.py`, `Makefile`); built and deployed to GitHub Pages by
  `.github/workflows/static.yml` (see [Documentation site](#documentation-site)).
- **`CITATION.cff`** — citation metadata (preferred citation: the ICANN 2026
  paper, arXiv `2604.14096`); keep it in sync with the BibTeX entry of `README.md`
  and `docs/index.md`.
- **`cached_data/`** — git-ignored cache of weights and scan results.

This file gives **rules, commands, and gotchas** for agents working in the
repository; it is not a place for reference data. The full parameter catalogue,
per-experiment results, cache inventory and notebook inventory live in the
documentation (`docs/index.md`, `docs/installation.md`, `docs/experiments.md`) —
point there for "what value / what result", and to the paper
(`tex/Perrinet26mnesis.pdf`, built locally and git-ignored) for the camera-ready
numbers.

An AI assistant was used to improve the readability of the code, not to create
it; these rules guide how that formatting and testing was done.

---

## Architecture Overview

### Module Dependency Graph (`src/`)

```
notebooks (*.ipynb)
   └── mnesis_chains.py         (HD_SNN, SpikingPattern, StochasticSpikingPattern, load())
         └── mnesis_boilerplate.py    (imports, Params dataclass, utilities, SpikeF1scoreLoss)
               └── snntorch / torch / numpy / matplotlib / ...
```

| File | Role |
|---|---|
| `src/mnesis_boilerplate.py` | Entry-point imports (torch, snntorch, …), device auto-detection (MPS → CUDA → CPU), `DEBUG` / `RECOMPUTE` / `datetag` switches, `Params` dataclass (single source of hyperparameter truth), global matplotlib style (`STYLE`, Latin Modern to match the paper), utilities (`flip_bits`, `printfig`, `get_scores`, `SpikeF1scoreLoss`, `get_f1_ignition` (F1 expected from theory for a noisy, truncated or lesioned trigger: the replay is a cascade decided by its first evoked spike), `get_f1_cascade` (the same cascade, simulated), `get_recall_time` / `plot_recall_time` (recall of each motif against time), cosine LR schedule, `cache_lock` (context manager holding a cache file's `.lock`), `purge_unfinished_scans`). |
| `src/mnesis_chains.py` | Pattern generators (`SpikingPattern`, `StochasticSpikingPattern`) and the `HD_SNN` class (`forward_pass`, with `feedback=False` for the open-loop control, `get_W_init`, `update_weight`, `get_recall_time`, `learn_model`); also a `load()` helper for trained checkpoints. |

Notebooks (`src/*.ipynb`) are numbered simple → complex, each loading cached
artifacts from the previous one. They **import** from the two modules — they do
**not** carry standalone copies of `Params`, `HD_SNN`, etc. Notebooks 01, 05, and
08 (former `boilerplate`, `parameters`, `generative-model`) were refactored into
the two modules; any reference to those notebook numbers in old documents is stale.

Notebooks 32, 34, 36 (all its variants) and 38 were removed: they moved to the SCALA
project and are not in the ICANN paper. `figure_wave` and the Lorenz experiment were
removed as well; any reference to them is stale.

---

## `src/` — Python source code

### Install

```bash
pip install -r src/requirements.txt     # Python 3.10+; toolchain lives in .venv/
```

### Conventions

- Imports belong at the top of `mnesis_boilerplate.py`; downstream modules and
  notebooks re-export via `from mnesis_boilerplate import …`.
- Notebooks `from mnesis_boilerplate import …` / `from mnesis_chains import …`, so
  they must be **run from inside `src/`** (that is the expected CWD).
- `src/mnesis_boilerplate.py` `Params` is the authoritative hyperparameter source.
- `.venv/` is git-ignored; do not commit it.
- Figure/sandbox output goes to `../figures/` (relative to `src/`); on the Jean Zay
  cluster (`USER == "uvb28bo"`) figure saving is disabled automatically.
- `datetag` controls the cache-filename prefix — change it to start a new batch
  (current run: `2026-09-07`; as of 2026-10-27 no new `2026-10-26` run is planned).
- `seed = 2026` (in `Params`) is fixed for reproducibility.
- Cache keys must include every parameter that changes a result: a parameter left
  out of the key silently reuses a stale result.
- Repository text (code, comments, docstrings, documentation, commit messages) is
  written in British English (`colour`, `normalise`, `behaviour`), in SI units (cm,
  never inches), with ISO 8601 dates (`2026-10-27`) in file names and labels.
- `Params.fig_width` is the figure width in cm (`15.92`, the text width of the paper);
  `fig_height = fig_width / phi` (golden section).

### Debug mode

Set `DEBUG > 1` in `mnesis_boilerplate.py` to shrink the problem: `N_neuron`,
`N_pattern`, `N_time`, and `num_epochs` are divided by `DEBUG`. Use `DEBUG = 1` (the
default) only for production/camera-ready runs.

### Cache invalidation

Artifacts are cached in `cached_data/` (git-ignored). To recompute:
- delete the individual `.pth`, `.json`, `.npz`, or `.sqlite3` file, or
- set `RECOMPUTE = True` at the top of the notebook to invalidate the whole cache.

**File-based locking** (used by the scan/optimisation notebooks): a `.lock`
sentinel file guards against concurrent runs of the same scan.
1. If `RECOMPUTE=True`, delete both the data and lock files to force a restart.
2. Load existing results if present.
3. If no lock file exists, create one and start processing.
4. For each parameter value, skip it if already computed; otherwise run the scan and append to the JSON.
5. Delete the lock file when done.
Steps 3 and 5 go through `with cache_lock(filename) as acquired:` (`mnesis_boilerplate`):
the lock is created atomically and released in a `finally`, so an exception or an
interrupt no longer leaves a stale lock (notebook 20 does the same inline).
This avoids data corruption and allows safe incremental resumption — never delete
a `.lock` out from under a running scan. A scan reserves a placeholder row
(`loss=null`, `time=0.0`) before computing it; if the process dies, that row would
count as computed forever. `purge_unfinished_scans()` (run at the start of
notebook 99, skipping locked scans unless `remove_locks=True`) deletes those rows.
Only purge when no job is running (`squeue`): the row reserved by a live job looks
like an orphan, and purging it loses that job's result when it ends (or computes the
run twice). The reserved rows, not the lock, keep several jobs from computing the same
run, so that several jobs can share one scan once its lock is removed by hand.

**Jean Zay.** Scans are computed on the cluster, not on the Mac. To copy `cached_data/`
back, run `rsync` with `--exclude '*lock' --exclude '._*'` so that lock files and
AppleDouble `._*` files are not copied.

### Test / compile the Python side

There is **no unit-test framework** (no pytest), and the only CI workflow builds
the documentation (see [Documentation site](#documentation-site)), it does not
run the notebooks. The checks below *are* the tests; run the fast one first, then a representative notebook, and only the full
orchestrator when you need end-to-end verification.

```bash
cd src

# 1. Fast: modules import and build cleanly
python -c "import mnesis_boilerplate, mnesis_chains"

# 2. Representative: execute the smallest notebook (set DEBUG > 1 in mnesis_boilerplate.py first)
jupyter nbconvert --to notebook --execute --inplace 10_MNESIS_generative-model.ipynb

# 3. Full pipeline (slow): the orchestrator runs the whole notebook chain
jupyter nbconvert --to notebook --execute --inplace 99_MNESIS_run-all.ipynb
```

`nbconvert` exiting without error and writing updated outputs is a green build.

---

## `tex/` — LaTeX source code

### Files

| File | Role |
|---|---|
| `Perrinet26mnesis.tex` | Main paper (`\documentclass{scrartcl}`). `\input{metadata}` + biblatex (`\addbibresource{mnesis.bib}`, `\printbibliography`). Uses `fontspec` — **must be built with `lualatex`**, not `pdflatex`. |
| `metadata.tex` | `\input` fragment (author/affiliation/AI-statement macros); **no** `\documentclass`, do not compile standalone. |
| `tikz_izhikevich.tex`, `tikz_snntorch.tex` | Standalone TikZ figures (`\documentclass{standalone}`); compiled to their own PDFs in `tex/` (the PDFs are versioned). |
| `figure_*.tex` | `\input` figure environments (caption + `\label`). `figure_izhikevich.tex` and `figure_snntorch.tex` wrap the TikZ PDFs of `tex/` (`figure_snntorch.tex` also includes `pattern.pdf`); `figure_target.tex`, `figure_retrieval.tex`, `figure_robustness.tex` (→ `fig:noise`), `figure_scan.tex` and `figure_periodic.tex` (→ `fig:periodic`) take their graphics from `../figures/`. |
| `mnesis.bib` | Bibliography database. |

The main paper resolves experiment figures through `\graphicspath{{../figures/}}`
(in the preamble of `Perrinet26mnesis.tex`): `pattern.pdf`, `target.pdf`,
`retrieval.pdf`, the `*_score.pdf` curves and the `scan_*.pdf` bars are
**generated by the `src/` notebooks**, not built in `tex/`. The two TikZ figures
are built in `tex/`. Build everything from inside
`tex/`. Engine is `lualatex` (fontspec); toolchain: `lualatex`, `latexmk -pdflua`,
`biber`.

### Manuscript conventions

- Use the `\ShortTitle` macro (`metadata.tex`) for the running head and the title of the
  memory-capacity subsection, not the long title.
- Avoid first-person voice (`I`, `we`) where the passive or the model as subject reads
  naturally. Keep `we` when the passive is heavy, for example "We used the simplest LIF
  model with a fixed membrane decay $\beta$." The AI-use statement uses the plural `the authors`.
- Each sentence rewritten from `we` must read naturally in its paragraph: prefer
  "The trigger window was corrupted with ..." to "We corrupted the trigger window with ...",
  and check the result in context rather than applying the rule word for word.
- The paper's main claim is robust storage of many context windows, made possible by the
  heterogeneous delays and shown by the parameter scan. Compression is not the argument.
- Check an interpretation against per-motif data before it enters the text (for example,
  a flat average F1 can hide some motifs learned and others not).
- Wherever an average over motifs is stated, give the confidence interval and the number
  of repeats.
- Present results in order of effect size: the most important effect comes first.
- Remove obsolete commented-out blocks. A `TODO` is never deleted: when it is resolved, add
  a `% DONE <ISO date>: ...` line after it (`# DONE ...` in Python).

### Test / compile the LaTeX side

Full build (reproduces `tex/Perrinet26mnesis.pdf`):

```bash
cd tex

# 1. Standalone TikZ figures
lualatex -interaction=nonstopmode tikz_izhikevich.tex
lualatex -interaction=nonstopmode tikz_snntorch.tex

# 2. Experiment figures — must already exist in ../figures/
#    generate them by running the relevant notebooks in src/ (see above), e.g.
cd ../src && jupyter nbconvert --to notebook --execute --inplace 11_MNESIS_learn-synthetic.ipynb && cd ..

# 3. Build the paper (biber handled automatically by latexmk)
cd tex && latexmk -pdflua -pdf -interaction=nonstopmode Perrinet26mnesis.tex
```

If latexmk refuses ("gave an error in previous invocation"), `latexmk -g` or run
`lualatex` twice directly. Quick, figure-independent parse check (tolerates
missing `../figures/*.pdf`):

```bash
cd tex && lualatex -interaction=nonstopmode Perrinet26mnesis.tex
```

A clean run produces `tex/Perrinet26mnesis.pdf`. (A `Missing character: ^^V
(U+0016)` warning from the `dsrom`/microtype font is a benign known quirk, not a
control character in the sources.) Generated byproducts
(`.aux`, `.bbl`, `.blg`, `.log`, `.fls`, `.fdb_latexmk`, `.synctex.gz`, and the
`Perrinet26mnesis_copy.tex*` scratch copy) are git-ignored — never commit them.
The compiled `tex/Perrinet26mnesis.pdf` is git-ignored too.

---

## Documentation site

`docs/` holds the Sphinx sources only (`conf.py`, `Makefile`, `requirements.txt`,
and the pages `index.md`, `installation.md`, `api.md`, `experiments.md`, written
in MyST Markdown). `make -C docs html` symlinks `src/*.ipynb` into
`docs/notebooks/` and renders them with their stored outputs (`nbsphinx_execute =
"never"`), and builds the API pages from the docstrings (`autodoc` + `napoleon`,
Google style). `docs/_build/` and `docs/notebooks/` are git-ignored.

The workflow `.github/workflows/static.yml` runs that build on every push to `main`
and deploys `docs/_build/html` to GitHub Pages
(<https://spikeai.github.io/MNESIS/>); never commit built HTML.

- `docs/index.md` is the reference for the model equations, the variables table,
  the parameter drift note and the results. `README.md` stays short: it repeats
  only the membrane equation, the notebook list (linking to the rendered pages)
  and the BibTeX entry — keep those in sync with the docs.
- `docs/api.md` lists the documented members of `mnesis_boilerplate` explicitly:
  add any new public helper there.
- The CI installs its own dependency list (CPU-only `torch`, `snntorch`, …), not
  `docs/requirements.txt`: a new import in the modules must be added to both.
- `docs/` contains sources only: the site, notebook pages included, is generated by the
  CI workflow, so no generated or static HTML belongs in `docs/`.
- The documentation covers the whole repository: every notebook appears in the sidebar,
  numbered and linked to its rendered page, so that navigation keeps the notebook numbers.
  `README.md` points to the documentation instead of repeating it.

```bash
pip install -r docs/requirements.txt && make -C docs html   # green if Sphinx exits without error
```

---

## Notebook annotations

Each experiment notebook carries markdown documentation in a standard shape
(see notebooks 11 and 14–16 for models to follow). Notebooks 10 and 13 follow the
shape but lack the `<!-- MNESIS-annotated -->` marker; notebooks 18, 25 and 99 do
not follow it yet (18 has a summary without the H1 title, 25 starts with code).

- **H1 title** (`# ...`) as the first markdown cell, stating the experiment in
  one paragraph; end it with the `<!-- MNESIS-annotated -->` idempotency marker
  so annotation passes are not applied twice.
- **`## Notebook summary`** with exactly three `###` sub-sections:
  `### Pipeline` (numbered stages), `### Main outputs` (caches + figures),
  `### Notes` (gotchas; keep known bugs flagged here rather than silently
  fixing scan cells).
- One **`##` header per experimental stage** (`## Parameters`,
  `## Analytic initialisation of the readout`, `## Supervised learning of the
  readout`, `## Sweep: ...`), with `###` for raster legends, evaluation cells
  and animations. Match the heading style already used in 18/20-series
  notebooks (`## working memory with periodic targets`).
- Inline `#` code comments may be added, but **commenting must never change
  executable code** (verify by stripping comments and diffing against HEAD).
- Update a notebook by running it globally in a fresh kernel
  (`jupyter nbconvert --to notebook --execute --inplace <notebook>`), never cell by cell
  in a loaded kernel: the state must be clean and reproducible. Long results come from
  the cache, so only the changed parameters are recomputed.
- Never edit notebook outputs by hand; they come only from execution.
- Do not modify `mnesis_boilerplate.py` or `mnesis_chains.py` while a run imports them:
  an edit would change the code under a running job.

**Terminology (consistency with the paper):**

| Prefer | Avoid |
|---|---|
| *motif* (a stored spiking pattern, `N_pattern` of them) | "pattern" when referring to what is memorised |
| *trigger window* (first `num_delay` ms, clamped input) | "cue", "prompt" |
| *replay* / *re-emit* | "output the duration" |
| *evoked window* = `spikes[:, :, N_pretime+num_delay : N_time+N_pretime]` vs `frozen_target[:, :, num_delay:]` (the trigger comes from a seeded noisy realisation `pattern_object(seed=...)`, the score always from the frozen motif) | ad-hoc scoring slices, scoring against the noisy realisation |
| *spiking motif (SM) window* for the `num_delay`-wide history | "stimulus window" |
| *period* (`N_periods`, the number of periodic repetitions) | "chunk" (`N_chunks`, renamed) |
| *F1 score* (the reported score; `SpikeF1scoreLoss` is only the training objective) | "loss", when the figure shows F1 |
| *cascade* (replay decided by its first evoked spike, positive feedback) and *mean-field* (Poisson average), the two predicted theories | "compression" for what the network does: the claim is robust storage of many context windows, enabled by the delays |

**Raster legend (used by every retrieval notebook):** green `x` = input spikes
(spontaneous pad + trigger), open red `o` = target (hidden from the network
after the trigger), blue `+` = evoked/replayed spikes; red/blue dashed vlines
mark `N_pretime` and `N_pretime + num_delay`. The paper captions in
`figure_*.tex` use the same colours — keep both sides in sync when changing a
plot.

**Figures → generating notebook:**

| figure | notebook |
|---|---|
| `ssp_frozen_pattern`, `ssp_two_frozen_pattern` | 10 |
| `pattern`, `target_init`, `target`, `histogram_log` | 11 |
| `retrieval` (+ `.mp4`) | 13 |
| `p_flip_target`, `p_flip_score`, `p_flip_retrain_score`, `p_flip_cascade` | 14 |
| `trigger_time`, `trigger_time_score`, `trigger_time_cascade` | 15 |
| `fraction_target`, `fraction_target_score`, `fraction_target_cascade` | 16 |
| `scan_{key}` (one per scanned parameter) | 20 |
| `periodic_scores` | 18 |
| other `periodic_*` / `optuna` (displayed only, `figpath=None`) | 18 / 25 |

`printfig` overwrites existing files by default (`do_overwrite=True`) and saves
the exact `fig_width` x `fig_height` canvas (`bbox=None`, constrained layout), so
a figure prints at scale 1 when `\includegraphics` uses the same width.
Text in figures prints at the manuscript sizes (11 pt class): `STYLE` sets axis
labels to `normalsize` (10.95 pt) and ticks/legends to `small` (10 pt). The TikZ
figures are scaled by `\includegraphics`, so their fonts (`\papernormalsize`,
`\papersmall`) are divided by `\printscale` (printed width / natural width);
update `\printscale` in `tikz_izhikevich.tex` if its natural width changes.
`pattern` is pinned on the rows of `tikz_snntorch.tex` (`\xY`, `\tLabelY`, bounding
box): change both together.

**Figure layout.** Choose the width from the panel layout (for example one third of
`\textwidth` for a row of three panels). The height stays the golden-section height
(`fig_height = fig_width / phi`) whatever the layout, and the font sizes stay those of
the manuscript. Individual realisations are drawn as very thin lines behind the mean
curve, and the motif number is printed above each motif in figures showing several
motifs. Mark heavy graphic elements (`pcolormesh`, `contourf`, dense `scatter` or
`plot`, `eventplot`) with `rasterized=True`. In the paper, place each figure close to
the paragraph that first describes it, at balanced positions rather than clustered.

**Character hygiene:** no non-breaking/narrow spaces (U+00A0/U+202F), zero-width
characters, curly quotes or non-ASCII hyphens in `.py`, `.ipynb`, `.tex`
comments/docstrings — use plain ASCII there (legitimate em/en dashes in README
prose are fine). The `^^V (U+0016)` lualatex warning is the `dsrom` font quirk,
not a stray character.

---

## Conventions & gotchas

- **Parameter drift.** As of the reconciliation run `2026-09-07`, the paper
  methods text (β = 0.7, τ ≈ 2.8 steps; ϑ = 0.75, δ = 0.25; p_flip = 0.01;
  FastSigmoid α = 5; adadelta; η 40→0.4 mLR; dropout 0.6) matches the
  `Params` production defaults in `mnesis_boilerplate.py`. Older documents
  may still cite camera-ready values (β = 0.8, α = 12, SGD, dropout 0.10);
  robustness results (Fig. `fig:noise`) keep their own run's numbers. **Do not
   "autocorrect" code defaults to match stale prose** — consult `docs/index.md` /
   `tex/Perrinet26mnesis.pdf` for the exact value a given figure used.
  Each notebook may also override defaults with its own local `opt_dict`.
- **Cache protocol change.** The caches of `datetag = 2026-09-07` predate the
  seeded-cue / frozen-target protocol (2026-10-26): their weights were initialised
  on a noisy realisation and trained with unseeded cues, and their scan rows were
  scored against an unseeded noisy realisation. The cache key is the `datetag`
  alone, so rerunning with it reuses (or appends to) these results. The seed
  streams changed again on 2026-10-26 at 18:54 (commit `5674e00`): `flip_bits`
  (stream 1) and the spontaneous pad of `get_input_spikes` (stream 2) now derive
  their generator state from the seed, which before collided with the frozen
  motif (the cue lost every spike of the motif on CPU, and the pad copied its
  first entries on every device). Any cache written before that time holds stale cues and pads. The module
  deliberately keeps running with `datetag = '2026-09-07'`
  (`mnesis_boilerplate.py`): **do not bump it** on your own initiative. The
  `2026-10-26` line next to it stays commented out, and 2026-10-26 is the date of
  the protocol change, not the current batch; uncommenting that line is what
  recomputes everything consistently, when the maintainer asks for it.
- **Bibliography toolchain.** The paper `.tex` file was migrated from `natbib`/`bibtex` to `biblatex`/`biber` (updates to `Perrinet26mnesis.tex`: replace `\usepackage{natbib}` with `\usepackage[style=numeric, sorting=none]{biblatex}` + `\addbibresource{mnesis.bib}`, and replace `\bibliographystyle{plainnat}\bibliography{mnesis}` with `\addbibresource{mnesis.bib}\printbibliography`; bibliography line `%!BIB TS-program = bibtex` → `%!BIB TS-program = biber). Do not revert these changes.
- The analytical initialisation in `HD_SNN.get_W_init` is controlled by two flags:
  `do_pinv` (default `True`; `False` switches to the Hebbian cross-correlation rule)
  and `do_deconv` (default `True`; LIF-membrane deconvolution of the target).
- `num_delay` has **no parity constraint**: the odd-value rule came from the
  former centred `conv1d` generator (removed in May 2026), whereas the causal
  delay windows of `forward_pass` and `get_W_init` accept any positive value
  (the `num_delay` scan of notebook 20 includes even values).
- Keep notebooks' shared code **in the two modules**, not inlined — notebooks are
  meant to stay import-only.

---

## Review and commits

- After changing a default value, an equation or a name, run `/harmonize-code` before
  committing. It checks comments, docstrings, notebook markdown, README, documentation,
  cache keys and unused imports, and reports the inconsistencies for approval.
- Commit with `/commit-session`: only the paths changed in the session, with the message
  and the file list previewed and validated first. The message is in English, the title is
  imperative and at most 72 characters, the body says why, and no `Co-Authored-By` or other
  attribution line is added.
- Never commit `.venv/`, `cached_data/`, LaTeX byproducts, the compiled paper PDF or built HTML.
