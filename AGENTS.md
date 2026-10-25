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
per-experiment results, cache inventory and notebook inventory live in
`README.md` — point there for "what value / what result", and to the paper
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
| `src/mnesis_boilerplate.py` | Entry-point imports (torch, snntorch, …), device auto-detection (MPS → CUDA → CPU), `DEBUG` / `RECOMPUTE` / `datetag` switches, `Params` dataclass (single source of hyperparameter truth), global matplotlib style (`STYLE`, Latin Modern to match the paper), utilities (`flip_bits`, `printfig`, `get_scores`, `SpikeF1scoreLoss`, cosine LR schedule, `purge_unfinished_scans`). |
| `src/mnesis_chains.py` | Pattern generators (`SpikingPattern`, `StochasticSpikingPattern`) and the `HD_SNN` class (`forward_pass`, `get_W_init`, `update_weight`, `learn_model`); also a `load()` helper for trained checkpoints. |

Notebooks (`src/*.ipynb`) are numbered simple → complex, each loading cached
artifacts from the previous one. They **import** from the two modules — they do
**not** carry standalone copies of `Params`, `HD_SNN`, etc. Notebooks 01, 05, and
08 (former `boilerplate`, `parameters`, `generative-model`) were refactored into
the two modules; any reference to those notebook numbers in old documents is stale.

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
  (current run: `2026-09-07`).
- `seed = 2026` (in `Params`) is fixed for reproducibility.

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
This avoids data corruption and allows safe incremental resumption — never delete
a `.lock` out from under a running scan. A scan reserves a placeholder row
(`loss=null`, `time=0.0`) before computing it; if the process dies, that row would
count as computed forever. `purge_unfinished_scans()` (run at the start of
notebook 99, skipping locked scans unless `remove_locks=True`) deletes those rows.

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
| `figure_*.tex` | `\input` figure environments (caption + `\label`). `figure_izhikevich.tex` and `figure_snntorch.tex` wrap the TikZ PDFs of `tex/` (`figure_snntorch.tex` also includes `pattern.pdf`); `figure_target.tex`, `figure_retrieval.tex`, `figure_robustness.tex` (→ `fig:noise`) and `figure_scan.tex` take their graphics from `../figures/`. |
| `mnesis.bib` | Bibliography database. |

The main paper resolves experiment figures through `\graphicspath{{../figures/}}`
(in the preamble of `Perrinet26mnesis.tex`): `pattern.pdf`, `target.pdf`,
`retrieval.pdf`, the `*_score.pdf` curves and the `scan_*.pdf` bars are
**generated by the `src/` notebooks**, not built in `tex/`. The two TikZ figures
are built in `tex/`. Build everything from inside
`tex/`. Engine is `lualatex` (fontspec); toolchain: `lualatex`, `latexmk -pdflua`,
`biber`.

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
(<https://laurentperrinet.github.io/MNESIS/>); never commit built HTML.

- `docs/index.md` repeats the model equations, the variables table, the parameter
  drift note and the results of `README.md`: change both together.
- `docs/api.md` lists the documented members of `mnesis_boilerplate` explicitly:
  add any new public helper there.
- The CI installs its own dependency list (CPU-only `torch`, `snntorch`, …), not
  `docs/requirements.txt`: a new import in the modules must be added to both.

```bash
pip install -r docs/requirements.txt && make -C docs html   # green if Sphinx exits without error
```

---

## Notebook annotations

Each experiment notebook carries markdown documentation in a standard shape
(see notebooks 11 and 14–16 for models to follow). Notebooks 10 and 13 follow the
shape but lack the `<!-- MNESIS-annotated -->` marker; notebooks 18, 20, 25 and
99 do not follow it yet (18 has a summary without the H1 title, 20 and 25 start
with code).

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

**Terminology (consistency with the paper):**

| Prefer | Avoid |
|---|---|
| *motif* (a stored spiking pattern, `N_pattern` of them) | "pattern" when referring to what is memorised |
| *trigger window* (first `num_delay` ms, clamped input) | "cue", "prompt" |
| *replay* / *re-emit* | "output the duration" |
| *evoked window* = `spikes[:, :, N_pretime+num_delay : N_time+N_pretime]` vs `target[:, :, num_delay:]` | ad-hoc scoring slices |
| *spiking motif (SM) window* for the `num_delay`-wide history | "stimulus window" |

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
| `p_flip_target`, `p_flip_score`, `p_flip_retrain_score` | 14 |
| `trigger_time`, `trigger_time_score` | 15 |
| `fraction_target`, `fraction_target_score` | 16 |
| `scan_{key}` (one per scanned parameter) | 20 |
| `periodic_*` / `optuna` (displayed only, `figpath=None`) | 18 / 25 |

`printfig` skips an existing file unless `DEBUG > 1`: to refresh a figure in
production, delete it from `figures/` first.

**Character hygiene:** no non-breaking/narrow spaces (U+00A0/U+202F), zero-width
characters, curly quotes or non-ASCII hyphens in `.py`, `.ipynb`, `.tex`
comments/docstrings — use plain ASCII there (legitimate em/en dashes in README
prose are fine). The `^^V (U+0016)` lualatex warning is the `dsrom` font quirk,
not a stray character.

---

## Conventions & gotchas

- **Parameter drift.** As of the reconciliation run `2026-09-07`, the paper
  methods text (β = 0.7, τ ≈ 2.8 steps; ϑ₀ = 0.75, δ = 0.25; p_flip = 0.01;
  FastSigmoid α = 5; adadelta; η 40→0.4 mLR; dropout 0.6) matches the
  `Params` production defaults in `mnesis_boilerplate.py`. Older documents
  may still cite camera-ready values (β = 0.8, α = 12, SGD, dropout 0.10);
  robustness results (Fig. `fig:noise`) keep their own run's numbers. **Do not
   "autocorrect" code defaults to match stale prose** — consult `README.md` /
   `tex/Perrinet26mnesis.pdf` for the exact value a given figure used.
  Each notebook may also override defaults with its own local `opt_dict`.
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
