# Installation & Quickstart

## Install dependencies

Requires Python 3.10+. From the repository root:

```bash
pip install -r src/requirements.txt
```

Core dependencies: `torch`, `snntorch`, `numpy`, `scipy`, `matplotlib`, `jupyter`.
The optimisation/scan notebooks additionally use `optuna` (bundled in
`src/requirements.txt`).

## Run the notebooks in order

The notebooks are numbered and designed to be run sequentially. They import shared
infrastructure from the Python modules `mnesis_boilerplate` (imports, device
detection, `Params` dataclass, utilities) and `mnesis_chains` (`HD_SNN` class,
analytical initialisation). Each notebook saves its outputs (model weights, scan
results) to `cached_data/` so that downstream notebooks can load them without
recomputation. Notebook 99 can orchestrate a full multi-notebook run in one place.

```bash
cd src
jupyter notebook
```

Each notebook opens with its own `opt_dict = dict(...)` that overrides the global
`Params` defaults for that experiment (a different `N_neuron`, `num_delay`,
`N_pretime`, threshold, …). The global defaults in `Params` are the canonical
reference; per-notebook `opt_dict`s tune the run for the pattern family being
studied.

See {doc}`experiments` for the annotated notebook pipeline.

## Debug mode

Set `DEBUG > 1` in `src/mnesis_boilerplate.py` to shrink the problem: `N_neuron`,
`N_pattern`, `N_time`, and `num_epochs` are divided by `DEBUG`. Use `DEBUG = 1` (the
default) only for production/camera-ready runs.

(cached-data)=
## Cached data

Results are saved to `cached_data/` (excluded from git via `.gitignore`):

| File pattern | Content |
|---|---|
| `*_init.pth` | Analytically initialised weights (pseudo-inverse or Hebbian, deconvolved or not) |
| `*.pth` | Trained model weights after gradient steps |
| `*_scan_*.json` | Parameter sweep results (loss, precision, recall per condition) |
| `*_periodic-with-noise.npz` | Periodic-memory robustness curves across noise levels and time chunks |
| `*_optuna.sqlite3` | Optuna studies (synthetic motifs) |

Delete a cached `.pth`, `.json`, `.npz`, or `.sqlite3` file to force recomputation;
set `RECOMPUTE = True` at the top of any notebook to invalidate the full cache for
that notebook. The scan/optimisation notebooks additionally guard each run with a
`.lock` sentinel file to allow safe incremental resumption: a `.lock` sentinel is
created when a scan starts, each computed parameter value is appended to the JSON
(not re-running values already present), and the lock is deleted when done — never
delete a `.lock` out from under a running scan.

## Build this documentation

The site is built *in place* inside `docs/` (doctree caches go to the git-ignored
`docs/_build/`), then published by committing and pushing that folder; GitHub Pages
is configured to serve `/docs` from the `main` branch:

```bash
pip install -r docs/requirements.txt
make -C docs html
git add docs && git commit -m "docs: rebuild" && git push
```

`make -C docs help` lists the available targets (`notebooks`, `html`, `clean`,
`all`). The notebooks are symlinked from `src/` into `docs/notebooks/` and rendered
with their stored outputs (they are *not* re-executed by the docs build).
