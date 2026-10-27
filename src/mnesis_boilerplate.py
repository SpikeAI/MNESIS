"""Shared boilerplate for the MNESIS experiments.

Provides the entry-point imports (torch, snntorch, numpy, matplotlib, ...),
automatic device detection (MPS -> CUDA -> CPU), the :class:`Params`
dataclass that is the single source of hyperparameter truth, and the
utilities reused by :mod:`mnesis_chains` and all notebooks: random bit
flipping (:func:`flip_bits`), figure saving (:func:`printfig`), spike-score
metrics (:func:`get_scores`, :func:`get_f1score`,
:class:`SpikeF1scoreLoss`), the F1 scores expected from theory for a degraded
trigger (:func:`get_f1_noisy_copy`, :func:`get_f1_ignition`, and the simulated
cascade :func:`get_f1_cascade`), the recall of each motif against time
(:func:`get_recall_time`, :func:`plot_recall_time`) and the cosine learning-rate
schedule (:func:`get_cosine_schedule_with_warmup`).
"""

from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import os
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
import snntorch as snn
import snntorch.surrogate as surrogate
from snntorch import utils as snn_utils
import snntorch.spikeplot as splt
from tqdm import trange
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams
import seaborn as sns
import pandas as pd
import datetime
# --- Configuration & Paths ---
RECOMPUTE = False
# RECOMPUTE = True 
DEBUG = 1 # production
if DEBUG > 1:
    print(f'running in debug mode with DEBUG = {DEBUG}')

datetag = '2026-07-11' # run with new parameters from the camera ready
datetag = '2026-08-06' # novel run on the revamped code
datetag = '2026-09-07' # post poster RT neurocomp - preparing ICANN poster, INT seminar and paper submission
# datetag = '2026-10-26' # seeded noisy cues, scoring against the frozen motifs, initialisation on the frozen motifs
print(f"datetag = '{datetag}'")

# --- Torch Setup ---
torch.set_float32_matmul_precision("medium")
torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)
torch.autograd.set_detect_anomaly(True)

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
print(f'Using device: {device}')


@dataclass
class Params:
    """Single source of hyperparameter truth for the MNESIS experiments.

    Instances also seed ``torch`` and ``numpy`` (in ``__post_init__``) so
    that a given ``seed`` reproduces a run. Field groups:

    Attributes:
        N_neuron, num_delay, N_pattern,
            N_time, N_pretime, p_A, p_flip, seed: network size and statistics.
        lif_beta, lif_threshold, learn_beta, learn_threshold, do_pinv,
            do_deconv: membrane dynamics and analytical-init flags.
        num_epochs, num_warmup_epochs, base_lr, final_lr, delta1, delta2,
            dropout, alpha_surrogate, surrogate_name, loss_name,
            reset_mechanism, optimizer: learning dynamics.
        verbose, fig_width, fig_height, phi, N_time_show, N_neuron_show,
            i_pattern, N_scan, N_cv: figure/scan settings.
    """
    datetag: str = datetag  
    N_neuron: int = 1024 // DEBUG        # number of presynaptic inputs
    num_delay: int = 41                  # number of timesteps in SM (no parity constraint: the delay windows are causal)
    N_pattern: int = 16 // DEBUG         # number of spiking motifs
    N_time: int = 1000 // DEBUG          # number of timebins for the WM patterns
    N_pretime: int = 50                  # number of timebins for spontaneous activity before the stimulus (the input then ends with as many silent bins)
    p_A: float = 0.00016                 # prior probability of firing for postsynaptic raster plot (spike per timebin)
    p_flip: float = 0.01                 # the default probability of flipping a bit in the stochastic pattern generator
    seed: int = 2026                     # seed
    device = device

    # network
    lif_beta: float = 0.70
    lif_threshold: float = 0.75
    learn_beta: bool = False
    learn_threshold: bool = False
    do_pinv: bool = True
    do_deconv: bool = True

    # learning
    num_epochs: int = 256 // DEBUG
    num_warmup_epochs: int = 16          # 2**4
    base_lr: float = 40.0e-3
    final_lr: float = 4.e-4
    delta1: float = 3.e-3
    delta2: float = 10.e-6
    dropout: float = 0.60
    alpha_surrogate: float = 5.0
    surrogate_name: str = "FastSigmoid"
    loss_name: str = "SpikeF1scoreLoss"  # 'MSELoss' #'L1Loss'
    reset_mechanism: str = "zero"    # "zero"
    optimizer: str = "adadelta"              # 'adamw' #adam

    # figures
    verbose: bool = False                # Displays more verbose output.
    fig_width: float = 15.92              # width of figure in cm: \textwidth of the paper (A4, 1 in margins), so figures print at scale 1
    phi: float = 1.61803                 # beauty is gold
    fig_height: float = 15.92/phi             # height of figure in cm
    N_time_show: int = N_time                # number of time points to show in plots
    N_neuron_show: int = N_neuron              # number of SM to show in plots
    i_pattern: int = 1                     # index of the motif shown in plots
    N_scan: int = 13 // DEBUG + 1        # number of values to scan
    N_cv: int = 10 // DEBUG + 1        # number of independent repetitions (random seeds) per scan value

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)



data_cache = Path('../cached_data')
data_cache.mkdir(exist_ok=True)

figpath = Path('../figures')
if os.environ.get("USER") == "uvb28bo": 
    figpath = None # Jean Zay

# --- Figure style (single source of truth, applied globally at import) ---
# Figures are typeset with the same face as the LaTeX manuscript
# (lualatex -> Latin Modern), at Physical-Review-like single-column size.
def _register_tex_fonts():
    faces = []
    roots = [Path('/usr/local/texlive'), Path('/opt/homebrew/texlive'),
             Path('/usr/share/texlive'), Path.home() / 'texmf']
    try:
        from matplotlib import font_manager
        for root in roots:
            if not root.exists():
                continue
            for otf in sorted(root.glob('*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf')):
                try:
                    font_manager.fontManager.addfont(str(otf))
                    name = font_manager.FontProperties(fname=str(otf)).get_name()
                    if name not in faces:
                        faces.append(name)
                except Exception:
                    pass
            if faces:
                break
    except Exception:
        pass
    return faces

_tex_faces = _register_tex_fonts()

# Font sizes of the paper (scrartcl, 11 pt class option), in pt. Figures are printed
# at scale 1 (see Params.fig_width), so these are also the sizes read in the paper.
normalsize = 10.95  # body text and captions
small = 10.  # \small

STYLE = {
     'font.family': 'serif',
     'font.serif': ['Latin Modern Roman'] + _tex_faces + ['DejaVu Serif', 'STIXGeneral'],
     'mathtext.fontset': 'cm',
     'mathtext.default': 'regular',
     'text.usetex': False,
     'font.size': normalsize,
     'axes.linewidth': 0.8,
     'axes.labelsize': normalsize,
     'axes.titlesize': normalsize,
     'axes.spines.left': True,
     'axes.spines.right': False,
     'axes.spines.top': False,
     'axes.spines.bottom': True,
     'xtick.labelsize': small,
     'ytick.labelsize': small,
     'xtick.direction': 'in',
     'ytick.direction': 'in',
     'xtick.major.size': 3.0,
     'ytick.major.size': 3.0,
     'xtick.major.width': 0.8,
     'ytick.major.width': 0.8,
     'errorbar.capsize': 1.5,
     'lines.linewidth': 1.0,
     'lines.markersize': 3.5,
     'legend.fontsize': small,
     'legend.frameon': False,
     'savefig.transparent': True,
}
plt.style.use(STYLE)

# --- Constants ---
phi = np.sqrt(5)/2 + 1/2
subplotpars = SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)

# --- Utility Functions ---
def pprint(s):
    """Print a headline surrounded by '=' banners of the same length."""
    print(len(s)*'=')
    print(s)
    print(len(s)*'=')

def printfig(fig, name='', fig_width=12, fig_height=None, exts=['pdf', 'png', 'svg'], figpath=figpath, dpi_exp=None, bbox=None, layout='constrained', verbose=True, do_overwrite=True):
    """Save a matplotlib figure to ``figpath`` under several extensions.

    Sizes the figure in centimetres, defaulting the height to
    ``fig_width / phi`` (golden ratio). On the Jean Zay cluster
    (``figpath is None``) saving is silently skipped. Existing files are
    overwritten only if ``do_overwrite`` is true.

    Args:
        fig_width: Width of the saved figure (cm); match it to the width the
            paper gives to ``\\includegraphics`` so fonts print at their nominal size.
        fig_height: Height of the saved figure (cm); defaults to ``fig_width / phi``.
        bbox: Passed to ``savefig(bbox_inches=...)``. The default ``None`` keeps
            the exact ``fig_width`` x ``fig_height`` canvas, whereas ``'tight'``
            crops it to the artists and so changes the printed scale.
        layout: Layout engine set before saving (``'constrained'`` fits labels
            inside the canvas); ``None`` keeps a manual ``subplots_adjust``.
    """
    if fig_height is None: fig_height = fig_width/phi
    cm = 1/2.54  # centimeters in inches
    fig.set_size_inches((fig_width*cm, fig_height*cm))
    if layout is not None: fig.set_layout_engine(layout)
    if figpath is not None: 
        figpath.mkdir(exist_ok=True)
        for ext in exts:
            filename = figpath / f'{name}.{ext}'
            if filename.exists() and not do_overwrite:
                if verbose: print(f'File {filename} already exists. Skipping save.')
            else:
                if verbose: print(f'Saving as {filename}')
                fig.savefig(filename, dpi=dpi_exp, bbox_inches=bbox, transparent=True)

def flip_bits(a, p_flip, seed=None, verbose=False):
    """Balanced bit flipping preserving the marginal firing rate of ``a``.

    Each entry of the binary tensor ``a`` is replaced, with probability
    ``p_flip``, by a fresh Bernoulli draw of rate ``a.mean()``, so the
    expected spike rate is preserved while the pattern is perturbed.

    Args:
        a (torch.Tensor): binary tensor to perturb.
        p_flip (float): probability of redrawing an entry (dimensionless).
        seed (int): seed of the flips; ``None`` draws a fresh random seed.
            The generator is seeded with a value derived from ``seed``, not
            with ``seed`` itself, so that a flip seed equal to the seed of the
            frozen motif does not read the same uniform draws.
        verbose (bool): if ``True``, report the number of flipped bits.

    Returns:
        torch.Tensor: A new tensor of the same shape as ``a``.
    """
    generator = torch.Generator(device=a.device)
    if seed is None:
        seed = generator.seed()
    else:
        # The frozen motif is drawn as U < p_A with a generator seeded by opt.seed: seeded
        # with the same value, the mask U < p_flip (p_flip > p_A) would flip every spike of
        # the motif. Deriving the generator state from the seed decorrelates the two streams.
        generator.manual_seed(int(np.random.SeedSequence([int(seed), 1]).generate_state(1)[0]))
    mask = torch.bernoulli(torch.ones_like(a) * p_flip, generator=generator)
    flipped = torch.bernoulli(torch.ones_like(a) * a.mean(), generator=generator)
    out = torch.where(mask == 1., flipped, a)
    if verbose:
        print(f"Flipping {mask.sum().item()} bits out of {a.numel()} (p_flip={p_flip}, flip seed={seed}), {a.mean().item():.3e} -> {out.mean().item():.3e}")
    return out

def stop(): 
    """Raise an AssertionError on purpose: temporary end of the script."""
    assert False, "Temporary end of the road"

def approx_equals(series, value, rtol=1e-6, atol=1e-12):
    """Elementwise ``np.isclose`` for a pandas Series, falling back to ``==``.

    Returns:
        Boolean mask selecting the entries close to ``value``.
    """
    try:
        return np.isclose(series.astype(float), float(value), rtol=rtol, atol=atol)
    except (TypeError, ValueError):
        return series == value

@contextmanager
def cache_lock(filename):
    """Hold the ``.lock`` sentinel of a cache file while this file is computed.

    The lock is taken atomically (``O_EXCL``), so two processes started together
    cannot both compute the same file. It is released in a ``finally`` clause, so
    an exception or an interrupt during the computation does not leave a stale
    lock behind, which would make every later run skip the computation.

    Args:
        filename: Path of the cache file to compute; the lock is the same path with a ``.lock`` suffix.

    Yields:
        True if this process now holds the lock and should compute the file;
        False if the lock already exists (another process is computing the file,
        or one died before releasing it), in which case nothing is removed.
    """
    lock_filename = Path(filename).with_suffix('.lock')
    try:
        lock_filename.touch(exist_ok=False) # atomic: fails if the lock already exists
    except FileExistsError:
        yield False
        return
    try:
        yield True
    finally:
        lock_filename.unlink(missing_ok=True)

def purge_unfinished_scans(datetag=None, data_cache=data_cache, ignore_locks=False, remove_locks=False, dry_run=False):
    """Remove the placeholder rows left by interrupted scans, so they are recomputed.

    A scan notebook reserves a row ``loss=inf, time=0.0`` (stored as ``null`` in
    JSON) before computing it, and overwrites it when done. A row still in that
    state means the process died in between, yet the scan loop treats it as
    "already computed" and never retries it. Deleting the row lifts that block.
    A failed evaluation also stores ``loss=inf`` but with a non-zero ``time``, so
    it is kept.

    Args:
        datetag: Cache-filename prefix of the batch to clean (ISO 8601 date); None cleans the scans of all batches.
        data_cache: Directory holding the ``{datetag}_*.json`` scan files.
        ignore_locks: If False, skip the scans that have a ``.lock`` file, as they may still be running.
        remove_locks: If True, also delete every ``.lock`` file of the selected batches, which lifts the block on resuming the scans. Only use it when no scan is running (e.g. at the start of the orchestrator), as it implies ``ignore_locks``.
        dry_run: If True, only report what would be removed.

    Returns:
        Dictionary mapping each file name to the number of rows removed (or to be removed).
    """
    ignore_locks = ignore_locks or remove_locks
    pattern = f'{datetag}_*' if datetag else '*'
    removed = {}
    for filename in sorted(Path(data_cache).glob(f'{pattern}.json')):
        rows = json.loads(filename.read_text())
        if not isinstance(rows, list) or not all(isinstance(r, dict) and 'i_cv' in r for r in rows):
            continue # not a scan file
        kept = [r for r in rows if not (r.get('loss') is None and r.get('time') == 0.0)]
        n_bad = len(rows) - len(kept)
        if n_bad == 0:
            continue
        if filename.with_suffix('.lock').exists() and not ignore_locks:
            print(f"{filename.name}: {n_bad} unfinished row(s) kept, scan is locked")
            continue
        print(f"{filename.name}: {n_bad}/{len(rows)} unfinished row(s) {'to remove' if dry_run else 'removed'}")
        removed[filename.name] = n_bad
        if not dry_run:
            tmp = filename.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(kept, indent=2))
            tmp.replace(filename) # atomic: a concurrent reader never sees a half-written file
    if remove_locks:
        # done after the purge so that a lock never disappears while its file is being rewritten
        for lock in sorted(Path(data_cache).glob(f'{pattern}.lock')):
            print(f"{lock.name}: lock {'to remove' if dry_run else 'removed'}")
            if not dry_run:
                lock.unlink(missing_ok=True)
    return removed

def get_scores(pred, target, epsilon=1e-12, dim=None):
    """Precision, recall and F1 between two (soft) spike tensors.

    Sums the elementwise true/false positives and negatives; high
    precision means few false positives, high recall few false
    negatives. ``epsilon`` avoids division by zero. High precision
    means few false positives (FP), high recall few false negatives (FN).

    Args:
        pred (torch.Tensor): predicted spikes.
        target (torch.Tensor): target spikes, of the same shape.
        epsilon (float): regularisation of the divisions.
        dim (int or tuple): dimensions to sum over; ``None`` sums over all
            of them, ``(1, 2)`` gives one score per motif for tensors of
            shape ``(N_pattern, N_neuron, N_time)``.

    Returns:
        tuple: (precision, recall, f1_score), scalar tensors or tensors over
        the dimensions left by ``dim``.
    """
    TP = (pred * target).sum(dim=dim)
    FP = (pred * (1 - target)).sum(dim=dim)
    FN = ((1-pred) * target).sum(dim=dim)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    return precision, recall, f1_score

def get_f1score(pred, target, epsilon=1e-12, dim=None):
    """F1 score: the harmonic mean of precision and recall.

    High only when both precision and recall are high; one minus this
    value is the training loss (:class:`SpikeF1scoreLoss`). ``dim`` is
    passed to :func:`get_scores` (``(1, 2)``: one score per motif).
    """
    _, _, f1_score = get_scores(pred, target, epsilon=epsilon, dim=dim)
    return f1_score

def get_beta_errorbar(scores, quantiles=(0.05, 0.95), axis=1):
    """Mean and asymmetric error bars of repeated scores bounded in [0, 1].

    An F1 score lives in [0, 1], so a symmetric mean +/- std bar can cross
    the bounds; instead a beta distribution is fitted to the repetitions by
    the method of moments (mean ``m``, variance ``v``):
    ``k = m (1 - m) / v - 1``, ``a = m k``, ``b = (1 - m) k``, and the bars
    span its ``quantiles``. Where no beta fit exists (``v = 0``, e.g. all
    repetitions at F1 = 0, or ``k <= 0`` for data piled at 0 and 1), the
    empirical quantiles are used instead (equal to ``m`` when ``v = 0``).

    Args:
        scores (np.ndarray): repeated scores in [0, 1], repetitions along ``axis``.
        quantiles (tuple): lower and upper quantiles of the error bars.
        axis (int): axis holding the repetitions.

    Returns:
        tuple: (mean, yerr), ready for ``ax.errorbar(x, mean, yerr=yerr)``;
        ``yerr`` has shape ``(2, ...)``: distances from the mean down to the
        lower quantile and up to the upper one.
    """
    m = scores.mean(axis=axis)
    v = scores.var(axis=axis) # biased variance, guarantees v <= m (1 - m)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = m * (1 - m) / v - 1
    fit = (v > 0) & (k > 0)
    q = np.quantile(scores, quantiles, axis=axis)
    for i, quantile in enumerate(quantiles):
        q[i][fit] = stats.beta.ppf(quantile, m[fit] * k[fit], (1 - m[fit]) * k[fit])
    # a strongly skewed beta can put a quantile on the wrong side of the mean;
    # clip so that errorbar never receives a negative length
    yerr = np.clip(np.stack([m - q[0], q[1] - m]), 0, None)
    return m, yerr

def get_f1_noisy_copy(p_flip, p_A):
    """F1 score of a balanced-flipped motif against the frozen one.

    A balanced flip (:func:`flip_bits`) keeps a spike of the motif with
    probability ``q = 1 - p_flip (1 - p_A)`` and turns an empty bin into a
    spike with probability ``p_flip p_A``, so the expected precision and
    recall are both equal to ``q``. This is the score of a readout that would
    copy the corrupted motif as is, without any pattern completion: it falls
    linearly from 1 to chance level (``F1 = p_A``) at ``p_flip = 1``.

    Args:
        p_flip (float or np.ndarray): bit-flip probability (dimensionless).
        p_A (float): firing probability of the motif (spike per time bin).

    Returns:
        np.ndarray: expected F1 score, shaped like ``p_flip``.
    """
    return 1 - np.asarray(p_flip, dtype=float) * (1 - p_A)

def get_f1_from_ignition(R, precision_0=1., recall_0=1.):
    """F1 score of the replay when a fraction ``R`` of the motifs ignites.

    An ignited motif is replayed with the precision and recall of the intact
    trigger, a silent one only adds false negatives. Per spike of the motifs,
    the true positives are ``R recall_0``, the false negatives ``1 - R
    recall_0`` and the false positives ``R recall_0 (1 / precision_0 - 1)``,
    so that ``F1 = 2 R recall_0 / (1 + R recall_0 / precision_0)``.

    Args:
        R (float or np.ndarray): ignition probability of a motif (dimensionless).
        precision_0 (float): precision of the replay with the intact trigger.
        recall_0 (float): recall of the replay with the intact trigger.

    Returns:
        np.ndarray: expected F1 score, shaped like ``R``.
    """
    return 2 * R * recall_0 / (1 + R * recall_0 / precision_0)

def get_trigger_presence(p_flip, p_A, num_delay, fraction=1., N_trigger_time=None):
    """Probability that a spike of the spiking-motif window is present in a degraded trigger.

    A spike of the motif lying in bin ``i`` of the window (``0 <= i <
    num_delay``) is present if it falls in the part that the trigger shows,
    the last ``N_trigger_time`` bins (a truncated trigger) on the ``fraction``
    of neurons left active (a lesioned one), and survives the balanced flips,
    which keep it with probability ``1 - p_flip (1 - p_A)``.

    Args:
        p_flip (float or np.ndarray): bit-flip probability of the trigger (dimensionless).
        p_A (float): firing probability of the motif (spike per time bin).
        num_delay (int): width of the spiking-motif window (time bins).
        fraction (float or np.ndarray): fraction of the neurons shown by the
            trigger (dimensionless, 1 for an intact trigger).
        N_trigger_time (float or np.ndarray): number of bins shown at the end
            of the window (time bins); ``None`` shows the whole window.

    Returns:
        np.ndarray: presence probability, shaped like the broadcast of the
        arguments with a last axis of ``num_delay`` bins.
    """
    if N_trigger_time is None: N_trigger_time = num_delay
    q = np.asarray(fraction, dtype=float) * (1 - np.asarray(p_flip, dtype=float) * (1 - p_A))
    shown = np.arange(num_delay) >= num_delay - np.asarray(N_trigger_time, dtype=float)[..., None]
    return q[..., None] * shown

def get_f1_ignition(p_flip, p_A, N_neuron, num_delay, lif_threshold, precision_0=1., recall_0=1.,
                    fraction=1., N_trigger_time=None):
    """F1 score expected when a degraded trigger ignites, or not, the replay of each motif.

    Mean-field model of the pseudo-inverse readout (:meth:`mnesis_chains.HD_SNN.get_W_init`):
    each spike of a motif is supported by the ``K`` spikes of that motif in the
    ``num_delay`` bins before it. The minimum-norm solution gives these
    supporting spikes equal weights, so the membrane reaches 1 when all of
    them are present and ``k / K`` when only ``k`` of them are: the spike is
    emitted if ``k >= ceil(lif_threshold K)``. A spurious spike of the trigger
    falls on a near-zero weight, as the motifs are sparse.

    The replay is a cascade, a positive feedback loop: each emitted spike
    joins the support of the next ones, so that once the first evoked spike of
    a motif is emitted, the following ones see more and more of their support
    and the replay ignites as a whole (the noise-free motif being a stable
    fixed point); if it fails, the support of the next spikes keeps shrinking
    and the replay dies out (:func:`get_f1_cascade` simulates the whole
    cascade and checks that its first step decides it). The ignition
    probability is thus that of the first evoked spike. The motif fires
    ``mu = p_A N_neuron`` spikes per bin, so this spike comes ``j`` bins after
    the trigger with probability ``P(j) = exp(-mu j) (1 - exp(-mu))``; its
    support then lies in the last ``num_delay - j`` bins of the trigger window,
    so that ``K`` is Poisson of mean ``lam_j = mu (num_delay - j)`` and a
    supporting spike is present with the mean ``q_j`` of
    :func:`get_trigger_presence` over these bins. Hence

        ``R = sum_j P(j) sum_{K >= 1} Poisson(K; lam_j) P(Binomial(K, q_j) >= ceil(lif_threshold K))``,

    normalised by its value for an intact trigger, whose failures (motifs with
    a gap longer than the window) are already counted in ``recall_0``. For
    scattered losses (flips, lesioned neurons), ``q_j = q`` does not depend on
    ``j``; for a truncated trigger, the support of the first evoked spike is
    more recent than a whole window, and so better shown than the fraction
    ``N_trigger_time / num_delay`` of the window. The F1 score follows from
    ``R`` by :func:`get_f1_from_ignition`.

    Args:
        p_flip (float or np.ndarray): bit-flip probability of the trigger (dimensionless).
        p_A (float): firing probability of the motif (spike per time bin).
        N_neuron (int): number of neurons.
        num_delay (int): width of the spiking-motif window (time bins).
        lif_threshold (float): firing threshold, relative to the membrane
            potential reached on the motif (dimensionless).
        precision_0 (float): precision of the replay with the intact trigger.
        recall_0 (float): recall of the replay with the intact trigger.
        fraction (float or np.ndarray): fraction of the neurons shown by the
            trigger (dimensionless, 1 for an intact trigger).
        N_trigger_time (float or np.ndarray): number of bins shown at the end
            of the window (time bins); ``None`` shows the whole window.

    Returns:
        np.ndarray: expected F1 score, shaped like the broadcast of ``p_flip``,
        ``fraction`` and ``N_trigger_time``.
    """
    mu = p_A * N_neuron
    j = np.arange(num_delay)
    P_j = np.exp(- mu * j) * (1 - np.exp(- mu)) # delay of the first evoked spike (beyond the window, K = 0: it never fires)
    lam_j = mu * (num_delay - j)
    # q_j: mean presence over the last num_delay - j bins of the window
    presence = get_trigger_presence(p_flip, p_A, num_delay, fraction, N_trigger_time)
    q_j = np.cumsum(presence[..., ::-1], axis=-1)[..., ::-1] / (num_delay - j)
    # K = 0 is left out: with no supporting spike, the bias-free readout never fires
    K = np.arange(1, int(mu * num_delay + 10 * np.sqrt(mu * num_delay)) + 10)
    # rounding first, so that e.g. 0.75 * 4 = 3 is not pushed to 4 by a float error
    k_min = np.ceil(np.round(lif_threshold * K, 9))
    p_K = stats.poisson.pmf(K, lam_j[:, None])
    Phi_j = np.sum(p_K * stats.binom.sf(k_min - 1, K, q_j[..., None]), axis=-1)
    R = np.sum(P_j * Phi_j, axis=-1) / np.sum(P_j * p_K.sum(axis=-1))
    return get_f1_from_ignition(R, precision_0, recall_0)

def get_f1_cascade(p_flip, p_A, N_neuron, num_delay, lif_threshold, precision_0=1., recall_0=1.,
                   fraction=1., N_trigger_time=None, N_motif=2000, seed=2026):
    """F1 score expected from a simulated cascade of the replay (Monte Carlo).

    Simulates the threshold network that :func:`get_f1_ignition` reduces to
    its first step: ``N_motif`` motifs are drawn with ``Binomial(N_neuron,
    p_A)`` spikes per bin; in the trigger window each spike is kept with the
    probability of :func:`get_trigger_presence`; then, bin after bin, the
    spikes of the motif are emitted if at least ``ceil(lif_threshold K)`` of
    the ``K`` spikes of the motif in the ``num_delay`` bins before them are
    present, the emitted spikes joining the support of the next ones (the
    positive feedback loop). The identity of the neurons does not matter in
    this mean-field picture, only the number of spikes per bin. After a
    transient of ``2 num_delay`` bins, the fraction of the spikes of the motifs
    re-emitted in the next ``num_delay`` bins is the ignition probability
    ``R``, normalised by its value for an intact trigger on the same motifs,
    and the F1 score follows from :func:`get_f1_from_ignition`. All the
    degradation levels share the same motifs and draws (common random
    numbers), so that the curve is smooth.

    Args:
        p_flip (float or np.ndarray): bit-flip probability of the trigger (dimensionless).
        p_A (float): firing probability of the motif (spike per time bin).
        N_neuron (int): number of neurons.
        num_delay (int): width of the spiking-motif window (time bins).
        lif_threshold (float): firing threshold, relative to the membrane
            potential reached on the motif (dimensionless).
        precision_0 (float): precision of the replay with the intact trigger.
        recall_0 (float): recall of the replay with the intact trigger.
        fraction (float or np.ndarray): fraction of the neurons shown by the
            trigger (dimensionless, 1 for an intact trigger).
        N_trigger_time (float or np.ndarray): number of bins shown at the end
            of the window (time bins); ``None`` shows the whole window.
        N_motif (int): number of simulated motifs.
        seed (int): seed of the random generator.

    Returns:
        np.ndarray: expected F1 score, shaped like the broadcast of ``p_flip``,
        ``fraction`` and ``N_trigger_time``.
    """
    presence = get_trigger_presence(p_flip, p_A, num_delay, fraction, N_trigger_time)
    shape = presence.shape[:-1]
    # the last level is the intact trigger, used for the normalisation
    presence = np.concatenate([presence.reshape(-1, num_delay), np.ones((1, num_delay))])
    N_time = 4 * num_delay
    rng = np.random.default_rng(seed)
    counts = rng.binomial(N_neuron, p_A, size=(N_motif, N_time))
    # one uniform per spike of the trigger window, shared by all the levels
    u = rng.random((N_motif, num_delay, max(counts[:, :num_delay].max(), 1)))
    in_bin = np.arange(u.shape[-1]) < counts[:, :num_delay, None]
    present = np.zeros((presence.shape[0], N_motif, N_time), dtype=np.int16) # spike counts per bin, a few at most
    present[:, :, :num_delay] = np.sum(in_bin & (u < presence[:, None, :, None]), axis=-1)
    for t in range(num_delay, N_time):
        K = counts[:, t-num_delay:t].sum(axis=-1)
        k = present[:, :, t-num_delay:t].sum(axis=-1)
        fire = (K > 0) & (k >= np.ceil(np.round(lif_threshold * K, 9)))
        present[:, :, t] = counts[:, t] * fire
    ignited = present[:, :, -num_delay:].sum(axis=(1, 2)) / counts[:, -num_delay:].sum()
    R = (ignited[:-1] / ignited[-1]).reshape(shape)
    return get_f1_from_ignition(R, precision_0, recall_0)

def get_recall_time(spikes, target, bin_width):
    """Recall of each motif in successive time bins of the evoked window.

    Shows the dynamics of the replay: once ignited, the recall of a motif
    stays near that of the intact trigger; otherwise it falls to zero.

    Args:
        spikes (torch.Tensor): evoked spikes, of shape ``(N_pattern, N_neuron, N_time)``.
        target (torch.Tensor): target spikes, of the same shape.
        bin_width (int): width of a time bin (time bins); the last incomplete
            bin is left out.

    Returns:
        np.ndarray: recall of shape ``(N_pattern, N_time // bin_width)``,
        ``nan`` in a bin without any target spike.
    """
    N_pattern, N_neuron, N_time = target.shape
    N_bin = N_time // bin_width
    def binned(x): # number of spikes per motif and per bin
        x = x[:, :, :N_bin * bin_width].reshape(N_pattern, N_neuron, N_bin, bin_width)
        return x.sum(dim=(1, 3)).cpu().numpy()
    TP, N_target = binned(spikes * target), binned(target)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(N_target > 0, TP / N_target, np.nan)

def plot_recall_time(ax, recall_time, labels, recall_open=None, bin_width=1, cmap='viridis'):
    """Plot the recall of each motif against time, one colour per degradation level.

    Thin lines are single motifs (bistable: they stay near the recall of the
    intact trigger or fall to zero), thick lines their mean; the dashed black
    line is the mean recall of the open-loop control.

    Args:
        ax (matplotlib.axes.Axes): axes to draw in.
        recall_time (list of np.ndarray): one ``(N_pattern, N_bin)`` array per
            level, as returned by :func:`get_recall_time`.
        labels (list of str): legend label of each level.
        recall_open (np.ndarray): recall of the open-loop control, or ``None``.
        bin_width (float): width of a time bin (ms).
        cmap (str): colormap of the levels.
    """
    colors = plt.get_cmap(cmap)(np.linspace(0, .85, len(recall_time)))
    for recall, label, color in zip(recall_time, labels, colors):
        time = np.arange(recall.shape[1]) * bin_width # start of each bin, drawn as steps
        ax.step(time, recall.T, where='post', color=color, alpha=.4, lw=.5, rasterized=True)
        ax.step(time, np.nanmean(recall, axis=0), where='post', color=color, lw=1.5, label=label)
    if recall_open is not None:
        time = np.arange(recall_open.shape[1]) * bin_width
        ax.step(time, np.nanmean(recall_open, axis=0), where='post', color='k', ls='--', lw=1.2, label='open loop')
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("recall")
    ax.set_ylim(-0.05, 1.05)

class SpikeF1scoreLoss(nn.Module):
    """Training loss ``L = 1 - F1`` between predicted and target spikes."""
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, pred, target):
        """Return ``1 - F1score(pred, target)``, differentiable via the surrogate."""
        return 1 - get_f1score(pred, target, self.epsilon)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_epochs, rel_final_lr):
    """Cosine learning-rate decay with warmup, as a ``torch`` ``LambdaLR``.

    The learning-rate multiplier is constant (1) during
    ``num_warmup_epochs``, then follows a half-cosine from 1 down to
    ``rel_final_lr`` (``final_lr / base_lr``) at ``num_epochs``.
    """
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return 1
        else:
            progress = (current_epoch - num_warmup_epochs) / max(1, num_epochs - num_warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return (cosine_decay + rel_final_lr) / (1 + rel_final_lr)
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

loss_fn = SpikeF1scoreLoss()
