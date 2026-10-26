"""Shared boilerplate for the MNESIS experiments.

Provides the entry-point imports (torch, snntorch, numpy, matplotlib, ...),
automatic device detection (MPS -> CUDA -> CPU), the :class:`Params`
dataclass that is the single source of hyperparameter truth, and the
utilities reused by :mod:`mnesis_chains` and all notebooks: random bit
flipping (:func:`flip_bits`), figure saving (:func:`printfig`), spike-score
metrics (:func:`get_scores`, :func:`get_f1score`,
:class:`SpikeF1scoreLoss`), the F1 scores expected from theory under trigger
noise (:func:`get_f1_noisy_copy`, :func:`get_f1_ignition`) and the cosine learning-rate schedule
(:func:`get_cosine_schedule_with_warmup`).
"""

from pathlib import Path
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
    i_pattern: int = 0                     # index of the motif shown in plots
    N_scan: int = 13 // DEBUG + 1        # number of values to scan
    N_cv: int = 10 // DEBUG + 1        # number of cross-validation steps

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

    Returns:
        torch.Tensor: A new tensor of the same shape as ``a``.
    """
    generator = torch.Generator(device=a.device)
    if seed is None:
        seed = generator.seed()
    else:
        generator.manual_seed(seed)
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

def get_scores(pred, target, epsilon=1e-12):
    """Precision, recall and F1 between two (soft) spike tensors.

    Sums the elementwise true/false positives and negatives; high
    precision means few false positives, high recall few false
    negatives. ``epsilon`` avoids division by zero. High precision
    means few false positives (FP), high recall few false negatives (FN).

    Returns:
        tuple: (precision, recall, f1_score) as scalar tensors.
    """
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1-pred) * target).sum()
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    return precision, recall, f1_score

def get_f1score(pred, target, epsilon=1e-12):
    """F1 score: the harmonic mean of precision and recall.

    High only when both precision and recall are high; one minus this
    value is the training loss (:class:`SpikeF1scoreLoss`).
    """
    _, _, f1_score = get_scores(pred, target, epsilon=epsilon)
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

def get_f1_ignition(p_flip, p_A, N_neuron, num_delay, lif_threshold, precision_0=1., recall_0=1.):
    """F1 score expected when a noisy trigger ignites, or not, the replay of each motif.

    Mean-field model of the pseudo-inverse readout (:meth:`mnesis_chains.HD_SNN.get_W_init`):
    each spike of a motif is supported by the ``K`` spikes of that motif in its
    spiking-motif window (``N_neuron`` neurons by ``num_delay`` time bins), and
    ``K`` follows a Poisson law of mean ``lam = p_A N_neuron num_delay``. The
    minimum-norm solution gives these supporting spikes equal weights, so the
    membrane reaches 1 when all of them are present and ``k / K`` when only
    ``k`` of them are; a spurious spike of the trigger falls on a near-zero
    weight, as the motifs are sparse. Each supporting spike of the trigger
    survives the flips with probability ``q = 1 - p_flip (1 - p_A)``.

    The first evoked spikes are driven by the trigger alone and, ``lam`` being
    small, they share the same few supporting spikes: the replay of a motif
    therefore ignites as a whole if ``k >= ceil(lif_threshold K)`` (it then
    sustains itself, the noise-free motif being a stable fixed point) and
    dies out otherwise, with an ignition probability

        ``R = sum_{K >= 1} Poisson(K; lam) P(Binomial(K, q) >= ceil(lif_threshold K))``.

    An ignited motif is replayed with the noise-free precision and recall, a
    silent one only adds false negatives, so that summed over the motifs
    ``F1 = 2 R recall_0 / (1 + R recall_0 / precision_0)``.

    Args:
        p_flip (float or np.ndarray): bit-flip probability of the trigger (dimensionless).
        p_A (float): firing probability of the motif (spike per time bin).
        N_neuron (int): number of neurons.
        num_delay (int): width of the spiking-motif window (time bins).
        lif_threshold (float): firing threshold, relative to the membrane
            potential reached on the motif (dimensionless).
        precision_0 (float): precision of the replay at ``p_flip = 0``.
        recall_0 (float): recall of the replay at ``p_flip = 0``.

    Returns:
        np.ndarray: expected F1 score, shaped like ``p_flip``.
    """
    p_flip = np.asarray(p_flip, dtype=float)
    lam = p_A * N_neuron * num_delay
    # K = 0 is left out: with no supporting spike, the bias-free readout never fires
    K = np.arange(1, int(lam + 10 * np.sqrt(lam)) + 10)
    # rounding first, so that e.g. 0.75 * 4 = 3 is not pushed to 4 by a float error
    k_min = np.ceil(np.round(lif_threshold * K, 9))
    q = 1 - p_flip[..., None] * (1 - p_A)
    R = np.sum(stats.poisson.pmf(K, lam) * stats.binom.sf(k_min - 1, K, q), axis=-1)
    return 2 * R * recall_0 / (1 + R * recall_0 / precision_0)

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
