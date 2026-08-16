#!/usr/bin/env python3
"""Mechanics behind the Figure 5 panels: loading the tables written by figure5_data.py,
applying the project's reliability conventions, and saving a panel under the naming rule.

Deliberately contains NO styling: no colours, no limits, no colour maps, no figure sizes.
Every one of those choices belongs in the notebook cell that draws the panel, so that each
cell of analysis_figure5.ipynb is self-contained and can be run on its own after a kernel
restart. This module only does the things that must not be re-typed (and so silently
diverge) from panel to panel: parsing, masking, truncation, file naming.

    import figure5_plot as f5
    q, h, mean, std, cnt = f5.load2d("C_pmf2d_qcore_total.dat")
    F = f5.reliable_2d(mean, std, cnt, std_ceil=10.0, min_visits=5)
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import dna_bending_wall_compare as dw   # _reliable_run / _reliable_region2d: the truncation
                                        # rules, defined once for the whole project

# Importing dna_bending_wall_compare applies seaborn's whitegrid theme globally. Undo it here,
# so that importing this module leaves matplotlib in its default state and each notebook cell
# gets exactly the style it sets for itself -- and nothing it did not ask for.
mpl.rcParams.update(mpl.rcParamsDefault)

DATA = "analysis/figure5"
FIGS = "figures"          # project convention: figures live in the root figures/

# which figure5_data.py run writes each table -- so a missing file says how to make it
MADE_BY = {"A_": "--panels A", "B_": "--panels B", "C_": "--panels C",
           "D_": "--panels D", "E_": "--panels E", "F1": "--panels F", "F2": "--panels F"}


def _path(name, data):
    """Resolve a table, and if it is not there yet say what produces it rather than
    dying on a bare FileNotFoundError three lines into a plotting cell."""
    path = os.path.join(data, name)
    if not os.path.exists(path):
        how = MADE_BY.get(name[:2], "")
        raise FileNotFoundError(
            f"{path} does not exist yet.\nGenerate it with:\n"
            f"    python analysis/scripts/figure5_data.py {how}".rstrip())
    return path


# ────────────────────────────────────────────────────────────────────── loading
def load1d(name, data=DATA):
    """-> dict of columns, named from the last header line, units stripped
    ('F_mean(kJ/mol)' -> 'F_mean')."""
    path = _path(name, data)
    with open(path) as fh:
        names = [l for l in fh if l.startswith("#")][-1].lstrip("# ").split()
    names = [n.split("(")[0] for n in names]
    d = np.loadtxt(path)
    return {n: d[:, i] for i, n in enumerate(names)}


def load2d(name, data=DATA, with_cov=False):
    """Long format (cv1 cv2 mean std count [n_walkers]) -> x, y, mean, std, count; grids
    indexed [x, y]. with_cov=True additionally returns the per-bin walker-coverage grid
    (6th column) when the table has one -- used by the SI panels' ergodicity mask."""
    d = np.loadtxt(_path(name, data))
    x, y = np.unique(d[:, 0]), np.unique(d[:, 1])
    cols = (2, 3, 4, 5) if with_cov else (2, 3, 4)
    if with_cov and d.shape[1] < 6:
        raise ValueError(f"{name} has no walker-coverage column; regenerate the table with "
                         "dna_bending_wall_compare.py --idrdna-bend-2d")
    return (x, y) + tuple(d[:, k].reshape(len(x), len(y)) for k in cols)


def load_meta(data=DATA):
    import json
    with open(os.path.join(data, "meta.json")) as fh:
        return json.load(fh)


def load_medoids(name="C_medoids_cvs.dat", data=DATA):
    """State medoids from select_snapshots.py -> (state, cv1, cv2) arrays, plus the column
    names read from the header. For panel C the columns are (total_helicity, q_core) and each
    row is a real frame, so the markers sit where the structure actually is on the surface."""
    path = _path(name, data)
    with open(path) as fh:
        cols = [l for l in fh if l.startswith("#")][-1].lstrip("# ").split()
    d = np.atleast_2d(np.loadtxt(path))
    ix = {c: i for i, c in enumerate(cols)}
    return d[:, ix["state"]].astype(int), d[:, 3], d[:, 4], cols[3:5]


# ─────────────────────────────────────────────────── project reliability conventions
def reliable_1d(mean, std, std_ceil):
    """Mask of the contiguous std <= std_ceil run around the minimum (the rest of the profile
    is too thinly sampled to plot). See dna_bending_wall_compare._reliable_run."""
    return np.isfinite(mean) & dw._reliable_run(mean, std, std_ceil)


def _component_with_min(keep, mean):
    """4-connected component of `keep` (a boolean mask) that contains the minimum of
    `mean` restricted to keep. Same flood fill as dw._reliable_region2d, but over an
    arbitrary mask rather than a std threshold."""
    if not keep.any():
        return keep
    i0, j0 = np.unravel_index(int(np.nanargmin(np.where(keep, mean, np.nan))), mean.shape)
    comp = np.zeros_like(keep)
    ni, nj = mean.shape
    stack = [(int(i0), int(j0))]
    comp[i0, j0] = True
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < ni and 0 <= b < nj and keep[a, b] and not comp[a, b]:
                comp[a, b] = True
                stack.append((a, b))
    return comp


def reliable_2d(mean, std, counts, std_ceil, min_visits, walker_cov=None, min_walkers=None):
    """Sampling mask (>= min_visits colvar frames per bin) + the std <= std_ceil region
    connected to the minimum, re-zeroed at its minimum. Bins outside -> NaN (drawn white).

    walker_cov / min_walkers (optional): a multi-walker ergodicity requirement. walker_cov is
    the per-bin count of independent walkers that sample the bin (column 6 of the *_pmf2d_*.dat
    tables, via load2d(..., with_cov=True)); a bin is kept only if at least min_walkers of them
    do, after which the connected component containing the minimum is re-taken. This drops
    regions reached by only one or two walkers -- densely populated but NOT equilibrated, so
    their reweighted free energy still drifts in time -- which a per-bin std ceiling cannot
    catch (two walkers that sample a bin consistently give it a LOW cross-snapshot std). Off
    (None) by default, so the main-text Figure 5 panels are unaffected; used for the SI mutant
    panels, where the S26D low-bend/high-contact corner is a 2/6-walker, late-entering artifact."""
    m = np.where(counts >= min_visits, mean, np.nan)
    out = np.where(dw._reliable_region2d(m, std, std_ceil), m, np.nan)
    if walker_cov is not None and min_walkers is not None:
        keep = np.isfinite(out) & (walker_cov >= min_walkers)
        out = np.where(_component_with_min(keep, out), out, np.nan)
    return out - np.nanmin(out)


def smooth_masked(F, sigma):
    """Light Gaussian smoothing of a 2D free-energy grid *for display only*, without letting the
    blur cross the reliability mask. Purely cosmetic: it takes the bin-to-bin ripple off the
    contour lines; it does not move a basin, and the reported minimum should still be read off the
    UNsmoothed grid.

    NaN bins (outside the sampling / std-ceiling region) stay NaN, so the white area is unchanged.
    Inside the valid region a normalised (Nadaraya-Watson) convolution is used -- the field and its
    validity mask are each blurred and then divided -- so a bin on the edge of the reliable region
    is averaged only with its valid neighbours and is not dragged toward the empty side.

    sigma is in bins; 0 or None returns F untouched (smoothing off).
    """
    if not sigma:
        return F
    from scipy.ndimage import gaussian_filter
    valid = np.isfinite(F)
    num = gaussian_filter(np.where(valid, F, 0.0), sigma, mode="nearest")
    den = gaussian_filter(valid.astype(float), sigma, mode="nearest")
    return np.where(valid, num / np.where(den > 0, den, 1.0), np.nan)


def to_bin_edges(x, y, F):
    """Extend a centre-sampled grid out to the bins' outer EDGES, so a filled contour covers the
    full extent of the histogram instead of stopping half a bin short of it.

    contourf places the surface on the grid *points*, i.e. the bin centres, so a Q_core grid whose
    last bin is centred on 0.99 (bin 0.98-1.00) paints nothing between 0.99 and the physical bound
    Q = 1. The edge value is not new information -- it is the value of the bin that reaches the
    edge -- so each border row/column is duplicated outward by half a bin. NaNs stay NaN, so the
    sampling mask and the reliability truncation are carried out to the edge unchanged.
    """
    def ext(c):
        h = 0.5 * (c[1] - c[0])
        return np.concatenate([[c[0] - h], c, [c[-1] + h]])

    G = np.pad(F, 1, mode="edge")
    return ext(np.asarray(x)), ext(np.asarray(y)), G


# ────────────────────────────────────────────────────────────────────── saving
def savepanel(fig, name, version, formats=("pdf", "png"), dpi=300, figs=FIGS,
              transparent=None):
    """figures/<system>_<observable>[_<variant>]_v<N>.<ext> -- the project naming rule.

    Background: transparent by default, in BOTH formats -- these panels are assembled into
    the paper figure and must drop cleanly onto an Inkscape canvas / slide / coloured
    background.

    Pass transparent=False for anything that is NOT going into the paper figure (diagnostic
    or comparison panels): a transparent PNG renders against whatever the viewer's background
    is, which makes axes and black text unreadable in a dark-themed editor (VS Code).
    """
    os.makedirs(figs, exist_ok=True)
    for ext in formats:
        tr = True if transparent is None else bool(transparent)
        fig.savefig(f"{figs}/{name}_{version}.{ext}", dpi=dpi, bbox_inches="tight",
                    transparent=tr,
                    **({} if tr else {"facecolor": "white", "edgecolor": "none"}))
    print(f"  saved {figs}/{name}_{version}.[{'|'.join(formats)}]"
          f"{'' if tr else '  (opaque white -- not a paper panel)'}")


# ─────────────────────────────────────────────────────── panel F1 (per replicate)
F1_LABEL = {"H3_helicity": "H3 helicity", "H3_RMSD_nm": "H3 RMSD (nm)",
            "H3_kink_deg": "H3 kink (deg)", "H3_displacement_nm": "H3 displacement (nm)",
            "HMG_helicity": "HMG helicity"}
F1_COLOR = {"A": "#4C72B0", "B": "#C44E52"}      # the two protein copies


def f1_panel(dat, out, version, obs=("H3_RMSD_nm",), figsize=(3.9, 2.7),
             n_blocks=150, title="", data=DATA, transparent=None):
    """Render the Figure-5 F1 panel (2Nhp6A H3 distortion vs time, per protein copy).

    Factored out of analysis_figure5.ipynb so the SAME plotting code can be pointed at any
    replicate's table (F1_equil_2Nhp6A_h3_rep<N>.dat) -- the replicates must be compared on
    identical axes and block averaging, which is exactly what a second copy of the code
    would not guarantee. The notebook cell keeps its own settings and calls this.

    Returns the stats dict {(obs, copy): (mean, std)} so callers can tabulate replicates.
    """
    f1 = load1d(dat, data=data)
    t = f1["time_ns"]
    copies = sorted({k.rsplit("_", 1)[1] for k in f1 if k != "time_ns"})
    obs = list(obs)

    stats = {}
    fig, axes = plt.subplots(len(obs), 1, figsize=figsize, sharex=True)
    for ax, key in zip(np.atleast_1d(axes), obs):
        for cp in copies:
            y = f1[f"{key}_{cp}"]
            ax.plot(t, y, lw=0.4, alpha=0.30, color=F1_COLOR[cp])
            k = max(1, len(y) // n_blocks)
            ax.plot(t[:len(y) // k * k].reshape(-1, k).mean(1),
                    y[:len(y) // k * k].reshape(-1, k).mean(1),
                    lw=1.4, color=F1_COLOR[cp], label=f"copy {cp}")
            stats[(key, cp)] = (float(y.mean()), float(y.std()))
        ax.set_ylabel(F1_LABEL[key])
        ax.set_xlim(0, t[-1])
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=len(copies), loc="lower left")
    np.atleast_1d(axes)[-1].set_xlabel("time (ns)")
    if title:
        np.atleast_1d(axes)[0].set_title(title)
    fig.align_ylabels(np.atleast_1d(axes))
    savepanel(fig, out, version, transparent=transparent)
    return stats
