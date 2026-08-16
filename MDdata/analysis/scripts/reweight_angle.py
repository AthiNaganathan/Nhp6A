"""
Final-bias reweighting of a spectator CV (DNA bending angle theta) from
well-tempered metadynamics colvar files.

The bending angle theta = ANGLE(end1, COM_all_DNA, end2) is recorded in the
colvar files but was NOT biased (biased CVs: coord_tail_dna and/or de2e).
We recover the unbiased free energy along theta (and 2D maps theta vs de2e,
theta vs coord_tail_dna) by weighting each frame with the converged WT bias:

    w_i = exp(+beta * V(s_i))  with  V(s) = -(1 - 1/gamma) F(s)
        = exp(-(1 - 1/gamma) F(s_i) / kBT)

F(s) is obtained from `plumed sum_hills` over the biased CV(s) and
interpolated at each frame's biased-CV value(s).  This matches the
final-bias (Branduardi 2012) estimator and is consistent with the existing
2D FES notebooks in this project (which use sum_hills directly).

Validation: reweighting back onto a *biased* CV must reproduce the sum_hills
marginal along that CV.  Use validate_against_sumhills() as the gate before
trusting F(theta).

Units: energies kJ/mol; theta stored in radians (helpers return degrees).
"""
import os
import subprocess
import numpy as np
from scipy.interpolate import interp1d, griddata

KB = 0.00831446261815324  # kJ/mol/K


def kBT(T=298.0):
    return KB * T


# ---------------------------------------------------------------- loading
def load_colvar(path, cols):
    """Load a PLUMED colvar file.

    cols: dict {name: column_index}.  Column 0 is assumed to be time in ps;
    it is returned as 'time_ns'.  Rows with non-finite values in any
    requested column (or time) are dropped.
    """
    d = np.genfromtxt(path, comments="#", invalid_raise=False,
                      filling_values=np.nan)
    if d.ndim == 1:
        d = d[None, :]
    keep = np.isfinite(d[:, 0])
    for idx in cols.values():
        keep &= np.isfinite(d[:, idx])
    d = d[keep]
    out = {"time_ns": d[:, 0] * 1e-3}
    for name, idx in cols.items():
        out[name] = d[:, idx]
    return out


def hills_gamma(path, biasf_col):
    """Read the bias factor (gamma) from the first data line of a hills file."""
    with open(path) as f:
        for line in f:
            if line.startswith(("#", "!")) or not line.strip():
                continue
            return float(line.split()[biasf_col])
    raise RuntimeError(f"no data line in {path}")


# ------------------------------------------------------------- sum_hills
def run_sumhills(hills_files, outfile, mins, maxs, bins, mintozero=True):
    """Run `plumed sum_hills`. hills_files: list of paths (multiwalker)."""
    cmd = (f"plumed sum_hills --hills {','.join(hills_files)} "
           f"--outfile {outfile} "
           f"--min {','.join(map(str, mins))} "
           f"--max {','.join(map(str, maxs))} "
           f"--bin {','.join(map(str, bins))}")
    if mintozero:
        cmd += " --mintozero"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"sum_hills failed:\n{r.stderr[-1000:]}")
    return outfile


def fes_at_points(fes_path, points, ndim):
    """Interpolate a sum_hills FES at scattered frame points.

    ndim=1: points shape (N,); ndim>=2: points shape (N, ndim).
    Robust to PLUMED grid ordering (uses griddata / interp1d on raw columns).
    Returns F values (NaN outside the convex hull / grid range).
    """
    d = np.loadtxt(fes_path, comments="#")
    if ndim == 1:
        f = interp1d(d[:, 0], d[:, 1], bounds_error=False, fill_value=np.nan)
        return f(points)
    return griddata(d[:, :ndim], d[:, ndim], points, method="linear")


# ----------------------------------------------------------- reweighting
def frame_weights(F_at_frames, gamma, T=298.0):
    """Final-bias weights w_i = exp(-(1-1/gamma) F_i / kBT), normalised so
    max(w)=1 for numerical stability.  Frames with non-finite F get w=0."""
    F = np.asarray(F_at_frames, float)
    valid = np.isfinite(F)
    arg = np.full_like(F, -np.inf)
    arg[valid] = -(1.0 - 1.0 / gamma) * F[valid] / kBT(T)
    arg[valid] -= np.max(arg[valid])
    w = np.zeros_like(F)
    w[valid] = np.exp(arg[valid])
    return w


def fes_1d(x, w, rng, nbins, T=298.0):
    """Weighted 1D FES. Returns (bin_centers, F) with min shifted to 0."""
    h, edges = np.histogram(x, bins=nbins, range=rng, weights=w)
    c = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore"):
        F = -kBT(T) * np.log(h)
    F[h == 0] = np.nan
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    return c, F


def fes_2d(x, y, w, rngx, rngy, nbins, T=298.0, bw=None):
    """Weighted 2D FES. Returns (xc, yc, F) with F shape (nx, ny).

    If bw=(bwx, bwy) is given (a Gaussian bandwidth in CV units, per axis), the weighted
    DENSITY is smoothed at that bandwidth before the log -- i.e. a binned kernel density
    estimate on a fine grid, rather than a raw histogram. Smoothing the density (not the free
    energy) is what keeps barrier heights unbiased: -kT ln(H) is convex, so blurring it directly
    would systematically fill in wells. Edge mode is 'nearest', so a CV at a hard physical bound
    (e.g. Q_core = 1) is not depleted by zero-padding beyond the bound. A component of bw that is
    0 or None leaves that axis unsmoothed."""
    H, xe, ye = np.histogram2d(x, y, bins=nbins, range=[rngx, rngy], weights=w)
    if bw is not None and any(bw):
        from scipy.ndimage import gaussian_filter
        dx = (rngx[1] - rngx[0]) / H.shape[0]
        dy = (rngy[1] - rngy[0]) / H.shape[1]
        sig = (bw[0] / dx if bw[0] else 0.0, bw[1] / dy if bw[1] else 0.0)
        H = gaussian_filter(H, sig, mode="nearest")
    with np.errstate(divide="ignore"):
        F = -kBT(T) * np.log(H)
    F[H <= 0] = np.nan
    if np.any(np.isfinite(F)):
        F -= np.nanmin(F)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return xc, yc, F


# ------------------------------------------------------------ validation
def sumhills_marginal_1d(fes_path, ndim, project_axis, window=None, T=298.0):
    """Boltzmann-marginalise a sum_hills FES onto one CV.

    ndim=1 -> just return the curve. ndim=2 -> integrate out the other CV
    (optionally within `window` on that other CV). project_axis: 0 or 1
    (which of the two columns to KEEP).  Returns (cv, F) min-shifted to 0.
    """
    d = np.loadtxt(fes_path, comments="#")
    if ndim == 1:
        return d[:, 0], d[:, 1] - np.nanmin(d[:, 1])
    keep = project_axis
    other = 1 - project_axis
    cols = d[:, :2]
    Fcol = d[:, 2]
    if window is not None:
        m = (cols[:, other] >= window[0]) & (cols[:, other] <= window[1])
        cols, Fcol = cols[m], Fcol[m]
    cv_vals = np.unique(cols[:, keep])
    out = np.full(cv_vals.shape, np.nan)
    for i, v in enumerate(cv_vals):
        sel = cols[:, keep] == v
        if not sel.any():
            continue
        z = np.nansum(np.exp(-Fcol[sel] / kBT(T)))
        if z > 0:
            out[i] = -kBT(T) * np.log(z)
    out -= np.nanmin(out)
    return cv_vals, out
