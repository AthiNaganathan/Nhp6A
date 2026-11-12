#!/usr/bin/env python3
"""Compute every data product behind the article simulation panels and write it to analysis/article/.

This is the *analysis* half of the figure: it does all the expensive work (parsing
~25M colvar rows, ~30 `sum_hills` calls, the c(t) bias reconstruction, DSSP on the
equilibrium trajectory) and writes plain text tables. `notebooks/article/article_figures.ipynb` is the
*plotting* half: it only loads these tables, so re-styling a panel costs seconds and
never re-touches the trajectories.

Everything is written FULL-RANGE, with the cross-snapshot std and the raw bin counts
alongside the mean. The reliability truncation (std <= 5 kJ/mol, contiguous from the
minimum) and the sampling mask are applied at PLOT time -- per project convention the
.dat archives keep the whole profile.

Panels
  A   F(Q_core, n_IDR-core), apo          -- sum_hills (both CVs biased)
  B   <S_alpha> per helix vs Q_core, apo  -- c(t)-reweighted (helicity is a spectator)
  C   F(Q_core, S_alpha), apo            -- c(t)-reweighted; S_alpha is the TOTAL
                                            helicity (helix1+helix2+helix3) by default,
                                            or a single helix with --helix helix2
  D   F(bend) and F(d_e2e), DNA vs holo   -- final-bias reweighted
  E   F(n_IDR-DNA, bend), holo            -- final-bias reweighted
  F   2Nhp6A H3 distortion + bend, equil  -- unbiased MD (the MetaD trimer run fails
                                             its reweighting gate and is NOT used)
  M   m-state / mM re-binding (SI)        -- unbiased MD from the pulled starts: core-DNA
                                             COM distance vs time (1:1, 2:1 per copy), the
                                             holo references, and core PC1 (common frame)
                                             paired with d_core

Usage
    python analysis/scripts/article_data.py                  # all panels
    python analysis/scripts/article_data.py --panels D E     # just those
    python analysis/scripts/article_data.py --stride 20      # fast, for a smoke test
    python analysis/scripts/article_data.py --dna-source DNA_wall2
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reweight_angle as rw            # final-bias reweighting + weighted histograms
import reweight_metad as rwm           # rigorous c(t) (Tiwary-Parrinello) reweighting
import fes_marginal_meanstd as fm      # sum_hills up to a time cutoff, marginals
import dna_bending_wall_compare as dw  # wall-system reweighting + snapshot machinery
import equil_monitor as em             # equilibrium observables (the H3 definitions)
import article_plot as f5              # load2d / reliable_2d: the SAME masking the notebook plots
from reweight_angle import fes_1d, fes_2d

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

KB = 0.00831446261815324
# `plumed sum_hills` was run without --kt, i.e. with PLUMED's default 2.494339 kJ/mol
# (= kB*300 K). The apo reweighting is validated against those sum_hills marginals, so it
# must use the same beta or the comparison is inconsistent by construction. The holo/DNA
# pipeline stays on dna_bending_wall_compare's T = 298 K (the METAD TEMP), the beta its own
# gate was passed with. The two differ by 0.7%, far below the sampling error.
KBT_SUMHILLS = 2.494339
T_RW = KBT_SUMHILLS / KB      # 300.0 K -- apo weighted histograms
T_MTD = dw.T                  # 298.0 K -- holo/DNA

# Q_BIN MUST match GRID_BIN in plumed_meta_2D_1Nhp6A_large.dat: it is the grid the sum_hills
# reference and the reweighting gate (apo_weights) are computed on, so gate MADs stay bin-aligned.
APO_DIR = "data/1Nhp6A_apo_large"
APO_NWALK = 6
Q_MIN, Q_MAX, Q_BIN = 0.3, 1.0, 100
# Q_PLOT_BIN is the FINER q_core grid the plotted apo 2D surfaces (panels A and C) share, so both
# panels place the folded minimum in the same top bin (centre 0.9975, not A's old 100-bin 0.9965).
# It is a plot-resolution choice only -- decoupled from the gate, which stays on Q_BIN.
Q_PLOT_BIN = 140
NTC_MIN, NTC_MAX, NTC_BIN = 0, 150, 300
TEQ_NS = 5.0          # bias-filling transient discarded before reweighting
FOLDED_Q = 0.95       # q_core >= this defines the folded reference state
HELICES = ["helix1", "helix2", "helix3"]
# "total" = helix1+helix2+helix3, the whole HMG box: the same definition select_snapshots.py
# uses for the 1Nhp6A_apo_large_helicity basins, so the medoid snapshots drop straight onto
# the panel-C surface.
HELIX_CHOICES = HELICES + ["total"]
HELIX_RANGE = {"helix1": (0, 12), "helix2": (0, 10), "helix3": (0, 24),  # ~ n_res - 5
               "total": (0, 46)}                                          # their sum
NQ_BINS = 15          # q_core bins for the panel-B profile
# (q_core, helicity) bins for panel C. FINE grid (dq = 0.005, dS_alpha = 0.5) so the contours are
# not blocky, combined with a density-domain KDE smoothing (C_BW, in CV units) inside fes_2d -- the
# weighted density is Gaussian-smoothed BEFORE the log, which keeps barriers unbiased while removing
# the bin-to-bin ripple a fine raw histogram would show. The per-bin sampling count is smoothed with
# the same kernel (effective counts), so the notebook's min_visits mask still means "enough frames
# within a bandwidth". Set C_BW = [0, 0] (or --c-bw 0 0) for the raw fine histogram.
C_BINS = [Q_PLOT_BIN, 92]   # q_core bins shared with panel A (Q_PLOT_BIN); S_alpha bins panel-C only
C_BW = [0.01, 0.8]    # KDE bandwidth (q_core, S_alpha) in CV units


def helicity(apo, which):
    """S_alpha for a single helix, or the total over the three (the whole HMG box)."""
    return sum(apo[h] for h in HELICES) if which == "total" else apo[which]

HOLO_DIR = "data/1Nhp6A_wall"
IDR_DIR = "data/1IDR_wall"     # revision: IDR-only peptide (res 1-23) on the same DNA
D2_10_DIR = "data/1D2_10_wall"  # revision: Nhp6A delta2-10 deletion (renumbered 1-84)
DE2E_RANGE = (2.4, 6.0)
BEND_RANGE = (0, 90)      # bend = 180 - theta; 0 deg = straight DNA
COORD_RANGE = (0, 40)
E_BINS = [50, 50]

# Series key -> (directory under data/equil/, theta column in that system's colvar.dat).
# The key is NOT the directory name: it is the series label written into F2_equil_bend_hist.dat
# as `counts_<key>` and read back by the notebook, so it must stay stable. The directories are
# `DNA` and `1Nhp6A_holo` (not `DNA_wall`/`1Nhp6A_wall`) -- pointing the loop at the key alone
# silently found nothing for those two systems and left panel F2 showing 2Nhp6A only.
EQ_SYSTEMS = {
    "DNA_wall":    ("DNA",         2),
    "1Nhp6A_wall": ("1Nhp6A_holo", 3),
    "2Nhp6A":      ("2Nhp6A",      4),
}
EQ_REPS = 3
BEND_HIST_BINS = 180      # fine histogram -> the notebook can rebin


# ─────────────────────────────────────────────────────────────── helpers
def hills_paths(d, n=6):
    return [f"{d}/hills.dat.{i}" for i in range(n)]


def last_time_ns(path):
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 8192))
        return float(f.read().decode(errors="ignore").strip().split("\n")[-1].split()[0]) / 1000.0


def colvar_cached(paths, cols, stride=1, cachedir="analysis/cache"):
    """Pool PLUMED colvar files into one dict of arrays, cached as .npz.

    Parsing these files (up to 15M rows) dominates the runtime, so the result is cached,
    keyed on the paths, their mtimes and the stride: a re-run after new data lands
    re-parses automatically, a re-run after a code change does not.
    """
    os.makedirs(cachedir, exist_ok=True)
    key = "|".join(f"{p}:{os.path.getmtime(p):.0f}" for p in paths)
    key += "|" + ",".join(sorted(cols)) + f"|{stride}"
    cache = os.path.join(cachedir, "colvar_" + hashlib.md5(key.encode()).hexdigest()[:12] + ".npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    acc = {}
    for p in paths:
        try:                     # pandas' C parser is ~20x faster than genfromtxt here
            import pandas as pd
            df = pd.read_csv(p, sep=r"\s+", comment="#", header=None,
                             usecols=[0] + list(cols.values()), on_bad_lines="skip").dropna()
            d = {"time_ns": df[0].to_numpy() * 1e-3}
            d.update({name: df[idx].to_numpy() for name, idx in cols.items()})
        except Exception as e:   # fall back to the reference (slow, tolerant) loader
            print(f"    [pandas failed on {p} ({e}); using genfromtxt]")
            d = rw.load_colvar(p, cols)
        for k, v in d.items():
            acc.setdefault(k, []).append(v[::stride])
    out = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez_compressed(cache, **out)
    return out


def save2d(path, x, y, mean, std, counts, header):
    """Long format: cv1 cv2 F_mean F_std count -- one row per bin, full range."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    cols = np.column_stack([X.ravel(), Y.ravel(), mean.ravel(), std.ravel(), counts.ravel()])
    np.savetxt(path, cols, fmt="%14.6g", header=header)
    print(f"    wrote {path}  ({len(x)}x{len(y)} bins)")


def check_hills_monotonic(paths):
    """Duplicate hills from untrimmed -cpi restarts double-count the bias in sum_hills
    (the DATA_INVENTORY blocker, confirmed for the small-box apo sets)."""
    bad = 0
    for p in paths:
        t = np.loadtxt(p, comments=["#", "!"], usecols=0)
        bad += int((np.diff(t) <= 0).sum())
    return bad


# ─────────────────────────────────────────────────── apo sum_hills (gate reference)
def apo_sumhills(args):
    """sum_hills F(q_core, n_IDR-core) over n_snaps cumulative times. Both apo CVs are biased, so
    this is a valid FES -- but it is used only as the REFERENCE the reweighting gate validates
    against, not as a plotted panel. Panel A itself is c(t)-reweighted (see panel_A), because
    sum_hills misplaces the folded minimum: it sits at the q_core=1.0 grid bound (GRID_MAX), and a
    minimum on a boundary is reconstructed lopsidedly (hills only ever pile up on its low side), so
    the sum_hills minimum is pushed ~1-2 bins inward (0.986 vs the reweighted 0.998, which matches
    the folded median 0.9987). Returns (q, ntc, surfaces, t_max)."""
    print("[apo] sum_hills F(Q_core, n_IDR-core) -- reference for the reweighting gate")
    paths = hills_paths(APO_DIR, APO_NWALK)
    hills = [fm.load_hills(p) for p in paths]
    header = fm.hills_header(paths[0])
    t_max = max(h[:, 0].max() for h in hills) / 1000.0
    mins, maxs, bins = [Q_MIN, NTC_MIN], [Q_MAX, NTC_MAX], [Q_BIN, NTC_BIN]
    surfaces, q, ntc = [], None, None
    with tempfile.TemporaryDirectory(prefix="panelA_") as wd:
        for t_ns in np.linspace(t_max / args.n_snaps, t_max, args.n_snaps):
            f = fm.run_sumhills_upto(hills, header, t_ns * 1000, f"t{int(t_ns):05d}",
                                     wd, mins, maxs, bins)
            q, ntc, F = fm.load_fes(f, ndim=2)          # F indexed [q, ntc]
            surfaces.append(F - np.nanmin(F))
            print(f"    sum_hills up to {t_ns:7.1f} ns", flush=True)
    return q, ntc, surfaces, round(t_max, 1)


# ─────────────────────────────────────────────────────────────── panel A
def panel_A(args, meta, W, frame_t, t_max):
    """apo F(Q_core, n_IDR-core), c(t)-reweighted -- the SAME estimator as panels B and C. Both
    CVs are biased, so reweighting is valid, and unlike sum_hills it locates the folded minimum
    correctly at the q_core=1.0 bound (see apo_sumhills for why sum_hills does not)."""
    print("[A] apo F(Q_core, n_IDR-core) -- c(t)-reweighted")
    apo = load_apo(args)
    bins = [Q_PLOT_BIN, NTC_BIN]                         # finer q_core grid, shared with panel C
    #                                                      (the gate still validates on Q_BIN)
    grids, hx, hy = [], None, None
    for t_ns in np.linspace(t_max / args.n_snaps, t_max, args.n_snaps):
        m = (frame_t >= TEQ_NS * 1e3) & (frame_t <= t_ns * 1e3)
        hx, hy, F = fes_2d(apo["q_core"][m], apo["n_tail_core"][m], W[m],
                           (Q_MIN, Q_MAX), (NTC_MIN, NTC_MAX), bins, T=T_RW)
        grids.append(F)
    G = np.array(grids)
    mean = np.nanmean(G, axis=0)
    mean -= np.nanmin(mean)
    std = np.nanstd(G, axis=0)

    sel = frame_t >= TEQ_NS * 1e3
    cnt, _, _ = np.histogram2d(apo["q_core"][sel], apo["n_tail_core"][sel],
                               bins=bins, range=[(Q_MIN, Q_MAX), (NTC_MIN, NTC_MAX)])
    save2d(f"{args.out}/A_fes2d_qcore_nidr.dat", hx, hy, mean, std, cnt,
           f"apo F(Q_core, n_IDR-core), c(t)-reweighted, "
           f"{args.n_snaps} cumulative-time snapshots\n"
           "q_core  n_IDR_core  F_mean(kJ/mol)  F_std(kJ/mol)  colvar_counts")

    i, j = np.unravel_index(np.nanargmin(np.where(cnt >= 5, mean, np.nan)), mean.shape)
    print(f"    minimum at Q_core={hx[i]:.3f}, n_IDR-core={hy[j]:.1f}")
    meta["A"] = dict(data=APO_DIR, walkers=APO_NWALK, ns_per_walker=t_max,
                     method="c(t)-reweighted 2D PMF (both CVs biased)", n_snaps=args.n_snaps,
                     minimum=dict(q_core=round(float(hx[i]), 3), n_idr_core=round(float(hy[j]), 1)),
                     preliminary=t_max < 999)


_APO_CACHE = {}


def load_apo(args):
    if "d" not in _APO_CACHE:
        _APO_CACHE["d"] = colvar_cached(
            [f"{APO_DIR}/replica_{r}/colvar_r{r}.dat" for r in range(APO_NWALK)],
            {"q_core": 1, "n_tail_core": 2, "rmsd_core": 3,
             "helix1": 4, "helix2": 5, "helix3": 6}, stride=args.stride)
        print(f"    apo colvar: {_APO_CACHE['d']['q_core'].size:,} frames (stride {args.stride})")
    return _APO_CACHE["d"]


# ────────────────────────────────────────────────── apo c(t) reweighting (B, C)
def apo_weights(args, meta, q, ntc, surfaces):
    """c(t) weights for the apo run + the validation gate that must pass before any
    reweighted quantity (panels B, C) is trusted."""
    print("[B/C] apo c(t) reweighting")
    apo = load_apo(args)
    hills = [fm.load_hills(p) for p in hills_paths(APO_DIR, APO_NWALK)]
    frame_t = apo["time_ns"] * 1e3          # ps -- the HILLS deposition-time axis

    t_h, q0, n0, hgt, sig_q, sig_n, gamma = rwm.pool_hills(hills)
    print(f"    {t_h.size:,} hills | sigma=({sig_q}, {sig_n}) | gamma={gamma:.0f}")
    print(f"    reconstructing V(s,t) and c(t) over {frame_t.size:,} frames ...", flush=True)
    ct = rwm.metad_ct_weights(t_h, q0, n0, hgt, sig_q, sig_n,
                              frame_t, apo["q_core"], apo["n_tail_core"], q, ntc,
                              gamma=gamma, kBT=KBT_SUMHILLS, checkpoint_dt=1.0, batch_dt=1.0)

    # Discard the bias-filling transient: the system starts folded and is trapped there while
    # that basin fills, so those frames would carry spuriously huge weights and collapse the ESS.
    W = ct["weights"].copy()
    W[frame_t < TEQ_NS * 1e3] = 0.0
    used = int((frame_t >= TEQ_NS * 1e3).sum())
    ess = W.sum() ** 2 / np.sum(W ** 2)
    print(f"    c(t): {ct['ct'][0]:.1f} -> {ct['ct'][-1]:.1f} kJ/mol | discard {TEQ_NS:.0f} ns "
          f"-> ESS = {ess:,.0f} ({100 * ess / used:.2f}% of {used:,} frames)")

    # ---- validation gate: reweighting back onto the BIASED CVs must reproduce sum_hills
    fes_full = surfaces[-1]                                     # full-run surface, [q, ntc]
    sh_q = fm.marginal_fes(ntc, q, fes_full.T, (NTC_MIN, NTC_MAX), KBT_SUMHILLS)[1]
    sh_n = fm.marginal_fes(q, ntc, fes_full, (Q_MIN, Q_MAX), KBT_SUMHILLS)[1]
    cq, Fq = fes_1d(apo["q_core"], W, (Q_MIN, Q_MAX), Q_BIN, T=T_RW)
    cn, Fn = fes_1d(apo["n_tail_core"], W, (NTC_MIN, NTC_MAX), NTC_BIN, T=T_RW)
    shq_i, shn_i = np.interp(cq, q, sh_q), np.interp(cn, ntc, sh_n)

    sel = frame_t >= TEQ_NS * 1e3
    kq, _ = np.histogram(apo["q_core"][sel], bins=Q_BIN, range=(Q_MIN, Q_MAX))
    kn, _ = np.histogram(apo["n_tail_core"][sel], bins=NTC_BIN, range=(NTC_MIN, NTC_MAX))

    def mad(a, b, m):
        m = m & np.isfinite(a) & np.isfinite(b)
        return (float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan), int(m.sum())

    # compare only where both estimators mean anything: well sampled and thermally relevant
    mq = (kq >= 200) & (shq_i <= 30.0)
    mn = (kn >= 200) & (shn_i <= 30.0)
    mad_q, nq = mad(Fq, shq_i, mq)
    mad_n, nn = mad(Fn, shn_i, mn)
    mad_bulk, nb = mad(Fq, shq_i, mq & (cq < 0.93))
    mad_wall, nw = mad(Fq, shq_i, (cq >= 0.93) & (kq >= 200))

    passing = 1.2 * KBT_SUMHILLS
    print(f"    GATE (MAD <~ kBT = {KBT_SUMHILLS:.2f}, +20% -> {passing:.2f} kJ/mol)")
    print(f"      F(q_core)     {mad_q:5.2f} ({nq} bins) | bulk q<0.93 {mad_bulk:5.2f} ({nb}) "
          f"| wall q>=0.93 {mad_wall:5.2f} ({nw})")
    print(f"      F(n_IDR-core) {mad_n:5.2f} ({nn} bins)")
    ok = (mad_bulk <= passing) and (mad_n <= passing)
    print(f"      -> {'PASS' if ok else 'FAIL'}")
    if mad_wall > passing:
        print("      WARNING: the q>=0.93 residual is the folded minimum pinned against the")
        print("      q_core=1.0 grid edge -- a known sum_hills/grid limitation, not an estimator")
        print("      error. Panels B/C are read in the bulk Q range, where the gate passes.")
    if not ok:
        sys.exit("apo reweighting does not reproduce the sum_hills marginals in the bulk -- "
                 "panels B/C would be meaningless. Stopping.")

    meta["reweighting_apo"] = dict(estimator="c(t) Tiwary-Parrinello (REWEIGHT_METAD + CALC_RCT)",
                                   gamma=float(gamma), teq_ns=TEQ_NS, ess=float(ess),
                                   ess_percent=round(100 * ess / used, 2),
                                   gate_mad_qcore_bulk=round(mad_bulk, 2),
                                   gate_mad_nidr=round(mad_n, 2),
                                   gate_mad_qcore_wall=round(mad_wall, 2),
                                   gate_pass=bool(ok))
    return W, frame_t


# ─────────────────────────────────────────────────────────────── panels B, C
def panel_B(args, meta, W, frame_t, t_max):
    print("[B] apo <S_alpha> per helix vs Q_core")
    apo = load_apo(args)
    qb = np.linspace(Q_MIN, Q_MAX, NQ_BINS * 3 + 1)   # full range; the notebook trims for display
    qc = 0.5 * (qb[:-1] + qb[1:])

    folded = apo["q_core"] >= FOLDED_Q
    ref = {h: float(np.sum(W[folded] * apo[h][folded]) / W[folded].sum()) for h in HELICES}
    print("    folded-state <S_alpha> (q_core >= %.2f): " % FOLDED_Q
          + "  ".join(f"{h}={ref[h]:.2f}" for h in HELICES))

    def profile(mask):
        """Weighted <S_alpha>(q_core) per helix, plus the per-bin effective sample size --
        a reweighted mean carried by a handful of high-weight frames is noise, so the
        notebook drops bins below an ESS floor."""
        w = np.where(mask, W, 0.0)
        den, _ = np.histogram(apo["q_core"], bins=qb, weights=w)
        den2, _ = np.histogram(apo["q_core"], bins=qb, weights=w ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            ess_bin = np.where(den2 > 0, den ** 2 / den2, 0.0)
        out = {}
        for h in HELICES:
            num, _ = np.histogram(apo["q_core"], bins=qb, weights=w * apo[h])
            with np.errstate(invalid="ignore", divide="ignore"):
                out[h] = np.where(den > 0, num / den, np.nan)
        return out, ess_bin

    # cumulative-time snapshots: the c(t) weights are time-local, so a snapshot is just a
    # frame subset -- no re-reweighting needed
    stack = {h: [] for h in HELICES}
    ess_bin = None
    for t_ns in np.linspace(t_max / args.n_snaps, t_max, args.n_snaps):
        p, ess_bin = profile((frame_t >= TEQ_NS * 1e3) & (frame_t <= t_ns * 1e3))
        for h in HELICES:
            stack[h].append(p[h])

    cols = [qc]
    names = ["q_core"]
    for h in HELICES:
        s = np.array(stack[h])
        cols += [np.nanmean(s, axis=0), np.nanstd(s, axis=0)]
        names += [f"{h}_mean", f"{h}_std"]
    cols.append(ess_bin)
    names.append("ess_bin")

    out = f"{args.out}/B_helicity_perhelix_vs_qcore.dat"
    np.savetxt(out, np.column_stack(cols), fmt="%14.6g",
               header=("apo <S_alpha> per helix vs Q_core, c(t)-reweighted, "
                       f"{args.n_snaps} cumulative-time snapshots\n"
                       "folded-state reference (q_core >= %.2f): " % FOLDED_Q
                       + " ".join(f"{h}={ref[h]:.4f}" for h in HELICES) + "\n"
                       + "  ".join(names)))
    print(f"    wrote {out}")
    meta["B"] = dict(data=APO_DIR, method="c(t)-reweighted profile", folded_reference=ref,
                     folded_q=FOLDED_Q, n_snaps=args.n_snaps, preliminary=t_max < 999)


def panel_C(args, meta, W, frame_t, t_max):
    print(f"[C] apo F(Q_core, S_alpha) for {args.helix}")
    apo = load_apo(args)
    hr = HELIX_RANGE[args.helix]
    sa = helicity(apo, args.helix)
    bins = list(args.c_bins)
    bw = list(args.c_bw)
    smoothing = f", KDE bw ({bw[0]:g}, {bw[1]:g})" if any(bw) else ", raw histogram"
    print(f"    grid {bins[0]} x {bins[1]}  (dq = {(Q_MAX - Q_MIN) / bins[0]:.3f}, "
          f"dS_alpha = {(hr[1] - hr[0]) / bins[1]:.2f}){smoothing}")
    grids, hx, hy = [], None, None
    for t_ns in np.linspace(t_max / args.n_snaps, t_max, args.n_snaps):
        m = (frame_t >= TEQ_NS * 1e3) & (frame_t <= t_ns * 1e3)
        hx, hy, F = fes_2d(apo["q_core"][m], sa[m], W[m],
                           (Q_MIN, Q_MAX), hr, bins, T=T_RW, bw=bw)
        grids.append(F)
    G = np.array(grids)
    mean = np.nanmean(G, axis=0)
    mean -= np.nanmin(mean)
    std = np.nanstd(G, axis=0)

    sel = frame_t >= TEQ_NS * 1e3
    cnt, _, _ = np.histogram2d(apo["q_core"][sel], sa[sel],
                               bins=bins, range=[(Q_MIN, Q_MAX), hr])
    if any(bw):
        # smooth the sampling count with the SAME kernel, so the notebook's min_visits mask means
        # ">= min_visits frames within a bandwidth" -- the honest support of the smoothed surface.
        from scipy.ndimage import gaussian_filter
        dq, ds = (Q_MAX - Q_MIN) / bins[0], (hr[1] - hr[0]) / bins[1]
        cnt = gaussian_filter(cnt, (bw[0] / dq if bw[0] else 0.0,
                                    bw[1] / ds if bw[1] else 0.0), mode="nearest")
    cnt_label = "eff_counts(KDE-smoothed)" if any(bw) else "colvar_counts"
    label = "total (H1+H2+H3)" if args.helix == "total" else args.helix
    save2d(f"{args.out}/C_pmf2d_qcore_{args.helix}.dat", hx, hy, mean, std, cnt,
           f"apo F(Q_core, S_alpha[{label}]), c(t)-reweighted{smoothing}, "
           f"{args.n_snaps} cumulative-time snapshots\n"
           f"q_core  S_alpha_{args.helix}  F_mean(kJ/mol)  F_std(kJ/mol)  {cnt_label}")
    meta["C"] = dict(data=APO_DIR, method="c(t)-reweighted 2D PMF", helix=args.helix,
                     bins=bins, kde_bw=bw, n_snaps=args.n_snaps, preliminary=t_max < 999)
    if args.helix == "total":
        meta["C"]["medoids"] = copy_medoids(args)


SEG_FES = "analysis/1Nhp6A_apo_large_fes_helicity_qcore_panelC.dat"


def export_seg_fes(args):
    """Write panel C's PLOTTED surface in the 3-column form select_snapshots.py reads
    (cv1 = total_helicity, cv2 = q_core), so the basins are segmented on exactly the surface the
    figure shows.

    Why this exists: the basins used to be segmented on a finer, single-shot c(t) FES, on which the
    shallow second well splits into several sub-minima within ~4 kJ/mol of one another -- less than
    the +/-4-6 kJ/mol cross-snapshot uncertainty there. The medoid was picked from one of those
    sub-minima and so landed on the flank of the broad basin that the (coarser, snapshot-averaged)
    panel-C surface actually resolves. Segmenting on the plotted surface removes that mismatch by
    construction. Cheap: no reweighting, it just re-masks the table already on disk.
    """
    q, sa, mean, std, cnt = f5.load2d("C_pmf2d_qcore_total.dat", data=args.out)
    F = f5.reliable_2d(mean, std, cnt, args.std_ceil, args.min_visits)
    H, Q = np.meshgrid(sa, q, indexing="ij")          # cv1 = helicity, cv2 = q_core
    np.savetxt(SEG_FES, np.column_stack([H.ravel(), Q.ravel(), F.T.ravel()]), fmt="%12.5f",
               header=("panel-C surface, c(t)-reweighted, snapshot mean, "
                       f"masked at std<={args.std_ceil} kJ/mol and >={args.min_visits} visits\n"
                       "cols: total_helicity  q_core  F(kJ/mol)   (nan = outside the reliable region)"))
    n = int(np.isfinite(F).sum())
    print(f"wrote {SEG_FES}  ({n} reliable bins of {F.size})\n"
          f"  now: python select_snapshots.py --system 1Nhp6A_apo_large_helicity "
          f"--plot-only --f-ceil 20 --barrier 3")


def copy_medoids(args):
    """Bring the apo helicity-basin medoids (select_snapshots.py) next to the panel data, so
    the notebook can mark on panel C the states whose structures go into the figure. Their CVs
    are already (total_helicity, q_core) -- the panel-C axes -- and they are real frames, so
    they land where they belong whatever grid the PMF is drawn on."""
    src = "analysis/snapshots/1Nhp6A_apo_large_helicity"
    cvs = f"{src}_medoids_cvs.dat"
    if not os.path.exists(cvs):
        print(f"    [C] no medoids at {cvs} -- run select_snapshots.py "
              "--system 1Nhp6A_apo_large_helicity to mark the snapshot states")
        return None
    shutil.copy(cvs, f"{args.out}/C_medoids_cvs.dat")
    pdbs = sorted(glob.glob(f"{src}/state_*_medoid.pdb"))
    for p in pdbs:
        shutil.copy(p, os.path.join(args.out, os.path.basename(p)))
    print(f"    [C] medoids -> {args.out}/C_medoids_cvs.dat ({len(pdbs)} structures)")
    return dict(source=cvs, structures=[os.path.basename(p) for p in pdbs],
                note="basins segmented on the c(t) FES F(total_helicity, q_core); "
                     "medoid = min mean heavy-atom RMSD frame of each basin")


# ─────────────────────────────────────────────────────────────── panels D, E
def build_cfg(dna_dir):
    return {
        "DNA": dict(label="DNA alone", dir=dna_dir,
                    colvar=[f"{dna_dir}/replica_{i}/colvar_r{i}.dat" for i in range(6)],
                    hills=hills_paths(dna_dir),
                    cols={"de2e": 1, "theta": 2, "cmap": 3},
                    biased=["de2e"], gamma_col=4,
                    sh_min=[0], sh_max=[8], sh_bin=[200]),
        "holo": dict(label="Nhp6A:DNA", dir=HOLO_DIR,
                     colvar=[f"{HOLO_DIR}/replica_{i}/colvar_r{i}.dat" for i in range(6)],
                     hills=hills_paths(HOLO_DIR),
                     cols={"coord_tail_dna": 1, "de2e": 2, "theta": 3, "cmap": 4},
                     biased=["coord_tail_dna", "de2e"], gamma_col=6,
                     sh_min=[0, 0], sh_max=[150, 8], sh_bin=[150, 200]),
        # same 2D wall protocol, CVs and colvar layout as holo (LABNOTEBOOK 2026-10-08)
        "idr": dict(label="IDR:DNA", dir=IDR_DIR,
                    colvar=[f"{IDR_DIR}/replica_{i}/colvar_r{i}.dat" for i in range(6)],
                    hills=hills_paths(IDR_DIR),
                    cols={"coord_tail_dna": 1, "de2e": 2, "theta": 3, "cmap": 4},
                    biased=["coord_tail_dna", "de2e"], gamma_col=6,
                    sh_min=[0, 0], sh_max=[150, 8], sh_bin=[150, 200]),
        # same 2D wall protocol and colvar layout; its n_IDR-DNA counts 11 residues (WT 20)
        "d2_10": dict(label="\u03942-10:DNA", dir=D2_10_DIR,
                      colvar=[f"{D2_10_DIR}/replica_{i}/colvar_r{i}.dat" for i in range(6)],
                      hills=hills_paths(D2_10_DIR),
                      cols={"coord_tail_dna": 1, "de2e": 2, "theta": 3, "cmap": 4},
                      biased=["coord_tail_dna", "de2e"], gamma_col=6,
                      sh_min=[0, 0], sh_max=[150, 8], sh_bin=[150, 200]),
    }


def panels_DE(args, meta, cfgs, want_E=True):
    for name, cfg in cfgs.items():
        print(f"[D] {cfg['label']} ({cfg['dir']}) -- final-bias reweighting")
        full = colvar_cached(cfg["colvar"], cfg["cols"], stride=args.stride)
        full["theta_deg"] = np.degrees(full["theta"])
        full["bend_deg"] = 180.0 - full["theta_deg"]     # 0 deg = straight DNA

        g = dw.process(name, cfg, full=full)             # gamma, ESS, validation MAD on de2e
        flag = "OK" if g["mad"] < dw.kBT else "CHECK"
        print(f"    frames={g['nframes']:,} gamma={g['gamma']:.0f} "
              f"ESS={g['ess']:,.0f} ({100 * g['ess'] / g['nframes']:.0f}%) "
              f"MAD(de2e)={g['mad']:.2f} kJ/mol [{flag}]")
        if flag == "CHECK":
            print("    WARNING: reweighting does not reproduce the sum_hills marginal on de2e.")

        t_max = full["time_ns"].max()
        print(f"    {args.n_snaps} snapshot reweightings up to {t_max:.0f} ns ...", flush=True)
        full, snaps = dw.snapshot_weights(name, cfg, t_max, n_snaps=args.n_snaps, full=full)

        for key, rng, nb, unit in [("bend_deg", BEND_RANGE, 60, "deg"),
                                   ("de2e", DE2E_RANGE, 80, "nm")]:
            curves, x = [], None
            for s in snaps:
                x, F = fes_1d(full[key][s["mask"]], s["w"], rng, nb, T=T_MTD)
                curves.append(F)
            C = np.array(curves)
            mean = np.nanmean(C, axis=0)
            mean -= np.nanmin(mean)
            std = np.nanstd(C, axis=0)
            cnt, _ = np.histogram(full[key], bins=nb, range=rng)
            tag = "bend" if key == "bend_deg" else "de2e"
            out = f"{args.out}/D_pmf_{tag}_{name}.dat"
            np.savetxt(out, np.column_stack([x, mean, std, cnt]), fmt="%14.6g",
                       header=(f"{cfg['label']} F({tag}), final-bias reweighted, "
                               f"{args.n_snaps} snapshots, {t_max:.0f} ns/walker\n"
                               f"{tag}({unit})  F_mean(kJ/mol)  F_std(kJ/mol)  frame_counts"))
            print(f"    wrote {out}")

        meta.setdefault("D", {})[name] = dict(
            data=cfg["dir"], label=cfg["label"], ns_per_walker=round(float(t_max), 1),
            method="final-bias reweighted, snapshot mean+/-std",
            ess=float(g["ess"]), ess_percent=round(100 * g["ess"] / g["nframes"], 1),
            gate_mad_de2e=round(float(g["mad"]), 2), gate_pass=bool(flag == "OK"),
            preliminary=os.path.basename(cfg["dir"]) == "DNA_wall")

        if want_E and name == "holo":
            print("[E] holo F(n_IDR-DNA, bend)")
            grids, ex, ey = [], None, None
            for s in snaps:
                m, w = s["mask"], s["w"]
                ex, ey, F = fes_2d(full["coord_tail_dna"][m], full["bend_deg"][m], w,
                                   COORD_RANGE, BEND_RANGE, E_BINS, T=T_MTD)
                grids.append(F)
            G = np.array(grids)
            mean = np.nanmean(G, axis=0)
            mean -= np.nanmin(mean)
            std = np.nanstd(G, axis=0)
            cnt, _, _ = np.histogram2d(full["coord_tail_dna"], full["bend_deg"],
                                       bins=E_BINS, range=[COORD_RANGE, BEND_RANGE])
            save2d(f"{args.out}/E_pmf2d_idrdna_bend.dat", ex, ey, mean, std, cnt,
                   f"holo F(n_IDR-DNA, bend), final-bias reweighted, {args.n_snaps} snapshots\n"
                   "n_IDR_DNA  bend_deg(180-theta)  F_mean(kJ/mol)  F_std(kJ/mol)  frame_counts")
            meta["E"] = dict(data=HOLO_DIR, method="final-bias reweighted 2D PMF",
                             n_snaps=args.n_snaps)


# ─────────────────────────────────────────────────────────────── panel F
def panel_F(args, meta):
    """2Nhp6A from the UNBIASED equilibrium run only. The MetaD trimer run (6 x 200 ns) fails
    its reweighting gate (ESS ~ 0%, MAD = 4.7 kJ/mol), so no free energy may be quoted from it --
    which is consistent with the paper text, that only claims equilibrium observations here."""
    xtc = f"data/equil/2Nhp6A/rep{args.trimer_rep}/md.xtc"
    if not os.path.exists(xtc):
        print(f"[F1] SKIPPED -- no equilibrium trajectory at {xtc}")
    else:
        print("[F1] 2Nhp6A H3 distortion (unbiased MD, DSSP + kink/RMSD)")
        t = em.load_whole("2Nhp6A", args.trimer_rep, args.trimer_skip)
        copies = em.protein_copies(t)
        print(f"    {t.n_frames} frames, {t.time[-1]:.0f} ns, {len(copies)} protein copies")
        keys = ["HMG helicity", "H3 helicity", "H3 RMSD (nm)", "H3 kink (deg)",
                "H3 displacement (nm)"]
        cols, names = [t.time], ["time_ns"]
        summary = {}
        for ci, (r0, _) in enumerate(copies):
            cp = em.COPIES[ci]
            obs = em.structural_observables(t, r0)
            for k in keys:
                cols.append(obs[k])
                names.append(f"{k}_{cp}".replace(" ", "_").replace("(", "").replace(")", ""))
                summary[f"{k} {cp}"] = [round(float(obs[k].mean()), 3),
                                        round(float(obs[k].std()), 3)]
                print(f"    copy {cp}  {k:22s} {obs[k].mean():7.3f} +/- {obs[k].std():.3f}")
        out = f"{args.out}/F1_equil_2Nhp6A_h3.dat"
        np.savetxt(out, np.column_stack(cols), fmt="%12.5g",
                   header=("2Nhp6A unbiased equilibrium MD, rep%d, %.0f ns "
                           "(H3 = 1LWM Pro64-Ala93)\n" % (args.trimer_rep, t.time[-1])
                           + "  ".join(names)))
        print(f"    wrote {out}")
        meta["F1"] = dict(data=f"data/equil/2Nhp6A/rep{args.trimer_rep}", ns=round(float(t.time[-1]), 1),
                          method="unbiased equilibrium MD; H3 = 1LWM Pro64-Ala93",
                          mean_std=summary,
                          note="MetaD 2Nhp6A excluded by design: reweighting gate fails "
                               "(ESS~0%, MAD=4.7 kJ/mol)")

    # F2: bend angle, unbiased vs unbiased across the three systems
    print("[F2] equilibrium bend-angle distributions")
    bins = np.linspace(*BEND_RANGE, BEND_HIST_BINS + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    found, missing = {}, []
    for s, (subdir, tcol) in EQ_SYSTEMS.items():
        b = []
        for r in range(EQ_REPS):
            p = f"data/equil/{subdir}/rep{r}/colvar.dat"
            if not os.path.exists(p):
                missing.append(f"{subdir}/rep{r}")
                continue
            d = colvar_cached([p], {"theta": tcol}, stride=1)
            b.append(180.0 - np.degrees(d["theta"]))
        if b:
            found[s] = np.concatenate(b)
    if missing:
        print("    equilibrium replicas not yet available: " + ", ".join(missing))
    if not found:
        print("    SKIPPED -- no equilibrium colvars yet")
        return
    cols, names = [centres], ["bend_deg"]
    stats = {}
    for s, b in found.items():
        h, _ = np.histogram(b, bins=bins)
        cols.append(h)
        names.append(f"counts_{s}")
        stats[s] = dict(n_frames=int(b.size), mean=round(float(b.mean()), 2),
                        std=round(float(b.std()), 2), median=round(float(np.median(b)), 2))
        print(f"    {s:12s} bend = {b.mean():5.1f} +/- {b.std():4.1f} deg "
              f"(median {np.median(b):.1f}, {b.size:,} frames)")
    out = f"{args.out}/F2_equil_bend_hist.dat"
    np.savetxt(out, np.column_stack(cols), fmt="%12g",
               header=("unbiased equilibrium bend-angle histograms (bend = 180 - theta)\n"
                       + "  ".join(names)))
    print(f"    wrote {out}")
    meta["F2"] = dict(systems=stats, method="unbiased equilibrium MD, raw histograms",
                      complete=len(found) == len(EQ_SYSTEMS),
                      missing=missing)


# ─────────────────────────────────────────────────────────────── panel M (m-state SI)
# Same definition as the PLUMED monitor/pull inputs: d_core = |COM(Calpha 27-93) - COM(all DNA)|.
# Colvar columns (0 = time): 1:1 plumed_monitor_mstate.dat, 2:1 plumed_monitor_mstate_2Nhp6A.dat.
M_RUNS_1 = [f"c{k}_rep{r}" for k in (0, 1, 3, 4) for r in (0, 1)]
M_RUNS_2 = [f"c{k}_rep{r}" for k in (1, 3) for r in (0, 1)]
M_COLS_1 = {"d_core": 3, "coord_core_dna": 2, "theta": 5}
M_COLS_2 = {"d_core_A": 5, "d_core_B": 6, "coord_core_dna_A": 3, "coord_core_dna_B": 4, "theta": 8}
M_DT = 0.1                    # ns: the grid of the contact-PCA projections and the holo whole_skip10 xtc
M_CORE = (27, 93)             # PLAN.md decision 2
# solute atom layout (0-based, half-open), from the tleap topologies: protein copies of 1539 atoms,
# then DNA strand 1 (484 atoms) and strand 2 (466 atoms)
M_LAYOUT = {"1Nhp6A_holo": dict(copies=[0], strands=[(1539, 2023), (2023, 2489)]),
            "2Nhp6A": dict(copies=[0, 1539], strands=[(3078, 3562), (3562, 4028)])}
M_PCA = "analysis/mstate_probe/2Nhp6A_mstate_md_mmcompare_v2.npz"   # core-only common-frame basis
# Common length of every m-state run. c3_rep0 / c4_rep0 were extended to 500 ns; the article uses
# 250 ns for all runs (user, 2026-11-06), so every table is cut here and the bound-mode labels
# come from the register-Q table computed on the cut runs (mstate_contact_pca.py --t-max 250).
M_T_MAX = 250.0
M_QNATIVE = "analysis/mstate_probe/1Nhp6A_mstate_md_qnative_t250ns.dat"


def _min_image(v, box):
    """Shortest periodic image of displacement vectors v (n,3); box (n,3,3), rows = box vectors.
    -pbc whole leaves molecules in arbitrary images (raw COM separations > 13 nm seen in the holo
    runs), so the +-1 shell search is repeated until no vector shortens."""
    import itertools
    sh = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    v = np.array(v, dtype=float)
    for _ in range(10):
        cand = v[:, None, :] + np.einsum("ki,nij->nkj", sh, box)
        best = cand[np.arange(len(v)), np.argmin((cand ** 2).sum(-1), 1)]
        if np.allclose(best, v):
            return v
        v = best
    raise RuntimeError("minimum image did not converge")


def core_dna_distance(t, copy_starts, strands, core=M_CORE):
    """d_core per protein copy on a trajectory whose molecules are whole (trjconv -pbc whole) but
    may sit in different images. The duplex is two molecules, so strand 2 is first welded onto
    strand 1 by the minimum image of their COMs (project memory: -pbc whole is not enough), then
    each core COM is taken to the duplex COM by minimum image. mdtraj element masses; checked
    against PLUMED step 0 on the 8 m-state pull starts to <= 0.008 nm."""
    def com(idx):
        m = np.array([t.top.atom(i).element.mass for i in idx])
        return (t.xyz[:, idx] * m[:, None]).sum(1) / m.sum(), m.sum()
    box = t.unitcell_vectors
    (c1, m1), (c2, m2) = (com(range(*s)) for s in strands)
    c2 = c1 + _min_image(c2 - c1, box)
    dna = (m1 * c1 + m2 * c2) / (m1 + m2)
    out = []
    for lo in copy_starts:
        r0 = t.top.atom(lo).residue.index
        ca = [a.index for a in t.top.atoms if a.name == "CA" and lo <= a.index < lo + 1539
              and core[0] <= a.residue.index - r0 + 1 <= core[1]]
        assert len(ca) == core[1] - core[0] + 1, len(ca)
        out.append(np.linalg.norm(_min_image(com(ca)[0] - dna, box), axis=1))
    return out


def _on_grid(d, dt=M_DT):
    """Point samples of a colvar dict on the dt grid (restart overlaps: the last row wins)."""
    k = np.rint(d["time_ns"] / dt).astype(int)
    on = np.isclose(d["time_ns"], k * dt, rtol=0, atol=1e-6)   # rtol=0: default 1e-5 = 2.5 ps at 250 ns
    k = k[on]
    _, last = np.unique(k[::-1], return_index=True)
    sel = np.flatnonzero(on)[::-1][last]
    return {name: v[sel] for name, v in d.items()}


def panel_M(args, meta):
    """SI figure on the m / mM probes. Unbiased MD only: no weights, so time series and
    last-window statistics, never populations."""
    out = {}
    # (1) m-state colvars on the 0.1 ns grid
    for tag, base, runs, cols in [("1to1", "data/mstate/md", M_RUNS_1, M_COLS_1),
                                  ("2to1", "data/mstate_2Nhp6A/md", M_RUNS_2, M_COLS_2)]:
        sysname = "1Nhp6A" if tag == "1to1" else "2Nhp6A"
        print(f"[M] {sysname} m-state colvars")
        series = {}
        for run in runs:
            k, r = run.split("_")
            p = f"{base}/{k}/{r}/colvar_{sysname}_mstate_md_{run}.dat"
            g = _on_grid(colvar_cached([p], cols, stride=1))
            keep = g["time_ns"] <= M_T_MAX + 1e-6
            g = {c: v[keep] for c, v in g.items()}
            g["bend"] = 180.0 - np.degrees(g.pop("theta"))
            series[run] = g
            print(f"    {run}: {g['time_ns'][-1]:6.1f} ns, {g['time_ns'].size} points, "
                  + ", ".join(f"{c} t0 {g[c][0]:.3f}" for c in g if c.startswith("d_core")))
        tmax = max(g["time_ns"][-1] for g in series.values())
        tgrid = np.round(np.arange(0, tmax + M_DT / 2, M_DT), 6)
        colsout, names = [tgrid], ["time_ns"]
        for run, g in series.items():
            idx = np.rint(g["time_ns"] / M_DT).astype(int)
            for c in [c for c in g if c != "time_ns"]:
                y = np.full(tgrid.size, np.nan)
                y[idx] = g[c]
                colsout.append(y)
                names.append(f"{c}_{run}")
        f = f"{args.out}/M_mstate_{tag}_timeseries.dat"
        np.savetxt(f, np.column_stack(colsout), fmt="%10.5g",
                   header=(f"{sysname} m-state unbiased MD (pulled starts), 0.1 ns point samples; "
                           "NaN past a run's end. d_core = |COM(CA 27-93) - COM(DNA)| (nm), "
                           "bend = 180 - theta (deg)\n" + "  ".join(names)))
        print(f"    wrote {f}")
        out[tag] = series
        meta[f"M_{tag}"] = dict(data=base, t_max_ns=M_T_MAX,
                                runs={r: round(float(g["time_ns"][-1]), 1) for r, g in series.items()})

    # (2) holo references from the unbiased equilibrium campaign (whole_skip10 = 0.1 ns)
    holo = {}
    colsout, names = None, []
    for system, reps in [("1Nhp6A_holo", (0, 1, 2)), ("2Nhp6A", (0, 1, 2))]:
        lay = M_LAYOUT[system]
        for rep in reps:
            t = em.load_whole(system, rep, skip=10)
            ds = core_dna_distance(t, lay["copies"], lay["strands"])
            if colsout is None:
                colsout = [t.time]
                names = ["time_ns"]
            n = min(len(colsout[0]), t.n_frames)
            colsout = [c[:n] for c in colsout]
            for cp, d in zip("AB", ds):
                key = f"{system}_rep{rep}_{cp}"
                holo[key] = (t.time, d)
                colsout.append(d[:n])
                names.append(f"d_core_{key}")
                late = d[t.time >= 20]
                print(f"    holo {key}: d_core median {np.median(late):.3f} "
                      f"[p5 {np.percentile(late, 5):.3f}, p95 {np.percentile(late, 95):.3f}] (t>=20 ns)")
    f = f"{args.out}/M_holo_dcore.dat"
    np.savetxt(f, np.column_stack(colsout), fmt="%10.5g",
               header=("holo unbiased equilibrium MD (data/equil), 0.1 ns, d_core per copy (nm); "
                       "molecules whole + min-image duplex weld\n" + "  ".join(names)))
    print(f"    wrote {f}")
    meta["M_holo"] = {k: [round(float(np.percentile(d[tt >= 20], q)), 3) for q in (5, 50, 95)]
                      for k, (tt, d) in holo.items()}

    # (3) core PC1 (core-only common native-relative frame) paired with d_core and bend
    z = np.load(M_PCA)
    pcols = list(z["proj_columns"])
    rows = []
    for key in [k for k in z.files if "|" in k]:
        stoich, run, cp = key.split("|")
        P = z[key]
        if "m-state" in stoich:
            P = P[P[:, pcols.index("time_ns")] <= M_T_MAX + 1e-6]
        tP = P[:, pcols.index("time_ns")]
        if "m-state" in stoich:
            g = out["1to1" if stoich.startswith("1:1") else "2to1"][run]
            dkey = "d_core" if stoich.startswith("1:1") else f"d_core_{cp}"
            tt, dd, bb = g["time_ns"], g[dkey], g["bend"]
        else:
            system = "1Nhp6A_holo" if stoich.startswith("1:1") else "2Nhp6A"
            tt, dd = holo[f"{system}_rep{run}_{cp or 'A'}"]
            bb = np.full(tt.size, np.nan)       # bend not needed for the holo clouds
        i = np.searchsorted(np.round(tt, 4), np.round(tP, 4))
        ok = (i < tt.size) & np.isclose(tt[np.minimum(i, tt.size - 1)], tP, rtol=0, atol=1e-4)
        for j in np.flatnonzero(ok):
            rows.append((stoich.replace(" ", "_"), run, cp or "A", tP[j],
                         P[j, pcols.index("PC1")], P[j, pcols.index("Q_native")],
                         P[j, pcols.index("Q_best")], dd[i[j]], bb[i[j]]))
        if not ok.all():
            print(f"    [M3] {key}: {np.count_nonzero(~ok)} PCA frames without a d_core sample")
    f = f"{args.out}/M_pc1_dcore.dat"
    with open(f, "w") as fh:
        fh.write(f"# core PC1 ({os.path.basename(M_PCA)}, core-only common frame) paired with "
                 "d_core (nm) and bend (deg) at the same time; Q_native = 30 native core-DNA pairs in the "
                 "copy's own register (copy B: flip -2 bp), Q_best = per-frame max over the 10 "
                 "register variants\n"
                 "# set run copy time_ns PC1 Q_native Q_best d_core bend\n")
        for r in rows:
            fh.write("%-12s %-8s %s %8.1f %8.4f %7.4f %7.4f %7.4f %7.2f\n" % r)
    print(f"    wrote {f} ({len(rows)} rows)")
    meta["M_pc1"] = dict(basis=M_PCA, n_rows=len(rows))

    # (4) 1:1 bound-mode labels over the last 10 ns of the CUT runs, with the same rule as the
    # survival table (equil_timeseries.bound_mode: best register-variant native Q >= 0.5)
    import equil_timeseries as et
    hdr = [l for l in open(M_QNATIVE) if l.startswith("# run")][-1].split()[2:]
    regs = [h.replace("_bp", "").replace("native", "0") for h in hdr]
    f = f"{args.out}/M_mstate_1to1_modes.dat"
    labels = {}
    with open(f, "w") as fh:
        fh.write(f"# 1:1 m-state bound mode over {M_T_MAX - 10:g}-{M_T_MAX:g} ns, from {M_QNATIVE} "
                 f"(rule: equil_timeseries.bound_mode, REGISTER_Q {et.REGISTER_Q})\n"
                 "# run mode register q_best\n")
        for line in open(M_QNATIVE):
            if line.startswith("#") or not line.strip():
                continue
            run, q = line.split()[0], np.array(line.split()[1:], float)
            reg, qb, mode, _ = et.bound_mode((regs, q), {"H1": 1.0})
            labels[run] = mode
            fh.write(f"{run:8s} {mode:10s} {reg:6s} {qb:.3f}\n")
            print(f"    {run}: {mode} ({reg}), Q_best {qb:.3f}")
    print(f"    wrote {f}")
    meta["M_modes"] = dict(source=M_QNATIVE, t_max_ns=M_T_MAX, labels=labels)


# ─────────────────────────────────────────────────────────────── main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panels", nargs="+", default=["A", "B", "C", "D", "E", "F"],
                   choices=["A", "B", "C", "D", "E", "F", "M"])
    p.add_argument("--out", default="analysis/article")
    p.add_argument("--stride", type=int, default=1,
                   help="colvar frame stride (>1 for a fast smoke test; 1 for the real thing)")
    p.add_argument("--n-snaps", type=int, default=5,
                   help="cumulative-time snapshots behind every mean +/- std band")
    p.add_argument("--dna-source", default="auto",
                   help="'auto' -> DNA_wall2 if present, else DNA_wall (the buggy dna_end2)")
    p.add_argument("--helix", default="total", choices=HELIX_CHOICES,
                   help="helicity shown in panel C: 'total' (H1+H2+H3, default) or one helix")
    p.add_argument("--c-bw", nargs=2, type=float, default=C_BW, metavar=("BWQ", "BWS"),
                   help="panel-C KDE bandwidth in CV units (q_core, S_alpha); 0 0 = raw histogram")
    p.add_argument("--c-bins", nargs=2, type=int, default=C_BINS, metavar=("NQ", "NS"),
                   help=f"panel-C grid: (q_core, helicity) bins (default {C_BINS[0]} x {C_BINS[1]})")
    p.add_argument("--d-systems", nargs="+", default=["DNA", "holo"],
                   choices=["DNA", "holo", "idr", "d2_10"],
                   help="panel-D systems to (re)compute; e.g. '--panels D --d-systems idr' adds "
                        "the IDR-only curve without touching the published DNA/holo tables")
    p.add_argument("--trimer-rep", type=int, default=0)
    p.add_argument("--trimer-skip", type=int, default=10)
    p.add_argument("--export-seg-fes", action="store_true",
                   help="write panel C's plotted surface for select_snapshots.py and exit "
                        "(no reweighting; re-masks the existing table)")
    p.add_argument("--std-ceil", type=float, default=10.0,
                   help="kJ/mol, reliability truncation -- must match the notebook")
    p.add_argument("--min-visits", type=int, default=5,
                   help="colvar frames per bin -- must match the notebook")
    args = p.parse_args()

    if not os.path.isdir("analysis/scripts"):
        sys.exit("run from the project root (HMG/)")
    os.makedirs(args.out, exist_ok=True)

    if args.export_seg_fes:
        export_seg_fes(args)
        return

    dna_dir = ("data/DNA_wall2" if os.path.isdir("data/DNA_wall2") else "data/DNA_wall") \
        if args.dna_source == "auto" else \
        (args.dna_source if args.dna_source.startswith("data/") else f"data/{args.dna_source}")
    dna_prelim = os.path.basename(dna_dir) == "DNA_wall"

    # A partial re-run (--panels C) must not wipe the provenance of the panels it did not
    # recompute, so start from the existing meta.json and overwrite only what we regenerate.
    meta = {}
    if os.path.exists(f"{args.out}/meta.json"):
        with open(f"{args.out}/meta.json") as fh:
            meta = json.load(fh)
    stamp = f"{datetime.datetime.now():%Y-%m-%d %H:%M}"
    meta.update({"generated": stamp, "host": socket.gethostname(), "stride": args.stride,
                 "n_snaps": args.n_snaps, "dna_source": dna_dir})
    meta["panels_last_run"] = {**meta.get("panels_last_run", {}),
                               **{p: stamp for p in args.panels}}

    print("=" * 78)
    print("dataset                       walkers  ns/walker   non-monotonic hills")
    for tag, d, n in [("apo  (1Nhp6A_apo_large)", APO_DIR, APO_NWALK),
                      ("holo (1Nhp6A_wall)", HOLO_DIR, 6),
                      (f"DNA  ({os.path.basename(dna_dir)})", dna_dir, 6)]:
        hp = hills_paths(d, n)
        if not all(os.path.exists(x) for x in hp):
            print(f"{tag:28s}  MISSING")
            continue
        lens = [last_time_ns(x) for x in hp]
        bad = check_hills_monotonic(hp)
        print(f"{tag:28s} {n:5d} {np.mean(lens):10.1f} {bad:20d}"
              + ("  <-- DE-DUP BEFORE USE" if bad else ""))
    if dna_prelim:
        print("\n" + "!" * 78)
        print("PRELIMINARY: the DNA-only reference is 'DNA_wall', built with the BUGGY dna_end2")
        print("(4 backbone atoms, not the terminal base pair). It biases BOTH de2e and theta --")
        print("i.e. both quantities compared in panel D. The corrected re-run is DNA_wall2")
        print("(SLURM 9180, calmodulin). Do not quote free-DNA numbers from this run.")
        print("  rsync -av calmodulin:~/Research/Projects/Colabs/HMG/data/DNA_wall2/ data/DNA_wall2/")
        print("!" * 78)
    print("=" * 78)

    q = ntc = surfaces = W = frame_t = None
    t_max_apo = None
    need_apo = bool({"A", "B", "C"} & set(args.panels))   # A is now reweighted too, like B and C

    if need_apo:
        q, ntc, surfaces, t_max_apo = apo_sumhills(args)  # sum_hills = the reweighting-gate reference
        W, frame_t = apo_weights(args, meta, q, ntc, surfaces)
    if "A" in args.panels:
        panel_A(args, meta, W, frame_t, t_max_apo)
    if "B" in args.panels:
        panel_B(args, meta, W, frame_t, t_max_apo)
    if "C" in args.panels:
        panel_C(args, meta, W, frame_t, t_max_apo)
    if {"D", "E"} & set(args.panels):
        cfgs = {k: v for k, v in build_cfg(dna_dir).items() if k in args.d_systems}
        panels_DE(args, meta, cfgs, want_E="E" in args.panels)
    if "F" in args.panels:
        panel_F(args, meta)
    if "M" in args.panels:
        panel_M(args, meta)

    meta["flags"] = []
    if dna_prelim:
        meta["flags"].append("PRELIMINARY: DNA-only reference uses the buggy dna_end2 (DNA_wall); "
                             "re-run against DNA_wall2 (SLURM 9180) before submission.")
    if meta.get("A", {}).get("preliminary"):
        meta["flags"].append("PRELIMINARY: apo is 500 ns/walker; insulin (SLURM 3821) is "
                             "extending it to 1000 ns.")
    meta["flags"].append("2Nhp6A MetaD excluded by design (reweighting gate fails: ESS~0%, "
                         "MAD=4.7 kJ/mol); panel F uses unbiased equilibrium MD only.")

    with open(f"{args.out}/meta.json", "w") as f:
        # numpy scalars (np.bool_, np.float64) are not JSON-serializable
        json.dump(meta, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nwrote {args.out}/meta.json")
    for fl in meta["flags"]:
        print(f"  FLAG: {fl}")


if __name__ == "__main__":
    main()
