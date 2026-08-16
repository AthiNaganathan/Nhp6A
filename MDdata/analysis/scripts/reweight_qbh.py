#!/usr/bin/env python3
"""
reweight_qbh.py -- 2D PMF of a *genuine* Best-Hummer Q against n_IDR-core,
by rigorous c(t) reweighting of the apo well-tempered metadynamics.

Motivation
----------
The biased CV `q_core` is a Ca-only contact map with a 0.65 nm native cutoff
(gen_plumed_inputs.py: native_contacts(cutoff_nm=0.65, min_sep=4)).  Measured on
the reference structure, that map is a strict SUBSET of the project's own
all-atom map: 67 residue pairs vs 103, and only 14 tertiary (|i-j|>=10) vs 43.
It is nearly blind to the H1-H2 interface (2 contacts vs 17) and completely
blind to H2-H3 (0 vs 2), while 75% of its weight sits on i,i+4 intra-helical
pairs.  Consequently `q_core` falls steeply on helix fraying but barely responds
to loss of tertiary packing, and below q_core ~ 0.6 it decouples from RMSD
(RMSD moves < 0.4 A while q_core drops four-fold).

This script therefore recomputes Q as an UNBIASED SPECTATOR using the project's
own all-atom Best-Hummer protocol (contact_map_setup.ipynb: all atoms incl. H,
0.30 nm cutoff, BETA=50 nm^-1, LAMBDA=1.4, residue separation > 3) and, as an
independent cross-check, the literature default (heavy atoms, 0.45 nm, λ=1.8;
Best, Hummer & Eaton PNAS 2013).

Because Q_BH is a spectator, every number here lives or dies by the reweighting
gate.  The gate is the same one figure5_data.py applies: reweight back onto the
BIASED CVs and require the result to reproduce the sum_hills marginals to within
MAD <~ kBT + 20% = 2.99 kJ/mol.  IT IS REPORTED WHETHER IT PASSES OR NOT and the
PMF is written either way, flagged accordingly -- never silently.

Hard sampling limit, stated up front
------------------------------------
Q_BH needs coordinates, so it can only be evaluated on trajectory frames
(200 ps stride, 5001 per walker = 30,006 total).  The biased-CV analyses use the
0.2 ps colvar, i.e. ~1000x more frames.  The ESS reported here is therefore
NOT comparable to the colvar-based gates elsewhere in the project and is
expected to be far smaller.

Outputs
-------
  analysis/1Nhp6A_apo_large_qbh_cache.npz        per-frame Q_BH (both defs) + t
  analysis/1Nhp6A_apo_large_pmf2d_qbh_nidr.dat   2D PMF, mean and std over snapshots
  figures/1Nhp6A_apo_large_pmf2d_qbh_nidr_v<N>.png

Usage
-----
  reweight_qbh.py --version 1
  reweight_qbh.py --version 1 --recompute-q     # ignore the Q_BH cache
"""

import argparse
import itertools
import os
import sys

import numpy as np
import mdtraj as md
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import reweight_metad as rwm

APO_DIR = "data/1Nhp6A_apo_large"
NWALK = 6
PREFIX = "1Nhp6A_parambsc1_tip3p_ions0.15_meta_2D"
NATIVE = "1Nhp6A_parambsc1_large.pdb"
TOPPDB = f"{APO_DIR}/{PREFIX}_proc.pdb"

KBT = 2.494339            # kJ/mol, the value used for sum_hills throughout
GATE = 2.99               # kBT + 20%
TEQ_NS = 5.0              # bias-filling transient discarded (as figure5_data.py)
CORE_LO, CORE_HI = 21, 93
Q_MIN, Q_MAX, Q_BIN = 0.3, 1.0, 100
NTC_MIN, NTC_MAX, NTC_BIN = 0, 150, 300
BETA_SW = 50.0

CACHE = "analysis/1Nhp6A_apo_large_qbh_cache.npz"


# ---------------------------------------------------------------- contact maps
def build_map(native, sel_atoms, cutoff, minsep_res=4):
    """Best-Hummer native contact map over ATOM pairs, exactly as
    contact_map_setup.ipynb: pairs separated by more than 3 residues whose
    native distance is below `cutoff`."""
    idx = np.asarray(sel_atoms)
    res = np.array([native.top.atom(i).residue.index for i in idx])
    pairs = np.array([(idx[a], idx[b])
                      for a, b in itertools.combinations(range(len(idx)), 2)
                      if abs(res[a] - res[b]) >= minsep_res], dtype=int)
    d0 = md.compute_distances(native[0], pairs, periodic=False)[0]
    keep = d0 < cutoff
    return pairs[keep], d0[keep]


def q_of(traj, pairs, ref, lam):
    d = md.compute_distances(traj, pairs, periodic=False)
    return (1.0 / (1.0 + np.exp(BETA_SW * (d - lam * ref)))).mean(axis=1)


def load_driver(dirname, tmax_ps):
    """Q_BH from `plumed driver` run over the RAW 2 ps trajectories: 100x more
    frames than the 200 ps processed copies, which is what the reweighting ESS
    needs.

    Truncated at `tmax_ps` so that frames written past the end of the hills/colvar
    snapshot being reweighted (e.g. if the run was extended after the driver pass)
    are not included.
    """
    Qa, Qh, T, R = [], [], [], []
    for r in range(NWALK):
        p = os.path.join(dirname, f"qbh_r{r}.dat")
        d = np.loadtxt(p, comments="#")
        m = d[:, 0] <= tmax_ps + 1e-6
        Qa.append(d[m, 1]); Qh.append(d[m, 2])
        T.append(d[m, 0]); R.append(np.full(int(m.sum()), r))
        print(f"    r{r}: {len(d):,} rows -> {int(m.sum()):,} kept "
              f"(t <= {tmax_ps/1e3:.0f} ns)")
    out = {"q_all": np.concatenate(Qa), "q_heavy": np.concatenate(Qh),
           "t_ps": np.concatenate(T), "rep": np.concatenate(R)}
    print(f"    total {out['t_ps'].size:,} frames at "
          f"{out['t_ps'][1]-out['t_ps'][0]:.0f} ps stride")
    return out


def compute_q(recompute):
    if os.path.exists(CACHE) and not recompute:
        z = np.load(CACHE)
        print(f"  loaded Q_BH cache {CACHE} ({z['t_ps'].size:,} frames)")
        return {k: z[k] for k in z.files}

    native = md.load(NATIVE)
    top = md.load(TOPPDB)
    if native.n_atoms != top.n_atoms:
        native = native.atom_slice(range(top.n_atoms))
    core_all = [a.index for a in top.top.atoms
                if CORE_LO <= a.residue.resSeq <= CORE_HI]
    core_hv = [i for i in core_all if top.top.atom(i).element.symbol != "H"]

    p_all, r_all = build_map(native, core_all, 0.30)
    p_hv, r_hv = build_map(native, core_hv, 0.45)
    print(f"  project all-atom map (0.30 nm, incl. H): {len(p_all):,} atom pairs")
    print(f"  standard heavy map    (0.45 nm)        : {len(p_hv):,} atom pairs")

    Qa, Qh, T, R = [], [], [], []
    for r in range(NWALK):
        t = md.load(f"{APO_DIR}/replica_{r}/{PREFIX}_r{r}_proc.xtc", top=TOPPDB)
        Qa.append(q_of(t, p_all, r_all, 1.4))
        Qh.append(q_of(t, p_hv, r_hv, 1.8))
        T.append(t.time)
        R.append(np.full(t.n_frames, r))
        print(f"    r{r}: {t.n_frames} frames  <Q_all>={Qa[-1].mean():.3f}  "
              f"<Q_heavy>={Qh[-1].mean():.3f}")
    out = {"q_all": np.concatenate(Qa), "q_heavy": np.concatenate(Qh),
           "t_ps": np.concatenate(T), "rep": np.concatenate(R)}
    np.savez_compressed(CACHE, **out)
    print(f"  cached -> {CACHE}")
    return out


# ---------------------------------------------------------------- colvar match
COLVAR_EXTRA = ["rmsd_core", "helix1", "helix2", "helix3"]


def colvar_at(times_ps, rep, extra=False):
    """q_core / n_tail_core (and optionally rmsd_core + the three helices)
    at the trajectory frame times, per walker."""
    names = "time q_core n_tail_core rmsd_core helix1 helix2 helix3".split()
    q = np.full(times_ps.size, np.nan)
    n = np.full(times_ps.size, np.nan)
    ex = {k: np.full(times_ps.size, np.nan) for k in COLVAR_EXTRA} if extra else {}
    for r in range(NWALK):
        d = pd.read_csv(f"{APO_DIR}/replica_{r}/colvar_r{r}.dat", sep=r"\s+",
                        comment="#", names=names, header=None, engine="c",
                        on_bad_lines="skip")
        d = (d.apply(pd.to_numeric, errors="coerce").dropna(subset=["time"])
               .drop_duplicates("time", keep="last"))
        d["k"] = d.time.round(1)
        m = d.set_index("k")
        sel = rep == r
        key = np.round(times_ps[sel], 1)
        for k in ex:
            ex[k][sel] = m[k].reindex(key).to_numpy()
        q[sel] = m.q_core.reindex(key).to_numpy()
        n[sel] = m.n_tail_core.reindex(key).to_numpy()
    return (q, n, ex) if extra else (q, n)


# ---------------------------------------------------------------- free energy
def fes_1d(x, w, rng, nbin):
    h, e = np.histogram(x, bins=nbin, range=rng, weights=w)
    c = 0.5 * (e[1:] + e[:-1])
    with np.errstate(divide="ignore"):
        F = -KBT * np.log(h / h.sum())
    return c, F - np.nanmin(F[np.isfinite(F)])


def fes_2d(x, y, w, xr, yr, xb, yb):
    h, xe, ye = np.histogram2d(x, y, bins=[xb, yb], range=[xr, yr], weights=w)
    with np.errstate(divide="ignore"):
        F = -KBT * np.log(h / h.sum())
    F[~np.isfinite(F)] = np.nan
    return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), F - np.nanmin(F)




# ---------------------------------------------------------------- panel family
def make_panels(qbh, q_core, ntc, extra, tt, rep, w, args, focus, qdef):
    """The Q_BH counterpart of the q_core panel set produced by
    metad_timeseries.py, plus the reweighted panels only available here.

    Every Q_BH axis is drawn on `focus` (the populated/reliable window), because
    Q_BH does NOT share the q_core scale: the folded basin sits at Q_BH 0.948 and
    the panel-C shoulder at 0.835, so a 0-1 axis wastes ~80% of its width and
    smooths out exactly the structure of interest.
    """
    import matplotlib.gridspec as gs
    lo, hi = focus
    colors = sns.color_palette("viridis", NWALK)
    sel = tt >= TEQ_NS * 1e3
    ql = (r"$Q_\mathrm{BH}$ (core, heavy 0.45 nm)" if qdef == "heavy"
          else r"$Q_\mathrm{BH}$ (core, all-atom 0.30 nm)")
    V = args.version
    out = []

    # -- 1. time series, per walker ------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for r in range(NWALK):
        m = rep == r
        axes[0].plot(tt[m] / 1e3, qbh[m], lw=0.3, alpha=0.8,
                     color=colors[r], label=f"r{r}")
    axes[0].axhspan(lo, hi, color="0.85", zorder=0, label="focus window")
    axes[0].set_ylabel(ql); axes[0].legend(ncol=7, fontsize=8, loc="lower left")
    axes[0].set_title("full range (r1/r2 excursions are real dynamics)")
    for r in range(NWALK):
        m = rep == r
        axes[1].plot(tt[m] / 1e3, qbh[m], lw=0.3, alpha=0.8, color=colors[r])
    axes[1].set_ylim(lo, hi); axes[1].set_ylabel(ql)
    axes[1].set_xlabel("Time (ns)"); axes[1].set_title("focus window")
    fig.suptitle(f"apo 1Nhp6A - {ql} time series, 6 walkers x 1000 ns (2 ps)",
                 fontsize=11)
    fig.tight_layout(); out.append(save2(fig, f"qbh_timeseries", V))

    # -- 2. distributions: raw vs reweighted ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in range(NWALK):
        m = (rep == r) & sel
        sns.kdeplot(x=qbh[m], ax=axes[0], color=colors[r], lw=1.2, label=f"r{r}")
    axes[0].set_xlim(lo, hi); axes[0].set_xlabel(ql); axes[0].set_ylabel("Density")
    axes[0].set_title("per-walker, UNWEIGHTED (biased sampling)")
    axes[0].legend(fontsize=8, ncol=2)
    hraw, e = np.histogram(qbh[sel], bins=120, range=(lo, hi))
    hrw, _ = np.histogram(qbh[sel], bins=120, range=(lo, hi), weights=w[sel])
    c = 0.5 * (e[1:] + e[:-1])
    axes[1].plot(c, hraw / hraw.sum(), lw=2, color="0.5", label="raw (biased)")
    axes[1].plot(c, hrw / hrw.sum(), lw=2, color="crimson", label="c(t) reweighted")
    axes[1].set_xlim(lo, hi); axes[1].set_xlabel(ql)
    axes[1].set_ylabel("normalised population")
    axes[1].set_title("pooled: bias removed"); axes[1].legend(fontsize=9)
    fig.suptitle(f"apo 1Nhp6A - {ql} distributions", fontsize=11)
    fig.tight_layout(); out.append(save2(fig, "qbh_distributions", V))

    # -- 3. Q_BH vs q_core mapping -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    hb = axes[0].hexbin(q_core[sel], qbh[sel], gridsize=90, cmap="magma",
                        mincnt=1, bins="log")
    fig.colorbar(hb, ax=axes[0], label=r"$\log_{10}$ counts")
    axes[0].plot([0, 1], [0, 1], "w--", lw=1, label="y = x")
    axes[0].set_xlabel(r"$Q_\mathrm{core}$ (biased CV, C$\alpha$ 0.65 nm)")
    axes[0].set_ylabel(ql); axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title("joint density (all frames)")
    edges = np.arange(0.10, 1.001, 0.025)
    mu, sd, ctr = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = sel & (q_core >= a) & (q_core < b)
        if m.sum() < 500:
            continue
        mu.append(qbh[m].mean()); sd.append(qbh[m].std()); ctr.append(0.5*(a+b))
    mu, sd, ctr = map(np.array, (mu, sd, ctr))
    axes[1].plot(ctr, mu - ctr, "o-", color="teal", lw=1.5, ms=3)
    axes[1].fill_between(ctr, mu - ctr - sd, mu - ctr + sd, color="teal", alpha=0.2)
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    for tgt, lbl, cc in [(0.879, "panel-C shoulder", "crimson"),
                         (0.998, "folded basin", "navy")]:
        m = sel & (np.abs(q_core - tgt) < 0.01)
        if m.any():
            axes[1].axvline(tgt, color=cc, lw=1, ls=":")
            axes[1].annotate(f"{lbl}\n$Q_{{core}}$={tgt:.3f}\n"
                             r"$\to Q_{BH}$=" + f"{qbh[m].mean():.3f}",
                             (tgt, mu[np.argmin(abs(ctr-tgt))] - tgt),
                             textcoords="offset points", xytext=(-70, -34),
                             fontsize=7.5, color=cc)
    axes[1].set_xlabel(r"$Q_\mathrm{core}$"); axes[1].set_ylabel(r"$Q_\mathrm{BH}-Q_\mathrm{core}$")
    axes[1].set_title("systematic offset between the two coordinates")
    fig.suptitle(f"apo 1Nhp6A - {ql} vs the biased $Q_{{core}}$   "
                 f"(pearson r = {np.corrcoef(q_core[sel], qbh[sel])[0,1]:.4f})",
                 fontsize=11)
    fig.tight_layout(); out.append(save2(fig, "qbh_vs_qcore", V))

    # -- 4. 2D sampling (raw), Q_BH vs n_IDR ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sc = axes[0].scatter(qbh[sel][::20], ntc[sel][::20], c=tt[sel][::20]/1e3,
                         cmap="viridis", s=1, alpha=0.4, rasterized=True)
    fig.colorbar(sc, ax=axes[0], label="Time (ns)")
    axes[0].set_title("sampling coloured by time")
    hb = axes[1].hexbin(qbh[sel], ntc[sel], gridsize=70, cmap="magma",
                        mincnt=1, bins="log", extent=(lo, hi, 0, args.ntc_max))
    fig.colorbar(hb, ax=axes[1], label=r"$\log_{10}$ counts")
    axes[1].set_title("pooled 2D density")
    for ax in axes:
        ax.set_xlim(lo, hi); ax.set_ylim(0, args.ntc_max)
        ax.set_xlabel(ql); ax.set_ylabel(r"$n_\mathrm{IDR-core}$")
    fig.suptitle(f"apo 1Nhp6A - {ql} sampling (UNWEIGHTED)", fontsize=11)
    fig.tight_layout(); out.append(save2(fig, "qbh_cv2d_sampling", V))

    # -- 5. reweighted 1D profile --------------------------------------------
    d1 = np.loadtxt("analysis/1Nhp6A_apo_large_fes1d_qbh.dat")
    x1, f1, s1 = d1[:, 0], d1[:, 1], d1[:, 2]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ok1 = np.isfinite(f1)
    ax.plot(x1[ok1], f1[ok1], lw=2, color="navy")
    ax.fill_between(x1[ok1], (f1-s1)[ok1], (f1+s1)[ok1], color="navy", alpha=0.2,
                    label=r"$\pm\sigma$ across time snapshots")
    rel = ok1 & (s1 <= 5.0)
    ax.plot(x1[rel], f1[rel], lw=3, color="crimson",
            label=r"reliable ($\sigma \leq 5$ kJ/mol)")
    for tgt, lbl, cc in [(0.835, "shoulder", "darkorange"), (0.948, "folded", "green")]:
        ax.axvline(tgt, color=cc, ls=":", lw=1.2)
        ax.annotate(lbl, (tgt, ax.get_ylim()[1]*0.85), rotation=90,
                    fontsize=8, color=cc, ha="right")
    ax.set_xlim(lo, hi); ax.set_ylim(0, 30)
    ax.set_xlabel(ql); ax.set_ylabel("F (kJ/mol)")
    ax.legend(fontsize=8)
    ax.set_title(f"apo 1Nhp6A - reweighted F({qdef} $Q_{{BH}}$), gate PASS", fontsize=11)
    fig.tight_layout(); out.append(save2(fig, "qbh_fes1d", V))

    # -- 6. reweighted <helix> vs Q_BH ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bins = np.linspace(lo, hi, 40)
    ctr = 0.5*(bins[1:]+bins[:-1])
    for k, cc in zip(["helix1", "helix2", "helix3"], ["tab:blue", "tab:red", "tab:green"]):
        v = extra[k]
        mu, er = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = sel & (qbh >= a) & (qbh < b) & np.isfinite(v)
            ww = w[m]
            if ww.sum() <= 0 or m.sum() < 50:
                mu.append(np.nan); er.append(np.nan); continue
            av = np.average(v[m], weights=ww)
            va = np.average((v[m]-av)**2, weights=ww)
            ess = ww.sum()**2/np.sum(ww**2)
            mu.append(av); er.append(np.sqrt(va/max(ess, 1)))
        mu, er = np.array(mu), np.array(er)
        ax.plot(ctr, mu, "-", color=cc, lw=2, label=k)
        ax.fill_between(ctr, mu-er, mu+er, color=cc, alpha=0.25)
    ax.set_xlim(lo, hi); ax.set_xlabel(ql)
    ax.set_ylabel(r"$\langle S_\alpha \rangle$ (reweighted)")
    ax.legend(fontsize=9)
    ax.set_title("apo 1Nhp6A - per-helix helicity vs genuine Q", fontsize=11)
    fig.tight_layout(); out.append(save2(fig, "qbh_helicity_perhelix", V))
    return out


def save2(fig, name, version):
    p = f"figures/1Nhp6A_apo_large_{name}_v{version}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    wrote {p}")
    return p




# ------------------------------------------------------- Figure-5 panel variants
FIG5_DIR = "analysis/figure5"
HELICES = ["helix1", "helix2", "helix3"]
# q_core >= 0.95 defines the folded reference in figure5_data.py; the measured
# mapping q_core 0.95-1.00 -> Q_BH 0.930 puts the equivalent Q_BH cut at 0.93.
FOLDED_QBH = 0.93


def _save2d(path, x, y, mean, std, cnt, header):
    with open(path, "w") as fh:
        fh.write("# " + header.replace("\n", "\n# ") + "\n")
        for i, a in enumerate(x):
            for j, b in enumerate(y):
                fh.write(f"{a:9.5f} {b:10.4f} {mean[i,j]:12.4f} "
                         f"{std[i,j]:10.4f} {cnt[i,j]:10.0f}\n")
    print(f"    wrote {path}")


def fig5_panels(qbh, ntc, extra, tt, w, args, focus):
    """Write the Figure-5 A/B/C DATA PRODUCTS with the genuine Q on the axis in
    place of the biased q_core.

    Data only -- no plotting. analysis_figure5.ipynb renders these, exactly as it
    renders the q_core originals, so the panels inherit the notebook's specs
    (CMAP RdYlBu_r, VMAX 50, STD_CEIL 15, rcParams, box aspect) instead of this
    script inventing its own.

    File formats are byte-compatible with figure5_data.py's, so f5.load2d /
    f5.load1d read them unchanged. Axis label is plain "Q"; every file name
    carries `qbh` so these can never be confused with the q_core originals.
    """
    os.makedirs(FIG5_DIR, exist_ok=True)
    lo, hi = focus
    sel = tt >= TEQ_NS * 1e3
    tmax = tt.max()
    nb_q = int(round((hi - lo) / 0.005))       # dq = 0.005, matching Q_PLOT_BIN=140
    sa = extra["helix1"] + extra["helix2"] + extra["helix3"]

    def snap_masks():
        for k in range(1, args.nsnap + 1):
            cut = TEQ_NS * 1e3 + (tmax - TEQ_NS * 1e3) * k / args.nsnap
            yield (tt >= TEQ_NS * 1e3) & (tt <= cut)

    # ---------------- A: F(Q, n_IDR-core) ----------------
    print("  [A/qbh] F(Q, n_IDR-core)")
    G = []
    for m in snap_masks():
        x, y, F = fes_2d(qbh[m], ntc[m], w[m], (lo, hi),
                         (NTC_MIN, args.ntc_max), nb_q, 60)
        G.append(F)
    G = np.array(G); Am = np.nanmean(G, 0); Am -= np.nanmin(Am); As = np.nanstd(G, 0)
    cnt, _, _ = np.histogram2d(qbh[sel], ntc[sel], bins=[nb_q, 60],
                               range=[(lo, hi), (NTC_MIN, args.ntc_max)])
    _save2d(f"{FIG5_DIR}/A_fes2d_qbh_nidr.dat", x, y, Am, As, cnt,
            "apo F(Q, n_IDR-core), c(t)-reweighted, genuine Best-Hummer Q "
            "(core heavy atoms, 0.45 nm, lambda 1.8), "
            f"{args.nsnap} cumulative-time snapshots\n"
            "Q  n_IDR_core  F_mean(kJ/mol)  F_std(kJ/mol)  colvar_counts")

    # ---------------- B: <S_alpha> per helix vs Q ----------------
    print("  [B/qbh] <S_alpha> per helix vs Q")
    qb = np.linspace(lo, hi, 46)
    qc = 0.5 * (qb[1:] + qb[:-1])
    folded = qbh >= FOLDED_QBH
    ref = {h: float(np.sum(w[folded] * extra[h][folded]) / w[folded].sum())
           for h in HELICES}
    print("    folded <S_alpha> (Q >= %.2f): " % FOLDED_QBH
          + "  ".join(f"{h}={ref[h]:.2f}" for h in HELICES))

    def profile(mask):
        ww = np.where(mask, w, 0.0)
        den, _ = np.histogram(qbh, bins=qb, weights=ww)
        den2, _ = np.histogram(qbh, bins=qb, weights=ww ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            eb = np.where(den2 > 0, den ** 2 / den2, 0.0)
        out = {}
        for h in HELICES:
            num, _ = np.histogram(qbh, bins=qb, weights=ww * extra[h])
            with np.errstate(invalid="ignore", divide="ignore"):
                out[h] = np.where(den > 0, num / den, np.nan)
        return out, eb

    stack = {h: [] for h in HELICES}
    eb = None
    for m in snap_masks():
        pr, eb = profile(m)
        for h in HELICES:
            stack[h].append(pr[h])
    cols, names = [qc], ["Q"]
    for h in HELICES:
        arr = np.array(stack[h])
        cols += [np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)]
        names += [f"{h}_mean", f"{h}_std"]
    cols.append(eb); names.append("ess_bin")
    outB = f"{FIG5_DIR}/B_helicity_perhelix_vs_qbh.dat"
    np.savetxt(outB, np.column_stack(cols), fmt="%14.6g",
               header=("apo <S_alpha> per helix vs genuine Q, c(t)-reweighted, "
                       f"{args.nsnap} cumulative-time snapshots\n"
                       "folded-state reference (Q >= %.2f): " % FOLDED_QBH
                       + " ".join(f"{h}={ref[h]:.4f}" for h in HELICES) + "\n"
                       + "  ".join(names)))
    print(f"    wrote {outB}")

    # ---------------- C: F(Q, total S_alpha) ----------------
    print("  [C/qbh] F(Q, total S_alpha)")
    hr = (10.0, 46.0)
    G = []
    for m in snap_masks():
        x2, y2, F = fes_2d(qbh[m], sa[m], w[m], (lo, hi), hr, nb_q, 72)
        G.append(F)
    G = np.array(G); Cm = np.nanmean(G, 0); Cm -= np.nanmin(Cm); Cs = np.nanstd(G, 0)
    cnt2, _, _ = np.histogram2d(qbh[sel], sa[sel], bins=[nb_q, 72],
                                range=[(lo, hi), hr])
    _save2d(f"{FIG5_DIR}/C_pmf2d_qbh_total.dat", x2, y2, Cm, Cs, cnt2,
            "apo F(Q, total S_alpha = H1+H2+H3), c(t)-reweighted, genuine "
            f"Best-Hummer Q, {args.nsnap} cumulative-time snapshots\n"
            "Q  total_S_alpha  F_mean(kJ/mol)  F_std(kJ/mol)  colvar_counts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--recompute-q", action="store_true")
    ap.add_argument("--from-driver", metavar="DIR", default=None,
                    help="load Q_BH from `plumed driver` output over the RAW "
                         "2 ps trajectories (analysis/qbh) instead of "
                         "recomputing on the 200 ps processed copies")
    ap.add_argument("--tmax-ps", type=float, default=1.0e6,
                    help="truncate driver frames at this time (default 1e6 ps "
                         "= 1000 ns, the endpoint of the hills snapshot)")
    ap.add_argument("--qdef", choices=["all", "heavy"], default="all",
                    help="which Q_BH to put on the PMF axis (default: the "
                         "project's own all-atom 0.30 nm protocol)")
    ap.add_argument("--nsnap", type=int, default=5,
                    help="cumulative time snapshots for the mean/std band")
    ap.add_argument("--qbh-bin", type=int, default=60)
    ap.add_argument("--qbh-range", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="restrict/zoom the Q_BH axis. Q_BH does NOT share the "
                         "q_core scale, so resolving structure in the populated "
                         "region needs its own range + fine bins")
    ap.add_argument("--fig5", action="store_true",
                    help="emit Figure-5 panels A/B/C with the genuine Q in place "
                         "of q_core (axis label 'Q', file names carry 'qbh')")
    ap.add_argument("--panels", action="store_true",
                    help="also emit the full Q_BH panel family (the counterpart "
                         "of the q_core set from metad_timeseries.py)")
    ap.add_argument("--qbh-focus", type=float, nargs=2, default=(0.78, 1.00),
                    metavar=("LO", "HI"),
                    help="axis window for every Q_BH panel (default 0.78 1.00, "
                         "the populated/reliable region)")
    ap.add_argument("--reuse-weights", action="store_true",
                    help="reuse cached c(t) weights instead of recomputing")
    ap.add_argument("--ntc-max", type=float, default=120.0)
    args = ap.parse_args()

    os.makedirs("analysis", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    if args.from_driver:
        print(f"[1] Q_BH from plumed driver output ({args.from_driver})")
        Q = load_driver(args.from_driver, args.tmax_ps)
    else:
        print("[1] Q_BH on the trajectory frames (200 ps processed copies)")
        Q = compute_q(args.recompute_q)
    t_ps, rep = Q["t_ps"], Q["rep"].astype(int)
    qbh = Q["q_all"] if args.qdef == "all" else Q["q_heavy"]
    print(f"    correlation Q_all vs Q_heavy: r = "
          f"{np.corrcoef(Q['q_all'], Q['q_heavy'])[0,1]:.4f}")

    print("[2] matching colvar (biased CVs) at those frames")
    q_core, ntc, extra = colvar_at(t_ps, rep, extra=True)
    ok = np.isfinite(q_core) & np.isfinite(ntc)
    print(f"    matched {ok.sum():,} / {ok.size:,} frames")
    if ok.sum() < 0.95 * ok.size:
        sys.exit("ERROR: too many unmatched frames -- colvar/traj time mismatch")

    print("[3] c(t) reweighting (Tiwary-Parrinello)")
    hills = [np.loadtxt(f"{APO_DIR}/hills.dat.{i}", comments=["#", "!"])
             for i in range(NWALK)]
    t_h, q0, n0, hgt, sq, sn, gamma = rwm.pool_hills(hills)
    print(f"    {t_h.size:,} hills | sigma=({sq}, {sn}) | gamma={gamma:.0f}")
    qg = np.linspace(Q_MIN, Q_MAX, Q_BIN + 1)
    ng = np.linspace(NTC_MIN, NTC_MAX, NTC_BIN + 1)
    WCACHE = "analysis/1Nhp6A_apo_large_qbh_weights.npz"
    if args.reuse_weights and os.path.exists(WCACHE):
        z = np.load(WCACHE)
        if z["w"].size == int(ok.sum()):
            ct = {"weights": z["w"], "ct": z["ct"]}
            print(f"    reused cached weights ({WCACHE})")
        else:
            args.reuse_weights = False
    if not (args.reuse_weights and os.path.exists(WCACHE)):
        ct = rwm.metad_ct_weights(t_h, q0, n0, hgt, sq, sn,
                                  t_ps[ok], q_core[ok], ntc[ok], qg, ng,
                                  gamma=float(gamma), kBT=KBT,
                                  checkpoint_dt=1.0, batch_dt=1.0)
        np.savez_compressed(WCACHE, w=ct["weights"], ct=ct["ct"])
    w = ct["weights"].copy()
    tt = t_ps[ok]
    w[tt < TEQ_NS * 1e3] = 0.0
    used = int((tt >= TEQ_NS * 1e3).sum())
    ess = w.sum() ** 2 / np.sum(w ** 2)
    print(f"    c(t): {ct['ct'][0]:.1f} -> {ct['ct'][-1]:.1f} kJ/mol")
    print(f"    ESS = {ess:,.0f} ({100*ess/used:.2f}% of {used:,} usable frames)")

    print("[4] GATE: reweight back onto the BIASED CVs vs sum_hills")
    # sum_hills grid: FIELDS q_core n_tail_core file.free, q varying FASTEST,
    # so a row-major reshape gives (n_ntc, n_q).  The reference marginal must be
    # a proper BOLTZMANN projection (fm.marginal_fes), not a minimum-energy
    # projection along the other CV -- those differ systematically and would
    # manufacture a gate failure on their own.
    # The reference surface is REBUILT from the current hills, exactly as
    # figure5_data.py::apo_sumhills does.  It must NOT be read from
    # analysis/1Nhp6A_apo_large_fes_sumhills.dat: that file dates from 2026-07-06,
    # i.e. the ~500 ns era, whereas the hills now run to 1000 ns.  Comparing a
    # 1000 ns reweighting against a half-length reference is a SYSTEMATIC error
    # that no amount of extra sampling removes -- it is what made the MAD sit at
    # ~11 kJ/mol and refuse to improve when the frame count rose 100x.
    import tempfile
    import fes_marginal_meanstd as fm
    hp = [f"{APO_DIR}/hills.dat.{i}" for i in range(NWALK)]
    hl = [fm.load_hills(x) for x in hp]
    hdr = fm.hills_header(hp[0])
    t_max_ps = max(h[:, 0].max() for h in hl)
    print(f"    rebuilding sum_hills reference from the CURRENT hills "
          f"(t_max = {t_max_ps/1e3:.1f} ns)")
    with tempfile.TemporaryDirectory(prefix="qbh_gate_") as wd:
        fpath = fm.run_sumhills_upto(hl, hdr, t_max_ps, "full", wd,
                                     [Q_MIN, NTC_MIN], [Q_MAX, NTC_MAX],
                                     [Q_BIN, NTC_BIN])
        qax, nax, G_qn = fm.load_fes(fpath, ndim=2)      # F indexed [q, ntc]
    G_qn = G_qn - np.nanmin(G_qn)
    G_nq = G_qn.T                                        # (n_ntc, n_q)
    # Masking replicates figure5_data.py::apo_weights EXACTLY, and must: comparing
    # every bin (as an earlier version of this script did) is dominated by bins
    # that are barely sampled or thermally irrelevant, and by the folded minimum
    # pinned against the q_core = 1.0 grid edge -- a known sum_hills/grid
    # limitation, not an estimator error. That gave MAD ~ 11 kJ/mol on data whose
    # project gate passes at 2.48. Criteria: >= 200 raw counts in the bin, and
    # sum_hills F <= 30 kJ/mol; verdict taken on the BULK q < 0.93 region.
    qref, Fsh_q = fm.marginal_fes(nax, qax, G_nq, (NTC_MIN, NTC_MAX), KBT, axis=0)
    nref, Fsh_n = fm.marginal_fes(qax, nax, G_nq.T, (Q_MIN, Q_MAX), KBT, axis=0)
    cq, Fq = fes_1d(q_core[ok], w, (Q_MIN, Q_MAX), Q_BIN)
    cn, Fn = fes_1d(ntc[ok], w, (NTC_MIN, NTC_MAX), NTC_BIN)
    shq_i = np.interp(cq, qref, Fsh_q)
    shn_i = np.interp(cn, nref, Fsh_n)

    sel = tt >= TEQ_NS * 1e3
    kq, _ = np.histogram(q_core[ok][sel], bins=Q_BIN, range=(Q_MIN, Q_MAX))
    kn, _ = np.histogram(ntc[ok][sel], bins=NTC_BIN, range=(NTC_MIN, NTC_MAX))

    def _mad(a, b, m):
        m = m & np.isfinite(a) & np.isfinite(b)
        return (float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan), int(m.sum())

    mq = (kq >= 200) & (shq_i <= 30.0)
    mn = (kn >= 200) & (shn_i <= 30.0)
    mad_q, nq_ = _mad(Fq, shq_i, mq)
    mad_n, nn_ = _mad(Fn, shn_i, mn)
    mad_bulk, nb_ = _mad(Fq, shq_i, mq & (cq < 0.93))
    mad_wall, nw_ = _mad(Fq, shq_i, (cq >= 0.93) & (kq >= 200))
    print(f"      F(q_core)     {mad_q:5.2f} ({nq_} bins) | bulk q<0.93 "
          f"{mad_bulk:5.2f} ({nb_}) | wall q>=0.93 {mad_wall:5.2f} ({nw_})")
    print(f"      F(n_IDR-core) {mad_n:5.2f} ({nn_} bins)")
    passed = (mad_bulk <= GATE) and (mad_n <= GATE)
    mads = {"q_core_bulk": mad_bulk, "q_core_wall": mad_wall, "n_tail_core": mad_n}
    print(f"    GATE (bulk q<0.93 and n_IDR both <= {GATE}) -> "
          f"{'PASS' if passed else 'FAIL'}")

    print("[4b] Q_BH <-> q_core mapping (do the two coordinates share a scale?)")
    from scipy.stats import pearsonr, spearmanr
    qc, qb = q_core[ok], qbh[ok]
    sel5 = tt >= TEQ_NS * 1e3
    print(f"    pearson  r = {pearsonr(qc[sel5], qb[sel5])[0]:+.4f}   "
          f"spearman r = {spearmanr(qc[sel5], qb[sel5])[0]:+.4f}  (unweighted, all frames)")
    print(f"    {'q_core bin':>12} {'<Q_BH>':>8} {'sd':>6} {'offset':>8} {'n':>10}")
    edges = np.arange(0.1, 1.001, 0.05)
    for a, b in zip(edges[:-1], edges[1:]):
        m = sel5 & (qc >= a) & (qc < b)
        if m.sum() < 500:
            continue
        mu = qb[m].mean()
        print(f"    {a:.2f}-{b:.2f} {mu:8.3f} {qb[m].std():6.3f} "
              f"{mu - 0.5*(a+b):+8.3f} {int(m.sum()):10d}")
    for tgt, lbl in [(0.879, "panel-C SHOULDER"), (0.998, "folded basin")]:
        m = sel5 & (np.abs(qc - tgt) < 0.01)
        if m.any():
            print(f"    q_core = {tgt:.3f} ({lbl}) maps to Q_BH = "
                  f"{qb[m].mean():.3f} +/- {qb[m].std():.3f}  (n={int(m.sum()):,})")

    print("[4c] fine 1D reweighted profile F(Q_BH) over the populated region")
    lo, hi = args.qbh_range if args.qbh_range else (0.0, 1.0)
    nb1 = 120
    stack1 = []
    tmax1 = tt.max()
    for k in range(1, args.nsnap + 1):
        cut = TEQ_NS * 1e3 + (tmax1 - TEQ_NS * 1e3) * k / args.nsnap
        ww = np.where(tt <= cut, w, 0.0)
        if ww.sum() <= 0:
            continue
        c1, F1 = fes_1d(qbh[ok], ww, (lo, hi), nb1)
        stack1.append(F1)
    S1 = np.vstack(stack1)
    f1m, f1s = np.nanmean(S1, axis=0), np.nanstd(S1, axis=0)
    f1m -= np.nanmin(f1m)
    out1 = "analysis/1Nhp6A_apo_large_fes1d_qbh.dat"
    with open(out1, "w") as fh:
        fh.write(f"# reweighted F(Q_BH[{args.qdef}]) over {lo}-{hi}, {nb1} bins\n")
        fh.write("# Q_BH  F_mean(kJ/mol)  F_std(kJ/mol)\n")
        for x, a, b in zip(c1, f1m, f1s):
            fh.write(f"{x:9.5f} {a:12.4f} {b:10.4f}\n")
    print(f"    wrote {out1}")

    print("[5] 2D PMF  Q_BH vs n_IDR-core, with snapshot mean/std")
    xr = tuple(args.qbh_range) if args.qbh_range else (0.0, 1.0)
    yr = (NTC_MIN, args.ntc_max)
    stack = []
    tmax = tt.max()
    for k in range(1, args.nsnap + 1):
        cut = TEQ_NS * 1e3 + (tmax - TEQ_NS * 1e3) * k / args.nsnap
        ww = np.where(tt <= cut, w, 0.0)
        if ww.sum() <= 0:
            continue
        xc, yc, F = fes_2d(qbh[ok], ntc[ok], ww, xr, yr, args.qbh_bin, 60)
        stack.append(F)
    S = np.dstack(stack)
    Fm, Fs = np.nanmean(S, axis=2), np.nanstd(S, axis=2)
    Fm -= np.nanmin(Fm)

    out = "analysis/1Nhp6A_apo_large_pmf2d_qbh_nidr.dat"
    with open(out, "w") as fh:
        fh.write(f"# 2D PMF  Q_BestHummer({args.qdef}) vs n_IDR-core, "
                 f"c(t)-reweighted apo MetaD\n")
        fh.write(f"# ESS = {ess:.0f} ({100*ess/used:.2f}% of {used} frames, "
                 f"200 ps stride)\n")
        fh.write(f"# gate MAD on q_core vs sum_hills: "
                 f"{'PASS' if passed else 'FAIL' if passed is not None else 'n/a'}\n")
        fh.write("# Q_BH  n_IDR_core  F_mean(kJ/mol)  F_std(kJ/mol)\n")
        for i, x in enumerate(xc):
            for j, y in enumerate(yc):
                fh.write(f"{x:8.4f} {y:9.3f} {Fm[i,j]:12.4f} {Fs[i,j]:10.4f}\n")
    print(f"    wrote {out}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    lv = np.arange(0, 42, 3)
    cf = axes[0].contourf(xc, yc, Fm.T, levels=lv, cmap="viridis", extend="max")
    axes[0].contour(xc, yc, Fm.T, levels=lv, colors="k", linewidths=0.3)
    fig.colorbar(cf, ax=axes[0], label="F (kJ/mol)")
    axes[0].set_title("PMF (snapshot mean)")
    cs = axes[1].contourf(xc, yc, Fs.T, levels=np.arange(0, 13, 1),
                          cmap="magma", extend="max")
    fig.colorbar(cs, ax=axes[1], label=r"$\sigma_F$ (kJ/mol)")
    axes[1].set_title("uncertainty across time snapshots")
    for ax in axes:
        ax.set_xlabel(r"$Q_\mathrm{Best-Hummer}$ (core, all-atom 0.30 nm)"
                      if args.qdef == "all" else
                      r"$Q_\mathrm{Best-Hummer}$ (core, heavy 0.45 nm)")
        ax.set_ylabel(r"$n_\mathrm{IDR-core}$")
    flag = "PASS" if passed else "FAIL" if passed is not None else "n/a"
    fig.suptitle(f"apo 1Nhp6A - reweighted PMF, genuine Best-Hummer Q  "
                 f"[ESS {100*ess/used:.2f}%, gate {flag}]", fontsize=11)
    fig.tight_layout()
    p = f"figures/1Nhp6A_apo_large_pmf2d_qbh_nidr_v{args.version}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"    wrote {p}")

    if args.fig5:
        print("[7] Figure-5 panel variants with genuine Q")
        fig5_panels(qbh[ok], ntc[ok], {k: v[ok] for k, v in extra.items()},
                    tt, w, args, tuple(args.qbh_focus))

    if args.panels:
        print("[6] Q_BH panel family")
        make_panels(qbh[ok], q_core[ok], ntc[ok],
                    {k: v[ok] for k, v in extra.items()},
                    tt, rep[ok], w, args, tuple(args.qbh_focus), args.qdef)


if __name__ == "__main__":
    main()
