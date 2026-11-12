#!/usr/bin/env python
"""Protein-DNA contact landscape of the m-state production runs vs the native (holo) complex.

Features: for every (protein residue 1-93, nucleotide) pair the closest heavy-atom distance
(min-image, so no whole-duplex reconstruction is needed), turned into a smooth contact
s = (1 - (d/R0)^6) / (1 - (d/R0)^12), R0 = 0.45 nm (project contact cut-off, coordination
form), or a 0/1 contact with --binary.  Pairs never formed (s < --min-feature in every frame
of every run) are dropped.  Distances are cached per run as <P>_dna_resdist_skip<N>.npz.

Two products:

1. Native core-DNA Q.  Native pairs = core (27-93) x nucleotide pairs with d < 0.45 nm in
   >= --native-occ of the pooled holo frames (3 x 250 ns unbiased, t >= --equil ns; the same
   cut as mstate_select_frames.py).  Q(t) = fraction of native pairs formed.  As a register
   diagnostic Q is also scored after mapping the native nucleotides by a shift of +-1/+-2 bp
   and/or the duplex pseudo-dyad flip (strand swap).  The sequence is NOT palindromic, so a
   high flipped/shifted Q means the same protein-DNA geometry on different bases.
   -> analysis/mstate_probe/1Nhp6A_mstate_md_qnative.dat, figures/<system>/1Nhp6A_mstate_md_qnative_<qv>.png
   (written only if the --q-version figure is missing or with --redo-q; Q is always computed)

2. PCA of the contact features (IDR + linker + core, all 93 residues x 30 nucleotides).
   --fit all (default, v2): fitted on the POOLED unbiased frames, holo 3 x 250 ns + m-state
   8 runs (t >= --tmin), unweighted (the two sets are ~balanced in frame count).  --fit mstate
   reproduces v1 (m-state only, holo projected).  Every frame is projected and coloured by run,
   DNA bend (colvar theta, matched by time) and identity-register native core-DNA Q; the
   detached first --t-detached ns of each m-state run are marked.  The share of each PC's
   squared loading carried by IDR (1-20) / linker (21-26) / core (27-93) features is reported.
   tICA (reversible, symmetrized C0/Ctau, assumes stationarity) is opt-in with --tica: on the
   m-state data it only separated start configs (implied timescales > data length).
   --pca-from NPZ: no refit; every frame (incl. runs extended after that fit) is projected on
   the saved basis, after checking it reproduces its own fit frames (v4 = v3 basis, 500 ns).
   -> figures/<system>/1Nhp6A_mstate_md_contact_pca_<v>.png [+ _contact_tica_<v>.png],
      analysis/mstate_probe/1Nhp6A_mstate_md_contact_proj_<v>.npz (projections, loadings,
      region fractions, feature pairs; v1 file is the unversioned _contact_proj.npz)

   Row 4 of the figure: (j) PC1 vs DNA bend (colvar, nearest time) with free-DNA and holo
   p5-p95 bend bands (data/equil/{DNA,1Nhp6A_holo}/rep*/colvar.dat, t >= --equil; same theta
   atom definition in both inputs), (k) bend vs time.

3. --dissect RUN [RUN ...] (opt-in; no PCA refit, PC1 is projected with the saved --version
   npz): one run window (--window B E ns) at --dissect-skip (default 1 = 10 ps).  Only the
   window is converted (trjconv -pbc whole -b/-e on the raw xtc, non-Water ->
   <P>_solute_whole_skip1_t<B>-<E>ns.xtc; distances <P>_dna_resdist_skip1_t<B>-<E>ns.npz),
   rather than caching the whole run at 10 ps.  Frame times are the xtc stamps and are matched
   to the colvar and to the 100 ps cache by time (distances/Q/PC1 checked equal there).
   Panels: per-element contact counts (IDR, linker, H1, loops, H2, H3), native and best-register
   Q, n_core/d_core/bend/Q_DNA (0.2 ps colvar, --smooth ns running mean), stable-onset raster of
   the native core pairs (+ holo-native linker pairs, not in Q), IDR per-bp footprint, PC1.
   Stable onset = first frame formed AND formed in >= --stable-frac of the frames over the
   following --stable-hold ns (default 80 % over 1 ns).
   -> figures/<system>/1Nhp6A_mstate_md_<run>_rebinding_<dissect-version>.png
4. --sliding RUN:END_NS ... (100 ps cache): Q per register variant vs time, onset raster of the
   native pairs mapped into the final (highest last-10-ns Q) register, bend and n_core.
   -> figures/<system>/1Nhp6A_mstate_md_sliding_<sliding-version>.png

5. --project-metad SYS [SYS ...] (opt-in; NO PCA refit): project the 6-walker wall MetaD runs
   (data/<SYS>_wall, SYS in 1Nhp6A 1S26D 1T63D 1TM) onto the saved --version PCA (keep/mean/
   loadings from the npz; holo + m-state projections are first re-derived from the caches and
   checked against the npz).  Protein+DNA _proc.xtc (box vectors required: min-image distances)
   at --metad-stride-ps, t <= --metad-tmax; distances cached per walker as
   <stem>_dna_resdist_<stride>ps.npz in the replica dir.  The point mutants have the same 93
   residues (only the mutated residue names differ), so the residue-level features align by
   index.  Colvar rows are de-duplicated by time keeping the LAST occurrence (restart
   continuation), matched to the xtc frames by time, and checked by recomputing de2e from the
   frames.  Frame weights: Tiwary-Parrinello c(t) (reweight_metad.metad_ct_weights) from the
   pooled HILLS on (n_IDR, de2e), first --metad-teq ns discarded; the static cmap LOWER_WALL is
   NOT reweighted (the duplex-integrity restraint is part of the target ensemble, as in
   dna_bending_wall_compare.py).  Gate: reweighted F(n_IDR), F(de2e) vs the sum_hills marginal.
   PMFs along PC1 and native core-DNA Q: mean +/- std over --n-snaps cumulative-time snapshots;
   the plotted curve is truncated to the std <= 5 kJ/mol run contiguous with the minimum and to
   bins reached by >= --min-walkers walkers (full range in the .dat).
   -> figures/<system>/1Nhp6A_wall_contact_pca_proj_<mv>.png, figures/mutants_wall_contact_pca_proj_<mv>.png
      (raw coverage, unweighted), figures/wall_metad_contact_pc1_q_pmf_<mv>.png,
      analysis/mstate_probe/<SYS>_wall_metad_{pc1,qnative}_pmf_<mv>.dat,
      analysis/mstate_probe/<SYS>_wall_metad_contact_proj_<mv>.npz (per-frame PCs, Q, weights)

Usage:
    mstate_contact_pca.py [--fit all|mstate] [--skip 10] [--tmin 0] [--tica --lag 1.0]
                          [--version v2] [--q-version v1 --redo-q]
    mstate_contact_pca.py --dissect c0_rep1 c0_rep0 --window 0 25 --dissect-skip 1 \\
                          --sliding c4_rep0:70 c1_rep1:80
    mstate_contact_pca.py --project-metad 1Nhp6A 1S26D 1T63D 1TM [--metad-stride-ps 100]
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from figpaths import figpath
from equil_monitor import ROOT, CONTACT_CUTOFF, DNA_PREFIXES, PROBE_CORE, gmx, load_whole
from equil_timeseries import PULL_COLOR, SURVIVAL_TH, SYSTEMS, load_run, md_config, \
    read_md_colvar, run_dir, run_prefix, run_src

MD = "1Nhp6A_mstate_md"
HOLO = "1Nhp6A_holo"
HOLO_REPS = (0, 1, 2)
NBP = 15
OUTDIR = os.path.join(ROOT, "analysis", "mstate_probe")
SHIFTS = (-2, -1, 0, 1, 2)
REGIONS = (("IDR", 1, 20), ("linker", 21, 26), ("core", 27, 93))
HOLO_GREY = {0: "0.25", 1: "0.5", 2: "0.72"}


# ---------------------------------------------------------------- features
def residue_pairs(top, copy=None):
    """(protein residue, nucleotide) index pairs, protein-major; plus the residue labels.
    copy: 0-based protein copy of a 2:1 topology (93 residues each, in topology order);
    None = the single-copy 1:1 topology."""
    prot = [r for r in top.residues if r.is_protein]
    if copy is not None:
        prot = prot[93 * copy: 93 * (copy + 1)]
        assert [r.resSeq for r in prot] == list(range(1, 94)), f"copy {copy} is not residues 1-93"
    nts = [r for r in top.residues if r.name[:2] in DNA_PREFIXES]
    assert len(prot) == 93 and len(nts) == 2 * NBP, (len(prot), len(nts))
    pairs = np.array([(p.index, n.index) for p in prot for n in nts])
    return pairs, [p.resSeq for p in prot], [f"{'AB'[i // NBP]}{i % NBP + 1}" for i in range(2 * NBP)]


def resdist(kind, rep, skip):
    """(time_ns, d[n_frames, 93*30]) closest-heavy residue-pair distances, cached."""
    import mdtraj as md

    if kind == "md":
        cache = os.path.join(run_dir(MD, rep), f"{run_prefix(MD, rep)}_dna_resdist_skip{skip}.npz")
        src = run_src(MD, rep)
    else:
        d = os.path.join(ROOT, "data", "equil", HOLO, f"rep{rep}")
        cache = os.path.join(d, f"{HOLO}_rep{rep}_dna_resdist_skip{skip}.npz")
        src = os.path.join(d, "md.xtc")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        z = np.load(cache)
        return z["time"], z["d"]
    t = load_run(MD, rep, skip) if kind == "md" else load_whole(HOLO, rep, skip)
    pairs, _, _ = residue_pairs(t.top)
    d, _ = md.compute_contacts(t, pairs, scheme="closest-heavy", periodic=True,
                               ignore_nonprotein=False)
    np.savez_compressed(cache, time=t.time, d=d.astype(np.float32))
    print(f"  cached {cache}")
    return t.time, d


def switch(d, binary=False):
    if binary:
        return (d < CONTACT_CUTOFF).astype(np.float32)
    x = (d / CONTACT_CUTOFF) ** 6
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(np.isclose(x, 1.0), 0.5, (1 - x) / (1 - x * x))
    return s.astype(np.float32)


def equil_bend(system, rep):
    """(time_ns, bend_deg) from data/equil/<system>/rep<rep>/colvar.dat, unique times.
    The DNA-only and holo inputs define theta on the same atoms (dna_end1/centre/end2 COMs)."""
    path = os.path.join(ROOT, "data", "equil", system, f"rep{rep}", "colvar.dat")
    names = open(path).readline().split()[2:]
    a = np.genfromtxt(path, invalid_raise=False, usecols=range(len(names)))
    a = a[~np.isnan(a).any(axis=1)]
    tc, iu = np.unique(a[:, 0] / 1000.0, return_index=True)    # restart overlaps: match by time
    return tc, 180.0 - np.degrees(a[iu, names.index("theta")])


def nearest(tc, times):
    """Index of the nearest tc (sorted) to each of `times`."""
    i = np.clip(np.searchsorted(tc, times), 1, len(tc) - 1)
    return i - ((times - tc[i - 1]) < (tc[i] - times))


def bend_at(kind, rep, times):
    """DNA bend angle (deg, 180 - theta) from the colvar, nearest colvar time to each frame."""
    if kind == "md":
        cv = read_md_colvar(rep)
        tc, iu = np.unique(cv["time"], return_index=True)
        b = cv["bend"][iu]
    else:
        tc, b = equil_bend(HOLO, rep)
    return b[nearest(tc, times)]


def bend_refs(equil):
    """p5/p50/p95 of the DNA bend (deg) for free DNA and holo, colvar stride, t >= equil ns."""
    out = {}
    for name, system in (("free DNA", "DNA"), ("holo", HOLO)):
        b = np.concatenate([bb[tt >= equil] for tt, bb in
                            (equil_bend(system, r) for r in (0, 1, 2))])
        out[name] = np.percentile(b, [5, 50, 95])
    return out


# ---------------------------------------------------------------- native Q
def nt_map(shift, flip):
    """Nucleotide index map (0..29, strand A then B) for a register shift / dyad flip."""
    m = np.full(2 * NBP, -1)
    for i in range(2 * NBP):
        s, k = divmod(i, NBP)
        if flip:                       # bp k = A[k]:B[14-k] -> bp 14-k: A[k]->B[k], B[j]->A[j]
            s = 1 - s
        k = k + shift if s == 0 else k - shift      # B runs antiparallel to A
        if 0 <= k < NBP:
            m[i] = s * NBP + k
    return m


def native_q(holo, md_runs, resseq, occ, equil):
    """Native core-DNA pair set from holo; Q(t) for every run and register variant."""
    core = np.array([PROBE_CORE[0] <= r <= PROBE_CORE[1] for r in resseq])
    pool = np.concatenate([d[t >= equil] for t, d in holo.values()])
    freq = (pool < CONTACT_CUTOFF).mean(axis=0).reshape(93, 2 * NBP)
    native = np.argwhere((freq >= occ) & core[:, None])          # (residue row, nt)
    variants = [(s, f) for f in (False, True) for s in SHIFTS]
    out = {}
    for key, (t, d) in list(holo.items()) + list(md_runs.items()):
        c = (d < CONTACT_CUTOFF).reshape(len(t), 93, 2 * NBP)
        q = {}
        for s, f in variants:
            m = nt_map(s, f)[native[:, 1]]
            ok = m >= 0
            hit = np.zeros(len(t))
            if ok.any():
                hit = c[:, native[ok, 0], m[ok]].sum(axis=1)
            q[(s, f)] = hit / len(native)             # unmappable pairs count as broken
        out[key] = (t, q)
    return native, freq, out


def vlabel(s, f):
    return ("flip " if f else "") + (f"{s:+d} bp" if s else ("native" if not f else "0 bp"))


# ---------------------------------------------------------------- PCA / tICA
def tica(trajs, lag, dim=4, eps=1e-6):
    """Reversible tICA: symmetrized C0/Ctau pooled over trajectories, whitened via C0."""
    mean = np.concatenate(trajs).mean(axis=0)
    nf = trajs[0].shape[1]
    c0, ct, n = np.zeros((nf, nf)), np.zeros((nf, nf)), 0
    for x in trajs:
        if len(x) <= lag:
            continue
        x = x - mean
        a, b = x[:-lag], x[lag:]
        c0 += a.T @ a + b.T @ b
        ct += a.T @ b + b.T @ a
        n += 2 * len(a)
    c0, ct = c0 / n, ct / n
    s, u = np.linalg.eigh(c0)
    keep = s > eps * s.max()
    w = u[:, keep] / np.sqrt(s[keep])
    lam, v = np.linalg.eigh(w.T @ ct @ w)
    order = np.argsort(lam)[::-1][:dim]
    return mean, w @ v[:, order], lam[order]


def run_style(rep):
    k, r = md_config(rep)
    return PULL_COLOR[k], ("o" if r == 0 else "^"), ("-" if r == 0 else "--")


def plot_q(qres, native, version):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    hq = np.concatenate([q[(0, False)] for k, (t, q) in qres.items() if k[0] == "holo"])
    for ax in axes[:2]:
        ax.axhspan(*np.percentile(hq, [5, 95]), color="0.88", lw=0, label="holo p5-p95")
    for key, (t, q) in qres.items():
        if key[0] != "md":
            continue
        c, _, ls = run_style(key[1])
        lab = key[1].replace("_", " ")
        w = max(1, int(round(1.0 / np.median(np.diff(t)))))     # 1-ns block mean
        n = len(t) // w * w
        tb = t[:n].reshape(-1, w).mean(1)
        axes[0].plot(tb, q[(0, False)][:n].reshape(-1, w).mean(1), color=c, ls=ls, lw=1.2, label=lab)
        best = np.stack([q[v][:n].reshape(-1, w).mean(1) for v in q]).max(axis=0)
        axes[1].plot(tb, best, color=c, ls=ls, lw=1.2)
    axes[0].set_ylabel(r"native core-DNA $Q$")
    axes[1].set_ylabel(r"best register-variant $Q$")
    for ax in axes[:2]:
        ax.set_xlabel("time (ns)")
        ax.set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize="x-small", ncol=2)
    # end-state register: mean Q over the last 10 ns per variant
    rows = [k for k in qres if k[0] == "md"]
    vs = list(next(iter(qres.values()))[1])
    mat = np.array([[qres[k][1][v][qres[k][0] >= qres[k][0][-1] - 10].mean() for v in vs]
                    for k in rows])
    im = axes[2].imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[2].set_xticks(range(len(vs)), [vlabel(*v) for v in vs], rotation=60, fontsize="x-small")
    axes[2].set_yticks(range(len(rows)), [k[1].replace("_", " ") for k in rows], fontsize="small")
    axes[2].set_title("Q per register variant, last 10 ns", fontsize="small")
    fig.colorbar(im, ax=axes[2], fraction=0.04)
    fig.suptitle(f"1Nhp6A m-state production: native core(27-93)-DNA contacts "
                 f"({len(native)} native pairs from holo)", x=0.01, ha="left", fontsize="medium")
    fig.tight_layout()
    out = figpath(f"{MD}_qnative_{version}.png", root=ROOT)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")
    return rows, vs, mat


def region_fractions(load, keep, resseq):
    """Share of each component's squared loading on IDR / linker / core features."""
    rs = np.asarray(resseq)[keep // (2 * NBP)]
    w = load ** 2 / (load ** 2).sum(axis=0)
    return np.array([[w[(rs >= a) & (rs <= b), j].sum() for _, a, b in REGIONS]
                     for j in range(load.shape[1])])


def draw_loading(ax, vec, keep, ntlab, title):
    """Residue x nucleotide heatmap of one component; IDR | linker | core boundaries dotted."""
    L = np.zeros(93 * 2 * NBP)
    L[keep] = vec
    L = L.reshape(93, 2 * NBP)
    lim = np.abs(L).max()
    im = ax.imshow(L.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=(0.5, 93.5, 2 * NBP - 0.5, -0.5), interpolation="nearest")
    ax.set_yticks(range(0, 2 * NBP, 2), ntlab[::2], fontsize="x-small")
    ax.axhline(NBP - 0.5, color="k", lw=0.5)
    for x in (20.5, 26.5):
        ax.axvline(x, color="k", lw=0.6, ls=":")
    for name, a, b in REGIONS:
        ax.text((a + b) / 2, -0.9, name, ha="center", va="bottom", fontsize="x-small")
    ax.set_xlabel("protein residue")
    ax.set_title(title, fontsize="small", pad=12)
    return im


def running_mean(y, w):
    """Centred running mean over w samples (shrinking window at the edges)."""
    import pandas as pd
    return pd.Series(y).rolling(max(1, int(w)), center=True, min_periods=1).mean().to_numpy()


def bend_bands(ax, refs, axis="y"):
    span = ax.axhspan if axis == "y" else ax.axvspan
    for (name, (lo, mid, hi)), col in zip(refs.items(), ("#56B4E9", "0.8")):
        span(lo, hi, color=col, alpha=0.35, lw=0, zorder=0,
             label=f"{name} bend p5-p95 [{lo:.0f}, {hi:.0f}]$^\\circ$")


def pc1_bend(ax_pl, ax_t, P, refs, tdet):
    """(j) PC1 vs DNA bend trajectories over the holo cloud; (k) bend vs time."""
    runs = P["runs"]
    ev = P["ev"]
    bend_bands(ax_pl, refs)
    bend_bands(ax_t, refs)
    for k, (t, y, b, _) in runs.items():
        if k[0] != "holo":
            continue
        ax_pl.scatter(y[:, 0], b, s=2, color=HOLO_GREY[k[1]], lw=0, alpha=0.35, marker="s",
                      rasterized=True)
        ax_t.plot(t, running_mean(b, 10), color=HOLO_GREY[k[1]], lw=0.6)
    for k, (t, y, b, _) in runs.items():
        if k[0] != "md":
            continue
        c, mk, ls = run_style(k[1])
        w = 10                                          # 1-ns running mean (100 ps frames)
        ax_pl.scatter(y[:, 0], b, s=3, color=c, marker=mk, lw=0, alpha=0.3, rasterized=True)
        ax_pl.plot(running_mean(y[:, 0], w), running_mean(b, w), color=c, ls=ls, lw=0.9,
                   label=k[1].replace("_", " "))
        e = t >= t[-1] - 10
        ax_pl.scatter(y[0, 0], b[0], s=60, marker=mk, facecolor="w", edgecolor=c, lw=1.4,
                      zorder=5)
        ax_pl.scatter(y[e, 0].mean(), b[e].mean(), s=100, marker=mk, facecolor=c,
                      edgecolor="k", lw=1.0, zorder=6)
        ax_t.plot(t, running_mean(b, w), color=c, ls=ls, lw=0.9)
    ax_pl.set_xlabel(f"PC1 ({100 * ev[0]:.0f}%)")
    ax_pl.set_ylabel("DNA bend (deg)")
    ax_pl.set_title("(j) PC1 vs DNA bend: points = 100 ps frames, line = 1-ns running mean; "
                    "open = t=0, filled = last-10-ns mean; grey squares = holo", fontsize="small")
    ax_pl.legend(frameon=False, fontsize="x-small", ncol=3, loc="upper left")
    ax_t.axvspan(0, tdet, color="0.9", lw=0, zorder=0)
    ax_t.set_xlabel("time (ns)")
    ax_t.set_ylabel("DNA bend (deg), 1-ns running mean")
    ax_t.set_title("(k) DNA bend vs time (grey = holo)", fontsize="small")
    ax_t.legend(frameon=False, fontsize="x-small", loc="upper right")
    for ax in (ax_pl, ax_t):
        ax.set_ylim(-2, 70)


def plot_pca(P, ntlab, nfit, tdet, version, refs):
    """v2: 4 x 3 grid; PC planes (run / bend / Q), PC vs time, PC1-PC3, PC1-3 loadings,
    PC1 vs DNA bend + bend vs time."""
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(18, 21))
    gs = GridSpec(4, 3, figure=fig, height_ratios=(1, 1, 1, 1.05))
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)])
    ev, runs, frac = P["ev"], P["runs"], P["frac"]
    hold = [k for k in runs if k[0] == "holo"]
    mds = [k for k in runs if k[0] == "md"]
    lab = lambda j: f"PC{j + 1} ({100 * ev[j]:.0f}%)"          # noqa: E731

    def by_run(ax, i, j):
        for k in hold:
            y = runs[k][1]
            ax.scatter(y[:, i], y[:, j], s=2, color=HOLO_GREY[k[1]], lw=0, alpha=0.5,
                       marker="s", rasterized=True, label=f"holo rep{k[1]}")
        for k in mds:
            t, y = runs[k][0], runs[k][1]
            c, mk, _ = run_style(k[1])
            late, early = t >= tdet, t < tdet
            ax.scatter(y[late, i], y[late, j], s=3, color=c, marker=mk, lw=0, alpha=0.45,
                       rasterized=True, label=k[1].replace("_", " "))
            ax.scatter(y[early, i], y[early, j], s=7, color=c, marker=mk, edgecolors="k",
                       lw=0.3, alpha=0.8, rasterized=True)
        for k in mds:          # start / end markers on top of all clouds
            t, y = runs[k][0], runs[k][1]
            c, mk, _ = run_style(k[1])
            e = y[t >= t[-1] - 10].mean(axis=0)
            ax.scatter(y[0, i], y[0, j], s=55, marker=mk, facecolor="w", edgecolor=c, lw=1.4,
                       zorder=5)
            ax.scatter(e[i], e[j], s=90, marker=mk, facecolor=c, edgecolor="k", lw=1.0, zorder=6)
        ax.set_xlabel(lab(i))
        ax.set_ylabel(lab(j))

    def by_value(ax, idx, cmap, vmin, vmax, cblab):
        for k in hold + mds:
            y, v = runs[k][1], runs[k][idx]
            sc = ax.scatter(y[:, 0], y[:, 1], s=2, c=v, cmap=cmap, vmin=vmin, vmax=vmax, lw=0,
                            marker="s" if k[0] == "holo" else "o", rasterized=True)
        fig.colorbar(sc, ax=ax, label=cblab, fraction=0.05)
        ax.set_xlabel(lab(0))
        ax.set_ylabel(lab(1))

    a = axes[0]
    by_run(a[0], 0, 1)
    a[0].set_title(f"(a) runs; black edge = detached t < {tdet:g} ns,\n"
                   "open = t=0, filled black-edged = last-10-ns mean", fontsize="small")
    by_value(a[1], 2, "plasma", 0, 60, "DNA bend (deg)")
    a[1].set_title("(b) coloured by DNA bend (squares = holo)", fontsize="small")
    by_value(a[2], 3, "viridis", 0, 1, r"native core-DNA $Q$ (identity register)")
    a[2].set_title(r"(c) coloured by native core-DNA $Q$", fontsize="small")
    h, lb = a[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=11, frameon=False, fontsize="small",
               markerscale=4, bbox_to_anchor=(0.5, 0.955))

    b = axes[1]
    for col, j in ((0, 0), (1, 1)):
        for k in hold:
            b[col].plot(runs[k][0], runs[k][1][:, j], color=HOLO_GREY[k[1]], lw=0.5)
        for k in mds:
            c, _, ls = run_style(k[1])
            b[col].plot(runs[k][0], runs[k][1][:, j], color=c, ls=ls, lw=0.7)
        b[col].axvspan(0, tdet, color="0.9", lw=0, zorder=0)
        b[col].set_xlabel("time (ns)")
        b[col].set_ylabel(lab(j))
        b[col].set_title(f"({'de'[col]}) PC{j + 1} vs time (grey = holo; shaded = detached)",
                         fontsize="small")
    by_run(b[2], 0, 2)
    b[2].set_title("(f) PC1-PC3 plane (markers as in a)", fontsize="small")

    for j, ax in enumerate(axes[2]):
        f = frac[j]
        im = draw_loading(ax, P["load"][:, j], P["keep"], ntlab,
                          f"({'ghi'[j]}) PC{j + 1} loading; |w|$^2$ IDR {f[0]:.2f} / "
                          f"linker {f[1]:.2f} / core {f[2]:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.05)
    pc1_bend(fig.add_subplot(gs[3, :2]), fig.add_subplot(gs[3, 2]), P, refs, tdet)
    nh, nm = nfit
    span = P.get("span") or {g: f"{len(ks)} x {max(runs[k][0][-1] for k in ks):.0f} ns"
                             for g, ks in (("holo", hold), ("md", mds))}
    fig.suptitle(f"1Nhp6A protein(1-93) x DNA contact PCA, {len(P['keep'])} features, fit on "
                 f"{P['fitdesc']}: holo {nh} frames ({span['holo']}) + m-state {nm} frames "
                 f"({span['md']}), unweighted", x=0.01, ha="left", fontsize="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = figpath(f"{MD}_contact_pca_{version}.png", root=ROOT)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_tica(P, ntlab, version):
    """Opt-in tICA row: run-coloured tIC1/tIC2, bend-coloured, tIC1 vs time, tIC1 loading."""
    fig, row = plt.subplots(1, 4, figsize=(18, 4.2))
    runs, ev = P["runs"], P["ev"]
    sc = None
    for k, (t, y, bnd, _) in runs.items():
        if k[0] == "holo":
            row[0].scatter(y[:, 0], y[:, 1], s=2, color=HOLO_GREY[k[1]], lw=0, alpha=0.4)
            row[2].plot(t, y[:, 0], color="0.6", lw=0.5)
            continue
        c, mk, ls = run_style(k[1])
        row[0].scatter(y[:, 0], y[:, 1], s=3, color=c, marker=mk, alpha=0.5, lw=0,
                       label=k[1].replace("_", " "))
        sc = row[1].scatter(y[:, 0], y[:, 1], s=3, c=bnd, cmap="plasma", vmin=0, vmax=60, lw=0)
        row[2].plot(t, y[:, 0], color=c, ls=ls, lw=0.8)
    fig.colorbar(sc, ax=row[1], label="DNA bend (deg)")
    for ax in row[:2]:
        ax.set_xlabel(f"tIC1 ({ev[0]})")
        ax.set_ylabel(f"tIC2 ({ev[1]})")
    row[0].legend(frameon=False, fontsize="x-small", markerscale=3, ncol=2)
    row[2].set_xlabel("time (ns)")
    row[2].set_ylabel("tIC1 (grey = holo)")
    draw_loading(row[3], P["load"][:, 0], P["keep"], ntlab, "tIC1 loading")
    fig.tight_layout()
    out = figpath(f"{MD}_contact_tica_{version}.png", root=ROOT)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- event dissection
# finer regions for the dissection (helices = project canon from 1LWM, equil_monitor.HELICES)
FINE = (("IDR", 1, 20), ("linker", 21, 26), ("H1", 27, 42), ("loop H1-H2", 43, 47),
        ("H2", 48, 61), ("loop H2-H3", 62, 63), ("H3", 64, 93))
FINE_COLOR = {"IDR": "#999999", "linker": "#E69F00", "H1": "#0072B2", "loop H1-H2": "#56B4E9",
              "H2": "#009E73", "loop H2-H3": "#F0E442", "H3": "#D55E00"}


def element(resseq):
    return next(n for n, a, b in FINE if a <= resseq <= b)


def window_resdist(rep, b, e, skip):
    """Closest-heavy residue-pair distances for t in [b, e] ns of an m-state run at stride
    `skip` (10-ps frames).  Only the window is converted: trjconv -pbc whole -b/-e on the raw
    xtc (non-Water) -> <P>_solute_whole_skip<N>_t<b>-<e>ns.xtc, distances cached as
    <P>_dna_resdist_skip<N>_t<b>-<e>ns.npz.  Times are the xtc time stamps (not a frame
    counter), so they can be matched to the colvar and the 100 ps cache by time."""
    import mdtraj as md

    os.environ.setdefault("HWLOC_COMPONENTS", "-gl")
    d, P = run_dir(MD, rep), run_prefix(MD, rep)
    tag = f"skip{skip}_t{b:g}-{e:g}ns"
    src, tpr = run_src(MD, rep), os.path.join(d, P + ".tpr")
    cache = os.path.join(d, f"{P}_dna_resdist_{tag}.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        z = np.load(cache)
        return z["time"], z["d"]
    top = os.path.join(d, f"{P}_solute_end.gro")
    if not os.path.exists(top):
        load_run(MD, rep, 10)                         # makes the solute topology
    xtc = os.path.join(d, f"{P}_solute_whole_{tag}.xtc")
    if not os.path.exists(xtc) or os.path.getmtime(xtc) < os.path.getmtime(src):
        r = gmx(["trjconv", "-f", src, "-s", tpr, "-o", xtc, "-pbc", "whole",
                 "-b", f"{b * 1000:g}", "-e", f"{e * 1000:g}", "-skip", str(skip)],
                stdin="non-Water\n")
        if not os.path.exists(xtc):
            raise SystemExit(f"trjconv failed for {P}:\n{r.stderr[-2000:]}")
    t = md.load(xtc, top=top)
    pairs, _, _ = residue_pairs(t.top)
    dd, _ = md.compute_contacts(t, pairs, scheme="closest-heavy", periodic=True,
                                ignore_nonprotein=False)
    time = np.round(t.time / 1000.0, 6)              # xtc stamps, ps -> ns
    np.savez_compressed(cache, time=time, d=dd.astype(np.float32))
    print(f"  cached {cache}")
    return time, dd


def stable_onset(c, dt, hold, frac):
    """First frame i with c[i] formed AND formed in >= frac of the frames in [t_i, t_i + hold).
    Only onsets with a full hold window inside the data are accepted; NaN = never."""
    n = int(round(hold / dt))
    cs = np.vstack([np.zeros((1, c.shape[1])), np.cumsum(c, axis=0)])
    m = len(c) - n + 1
    if m <= 0:
        return np.full(c.shape[1], np.nan)
    occ = (cs[n:n + m] - cs[:m]) / n
    ok = (c[:m] > 0) & (occ >= frac)
    first = np.argmax(ok, axis=0).astype(float)
    first[~ok.any(axis=0)] = np.nan
    return first


def q_variants(c, native):
    """Q(t) per register variant for a binary contact array c[n, 93, 30]."""
    q = {}
    for s, f in [(s, f) for f in (False, True) for s in SHIFTS]:
        m = nt_map(s, f)[native[:, 1]]
        ok = m >= 0
        q[(s, f)] = (c[:, native[ok, 0], m[ok]].sum(axis=1) if ok.any()
                     else np.zeros(len(c))) / len(native)
    return q


def dissect(rep, b, e, skip, native, linker_nat, resname, pcz, binary, hold, frac):
    """All per-frame observables of the event window of one run (see --dissect)."""
    t, d = window_resdist(rep, b, e, skip)
    dt = np.median(np.diff(t))
    assert np.allclose(np.diff(t), dt, atol=1e-4), f"{rep}: uneven frame spacing in window"
    c = (d < CONTACT_CUTOFF).reshape(len(t), 93, 2 * NBP)
    rs = np.arange(1, 94)
    regions = {n: c[:, (rs >= a) & (rs <= bb)].sum(axis=(1, 2)) for n, a, bb in FINE}
    q = q_variants(c, native)
    best = np.stack(list(q.values())).max(axis=0)
    bestkey = list(q)[int(np.argmax([v[t >= t[-1] - 2].mean() for v in q.values()]))]
    # IDR per-base-pair footprint (= mstate_select_frames.footprint: bp k = A k : B 16-k)
    idr = c[:, :20]
    fp = np.stack([idr[:, :, k].sum(1) + idr[:, :, 2 * NBP - 1 - k].sum(1) for k in range(NBP)],
                  axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        centroid = (fp * np.arange(1, NBP + 1)).sum(1) / fp.sum(1)
    # order of formation: native core pairs + holo-native linker pairs (marked, not in Q)
    pairs = np.concatenate([native, linker_nat])
    pc = c[:, pairs[:, 0], pairs[:, 1]].astype(float)
    onset = stable_onset(pc, dt, hold, frac)
    onset_t = np.where(np.isnan(onset), np.nan, t[0] + np.nan_to_num(onset) * dt)
    plab = [f"{resname[r]}{r + 1} {element(r + 1)} : {'AB'[n // NBP]}{n % NBP + 1}"
            for r, n in pairs]
    # PC1 with the published (v2) PCA
    feat = switch(d, binary)[:, pcz["keep"]]
    pcs = (feat - pcz["pca_mean"]) @ pcz["pca_load"].T
    # colvar (0.2 ps), unique times, window
    cv = read_md_colvar(rep)
    tc, iu = np.unique(cv["time"], return_index=True)
    sel = (tc >= b - 1e-9) & (tc <= e + 1e-9)
    col = {k: cv[k][iu][sel] for k in ("n_core", "d_core", "contacts", "bend", "n_idr")}
    gap = np.abs(tc[nearest(tc, t)] - t).max()
    assert gap < 1e-4, f"{rep}: frame times not in colvar (max gap {gap} ns)"
    return dict(rep=rep, t=t, dt=dt, d=d, regions=regions, q=q, best=best, bestkey=bestkey,
                fp=fp, centroid=centroid, pairs=pairs, nnat=len(native), onset=onset_t,
                plab=plab, pc=pcs, tc=tc[sel], col=col)


def plot_dissect(D, out, smooth_ns, hold, frac, ylims):
    """One column of time-aligned panels for one run window."""
    t, tc = D["t"], D["tc"]
    w = int(round(smooth_ns / D["dt"]))
    wc = int(round(smooth_ns / np.median(np.diff(tc))))
    fig, ax = plt.subplots(8, 1, figsize=(11, 27), sharex=True,
                           gridspec_kw=dict(height_ratios=(1, 1, 1, 0.9, 0.9, 2.6, 1.2, 0.9)))
    k, rr = md_config(D["rep"])

    def trace(a, x, y, ww, **kw):
        a.plot(x, y, lw=0.3, alpha=0.3, color=kw.get("color"))
        a.plot(x, running_mean(y, ww), lw=1.3, **kw)

    for n in ("IDR", "linker"):
        trace(ax[0], t, D["regions"][n], w, color=FINE_COLOR[n], label=n)
    for n in ("H1", "loop H1-H2", "H2", "loop H2-H3", "H3"):
        trace(ax[1], t, D["regions"][n], w, color=FINE_COLOR[n], label=n)
    for a, lab in ((ax[0], "(a1) IDR / linker"), (ax[1], "(a2) core elements")):
        a.set_ylabel("residue-nt pairs\n< 0.45 nm")
        a.legend(frameon=False, fontsize="small", ncol=5, loc="upper left")
        a.set_title(f"{lab}: protein-DNA residue-pair contacts (closest heavy < "
                    f"{CONTACT_CUTOFF} nm); thin = 10 ps, thick = {smooth_ns * 1000:.0f} ps "
                    "running mean", fontsize="small", loc="left")
    ax[0].set_ylim(0, ylims["a1"])
    ax[1].set_ylim(0, ylims["a2"])

    trace(ax[2], t, D["q"][(0, False)], w, color="k", label="identity register (native)")
    trace(ax[2], t, D["best"], w, color="C3", label="best register variant (max over "
          r"$\pm$2 bp, flip)")
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel(r"core-DNA $Q$")
    ax[2].legend(frameon=False, fontsize="small", loc="upper left")
    ax[2].set_title(f"(b) native core(27-93)-DNA Q, {D['nnat']} holo-native pairs", fontsize="small",
                    loc="left")

    col = D["col"]
    trace(ax[3], tc, col["n_core"], wc, color="#0072B2", label=r"$n_{core}$ (colvar)")
    ax[3].axhline(SURVIVAL_TH, color="#0072B2", ls=":", lw=0.8)
    ax[3].set_ylabel(r"$n_{core}$", color="#0072B2")
    ax[3].set_ylim(0, ylims["n_core"])
    a3 = ax[3].twinx()
    trace(a3, tc, col["d_core"], wc, color="#D55E00")
    a3.set_ylabel(r"$d_{core}$ (nm)", color="#D55E00")
    ax[3].set_title(r"(c1) core-DNA coordination $n_{core}$ (blue; dotted = survival threshold "
                    f"{SURVIVAL_TH}) and core-COM-DNA distance $d_{{core}}$ (red); 0.2 ps colvar, "
                    f"thick = {smooth_ns * 1000:.0f} ps mean", fontsize="small", loc="left")
    trace(ax[4], tc, col["bend"], wc, color="#CC79A7")
    ax[4].set_ylabel("DNA bend (deg)", color="#CC79A7")
    ax[4].set_ylim(-2, 65)
    a4 = ax[4].twinx()
    trace(a4, tc, col["contacts"], wc, color="0.3")
    a4.set_ylabel(r"$Q_{DNA}$ (cmap)", color="0.3")
    a4.set_ylim(0.5, 1.05)
    ax[4].set_title("(c2) DNA bend (180 - theta, pink) and duplex integrity Q_DNA (grey)",
                    fontsize="small", loc="left")

    # (d) raster of native pairs, sorted by stable onset
    on = D["onset"]
    order = np.argsort(np.where(np.isnan(on), np.inf, on), kind="stable")
    c = (D["d"].reshape(len(t), 93, 2 * NBP)[:, D["pairs"][order, 0], D["pairs"][order, 1]]
         < CONTACT_CUTOFF)
    ax[5].imshow(c.T, aspect="auto", cmap="Greys", interpolation="nearest",
                 extent=(t[0] - D["dt"] / 2, t[-1] + D["dt"] / 2, len(order) - 0.5, -0.5))
    for i, j in enumerate(order):
        el = D["plab"][j].split()[1]
        if np.isfinite(on[j]):
            ax[5].plot(on[j], i, marker="D", ms=4.5, mew=0.8, mec="w", color=FINE_COLOR[el])
    ax[5].set_yticks(range(len(order)), [D["plab"][j] + ("  *" if j >= D["nnat"] else "")
                                         for j in order], fontsize=6.5)
    for lab, j in zip(ax[5].get_yticklabels(), order):
        lab.set_color(FINE_COLOR[D["plab"][j].split()[1]])
    ax[5].set_title(f"(d) holo-native contacts (black = formed); diamond = stable onset: formed and "
                    f"formed in >= {frac:.0%} of frames over the next {hold:g} ns; sorted by onset; "
                    "* = linker pair (not in Q)", fontsize="small", loc="left")

    im = ax[6].imshow(D["fp"].T, aspect="auto", cmap="magma_r", interpolation="nearest",
                      extent=(t[0] - D["dt"] / 2, t[-1] + D["dt"] / 2, NBP + 0.5, 0.5),
                      vmin=0, vmax=ylims["fp"])
    ax[6].plot(t, running_mean(D["centroid"], w), color="#56B4E9", lw=1.2)
    ax[6].set_ylabel("base pair\n(A k : B 16-k)")
    ax[6].set_yticks(range(1, NBP + 1, 2))
    ax[6].set_title(f"(e) IDR (1-20) footprint: residue-nt contacts per bp; blue = footprint "
                    f"centroid ({smooth_ns * 1000:.0f} ps mean)", fontsize="small", loc="left")
    cax = ax[6].inset_axes((1.01, 0.0, 0.012, 1.0))
    fig.colorbar(im, cax=cax)

    trace(ax[7], t, D["pc"][:, 0], w, color=PULL_COLOR[k])
    ax[7].axhspan(1.21, 2.95, color="0.88", lw=0, zorder=0, label="holo PC1 p5-p95")
    ax[7].set_ylabel("PC1")
    ax[7].set_ylim(-4, 3.5)
    ax[7].legend(frameon=False, fontsize="small", loc="upper left")
    ax[7].set_title("(f) PC1 of the pooled v2 contact PCA (projected, 10 ps)", fontsize="small",
                    loc="left")
    ax[7].set_xlabel("time (ns)")
    ax[7].set_xlim(t[0], t[-1])
    for a in ax[[0, 1, 2, 3, 4, 7]]:
        a.grid(axis="x", color="0.85", lw=0.5)
    fig.suptitle(f"1Nhp6A m-state {D['rep'].replace('_', ' ')}: {t[0]:g}-{t[-1]:.0f} ns at "
                 f"{D['dt'] * 1000:.0f} ps", x=0.01, ha="left", fontsize="large")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


def report_dissect(D, bins):
    """Text summary: onsets by element, region counts / bend / IDR centroid in time bins."""
    print(f"\n== {D['rep']}  {D['t'][0]:g}-{D['t'][-1]:g} ns, {len(D['t'])} frames, "
          f"dt {D['dt'] * 1000:.0f} ps; final best register {vlabel(*D['bestkey'])}")
    on = D["onset"]
    for j in np.argsort(np.where(np.isnan(on), np.inf, on), kind="stable"):
        print(f"   onset {on[j]:7.2f} ns  {D['plab'][j]}{'  (linker)' if j >= D['nnat'] else ''}")
    tc, col = D["tc"], D["col"]
    hdr = "  ".join(f"{n[:6]:>6s}" for n, _, _ in FINE)
    print(f"   {'bin (ns)':>10s}  {hdr}  Q_nat Q_best  PC1   n_core d_core  bend  Q_DNA  IDR_c")
    for a, bb in zip(bins[:-1], bins[1:]):
        m = (D["t"] >= a) & (D["t"] < bb)
        mc = (tc >= a) & (tc < bb)
        if not m.any():
            continue
        reg = "  ".join(f"{D['regions'][n][m].mean():6.1f}" for n, _, _ in FINE)
        print(f"   {a:4.1f}-{bb:5.1f}  {reg}  {D['q'][(0, False)][m].mean():5.2f} "
              f"{D['best'][m].mean():5.2f}  {D['pc'][m, 0].mean():5.2f}  "
              f"{col['n_core'][mc].mean():6.2f} {col['d_core'][mc].mean():6.2f} "
              f"{col['bend'][mc].mean():5.1f}  {col['contacts'][mc].mean():5.3f}  "
              f"{np.nanmean(D['centroid'][m]):5.2f}")


VCOL = {0: "k", -1: "#E69F00", -2: "#0072B2", 1: "#009E73", 2: "#CC79A7"}   # by shift


def plot_sliding(specs, qres, mdr, native, resname, hold, frac, version):
    """Register-variant Q vs time for slow runs (100 ps cache) + onset raster in the final
    register (the variant with the highest last-10-ns Q)."""
    fig, ax = plt.subplots(3, len(specs), figsize=(8 * len(specs), 13), sharex="col",
                           gridspec_kw=dict(height_ratios=(1, 2.2, 0.9)), squeeze=False)
    print("\n== sliding: register-variant Q (1-ns means), 100 ps cache")
    for col, (rep, tend) in enumerate(specs):
        t, q = qres[("md", rep)]
        m = t <= tend + 1e-9
        t = t[m]
        q = {v: x[m] for v, x in q.items()}
        dt = np.median(np.diff(t))
        final = max(q, key=lambda v: q[v][t >= t[-1] - 10].mean())
        a = ax[0, col]
        for v, x in q.items():
            if x.max() < 0.15 and v not in ((0, False), final):
                continue
            ls = "--" if v[1] else "-"
            lw = 2.0 if v == final else 1.0
            a.plot(t, running_mean(x, 10), ls=ls, lw=lw, color=VCOL[v[0]], label=vlabel(*v)
                   + (" (final)" if v == final else ""))
        a.set_ylim(0, 1)
        a.set_ylabel(r"core-DNA $Q$ in register variant")
        a.legend(frameon=False, fontsize="small", ncol=2, loc="upper left")
        a.set_title(f"{rep.replace('_', ' ')}: Q per register (1-ns running mean; variants "
                    "with max < 0.15 hidden)", fontsize="small", loc="left")
        # which variant leads, per 5-ns bin
        print(f"   {rep} final register {vlabel(*final)}")
        for b0 in np.arange(0, tend, 5):
            mm = (t >= b0) & (t < b0 + 5)
            vals = {v: x[mm].mean() for v, x in q.items()}
            top = sorted(vals, key=vals.get, reverse=True)[:3]
            print(f"     {b0:4.0f}-{b0 + 5:3.0f} ns  " + "  ".join(
                f"{vlabel(*v)} {vals[v]:.2f}" for v in top))
        # raster in the final register (core pairs only; mapped by nt_map)
        s, f = final
        mp = nt_map(s, f)[native[:, 1]]
        ok = mp >= 0
        pairs = np.column_stack([native[ok, 0], mp[ok]])
        _, dmd = mdr[("md", rep)]
        cc = (dmd[m].reshape(m.sum(), 93, 2 * NBP)[:, pairs[:, 0], pairs[:, 1]]
              < CONTACT_CUTOFF).astype(float)
        on = stable_onset(cc, dt, hold, frac)
        on = np.where(np.isnan(on), np.nan, t[0] + np.nan_to_num(on) * dt)
        order = np.argsort(np.where(np.isnan(on), np.inf, on), kind="stable")
        ax[1, col].imshow(cc[:, order].T, aspect="auto", cmap="Greys", interpolation="nearest",
                          extent=(t[0] - dt / 2, t[-1] + dt / 2, len(order) - 0.5, -0.5))
        labs = []
        for i, j in enumerate(order):
            r, n = pairs[j]
            el = element(r + 1)
            labs.append(f"{resname[r]}{r + 1} {el} : {'AB'[n // NBP]}{n % NBP + 1}")
            if np.isfinite(on[j]):
                ax[1, col].plot(on[j], i, marker="D", ms=4.5, mew=0.8, mec="w",
                                color=FINE_COLOR[el])
        ax[1, col].set_yticks(range(len(order)), labs, fontsize=6.5)
        for lab, j in zip(ax[1, col].get_yticklabels(), order):
            lab.set_color(FINE_COLOR[element(pairs[j, 0] + 1)])
        ax[1, col].set_title(f"native core pairs mapped to the final register ({vlabel(*final)}); "
                             f"diamond = stable onset ({frac:.0%} over next {hold:g} ns)",
                             fontsize="small", loc="left")
        cv = read_md_colvar(rep)
        tc, iu = np.unique(cv["time"], return_index=True)
        sel = tc <= tend
        a2 = ax[2, col]
        a2.plot(tc[sel][::50], running_mean(cv["bend"][iu][sel], 5000)[::50], color="#CC79A7")
        a2.set_ylabel("DNA bend (deg)", color="#CC79A7")
        a2.set_ylim(0, 65)
        a3 = a2.twinx()
        a3.plot(tc[sel][::50], running_mean(cv["n_core"][iu][sel], 5000)[::50], color="#0072B2")
        a3.set_ylabel(r"$n_{core}$", color="#0072B2")
        a2.set_xlabel("time (ns)")
        a2.set_title("DNA bend and n_core (colvar, 1-ns running mean)", fontsize="small",
                     loc="left")
        a2.set_xlim(0, tend)
    fig.suptitle("1Nhp6A m-state slow re-engagers: shifted-register native-like geometry, "
                 "100 ps frames", x=0.01, ha="left", fontsize="large")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.subplots_adjust(wspace=0.32)             # room for the twin n_core axis label
    out = figpath(f"{MD}_sliding_{version}.png", root=ROOT)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- MetaD projection
METAD_SYS = ("1Nhp6A", "1S26D", "1T63D", "1TM")
METAD_WALKERS = tuple(range(6))
METAD_T = 298.0                    # METAD TEMP; = dna_bending_wall_compare.T (holo/DNA pipeline)
KB = 0.00831446261815324
# c(t) bias grid: the METAD GRID_MIN/MAX domain (n_IDR 0-150, de2e 0-8 nm), finer than its
# GRID_BIN (1.5 / 0.04) so the bilinear read of V resolves sigma = 3.0 / 0.25 hills
METAD_GRID = ((0.0, 150.0, 0.5), (0.0, 8.0, 0.02))
METAD_SUMHILLS = ([0, 0], [150, 8], [150, 200])    # gate grid = dna_bending_wall_compare's
NONNATIVE_PC1 = (-3.4, -2.5)       # m-state non-native H3-docked cluster (v2 PCA)


def metad_stem(system, i):
    wall = "" if system == "1Nhp6A" else "_wall"
    return f"{system}_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D{wall}_r{i}"


def metad_rdir(system, i):
    return os.path.join(ROOT, "data", f"{system}_wall", f"replica_{i}")


def metad_top(system):
    """Protein+DNA topology in the _proc.xtc atom order (WT: the pdb2gmx Protein+DNA pdb;
    mutants: the trjconv reference frame written with the proc.xtc)."""
    if system == "1Nhp6A":
        return os.path.join(ROOT, "1Nhp6A_DNA_parambsc1.pdb")
    return os.path.join(ROOT, "data", f"{system}_wall", f"{system}_proc_ref.pdb")


def plumed_com_atoms(path, label):
    """0-based atom indices of a `label: COM ATOMS=a-b,c-d` line of a plumed input."""
    for line in open(path):
        if line.startswith(f"{label}:"):
            spec = line.split("ATOMS=")[1].split()[0]
            idx = []
            for part in spec.split(","):
                a, _, b = part.partition("-")
                idx += list(range(int(a), int(b or a) + 1))
            return np.array(idx) - 1
    raise SystemExit(f"{label} not found in {path}")


def metad_resdist(job):
    """(time_ps, d[n, 93*30], de2e_xtc) for one MetaD walker at stride_ps, cached as
    <stem>_dna_resdist_<stride>ps.npz in the replica dir.  de2e_xtc = |COM(dna_end1) -
    COM(dna_end2)| recomputed from the frames (plumed.dat atom lists, element masses) to
    validate the colvar time matching."""
    import mdtraj as md

    system, i, stride_ps = job
    d, stem = metad_rdir(system, i), metad_stem(system, i)
    src = os.path.join(d, stem + "_proc.xtc")
    with md.open(src) as f:
        _, t2, _, _ = f.read(n_frames=2, atom_indices=[0])
    dt = float(t2[1] - t2[0])
    stride_ps = max(stride_ps, dt)        # cannot go finer than the saved frames
    stride = int(round(stride_ps / dt))
    cache = os.path.join(d, f"{stem}_dna_resdist_{stride_ps:g}ps.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        z = np.load(cache)
        return z["time_ps"], z["d"], z["de2e_xtc"]
    top = md.load(metad_top(system)).top
    if stride < 1 or abs(stride * dt - stride_ps) > 1e-3:
        raise SystemExit(f"{src}: frame spacing {dt} ps does not divide {stride_ps} ps")
    t = md.load(src, top=top, stride=stride)
    if t.unitcell_vectors is None or not np.all(t.unitcell_volumes > 0):
        raise SystemExit(f"{src}: no box vectors -- refusing to compute non-periodic distances")
    time = np.round(t.time.astype(float), 3)
    assert np.allclose(np.diff(time), stride_ps, atol=1e-3), f"{src}: uneven/duplicated times"
    pairs, _, _ = residue_pairs(t.top)
    # chunked: closest-heavy expands to ~4.5e5 atom pairs per frame (memory, 30 GB host)
    dd = np.concatenate([md.compute_contacts(t[a:a + 200], pairs, scheme="closest-heavy",
                                             periodic=True, ignore_nonprotein=False)[0]
                         for a in range(0, t.n_frames, 200)]).astype(np.float32)
    mass = np.array([a.element.mass for a in t.top.atoms])
    plm = os.path.join(d, "plumed.dat")
    com = [(t.xyz[:, ix] * mass[ix, None]).sum(1) / mass[ix].sum()
           for ix in (plumed_com_atoms(plm, "dna_end1"), plumed_com_atoms(plm, "dna_end2"))]
    de2e = np.linalg.norm(com[0] - com[1], axis=1)
    np.savez_compressed(cache, time_ps=time, d=dd.astype(np.float32), de2e_xtc=de2e,
                        stride_ps=stride_ps, src_dt_ps=dt)
    print(f"  cached {cache}", flush=True)
    return time, dd, de2e


def metad_colvar(system, i):
    """Walker colvar (0.2 ps) de-duplicated by time, keeping the LAST occurrence: on a
    restart PLUMED appends re-computed rows while gmx truncates the xtc to the checkpoint,
    so the later rows belong to the trajectory that was kept.  -> dict, n duplicates."""
    import pandas as pd
    path = os.path.join(metad_rdir(system, i), f"colvar_r{i}.dat")
    names = open(path).readline().split()[2:]
    a = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=names,
                    usecols=range(len(names))).dropna()
    a["time"] = a["time"].round(3)
    ndup = int(a["time"].duplicated().sum())
    a = a.drop_duplicates(subset="time", keep="last").sort_values("time")
    return {k: a[k].to_numpy() for k in names}, ndup


def metad_weights(system, frame_t, n_idr, de2e, tmax_ps, teq_ps, gate_dir):
    """c(t) weights (Tiwary-Parrinello, reweight_metad) for the analysed frames of one
    system, frames with t < teq_ps set to 0; plus the gate: reweighted F(n_IDR), F(de2e)
    against the sum_hills marginals of the same hills at tmax."""
    import pandas as pd
    import fes_marginal_meanstd as fm
    import reweight_metad as rwm

    kbt = KB * METAD_T
    hdir = os.path.join(ROOT, "data", f"{system}_wall")
    paths = [os.path.join(hdir, f"hills.dat.{i}") for i in METAD_WALKERS]
    hills = [pd.read_csv(p, sep=r"\s+", comment="#", header=None).to_numpy() for p in paths]
    hills = [h[h[:, 0] <= tmax_ps + 1e-6] for h in hills]
    t_h, q0, n0, hgt, sq, sn, gamma = rwm.pool_hills(hills)
    (q_lo, q_hi, dq), (n_lo, n_hi, dn) = METAD_GRID
    qg = np.arange(q_lo, q_hi + dq / 2, dq)
    ng = np.arange(n_lo, n_hi + dn / 2, dn)
    print(f"  {system}: {t_h.size:,} hills <= {tmax_ps / 1e3:g} ns, sigma ({sq}, {sn}), "
          f"gamma {gamma:g}; c(t) over {frame_t.size} frames ...", flush=True)
    ct = rwm.metad_ct_weights(t_h, q0, n0, hgt, sq, sn, frame_t, n_idr, de2e, qg, ng,
                              gamma=gamma, kBT=kbt, checkpoint_dt=10.0, batch_dt=1.0)
    use = frame_t >= teq_ps
    rb = ct["rbias"]
    w = np.where(use, np.exp((rb - rb[use].max()) / kbt), 0.0)
    ess = w.sum() ** 2 / np.sum(w ** 2)

    # gate: frame-reweighted marginals on the BIASED CVs vs sum_hills at tmax
    header = fm.hills_header(paths[0])
    mins, maxs, bins = METAD_SUMHILLS
    tag = f"{system}_t{int(tmax_ps)}ps"
    fes = fm.run_sumhills_upto(hills, header, tmax_ps, tag, gate_dir, mins, maxs, bins)
    cv1, cv2, f2 = fm.load_fes(fes, ndim=2)
    os.remove(fes)
    gate = {}
    for name, x, rng, nb, (sx, sf) in (
            ("n_IDR", n_idr, (0, 40), 40, fm.marginal_fes(cv1, cv2, f2, (0, 8), kbt, axis=1)),
            ("de2e", de2e, (2.4, 6.0), 72, fm.marginal_fes(cv1, cv2, f2, (0, 150), kbt))):
        h, e = np.histogram(x[use], bins=nb, range=rng, weights=w[use])
        cnt, _ = np.histogram(x[use], bins=nb, range=rng)
        c = 0.5 * (e[:-1] + e[1:])
        with np.errstate(divide="ignore"):
            fr = -kbt * np.log(h)
        ref = np.interp(c, sx, sf)
        ok = (cnt >= 20) & np.isfinite(fr) & (ref - np.nanmin(ref) <= 15.0)
        fr, ref = fr - fr[ok].min(), ref - ref[ok].min()
        gate[name] = (float(np.mean(np.abs(fr[ok] - ref[ok]))), int(ok.sum()))
    return w, ess, ct, gamma, gate


def snap_profile(x, w, t_ps, walker, edges, n_snaps, tmax_ps, teq_ps, min_fr=5):
    """Reweighted F(x) mean/std over n_snaps cumulative-time snapshots (c(t) weights are
    time-local, so a snapshot is a frame subset t in [teq, t_snap]).  A bin's std is NaN
    unless it is finite in >= 3 snapshots (one or two values give a spuriously small std).
    Also: raw frame counts and walkers with >= min_fr frames per bin (t >= teq)."""
    kbt = KB * METAD_T
    curves = []
    for ts in np.linspace(tmax_ps / n_snaps, tmax_ps, n_snaps):
        m = (t_ps >= teq_ps) & (t_ps <= ts + 1e-6)
        h, _ = np.histogram(x[m], bins=edges, weights=w[m])
        with np.errstate(divide="ignore"):
            f = -kbt * np.log(h)
        f[h == 0] = np.nan
        curves.append(f - np.nanmin(f))
    G = np.array(curves)
    nfin = np.isfinite(G).sum(0)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(G, axis=0)
            std = np.nanstd(G, axis=0)
    mean -= np.nanmin(mean)
    std[nfin < 3] = np.nan
    u = t_ps >= teq_ps
    cnt, _ = np.histogram(x[u], bins=edges)
    nw = sum((np.histogram(x[u & (walker == k)], bins=edges)[0] >= min_fr).astype(int)
             for k in np.unique(walker))
    return mean, std, cnt, nw, nfin


def verify_pca(pcz, holo, mdr, min_feature):
    """The saved PCA must be exactly what the default (v2) run fitted: same kept features,
    same per-run projections, and the stored EVR re-derived from the stored loadings.  Runs
    extended since the fit are compared on the npz's own frames (a prefix, matched by time)."""
    runs = []
    for k, (t, d) in list(holo.items()) + list(mdr.items()):
        n = len(pcz[f"PC_{k[0]}_{k[1]}"])
        assert len(t) >= n, f"{k}: fewer frames than the npz"
        runs.append((k, (t[:n], d[:n])))
    feats = {k: switch(d) for k, (t, d) in runs}
    keep = np.flatnonzero(np.max([f.max(axis=0) for f in feats.values()], axis=0) >= min_feature)
    assert np.array_equal(keep, pcz["keep"]), "kept-feature set differs from the npz"
    worst = 0.0
    for k, (t, d) in runs:
        y = (feats[k][:, keep] - pcz["pca_mean"]) @ pcz["pca_load"].T
        ref = pcz[f"PC_{k[0]}_{k[1]}"]
        assert np.allclose(ref[:, 0], t), f"{k}: frame times differ from the npz"
        worst = max(worst, float(np.abs(y - ref[:, 1:5]).max()))
    X = np.concatenate([feats[k][:, keep] for k, _ in runs])
    assert np.allclose(X.mean(0), pcz["pca_mean"], atol=1e-5), "stored PCA mean != fit-set mean"
    Xc = X - pcz["pca_mean"]
    evr = (Xc @ pcz["pca_load"].T).var(0, ddof=1) / Xc.var(0, ddof=1).sum()
    print(f"  PCA check: {len(keep)} kept features identical; holo + m-state projections "
          f"reproduce the npz to max |dPC| = {worst:.2e}; EVR from stored loadings "
          f"{np.round(evr, 3)} vs stored {np.round(pcz['pca_evr'], 3)}")
    assert worst < 1e-3 and np.allclose(evr, pcz["pca_evr"], atol=1e-3)


def project_metad(args, holo, mdr, resseq):
    """--project-metad: see module docstring (5)."""
    from concurrent.futures import ProcessPoolExecutor

    pz = os.path.join(OUTDIR, f"{MD}_contact_proj_{args.version}.npz")
    pcz = dict(np.load(pz))
    print(f"  PCA from {pz}: {len(pcz['keep'])} features, EVR {np.round(pcz['pca_evr'], 3)}")
    verify_pca(pcz, holo, mdr, args.min_feature)
    keep, mean, load = pcz["keep"], pcz["pca_mean"], pcz["pca_load"]
    tmax_ps, teq_ps = args.metad_tmax * 1e3, args.metad_teq * 1e3

    import mdtraj as md
    ref_nt = [r.name for r in md.load(metad_top("1Nhp6A")).top.residues if r.name[:2] in DNA_PREFIXES]
    ref_res = [r.name for r in md.load(metad_top("1Nhp6A")).top.residues if r.is_protein]
    for s in args.project_metad:
        top = md.load(metad_top(s)).top
        nt = [r.name for r in top.residues if r.name[:2] in DNA_PREFIXES]
        res = [r.name for r in top.residues if r.is_protein]
        assert nt == ref_nt and len(res) == 93, s
        print(f"  {s}: {top.n_atoms} atoms, 93 residues, DNA identical to WT; residue-name "
              f"differences vs WT: {[(j + 1, a, b) for j, (a, b) in enumerate(zip(ref_res, res)) if a != b]}")

    jobs = [(s, i, args.metad_stride_ps) for s in args.project_metad for i in METAD_WALKERS]
    with ProcessPoolExecutor(max_workers=args.metad_procs) as ex:
        dist = dict(zip([(s, i) for s, i, _ in jobs], ex.map(metad_resdist, jobs)))

    rc = dict(matplotlib.rcParams)              # dw sets a seaborn style on import: restore
    import dna_bending_wall_compare as dw       # colours, STD_CEIL, _reliable_run
    matplotlib.rcParams.update(rc)
    res = {}
    for s in args.project_metad:
        print(f"\n== {s}_wall MetaD")
        T, D, W, CV = [], [], [], {k: [] for k in ("n_idr", "de2e", "bend", "cmap")}
        for i in METAD_WALKERS:
            t, d, de_x = dist[(s, i)]
            m = t <= tmax_ps + 1e-6
            t, d, de_x = t[m], d[m], de_x[m]
            cv, ndup = metad_colvar(s, i)
            j = np.searchsorted(cv["time"], t)
            j = np.clip(j, 0, len(cv["time"]) - 1)
            gap = np.abs(cv["time"][j] - t).max()
            assert gap < 1e-3, f"{s} r{i}: xtc frame times missing from the colvar ({gap} ps)"
            dde = np.abs(cv["de2e"][j] - de_x)
            print(f"  r{i}: {len(t)} frames {t[0] / 1e3:g}-{t[-1] / 1e3:g} ns; colvar "
                  f"{cv['time'][-1] / 1e3:g} ns ({ndup} duplicated times, last kept); "
                  f"de2e xtc vs colvar |diff| median {np.median(dde):.4f}, max {dde.max():.4f} nm")
            T.append(t)
            D.append(d)
            W.append(np.full(len(t), i))
            CV["n_idr"].append(cv["coord_tail_dna"][j])
            CV["de2e"].append(cv["de2e"][j])
            CV["bend"].append(180.0 - np.degrees(cv["theta"][j]))
            CV["cmap"].append(cv["cmap"][j])
        t, d, walker = np.concatenate(T), np.concatenate(D), np.concatenate(W)
        cv = {k: np.concatenate(v) for k, v in CV.items()}
        pcs = (switch(d)[:, keep] - mean) @ load.T
        _, qq = native_q(holo, {("metad", s): (t / 1e3, d)}, resseq, args.native_occ,
                         args.equil)[2][("metad", s)]
        qn = qq[(0, False)]
        qslide = np.max([v for k, v in qq.items() if k != (0, False)], axis=0)
        kslide = [k for k in qq if k != (0, False)]
        islide = np.argmax([qq[k] for k in kslide], axis=0)
        w, ess, ct, gamma, gate = metad_weights(s, t, cv["n_idr"], cv["de2e"], tmax_ps,
                                                teq_ps, args.scratch)
        res[s] = dict(t=t, walker=walker, pc=pcs, q=qn, qslide=qslide, islide=islide,
                      kslide=kslide, w=w, ess=ess, gate=gate, **cv)
        kbt = KB * METAD_T
        u = t >= teq_ps
        print(f"  frames used (t >= {args.metad_teq:g} ns): {u.sum()}; Kish ESS {ess:.0f} "
              f"({100 * ess / u.sum():.1f}%); c(t) {ct['ct'][0]:.1f} -> {ct['ct'][-1]:.1f} kJ/mol")
        print("  gate (reweighted vs sum_hills marginal, MAD kJ/mol [bins]): " + ", ".join(
            f"{k} {v[0]:.2f} [{v[1]}] {'OK' if v[0] < kbt else 'CHECK'}" for k, v in gate.items()))
        print(f"  cmap wall (AT 0.9) active in {100 * (cv['cmap'] < 0.9).mean():.2f}% of frames "
              f"(min cmap {cv['cmap'].min():.3f}; static, not reweighted)")
        early = t < 1e3
        print(f"  sanity: t < 1 ns PC1 mean {pcs[early, 0].mean():.2f} "
              f"(holo 2.24 +/- 0.55), Q {qn[early].mean():.2f}")
        wn = w / w.sum()

        def frac(mask):
            return f"{100 * mask[u].mean():6.2f}% / {100 * wn[mask].sum():6.2f}%"
        print("  fraction of frames (unweighted / reweighted), t >= teq:")
        for lab, mk in (("PC1 < 0", pcs[:, 0] < 0), ("PC1 < 1", pcs[:, 0] < 1),
                        ("PC1 < 1.21 (holo p5)", pcs[:, 0] < 1.21),
                        (f"PC1 in non-native m-state cluster {NONNATIVE_PC1}",
                         (pcs[:, 0] >= NONNATIVE_PC1[0]) & (pcs[:, 0] <= NONNATIVE_PC1[1])),
                        ("Q_native < 0.3", qn < 0.3), ("Q_native < 0.5", qn < 0.5),
                        ("slid/flipped register: best non-identity Q >= 0.5 and > Q_native",
                         (qslide >= 0.5) & (qslide > qn))):
            print(f"    {lab:70s} {frac(mk)}")
        i0 = int(np.argmin(pcs[:, 0]))
        print(f"  min PC1 {pcs[i0, 0]:.2f} (walker {walker[i0]}, t {t[i0] / 1e3:.1f} ns, Q "
              f"{qn[i0]:.2f}, PC2 {pcs[i0, 1]:.2f}, n_IDR {cv['n_idr'][i0]:.1f}, bend "
              f"{cv['bend'][i0]:.1f}); min Q_native {qn.min():.2f}; max best-non-identity Q "
              f"{qslide.max():.2f} ({vlabel(*kslide[islide[int(np.argmax(qslide))]])})")
        print(f"  {'walker':>6} {'frames':>6} {'minPC1':>7} {'p1 PC1':>7} {'f<1':>6} "
              f"{'minQ':>5} {'f(Q<0.5)':>8} {'w share':>8}")
        for k in METAD_WALKERS:
            m = (walker == k) & u
            print(f"  {k:6d} {m.sum():6d} {pcs[m, 0].min():7.2f} "
                  f"{np.percentile(pcs[m, 0], 1):7.2f} {(pcs[m, 0] < 1).mean():6.3f} "
                  f"{qn[m].min():5.2f} {(qn[m] < 0.5).mean():8.3f} {wn[m].sum():8.3f}")
        # Delta F(PC1 < 0 vs >= 0) per snapshot
        dfs = []
        for ts in np.linspace(tmax_ps / args.n_snaps, tmax_ps, args.n_snaps):
            m = u & (t <= ts + 1e-6)
            plo = w[m & (pcs[:, 0] < 0)].sum() / w[m].sum()
            dfs.append(np.inf if plo == 0 else -kbt * np.log(plo / (1 - plo)))
        dfs = np.array(dfs)
        print("  Delta F(PC1 < 0 vs >= 0): " + (
            "PC1 < 0 never sampled" if not np.isfinite(dfs).any() else
            f"{np.mean(dfs):.1f} +/- {np.std(dfs):.1f} kJ/mol" if np.isfinite(dfs).all() else
            f"sampled only in {np.isfinite(dfs).sum()}/{len(dfs)} snapshots: {np.round(dfs, 1)}"))
        res[s]["dF_pc1neg"] = dfs

        # PMFs (full-range .dat)
        nn = len(native_q(holo, {}, resseq, args.native_occ, args.equil)[0])
        prof = {}
        for obs, x, edges in (("pc1", pcs[:, 0], np.arange(-4.0, 4.0 + 1e-9, 0.1)),
                              ("qnative", qn, (np.arange(nn + 2) - 0.5) / nn)):
            out_r = (x < edges[0]) | (x > edges[-1])
            assert not out_r[u].any(), f"{s} {obs}: {out_r[u].sum()} frames outside the bins"
            mn, sd, cnt, nw, nf = snap_profile(x, w, t, walker, edges, args.n_snaps, tmax_ps,
                                               teq_ps)
            c = 0.5 * (edges[:-1] + edges[1:])
            prof[obs] = (c, mn, sd, cnt, nw)
            path = os.path.join(OUTDIR, f"{s}_wall_metad_{obs}_pmf_{args.metad_version}.dat")
            np.savetxt(path, np.column_stack([c, mn, sd, cnt, nw, nf]), fmt="%12.5g",
                       header=(f"{s}_wall MetaD, c(t)-reweighted (n_IDR, de2e hills; cmap wall "
                               f"not reweighted), frames {args.metad_teq:g}-{args.metad_tmax:g} ns "
                               f"at {np.median(np.diff(t[walker == 0])):g} ps x 6 walkers, {args.n_snaps} "
                               f"cumulative-time snapshots, T {METAD_T:g} K; Kish ESS {ess:.0f}\n"
                               f"PCA {pz} (projection only)\n"
                               f"{obs}  F_mean(kJ/mol)  F_std(kJ/mol; NaN if <3 snapshots)  "
                               "frame_count  n_walkers(>=5 frames)  n_snapshots_finite"))
            print(f"  wrote {path}")
        res[s]["prof"] = prof
        out = os.path.join(OUTDIR, f"{s}_wall_metad_contact_proj_{args.metad_version}.npz")
        np.savez_compressed(out, time_ps=t, walker=walker, pc=pcs, q_native=qn,
                            q_best_nonidentity=qslide, n_idr=cv["n_idr"], de2e=cv["de2e"],
                            bend_deg=cv["bend"], cmap=cv["cmap"], weight=w, ess=ess,
                            teq_ns=args.metad_teq, tmax_ns=args.metad_tmax,
                            stride_ps=np.median(np.diff(t[walker == 0])), pca_npz=pz)
        print(f"  wrote {out}")
    plot_metad(res, pcz, dw, args)


def plot_metad(res, pcz, dw, args):
    """Raw coverage figures (unweighted) + reweighted PC1 / Q PMF overlay."""
    holo = [pcz[f"PC_holo_{r}"] for r in HOLO_REPS]
    mst = {k[6:]: pcz[k] for k in pcz if k.startswith("PC_md_")}
    wcol = matplotlib.colormaps["viridis"](np.linspace(0, 0.9, 6))

    def coverage(systems, fname):
        n = len(systems)
        fig, ax = plt.subplots(n, 3, figsize=(16, 4.6 * n), squeeze=False)
        for row, s in enumerate(systems):
            R = res[s]
            a0, a1, a2 = ax[row]
            for a, col in ((a0, 2), (a1, 5)):
                for r, y in zip(HOLO_REPS, holo):
                    a.scatter(y[:, 1], y[:, col], s=2, color="0.75", lw=0, alpha=0.35,
                              marker="s", rasterized=True,
                              label="holo (unbiased, 3 x 250 ns)" if r == 0 else None)
                for k, y in mst.items():
                    c, mk, _ = run_style(k)
                    a.scatter(y[:, 1], y[:, col], s=3, color=c, marker=mk, lw=0, alpha=0.35,
                              rasterized=True, label=f"m-state {k.replace('_', ' ')}")
                yy = R["pc"][:, 1] if col == 2 else R["bend"]
                a.scatter(R["pc"][:, 0], yy, s=1.5, color="k", lw=0, alpha=0.25,
                          rasterized=True, label="MetaD, 6 walkers (system of the row)")
                a.axvline(0, color="k", ls=":", lw=0.8)
                a.set_xlabel(f"PC1 ({100 * pcz['pca_evr'][0]:.0f}%), native core engagement")
                a.set_xlim(-4.2, 4.0)
            a0.set_ylabel(f"PC2 ({100 * pcz['pca_evr'][1]:.0f}%), IDR register")
            a1.set_ylabel("DNA bend (deg)")
            a1.set_ylim(-2, 75)
            a0.set_title(f"{s}_wall MetaD: PC1-PC2 (unweighted: where sampled, not "
                         "populations)", fontsize="small", loc="left")
            a1.set_title(f"{s}_wall MetaD: PC1-bend (unweighted)", fontsize="small", loc="left")
            for k in METAD_WALKERS:
                m = R["walker"] == k
                a2.plot(R["t"][m] / 1e3, running_mean(R["pc"][m, 0], 10), color=wcol[k], lw=0.7,
                        label=f"walker {k}")
            a2.axhspan(1.21, 2.95, color="0.85", lw=0, zorder=0, label="holo PC1 p5-p95")
            a2.axhspan(*NONNATIVE_PC1, color="#D55E00", alpha=0.15, lw=0, zorder=0,
                       label="non-native m-state cluster")
            a2.axhline(0, color="k", ls=":", lw=0.8)
            a2.set_ylim(-4.2, 4.0)
            a2.set_xlabel("time (ns)")
            a2.set_ylabel("PC1, 1-ns running mean")
            a2.set_title(f"{s}_wall: PC1 per walker", fontsize="small", loc="left")
            a2.legend(frameon=False, fontsize="x-small", ncol=3, loc="lower left")
        h, lb = ax[0, 0].get_legend_handles_labels()
        fig.legend(h, lb, loc="upper center", ncol=6, frameon=False, fontsize="small",
                   markerscale=5, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.07 / n))
        out = figpath(fname, root=ROOT)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"  wrote {out}")

    mv = args.metad_version
    if "1Nhp6A" in res:
        coverage(["1Nhp6A"], f"1Nhp6A_wall_contact_pca_proj_{mv}.png")
    muts = [s for s in res if s != "1Nhp6A"]
    if muts:
        coverage(muts, f"mutants_wall_contact_pca_proj_{mv}.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    hq = np.concatenate([y[:, 6] for y in holo])
    ax[0].axvspan(1.21, 2.95, color="0.88", lw=0, zorder=0, label="holo PC1 p5-p95")
    ax[0].axvspan(*NONNATIVE_PC1, color="#D55E00", alpha=0.12, lw=0, zorder=0,
                  label="non-native m-state cluster")
    ax[1].axvspan(*np.percentile(hq, [5, 95]), color="0.88", lw=0, zorder=0,
                  label="holo Q p5-p95")
    for s, R in res.items():
        for a, obs in zip(ax, ("pc1", "qnative")):
            c, mn, sd, cnt, nw = R["prof"][obs]
            keep = dw._reliable_run(np.where(nw >= args.min_walkers, mn, np.nan), sd,
                                    dw.STD_CEIL)
            keep &= np.isfinite(mn) & (nw >= args.min_walkers)
            col = dw.colors[f"{s}_wall"]
            a.plot(c[keep], mn[keep], color=col, lw=1.8, label=f"{s} (ESS {R['ess']:.0f})")
            a.fill_between(c[keep], (mn - sd)[keep], (mn + sd)[keep], color=col, alpha=0.25,
                           lw=0)
    ax[0].set_xlabel(f"PC1 ({100 * pcz['pca_evr'][0]:.0f}%), native core engagement")
    ax[0].set_xlim(-4, 4)
    ax[1].set_xlabel(r"native core-DNA $Q$ (identity register)")
    ax[1].set_xlim(0, 1)
    for a in ax:
        a.set_ylabel(r"$\Delta F$ (kJ/mol)")
        a.set_ylim(bottom=0)
        a.legend(frameon=False, fontsize="small", loc="upper left")
    fig.suptitle(f"Wall MetaD, c(t)-reweighted, mean $\\pm$ std over {args.n_snaps} cumulative-"
                 f"time snapshots; plotted where std $\\leq$ {dw.STD_CEIL:g} kJ/mol (contiguous "
                 f"from the minimum) and >= {args.min_walkers} walkers", fontsize="small",
                 x=0.01, ha="left")
    fig.tight_layout()
    out = figpath(f"wall_metad_contact_pc1_q_pmf_{mv}.png", root=ROOT)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- Stage B (2:1)
SB_MD = "2Nhp6A_mstate_md"
SB_HOLO = "2Nhp6A"
SB_HOLO_REPS = (1, 2)       # rep0 excluded (PBC 0.553 nm + copy-B H3 swing), as for the start frames
SB_PBC = os.path.join(ROOT, "analysis", "pbc", "mstate_2Nhp6A_md")
SB_COPIES = "AB"
SB_HELICES = {"H1": (27, 42), "H2": (48, 61), "H3": (64, 93)}     # = equil_timeseries.HELICES (1LWM)
SB_IDR = (1, 26)            # IDR + linker residues for the per-residue DNA occupancy


def sb_features(t):
    """Per-frame Stage B features of one 2:1 trajectory (whole molecules; min-image distances).
    d_A, d_B: copy x DNA closest-heavy residue-pair distances (93*30, as resdist); d_AB: copy A x
    copy B closest-heavy residue-pair distances (93*93, float16); com_AB: core (Ca 27-93) centre
    separation, minimum image over the 27 neighbouring lattice shifts; H3 RMSD / kink per copy
    from equil_monitor.structural_observables (panel F1 definitions, reference = frame 0)."""
    import itertools
    import mdtraj as md
    from equil_monitor import protein_copies, structural_observables

    out = {}
    for c, cp in enumerate(SB_COPIES):
        pairs, _, _ = residue_pairs(t.top, copy=c)
        d, _ = md.compute_contacts(t, pairs, scheme="closest-heavy", periodic=True,
                                   ignore_nonprotein=False)
        out[f"d_{cp}"] = d.astype(np.float32)
    prot = [r.index for r in t.top.residues if r.is_protein]
    assert len(prot) == 2 * 93, len(prot)
    ab = np.array([(a, b) for a in prot[:93] for b in prot[93:]])
    d, _ = md.compute_contacts(t, ab, scheme="closest-heavy", periodic=True)
    out["d_AB"] = d.astype(np.float16)
    lo, hi = PROBE_CORE
    ca = [[a.index for a in t.top.atoms if a.name == "CA"
           and a.residue.index in set(prot[93 * c + lo - 1: 93 * c + hi])] for c in (0, 1)]
    assert len(ca[0]) == len(ca[1]) == hi - lo + 1
    dx = t.xyz[:, ca[0]].mean(axis=1) - t.xyz[:, ca[1]].mean(axis=1)
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
    cand = dx[:, None, :] + np.einsum("sk,fkj->fsj", shifts, t.unitcell_vectors)
    out["com_AB"] = np.linalg.norm(cand, axis=2).min(axis=1)
    copies = protein_copies(t)
    assert len(copies) == 2, copies
    for (r0, _), cp in zip(copies, SB_COPIES):
        obs = structural_observables(t, r0)
        out[f"h3_rmsd_{cp}"] = obs["H3 RMSD (nm)"]
        out[f"h3_kink_{cp}"] = obs["H3 kink (deg)"]
    return out


def sb_load(kind, key):
    """Cached Stage B features, 100 ps frames, time in ns.  m-state: the PBC-whole solute xtc
    from analysis/pbc/mstate_2Nhp6A_md (run_solute_whole.sh on the run host, -dt 100); 2:1 MM
    holo: equil_monitor.load_whole (whole_skip10.xtc)."""
    import mdtraj as md

    if kind == "md":
        k, _ = md_config(key)
        src = os.path.join(SB_PBC, f"{SB_MD}_{key}_solute_whole.xtc")
        cache = os.path.join(run_dir(SB_MD, key), f"{SB_MD}_{key}_stageb_100ps.npz")
    else:
        d = os.path.join(ROOT, "data", "equil", SB_HOLO, f"rep{key}")
        src = os.path.join(d, "md.xtc")
        cache = os.path.join(d, f"{SB_HOLO}_rep{key}_stageb_100ps.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        return dict(np.load(cache))
    if kind == "md":
        t = md.load(src, top=os.path.join(SB_PBC, f"{SB_MD}_c{k}_solute_ref.pdb"))
        tps = t.time.astype(np.float64)                # xtc stamps, ps (float32)
        assert np.allclose(np.diff(tps), 100.0, atol=0.01), "expected 100 ps frames"
        t.time = tps / 1000.0
    else:
        t = load_whole(SB_HOLO, key, 10)
    F = sb_features(t)
    F["time"] = t.time
    np.savez_compressed(cache, **F)
    print(f"  cached {cache} ({t.n_frames} frames, {t.time[0]:.1f}-{t.time[-1]:.1f} ns)")
    return F


def sb_reglabel(s, f):
    """Register label as read by equil_timeseries.read_qnative_registers ('0', '-1', 'flip_+2')."""
    return vlabel(s, f).replace(" ", "_").replace("_bp", "").replace("native", "0")


def stage_b(args, holo1, resseq):
    """Stage B (PLAN.md B1, Analysis): per-copy bound mode against BOTH native sets -- the copy's
    own 2:1 MM contacts (reps 1-2, primary) and the 1:1 holo native (secondary) -- plus
    inter-protein contacts, core separation, occlusion of each copy's 2:1 footprint, H3 and the
    per-residue IDR occupancy.  Labels over the last 10 ns, as in the 1:1 survival table."""
    from equil_timeseries import MD2_COLOR, bound_mode

    rs = np.asarray(resseq)
    core = (rs >= PROBE_CORE[0]) & (rs <= PROBE_CORE[1])
    regs = [(s, f) for f in (False, True) for s in SHIFTS]
    labels = [sb_reglabel(*v) for v in regs]
    runs = [r for r in SYSTEMS[SB_MD]["reps"]
            if os.path.exists(os.path.join(SB_PBC, f"{SB_MD}_{r}_solute_whole.xtc"))]
    H = {("holo", r): sb_load("holo", r) for r in SB_HOLO_REPS}
    M = {("md", r): sb_load("md", r) for r in runs}
    for k, F in list(H.items()) + list(M.items()):
        assert all(np.isfinite(F[x]).all() for x in F), f"non-finite features in {k}"
        print(f"  {k[0]:4s} {str(k[1]):8s} {len(F['time']):5d} frames, "
              f"{F['time'][0]:.1f}-{F['time'][-1]:.1f} ns")

    def contacts(F, cp):
        return (F[f"d_{cp}"] < CONTACT_CUTOFF).reshape(len(F["time"]), 93, 2 * NBP)

    def hwin(F):
        return F["time"] >= args.equil

    # native sets and Q(t): own 2:1 copy (primary), 1:1 holo (secondary)
    Q, NAT, foot = {}, {}, {}
    for cp in SB_COPIES:
        h2 = {k: (F["time"], F[f"d_{cp}"]) for k, F in H.items()}
        mc = {k: (F["time"], F[f"d_{cp}"]) for k, F in M.items()}
        NAT[cp, "2:1"], _, Q[cp, "2:1"] = native_q(h2, mc, resseq, args.native_occ, args.equil)
        NAT[cp, "1:1"], _, Q[cp, "1:1"] = native_q(holo1, mc, resseq, args.native_occ, args.equil)
        foot[cp] = np.unique(NAT[cp, "2:1"][:, 1])                   # footprint nucleotides
        print(f"  copy {cp}: {len(NAT[cp, '2:1'])} own-2:1 native core pairs on nucleotides "
              f"{[f'{chr(65 + n // NBP)}{n % NBP + 1}' for n in foot[cp]]}; "
              f"{len(NAT[cp, '1:1'])} 1:1 native pairs")

    def occl(F, by, cp):
        """Per frame: fraction of copy cp's 2:1 footprint nucleotides contacted by copy `by`."""
        return contacts(F, by).any(axis=1)[:, foot[cp]].mean(axis=1)

    def series(k, F):
        return dict(n_ab=(F["d_AB"] < CONTACT_CUTOFF).sum(axis=1), com=F["com_AB"],
                    **{f"q_{cp}": Q[cp, "2:1"][k][1][(0, False)] for cp in SB_COPIES},
                    **{f"qbest_{cp}": np.max([Q[cp, "2:1"][k][1][v] for v in regs], axis=0)
                       for cp in SB_COPIES},
                    **{f"occ_{by}_{cp}": occl(F, by, cp) for cp in SB_COPIES for by in SB_COPIES},
                    **{f"h3_{cp}": F[f"h3_rmsd_{cp}"] for cp in SB_COPIES})

    S = {k: series(k, F) for k, F in list(H.items()) + list(M.items())}
    ref = {name: np.percentile(np.concatenate([S[k][name][hwin(H[k])] for k in H]), [5, 50, 95])
           for name in S[next(iter(H))]}

    # ---- table
    out = os.path.join(OUTDIR, f"{SB_MD}_stageb_modes_{args.sb_version}.dat")
    with open(out, "w") as fh:
        fh.write(f"# Stage B 2:1 m-state bound modes, last 10 ns per run (100 ps frames). bound = "
                 f"core({PROBE_CORE[0]}-{PROBE_CORE[1]})-DNA heavy pair < {CONTACT_CUTOFF} nm in "
                 f"frame fraction; mode/register from the best register-variant native core-DNA "
                 f"Q (equil_timeseries.bound_mode: >= 0.5 native/shifted)\n"
                 f"# native sets (holo occupancy >= {args.native_occ}, t >= {args.equil} ns): "
                 f"'2:1' = the copy's own contacts in 2Nhp6A reps {SB_HOLO_REPS} (PRIMARY), "
                 f"'1:1' = 1Nhp6A holo reps {HOLO_REPS} (secondary)\n"
                 "# occ_A/occ_B = fraction of this copy's 2:1 footprint nucleotides contacted by "
                 "copy A / copy B; n_AB = inter-copy residue pairs < 0.45 nm; com_AB = core centre "
                 "separation (nm); h3 = H3 backbone RMSD to the run's frame 0 (nm)\n")
        fh.write("# 2:1 MM reference (reps 1-2, t >= equil) p5/median/p95: " + "; ".join(
            f"{n} {lo:.3g}/{mid:.3g}/{hi:.3g}" for n, (lo, mid, hi) in ref.items()) + "\n")
        fh.write("# run  copy  bound  mode_2:1  reg_2:1  Qbest_2:1  Qid_2:1  mode_1:1  reg_1:1  "
                 "Qbest_1:1  helices  H1  H2  H3  occ_A  occ_B  n_AB  com_AB  h3\n")
        for k, F in M.items():
            t = F["time"]
            last = t >= t[-1] - 10.0
            for cp in SB_COPIES:
                c = contacts(F, cp)[last]
                hel = {h: c[:, (rs >= a) & (rs <= b)].sum(axis=(1, 2)).mean()
                       for h, (a, b) in SB_HELICES.items()}
                row = [k[1], cp, f"{c[:, core].any(axis=(1, 2)).mean():.2f}"]
                for ns in ("2:1", "1:1"):
                    qv = np.array([Q[cp, ns][k][1][v][last].mean() for v in regs])
                    reg, qb, mode, hstr = bound_mode((labels, qv), hel)
                    row += [mode, reg, f"{qb:.3f}"] + ([f"{qv[labels.index('0')]:.3f}"]
                                                       if ns == "2:1" else [])
                row += [hstr] + [f"{v:.1f}" for v in hel.values()]
                s = S[k]
                row += [f"{s[f'occ_A_{cp}'][last].mean():.2f}", f"{s[f'occ_B_{cp}'][last].mean():.2f}",
                        f"{s['n_ab'][last].mean():.1f}", f"{s['com'][last].mean():.2f}",
                        f"{s[f'h3_{cp}'][last].mean():.3f}"]
                fh.write("  ".join(row) + "\n")
    print(open(out).read())

    # ---- per-residue IDR/linker DNA occupancy (last --sb-window ns; holo t >= equil)
    ir = (rs >= SB_IDR[0]) & (rs <= SB_IDR[1])
    occ = {}
    for k, F in M.items():
        win = F["time"] >= F["time"][-1] - args.sb_window
        for cp in SB_COPIES:
            occ[k, cp] = contacts(F, cp)[win][:, ir].any(axis=2).mean(axis=0)
    occ_ref = {cp: np.concatenate([contacts(F, cp)[hwin(F)][:, ir].any(axis=2)
                                   for F in H.values()]).mean(axis=0) for cp in SB_COPIES}
    h1 = np.concatenate([(d[t >= args.equil] < CONTACT_CUTOFF).reshape(-1, 93, 2 * NBP)[:, ir]
                         .any(axis=2) for t, d in holo1.values()]).mean(axis=0)
    opath = os.path.join(OUTDIR, f"{SB_MD}_idr_occupancy_{args.sb_version}.dat")
    cols = [f"{k[1]}_{cp}" for (k, cp) in occ] + ["2to1_A", "2to1_B", "1to1"]
    np.savetxt(opath, np.column_stack([rs[ir]] + list(occ.values())
                                      + [occ_ref["A"], occ_ref["B"], h1]),
               fmt=["%4d"] + ["%6.3f"] * len(cols),
               header=f"per-residue DNA occupancy (any nucleotide < {CONTACT_CUTOFF} nm): m-state "
                      f"last {args.sb_window:g} ns; 2:1 MM reps {SB_HOLO_REPS} and 1:1 holo t >= "
                      f"{args.equil:g} ns\nresseq  " + "  ".join(cols))
    print(f"  wrote {opath}")
    for r in (16, 22):
        i = int(np.flatnonzero(rs[ir] == r)[0])
        print(f"  K{r} occupancy: " + ", ".join(f"{k[1]} {cp} {v[i]:.2f}" for (k, cp), v in occ.items())
              + f" | 2:1 MM A {occ_ref['A'][i]:.2f} B {occ_ref['B'][i]:.2f} | 1:1 {h1[i]:.2f}")

    # ---- figure
    fig, axs = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    w = int(round(args.smooth_sb / 0.1))
    panels = [("q_B", "copy B Q, own 2:1 native (identity reg.)", axs[0, 0], "B"),
              ("qbest_B", "copy B Q, best register (own 2:1 native)", axs[0, 1], "B"),
              ("q_A", "copy A Q, own 2:1 native (identity reg.)", axs[0, 2], "A"),
              ("n_ab", "inter-copy residue contacts (< 0.45 nm)", axs[1, 0], ""),
              ("com", "core-core centre separation (nm)", axs[1, 1], ""),
              ("occ_A_B", "copy-B 2:1 footprint contacted by copy A", axs[1, 2], ""),
              ("h3", "H3 backbone RMSD to frame 0 (nm)", axs[2, 0], "AB")]
    for name, lab, ax, cps in panels:
        keys = [f"h3_{cp}" for cp in cps] if name == "h3" else [name]
        for key, ls in zip(keys, ("-", "--")):
            tag = f" {key[-1]}" if name == "h3" else ""
            lo, _, hi = ref[key]
            ax.axhspan(lo, hi, color="0.85" if ls == "-" else "0.93", lw=0, zorder=0,
                       label=f"2:1 MM p5-p95{tag}")
            for k in M:
                t, y = M[k]["time"], S[k][key]
                ax.plot(t, y, color=MD2_COLOR[k[1]], lw=0.3, alpha=0.35, ls=ls)
                ax.plot(t, running_mean(y, w), color=MD2_COLOR[k[1]], lw=1.4, ls=ls,
                        label=f"{k[1].replace('_', ' ')}{tag}")
        ax.set_xlabel("time (ns)")
        ax.set_ylabel(lab, fontsize="small")
        ax.legend(fontsize="x-small", ncol=2)
    ax = axs[1, 2]
    for k in M:
        ax.plot(M[k]["time"], running_mean(S[k]["occ_B_B"], w), color=MD2_COLOR[k[1]], lw=1.0, ls=":")
    ax.text(0.02, 0.97, "dotted: contacted by copy B itself", transform=ax.transAxes,
            va="top", fontsize="x-small")
    x = rs[ir]
    for ax, cp in ((axs[2, 1], "B"), (axs[2, 2], "A")):
        ax.plot(x, h1, "k--", lw=1.2, label="1:1 holo")
        ax.plot(x, occ_ref[cp], color="0.5", lw=2.5, alpha=0.7, label=f"2:1 MM copy {cp}")
        for (k, c), v in occ.items():
            if c == cp:
                ax.plot(x, v, "o-", ms=3, lw=1, color=MD2_COLOR[k[1]], label=k[1].replace("_", " "))
        for r in (16, 22):
            ax.axvline(r, color="0.7", lw=0.6, ls=":")
        ax.axvspan(20.5, 26.5, color="#E69F00", alpha=0.08, lw=0)
        ax.set_xlabel("residue (IDR 1-20, linker 21-26; K16/K22 dotted)")
        ax.set_ylabel(f"copy {cp} DNA occupancy (last {args.sb_window:g} ns)", fontsize="small")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize="x-small", ncol=2)
    fig.suptitle(f"Stage B 2:1 m-state ({SB_MD}), {len(M)} runs x "
                 f"{max(F['time'][-1] for F in M.values()):.0f} ns, 100 ps frames; colour = run, "
                 f"thick = {args.smooth_sb:g} ns running mean; grey band = 2:1 MM reps "
                 f"{'/'.join(map(str, SB_HOLO_REPS))}, t >= {args.equil:g} ns",
                 x=0.01, ha="left", fontsize="medium")
    fpath = figpath(f"{SB_MD}_stageb_{args.sb_version}.png", root=ROOT)
    fig.savefig(fpath, dpi=300)
    print(f"  wrote {fpath}")


# ---------------------------------------------------------------- 1:1 m-state vs 2:1 copy B
NO_DNA = 99.0               # nm: distance given to 1:1-frame nucleotides with no 2:1 source
SB_FLAG = {"c1_rep0": "*", "c1_rep1": "*"}   # start has bp A12:B4 open at copy A's wedge site


def sb_frame(nat1, nat2):
    """Register map (shift, flip) of a 2:1 copy's own native pairs onto the 1:1 native pairs:
    highest Jaccard over shifts -10..10 bp and the dyad flip (pairs mapped off the duplex count
    as unmatched).  Returns the ranked (jaccard, shift, flip, shared) list."""
    ref = set(map(tuple, nat1))
    score = []
    for f in (False, True):
        for s in range(-10, 11):
            m = nt_map(s, f)[nat2[:, 1]]
            mp = {(r, n) for r, n in zip(nat2[:, 0], m) if n >= 0}
            shared = len(mp & ref)
            score.append((shared / (len(nat2) + len(ref) - shared), s, f, shared))
    return sorted(score, reverse=True)


def sb_to_frame(d, shift, flip):
    """Copy distances (frames, 93*30, 2:1 nucleotide order) re-indexed into the 1:1 native frame:
    2:1 nucleotide i -> nt_map(shift, flip)[i].  1:1-frame nucleotides with no 2:1 source (off
    the duplex in this copy's frame) get NO_DNA, i.e. contact 0.  Also returns the per-frame
    fraction of the copy's switch contact mass on 2:1 nucleotides that fall outside the 1:1
    frame, and the 1:1-frame nucleotides without a source."""
    m = nt_map(shift, flip)
    src = d.reshape(len(d), 93, 2 * NBP).astype(np.float32)
    out = np.full_like(src, NO_DNA)
    ok = m >= 0
    out[:, :, m[ok]] = src[:, :, ok]
    s = switch(src)
    lost = s[:, :, ~ok].sum(axis=(1, 2)) / np.maximum(s.sum(axis=(1, 2)), 1e-9)
    return out.reshape(len(d), -1), lost, np.setdiff1d(np.arange(2 * NBP), m[ok])


def mm_compare(args, holo, mdr, resseq):
    """--mm-compare: see module docstring (7)."""
    from scipy.spatial import cKDTree
    from equil_timeseries import MD2_COLOR

    pcz = dict(np.load(args.pca_from))
    print(f"  PCA from {args.pca_from}: {len(pcz['keep'])} features, EVR "
          f"{np.round(pcz['pca_evr'], 3)} (no refit)")
    verify_pca(pcz, holo, mdr, args.min_feature)
    keep, mean, load, evr = pcz["keep"], pcz["pca_mean"], pcz["pca_load"], pcz["pca_evr"]
    drop = np.setdiff1d(np.arange(93 * 2 * NBP), keep)
    rs = np.asarray(resseq)
    fres = np.repeat(rs, 2 * NBP)                    # residue of each feature (protein-major)
    basis = os.path.basename(args.pca_from)
    if args.mm_basis == "core":
        # refit on core (27-93) features only, on exactly the v3 fit frames
        assert str(pcz["fit"]) == "all" and float(pcz["tmin"]) == 0.0, "expected a pooled, tmin 0 fit"
        nfit = {k: len(pcz[f"PC_{k[0]}_{k[1]}"]) for k in list(holo) + list(mdr)}
        X = np.concatenate([switch(d[:nfit[k]]) for k, (t, d) in list(holo.items()) + list(mdr.items())])
        assert len(X) == int(pcz["n_fit_holo"]) + int(pcz["n_fit_mstate"]), "fit set != v3 fit set"
        keep = np.flatnonzero((fres >= PROBE_CORE[0]) & (X.max(axis=0) >= args.min_feature))
        from sklearn.decomposition import PCA
        pca = PCA(n_components=4).fit(X[:, keep])
        mean, load, evr = pca.mean_, pca.components_.copy(), pca.explained_variance_ratio_
        hm = np.concatenate([switch(d[t >= args.equil])[:, keep] for t, d in holo.values()]).mean(0)
        sgn = np.sign((hm - mean) @ load.T)          # orientation: holo on the positive side
        load *= np.where(sgn == 0, 1, sgn)[:, None]
        drop = np.setdiff1d(np.flatnonzero(fres >= PROBE_CORE[0]), keep)   # core pairs never formed in the fit
        nat1 = native_q(holo, {}, resseq, args.native_occ, args.equil)[0]
        isnat = np.isin(keep, nat1[:, 0] * 2 * NBP + nat1[:, 1])
        print(f"  core-only PCA refit on the v3 fit frames ({len(X)}: holo {int(pcz['n_fit_holo'])}, "
              f"m-state {int(pcz['n_fit_mstate'])}), {len(keep)} core features; EVR {np.round(evr, 3)}")
        for j in range(4):
            l2 = load[j] ** 2
            print(f"    PC{j + 1}: squared loading H1 {l2[(keep // (2 * NBP) + 1 >= 27) & (keep // (2 * NBP) + 1 <= 42)].sum():.2f}"
                  f" H2 {l2[(keep // (2 * NBP) + 1 >= 48) & (keep // (2 * NBP) + 1 <= 61)].sum():.2f}"
                  f" H3 {l2[keep // (2 * NBP) + 1 >= 64].sum():.2f}; on the {len(nat1)} 1:1 native pairs "
                  f"{l2[isnat].sum():.2f}")
        basis = f"core-only refit on the {basis} fit frames"
        del X

    def project(d):
        """PC1-4, residual norm outside the 4-PC plane, contact mass on features not in the basis."""
        x = switch(d)
        xc = x[:, keep] - mean
        y = xc @ load.T
        return y, np.linalg.norm(xc - y @ load, axis=1), x[:, drop].sum(axis=1)

    # ---- common frame: each 2:1 copy's own native pairs mapped onto the 1:1 native pairs
    nat1 = native_q(holo, {}, resseq, args.native_occ, args.equil)[0]
    H = {r: sb_load("holo", r) for r in SB_HOLO_REPS}
    runs = [r for r in SYSTEMS[SB_MD]["reps"]
            if os.path.exists(os.path.join(SB_PBC, f"{SB_MD}_{r}_solute_whole.xtc"))]
    M = {r: sb_load("md", r) for r in runs}
    frame = {}
    for cp in SB_COPIES:
        nat2 = native_q({k: (F["time"], F[f"d_{cp}"]) for k, F in H.items()}, {}, resseq,
                        args.native_occ, args.equil)[0]
        rank = sb_frame(nat1, nat2)
        j, s, f, n = rank[0]
        assert j > 0.4 and j - rank[1][0] > 0.05, f"copy {cp}: no clear register map {rank[:3]}"
        frame[cp] = (s, f)
        print(f"  copy {cp} -> 1:1 frame: {vlabel(s, f)} (Jaccard {j:.2f}, {n} of {len(nat2)} "
              f"own native pairs on the {len(nat1)} 1:1 pairs; runner-up {vlabel(*rank[1][1:3])} "
              f"{rank[1][0]:.2f})")

    # ---- every set in the 1:1 frame: (time, mapped distances, lost fraction)
    sets = {("1:1 holo", r, ""): (t, d, np.zeros(len(t))) for (_, r), (t, d) in holo.items()}
    sets.update({("1:1 m-state", r, ""): (t, d, np.zeros(len(t))) for (_, r), (t, d) in mdr.items()})
    for kind, src in (("2:1 holo", H), ("2:1 m-state", M)):
        for key, F in src.items():
            for cp in SB_COPIES:
                dm, lost, nosrc = sb_to_frame(F[f"d_{cp}"], *frame[cp])
                sets[(kind, key, cp)] = (F["time"], dm, lost)
    print(f"  1:1-frame nucleotides with no 2:1 source: " + "; ".join(
        f"copy {cp} {[f'{chr(65 + n // NBP)}{n % NBP + 1}' for n in sb_to_frame(np.zeros((1, 93 * 2 * NBP)), *frame[cp])[2]]}"
        for cp in SB_COPIES))
    _, _, Qall = native_q(holo, {k: (v[0], v[1]) for k, v in sets.items()}, resseq,
                          args.native_occ, args.equil)
    regs = [(s, f) for f in (False, True) for s in SHIFTS]
    R = {}
    for k, (t, d, lost) in sets.items():
        y, res, out = project(d)
        q = Qall[k][1]
        qv = np.array([q[v] for v in regs])
        R[k] = dict(t=t, pc=y, res=res, out=out, lost=lost, q=q[(0, False)],
                    qbest=qv.max(axis=0), ibest=qv.argmax(axis=0))
    for (_, r), (t, _) in mdr.items():               # the 1:1 frames must match the saved basis
        ref = pcz[f"PC_md_{r}"]
        n = len(ref)
        if args.mm_basis == "pca-from":
            assert np.abs(R[("1:1 m-state", r, "")]["pc"][:n] - ref[:, 1:5]).max() < 1e-3

    # ---- nearest-neighbour distance in PC1-4 (frames of one start config vs all the others)
    def pcs(sel):
        return np.concatenate([R[k]["pc"] for k in R if sel(k)])
    mkeys = [k for k in R if k[0] == "1:1 m-state"]
    tree_m = cKDTree(pcs(lambda k: k[0] == "1:1 m-state"))
    tree_h = cKDTree(pcs(lambda k: k[0] == "1:1 holo"))
    for k in mkeys:                                   # leave-one-start-config-out baseline
        cfg = md_config(k[1])[0]
        R[k]["nn"] = cKDTree(pcs(lambda j: j[0] == "1:1 m-state" and md_config(j[1])[0] != cfg)
                             ).query(R[k]["pc"])[0]
    for k in R:
        if k[0] == "1:1 holo":                        # leave-one-rep-out holo baseline
            R[k]["nn_h"] = cKDTree(pcs(lambda j: j[0] == "1:1 holo" and j[1] != k[1])
                                   ).query(R[k]["pc"])[0]
        elif k[0] != "1:1 m-state":
            R[k]["nn"] = tree_m.query(R[k]["pc"])[0]
            R[k]["nn_h"] = tree_h.query(R[k]["pc"])[0]
    base = np.concatenate([R[k]["nn"] for k in mkeys])
    bh = np.concatenate([R[k]["nn_h"] for k in R if k[0] == "1:1 holo"])
    b50, b95 = np.percentile(base, [50, 95])
    h50, h95 = np.percentile(bh, [50, 95])
    print(f"  NN distance (PC1-4) baselines: 1:1 m-state, each start config vs the other three: "
          f"median {b50:.2f}, p95 {b95:.2f}; 1:1 holo, each rep vs the other two: median {h50:.2f}, "
          f"p95 {h95:.2f}")

    # ---- table
    hb = lambda k: k[0] in ("1:1 holo", "2:1 holo")
    out = os.path.join(OUTDIR, f"{SB_MD}_mmcompare_{args.mm_version}.dat")
    with open(out, "w") as fh:
        fh.write(f"# 1:1 m-state vs 2:1 copies in the 1:1 native frame, projected on "
                 f"{basis}.  Copy maps: " + ", ".join(
                     f"{cp} {vlabel(*frame[cp])}" for cp in SB_COPIES) + "\n"
                 f"# all frames from t = 0 (the pulled start) except holo (t >= {args.equil:g} ns); "
                 f"'end' = last 10 ns.  Q = 1:1 native core-DNA set in the 1:1 frame (identity / best "
                 f"of the +-2 bp, flip variants).  nn = PC1-4 distance to the nearest 1:1 m-state "
                 f"frame (1:1 m-state rows: to the other start configs), baseline median {b50:.2f} "
                 f"p95 {b95:.2f}; f95 = fraction of frames with nn <= baseline p95.  nn_h = to the "
                 f"nearest 1:1 holo frame (holo baseline median {h50:.2f} p95 {h95:.2f}).  res = norm "
                 f"outside the 4-PC plane; lost = fraction of the copy's contact mass on nucleotides "
                 f"outside the 1:1 frame; out = contact mass on features outside the basis.  * = c1 start, bp A12:B4 open\n")
        fh.write("# set  run  copy  frames  PC1_end  PC2_end  PC3_end  PC4_end  Q_end  Qbest_end  "
                 "reg_end  nn_med  f95  nn_end  nn_h_med  nn_h_end  res_med  lost_med  out_med\n")
        for k, v in R.items():
            m = v["t"] >= args.equil if hb(k) else np.ones(len(v["t"]), bool)
            e = v["t"] >= v["t"][-1] - 10.0
            reg = vlabel(*regs[np.bincount(v["ibest"][e], minlength=len(regs)).argmax()])
            nn = v.get("nn", np.full(len(m), np.nan))
            row = [k[0].replace(" ", "_"), f"{k[1]}{SB_FLAG.get(k[1], '') if k[0] == '2:1 m-state' else ''}", k[2] or "-",
                   str(m.sum())] + [f"{x:.2f}" for x in v["pc"][e].mean(axis=0)] + [
                f"{v['q'][e].mean():.2f}", f"{v['qbest'][e].mean():.2f}", reg.replace(" ", ""),
                f"{np.median(nn[m]):.2f}", f"{(nn[m] <= b95).mean():.2f}", f"{np.median(nn[e]):.2f}",
                f"{np.median(v['nn_h'][m]):.2f}" if "nn_h" in v else "nan",
                f"{np.median(v['nn_h'][e]):.2f}" if "nn_h" in v else "nan",
                f"{np.median(v['res'][m]):.2f}", f"{np.median(v['lost'][m]):.3f}",
                f"{np.median(v['out'][m]):.2f}"]
            fh.write("  ".join(row) + "\n")
    print(open(out).read())

    npz = os.path.join(OUTDIR, f"{SB_MD}_mmcompare_{args.mm_version}.npz")
    np.savez_compressed(npz, basis=basis, keep=keep, pca_mean=mean, pca_load=load, pca_evr=evr,
                        frame=np.array([(cp, *frame[cp]) for cp in SB_COPIES], dtype=object).astype(str),
                        proj_columns=np.array(["time_ns", "PC1", "PC2", "PC3", "PC4", "Q_native",
                                               "Q_best", "nn_mstate", "residual", "lost"]),
                        **{"|".join(map(str, k)): np.column_stack(
                            [v["t"], v["pc"], v["q"], v["qbest"], v.get("nn", np.full(len(v["t"]), np.nan)),
                             v["res"], v["lost"]]) for k, v in R.items()})
    print(f"  wrote {npz}")

    # ---- figure
    pl = "core PC" if args.mm_basis == "core" else "PC"
    w = int(round(args.smooth_sb / 0.1))
    fig, axs = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    hol = np.concatenate([R[k]["pc"][R[k]["t"] >= args.equil] for k in R if k[0] == "1:1 holo"])
    hB = np.concatenate([R[k]["pc"][R[k]["t"] >= args.equil] for k in R
                         if k[0] == "2:1 holo" and k[2] == "B"])
    allm = pcs(lambda k: k[0] == "1:1 m-state")
    mA = pcs(lambda k: k[0] == "2:1 m-state" and k[2] == "A")
    sbB = [k for k in R if k[0] == "2:1 m-state" and k[2] == "B"]

    def backdrop(ax, i, j, mstate):
        if mstate:
            ax.scatter(allm[:, i], allm[:, j], s=1, c="0.82", lw=0, rasterized=True,
                       label="1:1 m-state, all frames")
        ax.scatter(hol[:, i], hol[:, j], s=1, c="0.45", lw=0, rasterized=True, label="1:1 holo")
        ax.scatter(hB[:, i], hB[:, j], s=1, c="#CC79A7", alpha=0.5, lw=0, rasterized=True,
                   label="2:1 holo copy B (mapped)")
        ax.scatter(mA[:, i], mA[:, j], s=1, c="#56B4E9", alpha=0.35, lw=0, rasterized=True,
                   label="2:1 m-state copy A, all frames")
        ax.set_xlabel(f"{pl}{i + 1}")
        ax.set_ylabel(f"{pl}{j + 1}")

    def path(ax, v, i, j, color, ls="-", label=None):
        y = np.column_stack([running_mean(v["pc"][:, c], w) for c in (i, j)])
        ax.plot(y[:, 0], y[:, 1], color=color, lw=1.0, ls=ls, label=label)
        ax.plot(*y[-1], "o", color=color, ms=6, mec="k", mew=0.5)

    z0 = -mean @ load.T                  # a frame with no contact on any basis feature
    for k in [k for k in R if k[0] in ("1:1 m-state", "2:1 m-state") and k[2] != "A"]:
        assert np.linalg.norm(R[k]["pc"][0] - z0) < 0.1, f"{k}: start frame has core-DNA contacts"

    def start(ax, i, j):
        ax.plot(z0[i], z0[j], "*", mfc="white", mec="k", ms=14, mew=1.2, zorder=5,
                label="pulled starts")

    ax = axs[0, 0]
    backdrop(ax, 0, 1, False)
    for k in mkeys:
        c, _, ls = run_style(k[1])
        path(ax, R[k], 0, 1, c, ls, k[1].replace("_", " "))
    start(ax, 0, 1)
    ax.set_title("1:1 m-state (star = all pulled starts: no core-DNA contact; filled circle = end)",
                 fontsize="small")
    for ax, (i, j) in ((axs[0, 1], (0, 1)), (axs[0, 2], (0, 2))):
        backdrop(ax, i, j, True)
        for k in sbB:
            path(ax, R[k], i, j, MD2_COLOR[k[1]], "-", f"{k[1].replace('_', ' ')}{SB_FLAG.get(k[1], '')}")
        start(ax, i, j)
        ax.set_title("2:1 m-state copy B, mapped to the 1:1 frame", fontsize="small")
    for name, ax, lab in (("pc", axs[1, 0], f"{pl}1"), ("qbest", axs[1, 1],
                                                     "Q best register (1:1 native, 1:1 frame)")):
        for kind, col, nm in (("1:1 holo", "0.6", "1:1 holo"), ("2:1 holo", "#CC79A7", "2:1 holo B")):
            ys = np.concatenate([(R[k][name][:, 0] if name == "pc" else R[k][name])[R[k]["t"] >= args.equil]
                                 for k in R if k[0] == kind and k[2] in ("", "B")])
            lo, hi = np.percentile(ys, [5, 95])
            ax.axhspan(lo, hi, color=col, alpha=0.25, lw=0, label=f"{nm} p5-p95")
        for k in mkeys:
            y = R[k][name][:, 0] if name == "pc" else R[k][name]
            ax.plot(R[k]["t"], running_mean(y, w), color="0.55", lw=0.7)
        ax.plot([], [], color="0.55", lw=0.7, label="1:1 m-state (8 runs)")
        for k in sbB:
            y = R[k][name][:, 0] if name == "pc" else R[k][name]
            ax.plot(R[k]["t"], running_mean(y, w), color=MD2_COLOR[k[1]], lw=1.4,
                    label=f"{k[1].replace('_', ' ')}{SB_FLAG.get(k[1], '')} B")
        ax.set_xlabel("time since pulled start (ns)")
        ax.set_ylabel(lab, fontsize="small")
    ax = axs[1, 2]
    ax.axhspan(0, b95, color="0.85", lw=0, label=f"1:1 m-state baseline <= p95 ({b95:.2f})")
    ax.axhline(b50, color="0.5", lw=0.8, ls="--", label=f"baseline median ({b50:.2f})")
    for k in mkeys:
        ax.plot(R[k]["t"], running_mean(R[k]["nn"], w), color="0.6", lw=0.6)
    for k in sbB:
        ax.plot(R[k]["t"], running_mean(R[k]["nn"], w), color=MD2_COLOR[k[1]], lw=1.4,
                label=f"{k[1].replace('_', ' ')}{SB_FLAG.get(k[1], '')} B")
    ax.set_xlabel("time since pulled start (ns)")
    ax.set_ylabel(f"{pl}1-4 distance to nearest 1:1 m-state frame", fontsize="small")
    ax = axs[2, 0]
    bins = np.linspace(min(hol[:, 0].min(), hB[:, 0].min(), mA[:, 0].min()) - 0.5, hol[:, 0].max() + 0.5, 50)
    for kind, cp, col, nm in (("1:1 holo", "", "0.3", "1:1 holo"),
                              ("2:1 holo", "A", "#56B4E9", "2:1 holo copy A (identity)"),
                              ("2:1 holo", "B", "#CC79A7", f"2:1 holo copy B ({vlabel(*frame['B'])})"),
                              ("2:1 m-state", "A", "#0072B2", "2:1 m-state copy A, all frames")):
        ys = np.concatenate([R[k]["pc"][R[k]["t"] >= (args.equil if hb(k) else 0), 0] for k in R
                             if k[0] == kind and k[2] == cp])
        ax.hist(ys, bins=bins, density=True, histtype="step", color=col, lw=1.5, label=nm,
                ls="--" if kind == "2:1 m-state" else "-")
    ax.set_xlabel(f"{pl}1")
    ax.set_ylabel("density")
    ax.set_title("mapping check: native complexes", fontsize="small")
    ax = axs[2, 1]
    groups = [("1:1 holo", lambda k: k[0] == "1:1 holo"),
              ("1:1 m-state", lambda k: k[0] == "1:1 m-state"),
              ("2:1 holo B", lambda k: k[0] == "2:1 holo" and k[2] == "B"),
              ("2:1 m-state B", lambda k: k[0] == "2:1 m-state" and k[2] == "B")]
    data = [np.concatenate([R[k]["res"][R[k]["t"] >= (args.equil if hb(k) else 0)]
                            for k in R if sel(k)]) for _, sel in groups]
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(groups) + 1), [g for g, _ in groups], fontsize="small")
    ax.set_ylabel(f"contact norm outside the {pl}1-4 plane", fontsize="small")
    ax = axs[2, 2]
    for k in sbB:
        ax.plot(R[k]["t"], running_mean(R[k]["lost"], w), color=MD2_COLOR[k[1]], lw=1.4,
                label=f"{k[1].replace('_', ' ')}{SB_FLAG.get(k[1], '')} B")
    lb = np.concatenate([R[k]["lost"][R[k]["t"] >= args.equil] for k in R
                         if k[0] == "2:1 holo" and k[2] == "B"])
    ax.axhline(np.median(lb), color="#CC79A7", ls="--", lw=1, label="2:1 holo B median")
    ax.set_xlabel("time since pulled start (ns)")
    ax.set_ylabel("copy-B contact mass off the 1:1 frame (fraction)", fontsize="small")
    for ax in axs.flat:
        if ax.get_legend_handles_labels()[0]:
            if ax is axs[0, 0]:      # runs + start only; the clouds are labelled in (b), (c)
                hl = [(h, l) for h, l in zip(*ax.get_legend_handles_labels())
                      if not isinstance(h, matplotlib.collections.PathCollection)]
                leg = ax.legend(*zip(*hl), fontsize="x-small", ncol=3, loc="lower right")
            else:
                leg = ax.legend(fontsize="x-small", ncol=2)
            for h in leg.legend_handles:             # enlarge the 1-px scatter dots only
                if isinstance(h, matplotlib.collections.PathCollection):
                    h.set_sizes([16])
    fig.suptitle(f"1:1 m-state vs 2:1 copy B in the 1:1 native frame (copy B {vlabel(*frame['B'])}), "
                 f"projected on {basis};\nall frames from the pulled start; "
                 f"lines = {args.smooth_sb:g} ns running mean; * c1 start: bp A12:B4 open",
                 x=0.01, ha="left", fontsize="medium")
    fpath = figpath(f"{SB_MD}_mmcompare_{args.mm_version}.png", root=ROOT)
    fig.savefig(fpath, dpi=300)
    print(f"  wrote {fpath}")


def main():
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip", type=int, default=10, help="xtc frame stride (10 ps frames)")
    p.add_argument("--equil", type=float, default=20.0, help="holo ns discarded (native set)")
    p.add_argument("--native-occ", type=float, default=0.5,
                   help="holo contact occupancy defining a native core-DNA pair")
    p.add_argument("--fit", choices=("all", "mstate"), default="all",
                   help="PCA fit set: pooled holo + m-state (v2) or m-state only (v1)")
    p.add_argument("--tmin", type=float, default=0.0, help="m-state ns discarded before the fit")
    p.add_argument("--t-detached", type=float, default=10.0,
                   help="m-state ns marked as the detached-core start in the figure")
    p.add_argument("--tica", action="store_true", help="also fit tICA (separate figure)")
    p.add_argument("--lag", type=float, default=1.0, help="tICA lag (ns)")
    p.add_argument("--binary", action="store_true", help="0/1 contacts instead of the switch")
    p.add_argument("--min-feature", type=float, default=0.05)
    p.add_argument("--version", default="v2", help="PCA figure / projection npz version")
    p.add_argument("--q-version", default="v1", help="native-Q figure version")
    p.add_argument("--pca-from", default=None, metavar="NPZ",
                   help="no refit: project every frame onto the PCA saved in NPZ (checked to "
                        "reproduce its own fit first) and write --version outputs; for runs "
                        "extended after the fit")
    p.add_argument("--redo-q", action="store_true",
                   help="rewrite the native-Q figure/.dat even if the --q-version figure exists")
    g = p.add_argument_group("event dissection / sliding (opt-in; no PCA refit, no PCA figure)")
    g.add_argument("--dissect", nargs="+", metavar="RUN",
                   help="m-state runs (e.g. c0_rep1 c0_rep0) to dissect over --window")
    g.add_argument("--window", type=float, nargs=2, default=(0.0, 25.0), metavar=("B", "E"))
    g.add_argument("--dissect-skip", type=int, default=1, help="stride in 10-ps frames")
    g.add_argument("--dissect-version", default="v1")
    g.add_argument("--smooth", type=float, default=0.1, help="running-mean width (ns)")
    g.add_argument("--stable-hold", type=float, default=1.0,
                   help="ns over which a contact must persist to count as stably formed")
    g.add_argument("--stable-frac", type=float, default=0.8,
                   help="fraction of frames formed over --stable-hold")
    g.add_argument("--sliding", nargs="+", metavar="RUN:END_NS",
                   help="slow re-engagers for the register-variant figure, e.g. c4_rep0:70")
    g.add_argument("--sliding-version", default="v1")
    g.add_argument("--pca-npz", default=None,
                   help="projection npz with the fitted PCA used for PC1 in --dissect "
                        "(default: the --version one)")
    g = p.add_argument_group("MetaD projection (opt-in; no PCA refit)")
    g.add_argument("--project-metad", nargs="+", choices=METAD_SYS, metavar="SYS",
                   help=f"wall MetaD systems to project ({' '.join(METAD_SYS)})")
    g.add_argument("--metad-stride-ps", type=float, default=100.0,
                   help="analysed frame spacing (ps); must be a multiple of the xtc spacing")
    g.add_argument("--metad-tmax", type=float, default=300.0,
                   help="ns per walker analysed (common xtc range)")
    g.add_argument("--metad-teq", type=float, default=5.0,
                   help="bias-filling transient (ns) given zero weight")
    g.add_argument("--n-snaps", type=int, default=5, help="cumulative-time snapshots")
    g.add_argument("--min-walkers", type=int, default=3,
                   help="plotted PMF bins must be visited (>= 5 frames) by this many walkers")
    g.add_argument("--metad-procs", type=int, default=4, help="parallel distance jobs")
    g.add_argument("--metad-version", default="v1")
    g.add_argument("--scratch", default=None, help="sum_hills work dir (default: a tempdir)")
    g = p.add_argument_group("Stage B 2:1 (opt-in; no PCA)")
    g.add_argument("--stage-b", action="store_true",
                   help="2:1 m-state bound modes, inter-copy contacts, occlusion, H3, IDR occupancy")
    g.add_argument("--sb-window", type=float, default=100.0,
                   help="last ns of each run for the per-residue IDR occupancy")
    g.add_argument("--smooth-sb", type=float, default=2.0, help="running-mean width (ns)")
    g.add_argument("--sb-version", default="v1")
    g.add_argument("--mm-compare", action="store_true",
                   help="1:1 m-state vs 2:1 copies in the 1:1 native frame, on the --pca-from basis")
    g.add_argument("--mm-basis", choices=("core", "pca-from"), default="core",
                   help="core: refit a core(27-93)-only PCA on the --pca-from fit frames; "
                        "pca-from: project on the saved (IDR + core) basis as is (v1)")
    g.add_argument("--mm-version", default="v1")
    p.add_argument("--t-max", type=float, default=None, metavar="NS",
                   help="cut every m-state run at this time before anything is computed (e.g. 250: "
                        "the common length, as if c3/c4_rep0 were never extended); outputs are "
                        "tagged _t<NS>ns so the full-length tables are never overwritten")
    args = p.parse_args()
    if args.mm_compare and not args.pca_from:
        p.error("--mm-compare projects on a saved basis: give --pca-from NPZ")
    if args.pca_from and (args.tica or args.binary):
        p.error("--pca-from projects the saved switch-contact PCA; no --tica / --binary")

    import mdtraj as md
    top = md.load(os.path.join(ROOT, "1Nhp6A_DNA_parambsc1_tip3p.pdb")).top
    _, resseq, ntlab = residue_pairs(top)
    resname = [r.code or "X" for r in top.residues if r.is_protein]

    md_reps = list(SYSTEMS[MD]["reps"])
    holo = {("holo", r): resdist("holo", r, args.skip) for r in HOLO_REPS}
    if args.stage_b:
        stage_b(args, holo, resseq)
        return
    mdr = {("md", r): resdist("md", r, args.skip) for r in md_reps
           if os.path.exists(run_src(MD, r))}
    tag = ""
    if args.t_max is not None:
        tag = f"_t{args.t_max:g}ns"
        mdr = {k: (t[t <= args.t_max + 1e-6], d[t <= args.t_max + 1e-6]) for k, (t, d) in mdr.items()}
        args.q_version = f"{args.q_version}{tag}"
    for k, (t, d) in list(holo.items()) + list(mdr.items()):
        assert np.isfinite(d).all(), f"non-finite distances in {k}"
        print(f"  {k[0]:4s} {str(k[1]):8s} {len(t):5d} frames, {t[0]:.1f}-{t[-1]:.1f} ns")

    if args.mm_compare:
        mm_compare(args, holo, mdr, resseq)
        return
    if args.project_metad:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="metad_gate_") as tmp:
            args.scratch = args.scratch or tmp
            project_metad(args, holo, mdr, resseq)
        return

    # ---- native Q (identity register is also used to colour the PCA)
    native, freq, qres = native_q(holo, mdr, resseq, args.native_occ, args.equil)
    qfig = figpath(f"{MD}_qnative_{args.q_version}.png", root=ROOT)
    if args.redo_q or not os.path.exists(qfig):
        rows, vs, mat = plot_q(qres, native, args.q_version)
        hq = np.concatenate([q[(0, False)][t >= args.equil] for k, (t, q) in qres.items()
                             if k[0] == "holo"])
        qpath = os.path.join(OUTDIR, f"{MD}_qnative{tag}.dat")
        with open(qpath, "w") as fh:
            fh.write(f"# native core(27-93)-DNA Q: {len(native)} native residue x nucleotide pairs "
                     f"(holo occupancy >= {args.native_occ}, t >= {args.equil} ns, d < "
                     f"{CONTACT_CUTOFF} nm)\n# holo Q_native median {np.median(hq):.3f} "
                     f"[p5 {np.percentile(hq, 5):.3f}, p95 {np.percentile(hq, 95):.3f}]\n"
                     "# columns: last-10-ns mean Q per register variant (sequence NOT palindromic: "
                     "a shifted/flipped register is the same geometry on different bases)\n"
                     + (f"# runs cut at t <= {args.t_max:g} ns (--t-max): last 10 ns = "
                        f"{args.t_max - 10:g}-{args.t_max:g} ns\n" if tag else ""))
            fh.write("# run  " + "  ".join(vlabel(*v).replace(" ", "_") for v in vs) + "\n")
            for k, m in zip(rows, mat):
                fh.write(f"{k[1]:8s} " + " ".join(f"{x:6.3f}" for x in m) + "\n")
        print(open(qpath).read())
    else:
        print(f"  native Q: {len(native)} pairs; kept existing {qfig} (--redo-q to rewrite)")
    if tag:      # --t-max produces only the tagged register-Q table: no PCA / dissection outputs
        return

    if args.dissect or args.sliding:
        hold, frac = args.stable_hold, args.stable_frac
        if args.sliding:
            specs = [(s.split(":")[0], float(s.split(":")[1])) for s in args.sliding]
            plot_sliding(specs, qres, mdr, native, resname, hold, frac, args.sliding_version)
        if not args.dissect:
            return
        # linker (21-26) pairs that are holo-native by the same occupancy cut (raster only)
        rs = np.asarray(resseq)
        linker_nat = np.argwhere((freq >= args.native_occ) & ((rs >= 21) & (rs <= 26))[:, None])
        pz = args.pca_npz or os.path.join(OUTDIR, f"{MD}_contact_proj_{args.version}.npz")
        pcz = dict(np.load(pz))
        print(f"  PCA from {pz}: {len(pcz['keep'])} features, EVR {np.round(pcz['pca_evr'], 3)}")
        b, e = args.window
        runs = {}
        for rep in args.dissect:
            D = dissect(rep, b, e, args.dissect_skip, native, linker_nat, resname, pcz,
                        args.binary, hold, frac)
            # validation against the 100 ps cache at common time stamps (match by TIME)
            tc, dc = mdr[("md", rep)]
            ic = nearest(D["t"], tc[(tc >= b) & (tc <= e)])
            tcm = tc[(tc >= b) & (tc <= e)]
            assert np.abs(D["t"][ic] - tcm).max() < 1e-4, "100 ps frames missing from window"
            dcm = dc[(tc >= b) & (tc <= e)]
            qc = qres[("md", rep)][1][(0, False)][(tc >= b) & (tc <= e)]
            pnpz = pcz.get(f"PC_md_{rep}")
            print(f"  {rep}: window {len(D['t'])} frames {D['t'][0]:.2f}-{D['t'][-1]:.2f} ns "
                  f"(dt {D['dt'] * 1000:.1f} ps); vs 100 ps cache at {len(ic)} common times: "
                  f"max |d diff| {np.abs(D['d'][ic] - dcm).max():.2e} nm, "
                  f"max |Q diff| {np.abs(D['q'][(0, False)][ic] - qc).max():.3f}"
                  + ("" if pnpz is None else
                     f", max |PC1 diff| vs npz "
                     f"{np.abs(D['pc'][ic, 0] - pnpz[(pnpz[:, 0] >= b) & (pnpz[:, 0] <= e), 1]).max():.2e}"))
            runs[rep] = D
        ylims = dict(a1=max(D["regions"][n].max() for D in runs.values() for n in ("IDR", "linker")) + 2,
                     a2=max(D["regions"][n].max() for D in runs.values()
                            for n in ("H1", "loop H1-H2", "H2", "loop H2-H3", "H3")) + 2,
                     n_core=max(D["col"]["n_core"].max() for D in runs.values()) * 1.05,
                     fp=max(D["fp"].max() for D in runs.values()))
        bins = np.array([0, 2, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25], float)
        bins = np.unique(np.clip(bins, b, e))
        for rep, D in runs.items():
            report_dissect(D, bins)
            plot_dissect(D, figpath(f"{MD}_{rep}_rebinding_{args.dissect_version}.png", root=ROOT),
                         args.smooth, hold, frac, ylims)
        return

    # ---- PCA (+ opt-in tICA)
    allruns = list(holo.items()) + list(mdr.items())
    feats = {k: switch(d, args.binary) for k, (t, d) in allruns}
    times = {k: v[0] for k, v in allruns}
    if args.pca_from:
        from types import SimpleNamespace
        pcz = dict(np.load(args.pca_from))
        print(f"  PCA from {args.pca_from}: {len(pcz['keep'])} features, "
              f"EVR {np.round(pcz['pca_evr'], 3)} (no refit)")
        verify_pca(pcz, holo, mdr, args.min_feature)
        keep = pcz["keep"]
        args.fit, args.tmin = str(pcz["fit"]), float(pcz["tmin"])
        nfit = (int(pcz["n_fit_holo"]), int(pcz["n_fit_mstate"]))
        new = np.setdiff1d(np.flatnonzero(np.max([f.max(axis=0) for f in feats.values()], axis=0)
                                          >= args.min_feature), keep)
        print(f"  {len(new)} features reach s >= {args.min_feature} only in frames after the fit "
              f"(not in the basis, zero loading)")
        pca = SimpleNamespace(components_=pcz["pca_load"], explained_variance_ratio_=pcz["pca_evr"],
                              mean_=pcz["pca_mean"],
                              transform=lambda x: (x - pcz["pca_mean"]) @ pcz["pca_load"].T)
        tfit = {g: [pcz[f"PC_{k[0]}_{k[1]}"][-1, 0] for k in feats if k[0] == g] for g in ("holo", "md")}
        span = {g: f"{len(v)} x {max(v):.0f} ns" for g, v in tfit.items()}
        fitdesc = (f"pooled holo + m-state (basis {os.path.basename(args.pca_from)}, "
                   f"frames after the fit projected)")
    else:
        keep = np.flatnonzero(np.max([f.max(axis=0) for f in feats.values()], axis=0)
                              >= args.min_feature)
        fitkeys = list(mdr) if args.fit == "mstate" else list(holo) + list(mdr)
        fit = {k: feats[k][(times[k] >= args.tmin) if k[0] == "md" else slice(None)][:, keep]
               for k in fitkeys}
        nfit = (sum(len(fit[k]) for k in fit if k[0] == "holo"),
                sum(len(fit[k]) for k in fit if k[0] == "md"))
        X = np.concatenate(list(fit.values()))
        assert np.isfinite(X).all()
        print(f"  {len(keep)} of {feats[next(iter(feats))].shape[1]} contact features kept; "
              f"fit ({args.fit}) on {len(X)} frames: holo {nfit[0]}, m-state {nfit[1]} "
              f"(m-state t >= {args.tmin} ns), unweighted")

        from sklearn.decomposition import PCA
        pca = PCA(n_components=4).fit(X)
        span = None
        fitdesc = "pooled holo + m-state" if args.fit == "all" else "m-state only (holo projected)"
    frac = region_fractions(pca.components_.T, keep, resseq)
    P = dict(load=pca.components_.T, ev=pca.explained_variance_ratio_, frac=frac, keep=keep,
             fitdesc=fitdesc, span=span, runs={})
    for k, f in feats.items():
        P["runs"][k] = (times[k], pca.transform(f[:, keep]), bend_at(k[0], k[1], times[k]),
                        qres[k][1][(0, False)])
    print(f"  PCA explained variance: {np.round(pca.explained_variance_ratio_, 3)}")
    print("  squared-loading share   IDR(1-20) linker(21-26) core(27-93)")
    for j, f in enumerate(frac):
        print(f"    PC{j + 1}                  {f[0]:.3f}     {f[1]:.3f}       {f[2]:.3f}")

    # placement of every run relative to the holo cloud (holo t >= --equil as reference)
    href = np.concatenate([y[t >= args.equil] for k, (t, y, _, _) in P["runs"].items()
                           if k[0] == "holo"])
    mu, sd = href.mean(axis=0), href.std(axis=0)
    lo, hi = np.percentile(href[:, 0], [5, 95])
    print(f"  holo (t >= {args.equil} ns) PC mean {np.round(mu, 2)}, sd {np.round(sd, 2)}; "
          f"PC1 p5-p95 [{lo:.2f}, {hi:.2f}]")
    print(f"  {'run':14s} {'PC1-4, last-10-ns mean':>30s}   {'z vs holo':>24s}  "
          f"{'f(PC1 in holo p5-p95)':>10s}  early(t<{args.t_detached:g}) PC1-2   Q_end bend_end")
    for k, (t, y, bnd, q) in P["runs"].items():
        e = t >= t[-1] - 10
        last = y[e].mean(axis=0)
        early = y[t < args.t_detached, :2].mean(axis=0)
        inh = ((y[e, 0] >= lo) & (y[e, 0] <= hi)).mean()
        print(f"  {k[0]:4s} {str(k[1]):9s} {np.array2string(last, precision=2):>30s}   "
              f"{np.array2string((last - mu) / sd, precision=1):>24s}  {inh:10.2f}  "
              f"{np.array2string(early, precision=2):>16s}   {q[e].mean():.2f}  "
              f"{np.median(bnd[e]):5.1f}")
    allp = np.concatenate([v[1] for v in P["runs"].values()])
    allb = np.concatenate([v[2] for v in P["runs"].values()])
    allq = np.concatenate([v[3] for v in P["runs"].values()])
    for j in range(4):
        print(f"  PC{j + 1}: Pearson r with Q_native {np.corrcoef(allp[:, j], allq)[0, 1]:+.2f}, "
              f"with bend {np.corrcoef(allp[:, j], allb)[0, 1]:+.2f} (all frames)")
    refs = bend_refs(args.equil)
    for n, (lo, mid, hi) in refs.items():
        print(f"  {n} DNA bend (colvar, t >= {args.equil} ns): median {mid:.1f}, p5-p95 [{lo:.1f}, {hi:.1f}] deg")
    plot_pca(P, ntlab, nfit, args.t_detached, args.version, refs)

    extra = {}
    if args.tica:
        dt = np.median(np.diff(next(iter(times.values()))))
        lagf = max(1, int(round(args.lag / dt)))
        tmean, tvec, tlam = tica(list(fit.values()), lagf)
        its = -args.lag / np.log(np.clip(tlam, 1e-12, 1 - 1e-12))
        T = dict(load=tvec, ev=[f"t$_i$ {x:.1f} ns" for x in its], keep=keep,
                 runs={k: (times[k], (f[:, keep] - tmean) @ tvec) + P["runs"][k][2:]
                       for k, f in feats.items()})
        plot_tica(T, ntlab, args.version)
        print(f"  tICA implied timescales (ns, lag {args.lag}): {np.round(its, 1)}")
        extra = dict(tica_load=tvec.T, tica_eig=tlam, tica_lag_ns=args.lag,
                     **{f"tIC_{k[0]}_{k[1]}": np.column_stack([v[0], v[1]])
                        for k, v in T["runs"].items()})

    out = os.path.join(OUTDIR, f"{MD}_contact_proj_{args.version}.npz")
    np.savez_compressed(out, keep=keep, resseq=np.array(resseq), nt=np.array(ntlab),
                        fit=args.fit, pca_from=str(args.pca_from or ""), n_fit_holo=nfit[0], n_fit_mstate=nfit[1], tmin=args.tmin,
                        pca_load=pca.components_, pca_evr=pca.explained_variance_ratio_,
                        pca_mean=pca.mean_, region_frac=frac,
                        regions=np.array([f"{n} {a}-{b}" for n, a, b in REGIONS]),
                        proj_columns=np.array(["time_ns", "PC1", "PC2", "PC3", "PC4", "bend_deg",
                                               "Q_native"]),
                        **{f"PC_{k[0]}_{k[1]}": np.column_stack([v[0], v[1], v[2], v[3]])
                           for k, v in P["runs"].items()}, **extra)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
