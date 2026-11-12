"""
Reweighted per-residue Calpha RMSF of the apo 1Nhp6A HMG box from the
wall-restrained well-tempered metadynamics run (data/1Nhp6A_apo_large).

The bias acts on (q_core, n_tail_core) with an UPPER_WALLS on core RMSD at
0.5 nm, so the run samples ONLY the folded basin.  The RMSF reported here is
therefore a *folded-state* rigidity measure -- how much each part of the fold
fluctuates about the native structure -- NOT a measure that includes global
unfolding.

Weights are the rigorous Tiwary-Parrinello c(t) estimator (reweight_metad),
using the SAME grid, kBT and 5-ns transient discard as article_data.py, so the
weighted ensemble is identical to the one behind the paper's apo panels.

RMSF_a = sqrt( sum_t w_t |x_a(t) - <x_a>_w|^2 / sum_t w_t ),  after superposing
every frame on the rigid core Calpha (resSeq 21-93) and iterating the reference
once to the weighted-mean structure.

Outputs (figures/1Nhp6A_apo/ + a .dat table), version set by --fig-version:
  figures/1Nhp6A_apo/1Nhp6A_apo_rmsf_ca_reweighted_v<N>.png
  figures/1Nhp6A_apo/1Nhp6A_apo_rmsf_ca_reweighted_seq_v<N>.png
  figures/1Nhp6A_apo/1Nhp6A_apo_rmsf_ca_reweighted_v<N>.dat

The frame source is --traj (see reweight_qbh.py, same convention): the
per-replica *_proc.xtc go stale whenever the run is extended, while
analysis/pbc/solute_r{r}.xtc are rebuilt from the .trr and carry the current
length.  At 1250 ns:

  apo_rmsf.py --traj 'analysis/pbc/solute_r{r}.xtc' --fig-version 2

--equil mode: apo vs DNA-bound (1:1) Calpha RMSF from the UNBIASED equilibrium MD
campaign (data/equil/{1Nhp6A_apo_large,1Nhp6A_holo}/rep{0,1,2}, 250 ns
each, no weights).  Frames = the cached pbc-whole solute copies (100 ps stride,
equil_monitor.load_whole), first EQ_T0 ns discarded, superposed on the core
Calpha (21-93).  Per replica, RMSF is averaged over EQ_WIN-ns
windows (so slow drift does not inflate it); the full-run RMSF is tabulated too.
Error = std over the 3 independent replicas.

  apo_rmsf.py --equil [--plot-only] --fig-version 1
    -> figures/1Nhp6A_equil_rmsf_apo_vs_holo_v<N>.{png,dat}   (apo vs holo: root figures/)

--ensemble N mode: N structures that carry the spread the panel-C medoids hide
(MetaD by systematic resampling on the same c(t) weights; equil uniformly),
superposed on the basin medoid over Calpha 21-88:

  apo_rmsf.py --ensemble 20 --traj 'analysis/pbc/solute_r{r}.xtc'
    -> analysis/snapshots/1Nhp6A_apo_large_ensemble/
       1Nhp6A_apo_large_ensemble_{metad,equil}_n<N>.pdb + _frames.dat
"""
import argparse
import os
import sys
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reweight_metad as rwm
import fes_marginal_meanstd as fm
from figpaths import figpath

# ---- constants copied from article_data.py (keep the ensemble identical) -----
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APO_DIR = f"{ROOT}/data/1Nhp6A_apo_large"
NWALK = 6
Q_MIN, Q_MAX, Q_BIN = 0.3, 1.0, 100
NTC_MIN, NTC_MAX, NTC_BIN = 0, 150, 300
KBT = 2.494339          # kJ/mol, PLUMED default used for sum_hills
TEQ_PS = 5.0e3          # discard the 5-ns bias-filling transient
TOP = f"{APO_DIR}/1Nhp6A_parambsc1_tip3p_ions0.15_meta_2D_proc.gro"

# ---- fold segments (resSeq, 1-indexed; helices are the plumed-monitored ranges)
SEGMENTS = [
    ("IDR",        1, 20),
    ("loop 0-H1", 21, 26),
    ("H1",        27, 42),
    ("loop H1-H2",43, 47),
    ("H2",        48, 61),
    ("loop H2-H3",62, 63),
    ("H3",        64, 93),   # 1LWM project canon (CLAUDE.md)
]
CORE_LO, CORE_HI = 21, 93         # rigid core used for superposition


def load_cv_at_traj_times(colvar, traj_time_ps):
    """Return (q_core, n_tail_core) sampled at exactly the trajectory frame times.

    The run has a few restart overlaps, so colvar row-index != time/dt.  We map
    rounded-time -> LAST occurrence (post-restart value) and look up each traj
    time; every traj time is present (verified: 5001 unique 200-ps multiples)."""
    raw = np.loadtxt(colvar, comments=["#", "!"], usecols=(0, 1, 2))
    key = np.round(raw[:, 0], 1)
    lut = {}
    for k, q, n in zip(key, raw[:, 1], raw[:, 2]):
        lut[k] = (q, n)                      # later rows overwrite -> post-restart
    q = np.empty(traj_time_ps.size)
    n = np.empty(traj_time_ps.size)
    for i, t in enumerate(np.round(traj_time_ps, 1)):
        q[i], n[i] = lut[t]
    return q, n


def weighted_rmsf(xyz, w, super_idx):
    """Per-atom RMSF with per-frame weights w.  xyz: (F, A, 3) already superposed.
    Iterate the reference once: rebuild mean on weighted coords, re-superpose."""
    W = w / w.sum()
    mu = np.tensordot(W, xyz, axes=(0, 0))               # (A,3) weighted mean
    d2 = ((xyz - mu) ** 2).sum(axis=2)                   # (F,A)
    return np.sqrt(np.tensordot(W, d2, axes=(0, 0)))     # (A,)


DEFAULT_TRAJ = (APO_DIR + "/replica_{r}/"
                "1Nhp6A_parambsc1_tip3p_ions0.15_meta_2D_r{r}_proc.xtc")


def load_weighted(trajfmt=DEFAULT_TRAJ):
    """All walkers' frames + their c(t) weights (transient = weight 0).
    Returns traj, w, walker id, time (ps), q_core, n_tail_core, ESS, n_used."""
    # ---------- weights (c(t)) over all walkers, matched to traj frames --------
    hills = [fm.load_hills(f"{APO_DIR}/hills.dat.{r}") for r in range(NWALK)]
    t_h, q0, n0, hgt, sig_q, sig_n, gamma = rwm.pool_hills(hills)
    print(f"[weights] {t_h.size:,} pooled hills | sigma=({sig_q},{sig_n}) gamma={gamma:.0f}")
    q_grid = np.linspace(Q_MIN, Q_MAX, Q_BIN)
    n_grid = np.linspace(NTC_MIN, NTC_MAX, NTC_BIN)

    trajs, ftime, fq, fn, fwid = [], [], [], [], []
    for r in range(NWALK):
        xtc = trajfmt.format(r=r)
        if not os.path.isabs(xtc):                 # allow repo-relative templates
            xtc = os.path.join(ROOT, xtc)
        tr = md.load(xtc, top=TOP)
        q, n = load_cv_at_traj_times(f"{APO_DIR}/replica_{r}/colvar_r{r}.dat", tr.time)
        trajs.append(tr); ftime.append(tr.time); fq.append(q); fn.append(n)
        fwid.append(np.full(tr.n_frames, r))
        print(f"[traj] r{r}: {tr.n_frames} frames")
    traj = trajs[0].join(trajs[1:])
    frame_t = np.concatenate(ftime)
    frame_q = np.concatenate(fq)
    frame_n = np.concatenate(fn)
    wid = np.concatenate(fwid)

    print(f"[weights] reconstructing V(s,t), c(t) over {frame_t.size:,} frames ...", flush=True)
    ct = rwm.metad_ct_weights(t_h, q0, n0, hgt, sig_q, sig_n,
                              frame_t, frame_q, frame_n, q_grid, n_grid,
                              gamma=gamma, kBT=KBT, checkpoint_dt=1.0, batch_dt=1.0)
    w = ct["weights"].copy()
    w[frame_t < TEQ_PS] = 0.0                            # drop bias-filling transient
    used = int((frame_t >= TEQ_PS).sum())
    ess = w.sum() ** 2 / np.sum(w ** 2)
    print(f"[weights] c(t): {ct['ct'][0]:.1f} -> {ct['ct'][-1]:.1f} kJ/mol | "
          f"ESS = {ess:,.0f} ({100*ess/used:.2f}% of {used:,} frames)")
    return traj, w, wid, frame_t, frame_q, frame_n, ess, used


def main(trajfmt=DEFAULT_TRAJ):
    traj, w, wid, _, _, _, ess, used = load_weighted(trajfmt)

    # ---------- superpose on rigid core Calpha, then weighted RMSF -------------
    top = traj.top
    ca = top.select("name CA and protein")
    ca_res = np.array([top.atom(i).residue.resSeq for i in ca])
    ca_resname = [top.atom(i).residue.name for i in ca]
    core_ca = ca[(ca_res >= CORE_LO) & (ca_res <= CORE_HI)]

    keep = w > 0
    traj = traj[keep]; w = w[keep]; wid_k = wid[keep]
    traj.superpose(traj, frame=0, atom_indices=core_ca)          # pass 1
    W = w / w.sum()
    mu = np.tensordot(W, traj.xyz, axes=(0, 0))
    ref = md.Trajectory(mu[None], top)
    traj.superpose(ref, frame=0, atom_indices=core_ca)           # pass 2: to weighted mean
    mu = np.tensordot(W, traj.xyz, axes=(0, 0))                  # mean in the new frame

    # pooled reweighted RMSF about the global weighted-mean structure
    d2 = ((traj.xyz[:, ca] - mu[ca]) ** 2).sum(axis=2)           # (F, nCA)
    rmsf_ca = np.sqrt(np.tensordot(W, d2, axes=(0, 0))) * 10.0    # Angstrom

    # per-walker block RMSF (same reference frame + global mean) -> sampling spread.
    # The 6 walkers are independent trajectories, so their scatter is the error bar;
    # given the low ESS this is the honest uncertainty (each walker ~1/6 of the ESS).
    per_walker = []
    for r in range(NWALK):
        m = wid_k == r
        if not m.any() or w[m].sum() == 0:
            continue
        Wr = w[m] / w[m].sum()
        per_walker.append(np.sqrt(np.tensordot(Wr, d2[m], axes=(0, 0))) * 10.0)
    per_walker = np.array(per_walker)
    rmsf_err = per_walker.std(axis=0, ddof=1)                    # +/- band (n=6 walkers)

    # raw (uniform weight) over the SAME post-transient frames, for comparison
    rmsf_ca_raw = np.sqrt(d2.mean(axis=0)) * 10.0

    save_dat(ca_res, ca_resname, rmsf_ca, rmsf_err, rmsf_ca_raw, ess, used)
    make_figures(ca_res, ca_resname, rmsf_ca, rmsf_err, rmsf_ca_raw)


# ─────────────────────────────────────────────────────────── output / plotting
# set from --fig-version; the run length changes with every extension, so the
# figure/table names must be versionable rather than pinned to _v1.
FIGV = "1"
DAT = figpath("1Nhp6A_apo_rmsf_ca_reweighted_v1.dat", makedirs=False)   # root-relative

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "HIE": "H", "HID": "H", "HIP": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# article notebook style (notebooks/article/article_figures.ipynb): typography, helix palette, labels
ARTICLE_RC = {"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 10,
           "ytick.labelsize": 10, "legend.fontsize": 10, "axes.linewidth": 0.8,
           "figure.dpi": 150, "axes.grid": False, "pdf.fonttype": 42}
HELIX = {"H1": "#4C72B0", "H2": "#C44E52", "H3": "#55A868"}   # article COLOR dict
HLABEL = {"H1": "H1 (27–42)", "H2": "H2 (48–61)", "H3": "H3 (64–93)"}


def save_dat(ca_res, ca_resname, rmsf_ca, rmsf_err, rmsf_raw, ess, used):
    os.makedirs(os.path.dirname(f"{ROOT}/{DAT}"), exist_ok=True)
    print("\n[segment-averaged Calpha RMSF (Angstrom)]")
    print(f"  {'segment':<11} {'resSeq':>9} {'reweighted':>13} {'raw':>7}")
    for name, lo, hi in SEGMENTS:
        m = (ca_res >= lo) & (ca_res <= hi)
        if not m.any():
            continue
        print(f"  {name:<11} {lo:>3}-{hi:<3} "
              f"{rmsf_ca[m].mean():>6.2f} +/- {rmsf_err[m].mean():<4.2f} {rmsf_raw[m].mean():>7.2f}")
    with open(f"{ROOT}/{DAT}", "w") as fh:
        fh.write("# apo 1Nhp6A folded-state Calpha RMSF (Angstrom), c(t)-reweighted "
                 "wall-MetaD (data/1Nhp6A_apo_large)\n")
        fh.write(f"# ESS={ess:.0f} of {used} frames; superposed on core Calpha "
                 f"resSeq {CORE_LO}-{CORE_HI}; 5-ns transient discarded\n")
        fh.write("# err = std over the 6 independent walkers (per-walker block spread)\n")
        fh.write("# resSeq  resname  rmsf_reweighted  rmsf_err  rmsf_raw\n")
        for k in range(ca_res.size):
            fh.write(f"{ca_res[k]:4d}  {ca_resname[k]:<4s}  "
                     f"{rmsf_ca[k]:8.3f}  {rmsf_err[k]:8.3f}  {rmsf_raw[k]:8.3f}\n")
    print(f"[saved] {ROOT}/{DAT}")


def _shade(ax, top_labels=True):
    """Helix + IDR shading shared by both figure variants (article palette)."""
    ax.axvspan(0.5, 20.5, color="0.5", alpha=0.10, lw=0)
    for name, lo, hi in SEGMENTS:
        if name in HELIX:
            ax.axvspan(lo - 0.5, hi + 0.5, color=HELIX[name], alpha=0.13, lw=0)
            if top_labels:
                ax.text((lo + hi) / 2, 0.97, HLABEL[name], ha="center", va="top",
                        transform=ax.get_xaxis_transform(), fontsize=9, color=HELIX[name])
    if top_labels:
        ax.text(10.5, 0.97, "IDR", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9, color="0.45")


def make_figures(ca_res, ca_resname, rmsf_ca, rmsf_err, rmsf_raw):

    # ---- variant 1: compact panel, residue-number x axis (article style) --------
    with plt.rc_context(ARTICLE_RC):
        fig, ax = plt.subplots(figsize=(6.6, 2.8))
        _shade(ax)
        ax.plot(ca_res, rmsf_raw, color="0.6", lw=1.0, ls=(0, (4, 2)),
                label="raw (unweighted)", zorder=2)
        ax.fill_between(ca_res, rmsf_ca - rmsf_err, rmsf_ca + rmsf_err,
                        color="0.15", alpha=0.22, lw=0, zorder=3)
        ax.plot(ca_res, rmsf_ca, color="0.15", lw=2, label="c(t)-reweighted", zorder=4)
        ax.set_xlabel("residue")
        ax.set_ylabel(r"C$\alpha$ RMSF ($\mathrm{\AA}$)")
        ax.set_xlim(0.5, ca_res.max() + 0.5)
        ax.set_ylim(0, None)
        ax.legend(frameon=False, loc="center right")
        fig.tight_layout()
        p1 = figpath(f"1Nhp6A_apo_rmsf_ca_reweighted_v{FIGV}.png", root=ROOT)
        fig.savefig(p1, dpi=300); plt.close(fig)
    print(f"[saved] {p1}")

    # ---- variant 2: very stretched, one-letter sequence as x tick labels ------
    with plt.rc_context(ARTICLE_RC):
        fig, ax = plt.subplots(figsize=(0.20 * ca_res.size, 2.9))
        _shade(ax, top_labels=True)
        ax.fill_between(ca_res, rmsf_ca - rmsf_err, rmsf_ca + rmsf_err,
                        color="0.15", alpha=0.22, lw=0, zorder=3)
        ax.plot(ca_res, rmsf_raw, color="0.6", lw=1.0, ls=(0, (4, 2)),
                label="raw (unweighted)", zorder=2)
        ax.plot(ca_res, rmsf_ca, color="0.15", lw=2, label="c(t)-reweighted", zorder=4)
        seq = [THREE2ONE.get(n, "X") for n in ca_resname]
        ax.set_xticks(ca_res)
        ax.set_xticklabels(seq, fontsize=6, family="monospace")
        ax.tick_params(axis="x", length=2, pad=2)
        ax.set_xlabel("sequence")
        ax.set_ylabel(r"C$\alpha$ RMSF ($\mathrm{\AA}$)")
        ax.set_xlim(0.5, ca_res.max() + 0.5)
        ax.set_ylim(0, None)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        p2 = figpath(f"1Nhp6A_apo_rmsf_ca_reweighted_seq_v{FIGV}.png", root=ROOT)
        fig.savefig(p2, dpi=300); plt.close(fig)
    print(f"[saved] {p2}")


def plot_only():
    d = np.genfromtxt(f"{ROOT}/{DAT}", dtype=None, encoding=None,
                      names=["resSeq", "resname", "rw", "err", "raw"])
    make_figures(d["resSeq"].astype(int), list(d["resname"]),
                 d["rw"], d["err"], d["raw"])


# ───────────────────────────────────── --equil: apo vs holo from unbiased MD
EQ_T0 = 10.0       # ns discarded from each 250-ns replica
EQ_WIN = 25.0      # ns window for the drift-free (windowed) RMSF
EQ_SETS = [        # (key, equil system, label, colour)
    ("apo",   "1Nhp6A_apo_large", "apo",           "0.15"),
    ("holo",  "1Nhp6A_holo",      "1:1 Nhp6A:DNA", "#0072B2"),
]
EQ_DAT = figpath("1Nhp6A_equil_rmsf_apo_vs_holo_v1.dat", makedirs=False)


def _rmsf_A(xyz):
    """Per-atom RMSF (Angstrom) about the mean of already-superposed coords (nm)."""
    return np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(2).mean(0)) * 10.0


def equil_main():
    from equil_monitor import load_whole
    res, names, full, win = None, None, {}, {}
    for key, system, _, _ in EQ_SETS:
        full[key], win[key] = [], []
        for rep in range(3):
            t = load_whole(system, rep, 10, root=ROOT)          # .time in ns
            t = t[t.time >= EQ_T0]
            ca = t.top.select("name CA and protein")
            rs = np.array([t.top.atom(i).residue.resSeq for i in ca])
            sub = t.atom_slice(ca)
            core = np.where((rs >= CORE_LO) & (rs <= CORE_HI))[0]
            sub.superpose(sub, 0, atom_indices=core)              # pass 1
            ref = sub[0]; ref.xyz[0] = sub.xyz.mean(0)
            sub.superpose(ref, 0, atom_indices=core)              # pass 2: to mean
            full[key].append(_rmsf_A(sub.xyz))
            w = []
            for a in np.arange(EQ_T0, sub.time[-1], EQ_WIN):
                m = (sub.time >= a) & (sub.time < a + EQ_WIN)
                if m.sum() < 0.8 * EQ_WIN / 0.1:                  # drop a short tail window
                    continue
                s = sub[m]; s.superpose(s, 0, atom_indices=core)
                w.append(_rmsf_A(s.xyz))
            win[key].append(np.mean(w, axis=0))
            print(f"[equil] {key} rep{rep}: {sub.n_frames} frames, {len(w)} windows, "
                  f"core {win[key][-1][core].mean():.2f} A (windowed)", flush=True)
            if res is None:
                res = rs
                names = [sub.top.atom(i).residue.name for i in range(sub.n_atoms)]
            elif not np.array_equal(res, rs):
                sys.exit(f"residue numbering differs for {key} rep{rep}")
    save_equil_dat(res, names, {k: np.array(v) for k, v in win.items()},
                   {k: np.array(v) for k, v in full.items()})
    make_equil_figure(res, {k: (np.mean(v, 0), np.std(v, 0, ddof=1)) for k, v in win.items()})


# ─────────────────────────────────────────── --ensemble: random structural draws
# The panel-C medoids (basin + shoulder) are, by construction, the most central
# frames of their regions, so they show NO fluctuation where the RMSF is largest
# (the H3 C-terminus, res 89-93).  The ensemble draw shows that spread with the
# correct populations: MetaD frames by systematic resampling on the c(t) weight,
# equil frames drawn uniformly.  Every model is superposed on the basin medoid over the
# core MINUS the H3 C-tip, so the tip motion is shown rather than fitted away.
ENS_REF = "analysis/article/state_1_medoid.pdb"   # panel-C basin medoid
ENS_FIT = (21, 88)
ENS_DIR = "analysis/snapshots/1Nhp6A_apo_large_ensemble"
ENS_SEED = 20261103


def _write_ensemble(traj, meta, header, tag):
    ref = md.load(os.path.join(ROOT, ENS_REF))
    ca = ref.top.select("name CA")
    rs = np.array([ref.top.atom(i).residue.resSeq for i in ca])
    fit = ca[(rs >= ENS_FIT[0]) & (rs <= ENS_FIT[1])]
    traj.superpose(ref, 0, atom_indices=fit)
    os.makedirs(os.path.join(ROOT, ENS_DIR), exist_ok=True)
    base = os.path.join(ROOT, ENS_DIR, f"1Nhp6A_apo_large_ensemble_{tag}_n{traj.n_frames}")
    traj.save_pdb(base + ".pdb")                       # one MODEL per draw
    np.savetxt(base + "_frames.dat", meta, fmt="%10.4g", header=header)
    tip = (rs >= 89) & (rs <= 93)
    rmsf = np.sqrt(((traj.xyz[:, ca] - traj.xyz[:, ca].mean(0)) ** 2).sum(2).mean(0)) * 10
    print(f"[ensemble] {tag}: {traj.n_frames} models -> {base}.pdb\n"
          f"           Calpha RMSF of the draw, res 89-93: "
          + " ".join(f"{x:.2f}" for x in rmsf[tip]) + " A")


def ensemble_main(n, trajfmt=DEFAULT_TRAJ, sources=("metad", "equil")):
    rng = np.random.default_rng(ENS_SEED)
    fit_txt = f"superposed on {ENS_REF} Calpha {ENS_FIT[0]}-{ENS_FIT[1]}; seed {ENS_SEED}"
    if "metad" in sources:
        traj, w, wid, ft, fq, fn, ess, used = load_weighted(trajfmt)
        W = w / w.sum()
        # systematic resampling: n evenly spaced points (one random offset) on the
        # cumulative weight.  Unbiased like a multinomial draw, but the weight is
        # concentrated in two walkers (w4 53 %, w0 35 %), and a plain 20-frame draw
        # swings by +/-2 frames per walker and per q_core range; this one cannot.
        # No frame has W > 1/n, so no frame is drawn twice.
        u = (rng.random() + np.arange(n)) / n
        pick = np.minimum(np.searchsorted(np.cumsum(W), u), W.size - 1)
        print(f"[ensemble] metad: max single-frame weight {W.max():.4f}, "
              f"drawn weight sum {W[pick].sum():.4f}, ESS {ess:.0f}")
        _write_ensemble(traj[pick],
                        np.column_stack([wid[pick], ft[pick], fq[pick], fn[pick], W[pick]]),
                        f"apo MetaD, systematic resampling on c(t) weight (ESS {ess:.0f} of "
                        f"{used} post-{TEQ_PS/1e3:.0f}-ns frames); {fit_txt}\n"
                        "walker  time_ps  q_core  n_tail_core  weight", "metad")
    if "equil" in sources:
        from equil_monitor import load_whole
        trajs, meta = [], []
        for rep in range(3):
            t = load_whole("1Nhp6A_apo_large", rep, 10, root=ROOT)
            t = t[t.time >= EQ_T0]
            trajs.append(t.atom_slice(t.top.select("protein")))
            meta.append(np.column_stack([np.full(t.n_frames, rep), t.time]))
        traj = trajs[0].join(trajs[1:])
        meta = np.concatenate(meta)
        pick = np.sort(rng.choice(traj.n_frames, size=n, replace=False))
        _write_ensemble(traj[pick], meta[pick],
                        f"apo unbiased equil MD, 3 reps, first {EQ_T0:.0f} ns discarded, "
                        f"100-ps frames drawn uniformly from {traj.n_frames}; {fit_txt}\n"
                        "rep  time_ns", "equil")


def save_equil_dat(res, names, win, full):
    print(f"\n[segment-averaged Calpha RMSF (A), {EQ_WIN:.0f}-ns windows; mean +/- std over reps]")
    print(f"  {'segment':<11}" + "".join(f"{k:>15}" for k, *_ in EQ_SETS))
    for name, lo, hi in SEGMENTS + [("core", CORE_LO, CORE_HI)]:
        m = (res >= lo) & (res <= hi)
        row = f"  {name:<11}"
        for k, *_ in EQ_SETS:
            v = win[k][:, m].mean(1)
            row += f"  {v.mean():6.2f} +/- {v.std(ddof=1):4.2f}"
        print(row)
    with open(f"{ROOT}/{EQ_DAT}", "w") as fh:
        fh.write("# Nhp6A Calpha RMSF (Angstrom), unbiased equilibrium MD, 3 reps x 250 ns "
                 "(data/equil/<system>/rep<r>)\n")
        fh.write(f"# first {EQ_T0:.0f} ns discarded; 100-ps frames; superposed on core Calpha "
                 f"resSeq {CORE_LO}-{CORE_HI}\n")
        fh.write(f"# win = mean over {EQ_WIN:.0f}-ns windows, full = whole run; "
                 "mean/std over the 3 replicas\n")
        cols = "".join(f"  {k}_win  {k}_win_sd  {k}_full  {k}_full_sd" for k, *_ in EQ_SETS)
        fh.write(f"# resSeq  resname{cols}\n")
        for i in range(res.size):
            fh.write(f"{res[i]:4d}  {names[i]:<4s}")
            for k, *_ in EQ_SETS:
                fh.write(f"  {win[k][:, i].mean():7.3f}  {win[k][:, i].std(ddof=1):7.3f}"
                         f"  {full[k][:, i].mean():7.3f}  {full[k][:, i].std(ddof=1):7.3f}")
            fh.write("\n")
    print(f"[saved] {ROOT}/{EQ_DAT}")


def make_equil_figure(res, mean_sd):
    """mean_sd: {key: (mean, std)} per residue -> mean line + std band.
    Linear y (user 2026-11-06: no log scale for RMSF)."""
    os.makedirs(f"{ROOT}/figures", exist_ok=True)
    with plt.rc_context(ARTICLE_RC):
        fig, ax = plt.subplots(figsize=(6.6, 2.8))
        _shade(ax)
        for k, _, lab, col in EQ_SETS:
            mu, sd = mean_sd[k]
            ax.fill_between(res, np.clip(mu - sd, 0, None), mu + sd,
                            color=col, alpha=0.2, lw=0)
            ax.plot(res, mu, color=col, lw=1.6, label=lab)
        ax.set_ylim(0, None)
        ax.set_ylabel(r"C$\alpha$ RMSF ($\mathrm{\AA}$)")
        ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.88))
        ax.set_xlabel("residue")
        ax.set_xlim(0.5, res.max() + 0.5)
        fig.tight_layout()
        p = f"{ROOT}/{EQ_DAT[:-4]}.png"
        fig.savefig(p, dpi=300); plt.close(fig)
    print(f"[saved] {p}")


def equil_plot_only():
    d = np.loadtxt(f"{ROOT}/{EQ_DAT}", dtype=str)
    res, vals = d[:, 0].astype(int), d[:, 2:].astype(float)
    make_equil_figure(res, {k: (vals[:, 4 * j], vals[:, 4 * j + 1])     # *_win, *_win_sd
                            for j, (k, *_) in enumerate(EQ_SETS)})


# ───────────────────────────── --holo: reweighted RMSF of a DNA-bound wall MetaD
# Same estimator as the apo run (c(t) weights, 5-ns transient dropped, core-Calpha
# superposition iterated to the weighted mean, per-walker std band), applied to the
# 2D (n_IDR-DNA, de2e) wall runs so the holo curves compare with the apo one 1:1.
# Residues are reported in WT numbering: D2-10 keeps Met1 and new k>=2 is WT k+9.
HOLO = {
    "1Nhp6A_wall": dict(
        label="1:1 Nhp6A:DNA", top="data/1Nhp6A/1Nhp6A_protdna.gro",
        xtc="data/1Nhp6A_wall/replica_{r}/"
            "1Nhp6A_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D_r{r}_proc.xtc",
        to_wt=lambda r: r, color="#0072B2"),
    "1D2_10_wall": dict(
        label="Δ2-10:DNA", top="1D2_10_DNA_parambsc1_tip3p.pdb",
        xtc="data/1D2_10_wall/replica_{r}/"
            "1D2_10_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D_wall_r{r}_proc.xtc",
        to_wt=lambda r: np.where(r == 1, 1, r + 9), color="#882255"),
}
HOLO_N = (0.0, 150.0, 300)       # coord_tail_dna grid (sum_hills range of the wall runs)
HOLO_D = (0.0, 8.0, 200)         # de2e grid (nm)
APO_CMP_DAT = figpath("1Nhp6A_apo_rmsf_ca_reweighted_v2.dat", makedirs=False)  # 1250 ns


def _cv_at_times(colvar, traj_time_ps, cols=(0, 1, 2)):
    """CV values at the trajectory frame times (last occurrence wins, as above)."""
    import pandas as pd
    raw = pd.read_csv(colvar, sep=r"\s+", comment="#", header=None,
                      usecols=list(cols)).to_numpy()
    lut = dict(zip(np.round(raw[:, 0], 1), raw[:, 1:]))
    return np.array([lut[t] for t in np.round(traj_time_ps, 1)])


def holo_dat(name):
    return figpath(f"{name}_rmsf_ca_reweighted_v{FIGV}.dat", makedirs=False)


def holo_main(name):
    cfg = HOLO[name]
    d = f"{ROOT}/data/{name}"
    hills = [fm.load_hills(f"{d}/hills.dat.{r}") for r in range(NWALK)]
    t_h, n0, e0, hgt, sig_n, sig_e, gamma = rwm.pool_hills(hills)
    print(f"[weights] {name}: {t_h.size:,} pooled hills, last {t_h.max()/1e3:.1f} ns | "
          f"sigma=({sig_n},{sig_e}) gamma={gamma:.0f}")
    top = md.load(f"{ROOT}/{cfg['top']}").top
    ca = top.select("name CA and protein")
    res = np.array([top.atom(i).residue.resSeq for i in ca])
    names = [top.atom(i).residue.name for i in ca]
    xyz, ft, fn, fe, wid = [], [], [], [], []
    for r in range(NWALK):
        tr = md.load(f"{ROOT}/{cfg['xtc'].format(r=r)}", top=top, atom_indices=ca)
        cv = _cv_at_times(f"{d}/replica_{r}/colvar_r{r}.dat", tr.time)
        xyz.append(tr.xyz); ft.append(tr.time); fn.append(cv[:, 0]); fe.append(cv[:, 1])
        wid.append(np.full(tr.n_frames, r))
        print(f"[traj] r{r}: {tr.n_frames} frames, to {tr.time[-1]/1e3:.1f} ns", flush=True)
    xyz = np.concatenate(xyz); ft = np.concatenate(ft); wid = np.concatenate(wid)
    bond = np.linalg.norm(np.diff(xyz[0], axis=0), axis=1)        # topology/order sanity
    if not (0.36 < bond.min() and bond.max() < 0.40):
        sys.exit(f"consecutive Calpha distances {bond.min():.3f}-{bond.max():.3f} nm: "
                 "topology does not match the trajectory atom order")
    ct = rwm.metad_ct_weights(t_h, n0, e0, hgt, sig_n, sig_e, ft,
                              np.concatenate(fn), np.concatenate(fe),
                              np.linspace(*HOLO_N), np.linspace(*HOLO_D),
                              gamma=gamma, kBT=KBT, checkpoint_dt=1.0, batch_dt=1.0)
    w = ct["weights"].copy()
    w[ft < TEQ_PS] = 0.0
    used = int((ft >= TEQ_PS).sum())
    ess = w.sum() ** 2 / np.sum(w ** 2)
    print(f"[weights] c(t) {ct['ct'][0]:.1f} -> {ct['ct'][-1]:.1f} kJ/mol | "
          f"ESS = {ess:,.0f} ({100*ess/used:.2f}% of {used:,} frames)")

    keep = w > 0
    xyz, w, wid = xyz[keep], w[keep], wid[keep]
    sub = md.Trajectory(xyz, md.load(f"{ROOT}/{cfg['top']}").atom_slice(ca).top)
    wt = cfg["to_wt"](res)
    core = np.where((wt >= CORE_LO) & (wt <= CORE_HI))[0]
    sub.superpose(sub, 0, atom_indices=core)                      # pass 1
    W = w / w.sum()
    ref = sub[0]; ref.xyz[0] = np.tensordot(W, sub.xyz, axes=(0, 0))
    sub.superpose(ref, 0, atom_indices=core)                      # pass 2: weighted mean
    mu = np.tensordot(W, sub.xyz, axes=(0, 0))
    d2 = ((sub.xyz - mu) ** 2).sum(axis=2)
    rmsf = np.sqrt(np.tensordot(W, d2, axes=(0, 0))) * 10.0
    pw = np.array([np.sqrt(np.tensordot(w[wid == r] / w[wid == r].sum(),
                                        d2[wid == r], axes=(0, 0))) * 10.0
                   for r in range(NWALK) if w[wid == r].sum() > 0])
    err = pw.std(axis=0, ddof=1)
    raw = np.sqrt(d2.mean(axis=0)) * 10.0
    p = f"{ROOT}/{holo_dat(name)}"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(f"# {cfg['label']} Calpha RMSF (Angstrom), {name} wall MetaD, "
                 f"{NWALK} walkers, frames to {ft.max()/1e3:.0f} ns/walker (20 ps)\n")
        fh.write(f"# c(t) weights on (coord_tail_dna, de2e); ESS {ess:.0f} "
                 f"({100*ess/used:.2f}% of {used}); 5-ns transient discarded\n")
        fh.write(f"# superposed on core Calpha WT {CORE_LO}-{CORE_HI}; err = std over "
                 "per-walker RMSF\n# resSeq  resSeq_WT  resname  rmsf_reweighted  "
                 "rmsf_err  rmsf_raw\n")
        for i in range(res.size):
            fh.write(f"{res[i]:4d}  {wt[i]:4d}  {names[i]:<4s}  {rmsf[i]:7.3f}  "
                     f"{err[i]:7.3f}  {raw[i]:7.3f}\n")
    print(f"[saved] {p}")
    for seg, lo, hi in SEGMENTS + [("core", CORE_LO, CORE_HI)]:
        m = (wt >= lo) & (wt <= hi)
        if m.any():
            print(f"  {seg:<11} {rmsf[m].mean():6.2f} A  (raw {raw[m].mean():.2f})")


# ─────────────────── --helix: reweighted per-residue DSSP helix fraction (MetaD)
# Every 2D MetaD with a structural trajectory on rubisco.  Weights = c(t) on the
# run's own two biased CVs, 5-ns transient dropped (identical to the RMSF modes);
# they are cached per system in analysis/cache so a rerun only redoes DSSP.
# 2Nhp6A is the exception: 3D bias (c(t) here is 2D), only 1-ns frames on rubisco,
# and the run fails its reweighting gate -> reported UNWEIGHTED, per copy.
_WALL_GRID = dict(grid=(HOLO_N, HOLO_D), cvcols=(0, 1, 2))
HELIX_SYS = {
    "1Nhp6A_apo_large": dict(label="apo", color="0.55", grid=((Q_MIN, Q_MAX, Q_BIN),
                             (NTC_MIN, NTC_MAX, NTC_BIN)), cvcols=(0, 1, 2),
                             top=TOP, xtc="analysis/pbc/solute_r{r}.xtc",
                             to_wt=lambda r: r),
    "1Nhp6A_wall": dict(HOLO["1Nhp6A_wall"], **_WALL_GRID),
    "1IDR_wall": dict(label="IDR(1-23):DNA", color="#000000",
                      top="data/1IDR_wall/1IDR_DNA_parambsc1_tip3p.pdb",
                      xtc="data/1IDR_wall/replica_{r}/"
                          "1IDR_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D_wall_r{r}_proc.xtc",
                      to_wt=lambda r: r, **_WALL_GRID),
    "1D2_10_wall": dict(HOLO["1D2_10_wall"], **_WALL_GRID),
    **{f"{m}_wall": dict(label=f"{m[1:]}:DNA", color=c,
                         top=f"data/{m}_wall/{m}_proc_ref.pdb",
                         xtc=f"data/{m}_wall/replica_{{r}}/"
                             f"{m}_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D_wall_r{{r}}_proc.xtc",
                         to_wt=lambda r: r, **_WALL_GRID)
       for m, c in [("1S26D", "#CC79A7"), ("1T63D", "#E69F00"), ("1TM", "#56B4E9")]},
    "2Nhp6A": dict(label="2:1 Nhp6A:DNA (unweighted)", color="#009E73",
                   top="data/2Nhp6A/2Nhp6A_DNA_parambsc1_tip3p.gro",
                   xtc="data/2Nhp6A/replica_{r}/"
                       "2Nhp6A_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D_r{r}_vmd.xtc",
                   to_wt=lambda r: r, unweighted=True),
}
HELIX_SEGS = [("IDR", 1, 20), ("H1", 27, 42), ("H2", 48, 61), ("H3", 64, 93)]


def metad_ct_cached(name, cfg, ft, wid):
    """c(t) weights for frames (time ps, walker id), cached on the exact frame set."""
    cache = f"{ROOT}/analysis/cache/{name}_ct_weights_frames.npz"
    if os.path.exists(cache):
        c = np.load(cache)
        if np.array_equal(c["t"], ft) and np.array_equal(c["wid"], wid):
            print(f"[weights] {name}: cached ({cache})")
            return c["w"], float(c["ess"]), int(c["used"])
    d = f"{ROOT}/data/{name}"
    t_h, a0, b0, hgt, sa, sb, gamma = rwm.pool_hills(
        [fm.load_hills(f"{d}/hills.dat.{r}") for r in range(NWALK)])
    print(f"[weights] {name}: {t_h.size:,} hills to {t_h.max()/1e3:.1f} ns, "
          f"sigma=({sa},{sb}) gamma={gamma:.0f}", flush=True)
    cv = np.zeros((ft.size, 2))
    for r in range(NWALK):
        m = wid == r
        cv[m] = _cv_at_times(f"{d}/replica_{r}/colvar_r{r}.dat", ft[m], cfg["cvcols"])
    ga, gb = cfg["grid"]
    ct = rwm.metad_ct_weights(t_h, a0, b0, hgt, sa, sb, ft, cv[:, 0], cv[:, 1],
                              np.linspace(*ga), np.linspace(*gb), gamma=gamma, kBT=KBT,
                              checkpoint_dt=1.0, batch_dt=1.0)
    w = ct["weights"].copy()
    w[ft < TEQ_PS] = 0.0
    used = int((ft >= TEQ_PS).sum())
    ess = w.sum() ** 2 / np.sum(w ** 2)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, t=ft, wid=wid, w=w, ess=ess, used=used)
    print(f"[weights] c(t) {ct['ct'][0]:.1f} -> {ct['ct'][-1]:.1f} kJ/mol | "
          f"ESS = {ess:,.0f} ({100*ess/used:.2f}% of {used:,} frames)")
    return w, ess, used


def helix_dat(name):
    return figpath(f"{name}_helix_dssp_reweighted_v{FIGV}.dat", makedirs=False)


def helix_main(name):
    from equil_monitor import protein_copies
    """Per-residue DSSP: f_H = weighted fraction of 'H' (alpha); f_HGI adds 3-10 + pi."""
    cfg = HELIX_SYS[name]
    top = md.load(os.path.join(ROOT, cfg["top"])).top
    prot = top.select("protein")
    sub_top = top.subset(prot)
    copies = protein_copies(md.Trajectory(
        np.zeros((1, prot.size, 3)), sub_top))
    codes, ft, wid = [], [], []
    for r in range(NWALK):
        tr = md.load(os.path.join(ROOT, cfg["xtc"].format(r=r)), top=top, atom_indices=prot)
        _, last = np.unique(np.round(tr.time[::-1], 1), return_index=True)
        keep = np.sort(tr.n_frames - 1 - last)                    # dedupe restart overlaps
        tr = tr[keep]
        codes.append(md.compute_dssp(tr, simplified=False))
        ft.append(tr.time); wid.append(np.full(tr.n_frames, r))
        print(f"[dssp] {name} r{r}: {tr.n_frames} frames to {tr.time[-1]/1e3:.1f} ns "
              f"({keep.size} unique)", flush=True)
    codes = np.concatenate(codes); ft = np.concatenate(ft); wid = np.concatenate(wid)
    if cfg.get("unweighted"):
        w = np.where(ft >= TEQ_PS, 1.0, 0.0); ess = used = int(w.sum())
        wtxt = "UNWEIGHTED (3D bias, 1-ns frames, run fails its reweighting gate)"
    else:
        w, ess, used = metad_ct_cached(name, cfg, ft, wid)
        wtxt = f"c(t) weights, ESS {ess:.0f} ({100*ess/used:.2f}% of {used})"
    isH = codes == "H"
    isHGI = np.isin(codes, ["H", "G", "I"])
    W = w / w.sum()
    fH, fHGI = W @ isH, W @ isHGI
    pw = np.array([(w[wid == r] / w[wid == r].sum()) @ isH[wid == r]
                   for r in range(NWALK) if w[wid == r].sum() > 0])
    err = pw.std(axis=0, ddof=1)
    raw = isH[ft >= TEQ_PS].mean(0)
    res = np.array([rr.resSeq for rr in sub_top.residues])
    rname = [rr.name for rr in sub_top.residues]
    wt = cfg["to_wt"](res)
    copy = np.zeros(res.size, int)
    for k, (a, b) in enumerate(copies):
        copy[a:b + 1] = k
    p = f"{ROOT}/{helix_dat(name)}"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(f"# {cfg['label']} per-residue DSSP helix fraction, {name} MetaD, "
                 f"{NWALK} walkers, frames to {ft.max()/1e3:.0f} ns/walker\n")
        fh.write(f"# {wtxt}; first {TEQ_PS/1e3:.0f} ns dropped. f_H = alpha (DSSP H), "
                 "f_HGI = H+G+I; err = std over per-walker f_H\n")
        fh.write("# copy  resSeq  resSeq_WT  resname  f_H  f_H_err  f_HGI  f_H_raw\n")
        for i in range(res.size):
            fh.write(f"{copy[i]:2d}  {res[i]:4d}  {wt[i]:4d}  {rname[i]:<4s}  {fH[i]:6.3f}  "
                     f"{err[i]:6.3f}  {fHGI[i]:6.3f}  {raw[i]:6.3f}\n")
    print(f"[saved] {p}")
    rows = []
    for k in range(len(copies)):
        for seg, lo, hi in HELIX_SEGS:
            m = (copy == k) & (wt >= lo) & (wt <= hi)
            if not m.any():
                continue
            segpw = pw[:, m].mean(1)
            rows.append((name + ("" if len(copies) == 1 else "_" + "AB"[k]), seg,
                         fH[m].mean(), segpw.std(ddof=1), fHGI[m].mean(), raw[m].mean()))
            print(f"  {rows[-1][0]:<18} {seg:<4} f_H {rows[-1][2]:.3f} +/- {rows[-1][3]:.3f}"
                  f"  f_HGI {rows[-1][4]:.3f}  raw {rows[-1][5]:.3f}")
    return rows


# --helix figures: one standalone figure per comparison (name tag -> systems)
HELIX_FIGS = {
    "constructs": ("1Nhp6A_wall", "1Nhp6A_apo_large", "1D2_10_wall", "1IDR_wall"),
    "stoichiometry": ("1Nhp6A_wall", "2Nhp6A"),
    "mutants": ("1Nhp6A_wall", "1S26D_wall", "1T63D_wall", "1TM_wall"),
}


def helix_table(rows):
    tab = figpath(f"all_metad_helix_dssp_segments_v{FIGV}.dat", root=ROOT)
    with open(tab, "w") as fh:
        fh.write("# segment-mean DSSP helix fraction from MetaD (see per-system "
                 "*_helix_dssp_reweighted_v*.dat for weights/ESS)\n"
                 "# segments in WT numbering: IDR 1-20, H1 27-42, H2 48-61, H3 64-93\n"
                 "# system  segment  f_H  f_H_err(walker std)  f_HGI  f_H_raw\n")
        for r in rows:
            fh.write(f"{r[0]:<20} {r[1]:<4} {r[2]:6.3f} {r[3]:6.3f} {r[4]:6.3f} {r[5]:6.3f}\n")
    print(f"[saved] {tab}")


def helix_figures(names):
    """Per-residue f_H from the saved per-system .dat, one figure per HELIX_FIGS group."""
    for tag, group in HELIX_FIGS.items():
        g = [n for n in group if n in names]
        if not g:
            continue
        with plt.rc_context(ARTICLE_RC):
            fig, ax = plt.subplots(figsize=(6.6, 3.0))
            _shade(ax)
            for n in g:
                d = np.loadtxt(f"{ROOT}/{helix_dat(n)}", dtype=str)
                cp, x = d[:, 0].astype(int), d[:, 2].astype(int)
                mu, sd = d[:, 4].astype(float), d[:, 5].astype(float)
                for k in np.unique(cp):
                    mk = np.where(cp == k)[0]
                    lab = HELIX_SYS[n]["label"] + ("" if cp.max() == 0 else f" copy {'AB'[k]}")
                    ls = "-" if k == 0 else "--"
                    for j, s in enumerate(np.split(mk, np.where(np.diff(x[mk]) > 1)[0] + 1)):
                        ax.fill_between(x[s], np.clip(mu[s] - sd[s], 0, 1),
                                        np.clip(mu[s] + sd[s], 0, 1),
                                        color=HELIX_SYS[n]["color"], alpha=0.15, lw=0)
                        ax.plot(x[s], mu[s], color=HELIX_SYS[n]["color"], lw=1.4, ls=ls,
                                label=lab if j == 0 else None)
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel(r"helix fraction (DSSP $\alpha$)")
            ax.set_xlabel("residue (WT numbering)")
            ax.set_xlim(0.5, 93.5)
            ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=3,
                      bbox_to_anchor=(0.5, 1.0))
            fig.tight_layout()
            p = figpath(f"all_metad_helix_dssp_{tag}_v{FIGV}.png", root=ROOT)
            fig.savefig(p, dpi=300); plt.close(fig)
        print(f"[saved] {p}")


def holo_compare_figure(names):
    """apo (MetaD v2) + each holo system on WT numbering, linear y, per-walker std band."""
    a = np.genfromtxt(f"{ROOT}/{APO_CMP_DAT}", dtype=None, encoding=None,
                      names=["res", "rn", "rw", "err", "raw"])
    curves = [("apo", a["res"].astype(int), a["rw"], a["err"], "0.15")]
    for n in names:
        d = np.loadtxt(f"{ROOT}/{holo_dat(n)}", dtype=str)
        curves.append((HOLO[n]["label"], d[:, 1].astype(int), d[:, 3].astype(float),
                       d[:, 4].astype(float), HOLO[n]["color"]))
    with plt.rc_context(ARTICLE_RC):
        fig, ax = plt.subplots(figsize=(6.6, 2.8))
        _shade(ax)
        for lab, x, mu, sd, col in curves:
            seg = np.split(np.arange(x.size), np.where(np.diff(x) > 1)[0] + 1)
            for k, s in enumerate(seg):          # break the line across the deletion
                ax.fill_between(x[s], np.clip(mu[s] - sd[s], 0, None), mu[s] + sd[s],
                                color=col, alpha=0.2, lw=0)
                ax.plot(x[s], mu[s], color=col, lw=1.6, marker="o" if s.size == 1 else None,
                        ms=3, label=lab if k == len(seg) - 1 else None)
        ax.set_ylim(0, None)
        ax.set_ylabel(r"C$\alpha$ RMSF ($\mathrm{\AA}$)")
        ax.set_xlabel("residue (WT numbering)")
        ax.set_xlim(0.5, 93.5)
        ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.62, 0.88))
        fig.tight_layout()
        p = figpath(f"all_metad_rmsf_d2_10_wt_apo_v{FIGV}.png", root=ROOT)
        fig.savefig(p, dpi=300); plt.close(fig)
    print(f"[saved] {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--equil", action="store_true",
                    help="apo vs DNA-bound RMSF from the unbiased equilibrium MD "
                         "(3 reps x 250 ns), instead of the reweighted apo MetaD")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-render figures from the saved .dat (skip reweighting)")
    ap.add_argument("--traj", default=DEFAULT_TRAJ,
                    help="path template (with {r}) of the processed protein-only "
                         "trajectories. Default: the per-replica *_proc.xtc. Use "
                         "analysis/pbc/solute_r{r}.xtc for the copies rebuilt "
                         "from the .trr, which carry the current run length when "
                         "the _proc copies are stale (same 200 ps stride, same "
                         "1539-atom topology).")
    ap.add_argument("--fig-version", default="1",
                    help="figure/table version N -> _v<N>. Bump when the run "
                         "length grew; do not silently overwrite an older N.")
    ap.add_argument("--ensemble", type=int, default=0, metavar="N",
                    help="write N randomly drawn structures (MetaD by systematic "
                         "resampling on c(t) weight, equil uniformly) as multi-model PDBs in " + ENS_DIR +
                         ", to show the spread that the medoids hide")
    ap.add_argument("--ensemble-source", choices=["metad", "equil", "both"],
                    default="both")
    ap.add_argument("--holo", nargs="+", choices=list(HOLO), metavar="SYS",
                    help="c(t)-reweighted RMSF of DNA-bound wall MetaD runs (" +
                         ", ".join(HOLO) + "), then a WT-numbered comparison "
                         "figure against the apo MetaD v2 table")
    ap.add_argument("--helix", nargs="+", choices=list(HELIX_SYS), metavar="SYS",
                    help="per-residue DSSP helix fraction from the MetaD runs (" +
                         ", ".join(HELIX_SYS) + "), c(t)-reweighted (2Nhp6A unweighted)")
    args = ap.parse_args()
    if args.helix:
        FIGV = args.fig_version
        if not args.plot_only:
            rows = []
            for n in args.helix:
                rows += helix_main(n)
            helix_table(rows)
        helix_figures(args.helix)
        sys.exit()
    if args.holo:
        FIGV = args.fig_version
        if not args.plot_only:
            for n in args.holo:
                holo_main(n)
        holo_compare_figure(args.holo)
        sys.exit()
    if args.ensemble:
        src = ("metad", "equil") if args.ensemble_source == "both" else (args.ensemble_source,)
        ensemble_main(args.ensemble, args.traj, src)
        sys.exit()
    FIGV = args.fig_version
    DAT = figpath(f"1Nhp6A_apo_rmsf_ca_reweighted_v{FIGV}.dat", makedirs=False)
    EQ_DAT = figpath(f"1Nhp6A_equil_rmsf_apo_vs_holo_v{FIGV}.dat", makedirs=False)
    if args.equil:
        equil_plot_only() if args.plot_only else equil_main()
    else:
        plot_only() if args.plot_only else main(args.traj)
