"""
Reweighted per-residue Calpha RMSF of the apo 1Nhp6A HMG box from the
wall-restrained well-tempered metadynamics run (data/1Nhp6A_apo_large).

The bias acts on (q_core, n_tail_core) with an UPPER_WALLS on core RMSD at
0.5 nm, so the run samples ONLY the folded basin.  The RMSF reported here is
therefore a *folded-state* rigidity measure -- how much each part of the fold
fluctuates about the native structure -- NOT a measure that includes global
unfolding.

Weights are the rigorous Tiwary-Parrinello c(t) estimator (reweight_metad),
using the SAME grid, kBT and 5-ns transient discard as figure5_data.py, so the
weighted ensemble is identical to the one behind the paper's apo panels.

RMSF_a = sqrt( sum_t w_t |x_a(t) - <x_a>_w|^2 / sum_t w_t ),  after superposing
every frame on the rigid core Calpha (resSeq 21-93) and iterating the reference
once to the weighted-mean structure.

Outputs (figures/ + a .dat table):
  figures/1Nhp6A_apo_rmsf_ca_reweighted_v1.png
  figures/1Nhp6A_apo_rmsf_ca_reweighted_v1.dat
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

# ---- constants copied from figure5_data.py (keep the ensemble identical) -----
ROOT = os.environ.get("HMG_ROOT", os.getcwd())   # project root; override with $HMG_ROOT
APO_DIR = f"{ROOT}/data/1Nhp6A_apo_large"
NWALK = 6
Q_MIN, Q_MAX, Q_BIN = 0.3, 1.0, 100
NTC_MIN, NTC_MAX, NTC_BIN = 0, 150, 300
KBT = 2.494339          # kJ/mol, PLUMED default used for sum_hills
TEQ_PS = 5.0e3          # discard the 5-ns bias-filling transient
TOP = f"{APO_DIR}/1Nhp6A_parambsc1_tip3p_ions0.15_meta_2D_proc.gro"

# ---- fold segments (resSeq, 1-indexed; helices are the plumed-monitored ranges)
SEGMENTS = [
    ("IDR tail",   1, 20),
    ("loop 0-H1", 21, 26),
    ("H1",        27, 42),
    ("loop H1-H2",43, 47),
    ("H2",        48, 61),
    ("loop H2-H3",62, 63),
    ("H3",        64, 91),
    ("C-term",    92, 93),
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


def main():
    # ---------- weights (c(t)) over all walkers, matched to traj frames --------
    hills = [fm.load_hills(f"{APO_DIR}/hills.dat.{r}") for r in range(NWALK)]
    t_h, q0, n0, hgt, sig_q, sig_n, gamma = rwm.pool_hills(hills)
    print(f"[weights] {t_h.size:,} pooled hills | sigma=({sig_q},{sig_n}) gamma={gamma:.0f}")
    q_grid = np.linspace(Q_MIN, Q_MAX, Q_BIN)
    n_grid = np.linspace(NTC_MIN, NTC_MAX, NTC_BIN)

    trajs, ftime, fq, fn, fwid = [], [], [], [], []
    for r in range(NWALK):
        xtc = f"{APO_DIR}/replica_{r}/1Nhp6A_parambsc1_tip3p_ions0.15_meta_2D_r{r}_proc.xtc"
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
DAT = "figures/1Nhp6A_apo_rmsf_ca_reweighted_v1.dat"

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "HIE": "H", "HID": "H", "HIP": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Figure-5 notebook style (analysis_figure5.ipynb): typography, helix palette, labels
FIG5_RC = {"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 10,
           "ytick.labelsize": 10, "legend.fontsize": 10, "axes.linewidth": 0.8,
           "figure.dpi": 150, "axes.grid": False, "pdf.fonttype": 42}
HELIX = {"H1": "#4C72B0", "H2": "#C44E52", "H3": "#55A868"}   # fig5 COLOR dict
HLABEL = {"H1": "H1 (27–42)", "H2": "H2 (48–61)", "H3": "H3 (64–91)"}


def save_dat(ca_res, ca_resname, rmsf_ca, rmsf_err, rmsf_raw, ess, used):
    os.makedirs(f"{ROOT}/figures", exist_ok=True)
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
    """Helix + IDR-tail shading shared by both figure variants (fig5 palette)."""
    ax.axvspan(0.5, 20.5, color="0.5", alpha=0.10, lw=0)
    for name, lo, hi in SEGMENTS:
        if name in HELIX:
            ax.axvspan(lo - 0.5, hi + 0.5, color=HELIX[name], alpha=0.13, lw=0)
            if top_labels:
                ax.text((lo + hi) / 2, 0.97, HLABEL[name], ha="center", va="top",
                        transform=ax.get_xaxis_transform(), fontsize=9, color=HELIX[name])
    if top_labels:
        ax.text(10.5, 0.97, "IDR tail", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9, color="0.45")


def make_figures(ca_res, ca_resname, rmsf_ca, rmsf_err, rmsf_raw):
    os.makedirs(f"{ROOT}/figures", exist_ok=True)

    # ---- variant 1: compact panel, residue-number x axis (fig5 style) --------
    with plt.rc_context(FIG5_RC):
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
        p1 = f"{ROOT}/figures/1Nhp6A_apo_rmsf_ca_reweighted_v1.png"
        fig.savefig(p1, dpi=300); plt.close(fig)
    print(f"[saved] {p1}")

    # ---- variant 2: very stretched, one-letter sequence as x tick labels ------
    with plt.rc_context(FIG5_RC):
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
        p2 = f"{ROOT}/figures/1Nhp6A_apo_rmsf_ca_reweighted_seq_v1.png"
        fig.savefig(p2, dpi=300); plt.close(fig)
    print(f"[saved] {p2}")


def plot_only():
    d = np.genfromtxt(f"{ROOT}/{DAT}", dtype=None, encoding=None,
                      names=["resSeq", "resname", "rw", "err", "raw"])
    make_figures(d["resSeq"].astype(int), list(d["resname"]),
                 d["rw"], d["err"], d["raw"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="re-render figures from the saved .dat (skip reweighting)")
    args = ap.parse_args()
    plot_only() if args.plot_only else main()
