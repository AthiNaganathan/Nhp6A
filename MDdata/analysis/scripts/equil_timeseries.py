#!/usr/bin/env python
"""Per-system time-series overview of the unbiased equilibrium MD campaign.

One figure per system, the three replicas overlaid, for the observables that
physically exist for that system:

    DNA only          (DNA)               de2e, bend angle, DNA contacts
    protein only      (1Nhp6A_apo_large)  q_core, rmsd_core, IDR-core contacts,
                                           Rg(IDR), total + per-helix helicity
    protein:DNA (WT)  (1Nhp6A_holo)       all of the above + per-helix / DNA contacts

For every protein+DNA system (holo, and the 2:1 complex) three extra panels count the
per-helix / DNA contacts: the number of (helix residue x DNA residue) pairs whose closest
heavy-atom distance is < 0.45 nm (H1=27-42, H2=48-61, H3=64-93; residue-pair count, the
medoid_contacts.py convention).  Distances are minimum-image, so the count is robust to
how the duplex/copies are wrapped and does NOT need a whole-molecule PBC reconstruction.
For the 2:1 complex each helix is scored once per copy against the WHOLE duplex.

NOTE ON NAMING: these equilibrium runs are UNBIASED -- no wall, no restraint, no
metadynamics (PLUMED registers zero bias actions; apo runs without PLUMED at all).
The directories were once called DNA_wall / 1Nhp6A_wall after the MetaD systems
they were seeded from, which wrongly implied a cmap wall was acting.  Renamed
2026-07-30.  The walled systems are the top-level data/*_wall MetaD runs.

DNA CVs (de2e / bend / contacts) come straight from the force-free colvar.dat.
bend = 180 - deg(theta) (theta is a PLUMED ANGLE in radians); contacts is the
normalized CONTACTMAP fraction of native base-pair contacts.

The protein folding CVs are NOT in the equilibrium colvars (apo ran PLUMED-free;
holo printed only the DNA CVs), so they are recomputed post-hoc with `plumed
driver`, reusing the EXACT CV definitions from the apo well-tempered MetaD input
(plumed_meta_2D_1Nhp6A_large.dat) so the numbers are on the same scale as the
biased runs / article panels:

    q_core       CONTACTMAP, 67 CA-CA core contacts, Best-Hummer Q switch
    n_tail_core  COORDINATION IDR(tail)-core, R_0=0.5 NN=6 MM=12
    rmsd_core    RMSD of the core to ref_core_ca_1Nhp6A_large.pdb (TYPE=OPTIMAL)
    helix1/2/3   ALPHARMSD over 27-42, 48-61, 64-93   (H3 = 1LWM canonical)

All of these are distance-/occupancy-weighted (not mass-weighted), so the driver
reproduces the in-mdrun value WITHOUT a --mc mass file -- the one CV class exempt
from the project driver-masses caveat.  ALPHARMSD defaults to DRMSD (also mass
free).  The biased run uses absolute atom indices <= 1539 and the protein is
atoms 1-1539 (identical ordering in the apo and holo builds, verified), so the
whole CV block is evaluated on a protein-only slice that preserves those indices.

Rg(IDR) (IDR dimensions) and total helicity (H1+H2+H3) are added from mdtraj.
NOTE: H3 here is 64-93 (1LWM Pro64-Ala93); the apo MetaD helix3 CV used 64-91, so
this H3 Sa runs ~1.5 above apo panel-B helix3 by construction.

Reuses equil_monitor.load_whole for the trjconv -pbc whole + topology machinery.

STEERED PULLS (1Nhp6A_mstate_pull, PLAN.md A1): the m-state probe's BIASED construction
runs in data/mstate/pull/c<k>/ (--reps = config ids, default 0,1,3,4).  Same protein /
helix-DNA panels as above, plus the pull itself, read from colvar_<P>.dat
(P = 1Nhp6A_mstate_pull_c<k>): core-DNA COM distance d_core with the moving target
AT(t) = D0 + (D1 - D0) t/40 ns, the restraint force kappa (AT - d_core) along d_core
(positive = pulling the core off the DNA), the accumulated work, core-DNA and IDR-DNA
coordination, the IDR lower-wall bias, and the minimum core(27-93)-DNA heavy-atom distance
(mdtraj, min-image).  DNA contacts (cmap) come from <P>_qdna.dat (plumed driver, canonical
monitor cmap block).  Caches are written as <P>_<kind>_skip<N>.dat.

M-STATE PRODUCTION (1Nhp6A_mstate_md, PLAN.md A2): the UNBIASED runs started from the
equilibrated pulls, data/mstate/md/c<k>/rep<r>/ (--reps = run ids c<k>_rep<r>, default all
8; P = 1Nhp6A_mstate_md_c<k>_rep<r>).  The force-free monitor colvar already carries
coord_core_dna, coord_tail_dna, d_core, de2e, theta and the canonical cmap (identical block
to the pull <P>_qdna.dat), so no driver pass is needed for Q_DNA.  Colour = config,
solid/dashed = rep0/rep1.  Also writes the survival table
analysis/mstate_probe/1Nhp6A_mstate_md_survival.dat: re-binding (PLAN.md decision 3,
n_core > 4.85 sustained >= 1 ns) under four readings side by side, no primary (user,
2026-10-08) -- every 0.2 ps colvar frame, every 10 ps frame, 1-ns running mean, first
frame -- plus the first core-DNA heavy pair < 0.45 nm (at the --skip stride).  Bound-mode
columns (PLAN.md, 2026-10-09; last 10 ns): bound = sustained core-DNA heavy-atom contact,
mode = native / shifted / non-native from the register-variant native Q, plus the contacting
helices.  The register Q comes from analysis/mstate_probe/1Nhp6A_mstate_md_qnative.dat, so run
mstate_contact_pca.py FIRST (and never concurrently: both rebuild <P>_solute_whole_skip10.xtc).

STAGE B 2:1 M-STATE PRODUCTION (2Nhp6A_mstate_md, PLAN.md B1): copy A bound, copy B's core
pulled off, data/mstate_2Nhp6A/md/c<k>/rep<r>/ (runs c1/c3 x rep0/rep1; P =
2Nhp6A_mstate_md_c<k>_rep<r>).  COLVAR-ONLY for now (no trajectory pass): the force-free
monitor carries coord_core_dna_A/B, coord_tail_dna_A/B, d_core_A/B, de2e, theta, cmap.
Colour = run, solid/dashed = copy A/B.  Reference lines are copy B's own 2:1 values (PLAN.md
B1): survival threshold 5.31, bound p5-p95 n_core 16.0-26.4 and n_IDR 5.3-11.5.  Writes
analysis/mstate_probe/2Nhp6A_mstate_md_survival.dat: copy-B re-binding under the same four
readings as the 1:1 table, plus copy-A retention (its n_core vs its 2:1 p5 15.9) and DNA state.

Usage:
    equil_timeseries.py                       # all three systems
    equil_timeseries.py --system 1Nhp6A_holo  # just one
    equil_timeseries.py --skip 20             # coarser trajectory stride
    equil_timeseries.py --system 1Nhp6A_mstate_pull --version v1
    equil_timeseries.py --system 1Nhp6A_mstate_md --version v1
    equil_timeseries.py --system 2Nhp6A_mstate_md --version v1
"""
import argparse
import os
import subprocess
import sys
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equil_monitor import ROOT, TOPOL, gmx, load_whole, protein_copies  # pbc-whole + topology + copy split
from figpaths import figpath

PLUMED = "/usr/local/bin/plumed"
# the biased apo input whose CV definitions we reproduce verbatim
PROTEIN_CV_SRC = os.path.join(ROOT, "plumed_meta_2D_1Nhp6A_large.dat")
REF_CORE = os.path.join(ROOT, "ref_core_ca_1Nhp6A_large.pdb")
PROTEIN_ENTITY = "1-1539"        # protein atom range (identical apo/holo)
IDR_RESID = (0, 21)              # residues 1-22, 0-based, for Rg(IDR)

SYSTEMS = {
    "DNA":              dict(title="DNA only",           protein=False, colvar=True, ncopies=0, dna=True),
    "1Nhp6A_apo_large": dict(title="protein only (apo)", protein=True,  colvar=False, ncopies=1, dna=False),
    "1Nhp6A_holo":      dict(title="protein:DNA (WT)",   protein=True,  colvar=True,  ncopies=1, dna=True),
    # 2:1 complex: two independent Nhp6A copies on one duplex, so every protein CV is
    # evaluated once per copy and the two are plotted separately (solid = A, dashed = B).
    "2Nhp6A":           dict(title="2:1 protein:DNA",    protein=True,  colvar=True,  ncopies=2, dna=True),
    # m-state probe steered pulls (BIASED construction runs), one dir per start config
    "1Nhp6A_mstate_pull": dict(title="m-state steered pull, 1:1 (BIASED construction run)",
                               protein=True, colvar=True, ncopies=1, dna=True, mstate="pull",
                               reps=(0, 1, 3, 4)),
    # m-state probe unbiased production, 4 configs x 2 seeds
    "1Nhp6A_mstate_md": dict(title="m-state unbiased production, 1:1 (core detached at t=0)",
                             protein=True, colvar=True, ncopies=1, dna=True, mstate="md",
                             reps=tuple(f"c{k}_rep{r}" for k in (0, 1, 3, 4) for r in (0, 1))),
    # Stage B: 2:1, copy B's core detached at t=0; colvar-only (protein=False skips every
    # trajectory pass), copy-B reference lines/bands from PLAN.md B1
    "2Nhp6A_mstate_md": dict(title="m-state unbiased production, 2:1 (copy B core detached at t=0)",
                             protein=False, colvar=True, ncopies=2, dna=True, mstate="md",
                             datadir="mstate_2Nhp6A",
                             reps=tuple(f"c{k}_rep{r}" for k in (1, 3) for r in (0, 1)),
                             hlines={"n_core": (5.31, "copy-B survival threshold"),
                                     "contacts": (0.9, "Q$_{DNA}$ gate")},
                             bands={"n_core": (16.0, 26.4, "copy-B bound p5-p95 (2:1)"),
                                    "n_idr": (5.3, 11.5, "copy-B bound p5-p95 (2:1)")}),
    # Nhp6A delta2-10 (WT res 2-10 deleted, renumbered 1-84: res 1 = WT 1, res k = WT k+9).
    # The WT CV block is mapped onto it by (residue, atom name), see protein_cv_block();
    # helix / core residue numbers in labels stay in WT numbering.  n_tail_core uses the
    # deletion's own 11-residue tail group (its MetaD n_IDR group); Rg(IDR) = new res 1-13,
    # i.e. what is left of WT 1-22.
    "1D2_10_holo": dict(title="protein:DNA (Δ2-10)", protein=True, colvar=True, ncopies=1,
                        dna=True, wt_offset=9, prot_atoms=1374, idr_resid=(0, 12),
                        tail_src="plumed_meta_2D_1D2_10_wall.dat"),
}
# --pmf-overlay: equilibrium system -> the wall-MetaD system whose reweighted F(n_IDR-DNA, bend)
# table (analysis/<sys>_pmf2d_idrdna_bend.dat, dna_bending_wall_compare.py --idrdna-bend-2d) it is
# drawn on.  Same coord_tail_dna / theta definitions in the monitor and MetaD inputs (checked
# 2026-11-04).  2Nhp6A has no such table (two per-copy n_IDR CVs), so it is not listed.
PMF_OVERLAY = {"1Nhp6A_holo": "1Nhp6A_wall", "1D2_10_holo": "1D2_10_wall"}
# surface masking = the SI 1IDR panel (article_figures.ipynb cell 20)
OVL_STD_CEIL, OVL_MIN_VISITS, OVL_MIN_WALKERS = 15.0, 5, 3
OVL_BLOCK_NS = 1.0                 # trajectory line = block means over this window
OVL_COLOR = ["#0072B2", "#E69F00", "#009E73"]   # Okabe-Ito, rep0/1/2
WT_TOPOL = TOPOL["1Nhp6A_holo"]    # atom numbering the PROTEIN_CV_SRC indices refer to


def wt_offset(system):
    """WT resSeq - system resSeq for residues past the deletion (0 for WT systems)."""
    return SYSTEMS[system].get("wt_offset", 0)
COPIES = "AB"      # copy labels, in topology order (matches equil_monitor.COPIES)

# canonical HMG-box helices (1LWM; the RESIDUES ranges of the apo ALPHARMSD CVs). Used for
# the per-helix / DNA contact panels -- do NOT re-derive helix bounds from DSSP on the traj.
HELICES = {"h1_dna": (27, 42), "h2_dna": (48, 61), "h3_dna": (64, 93)}
CONTACT_CUTOFF = 0.45      # nm, closest-heavy-atom; project convention (medoid_contacts.py)
DNA_PREFIXES = ("DA", "DT", "DG", "DC")   # residue-name prefixes (also matches DA5/DG3 termini)

COLVAR_PANELS = ["de2e", "bend", "contacts", "n_idr"]   # n_idr only where the colvar has it
PROTEIN_PANELS = ["q_core", "rmsd_core", "n_tail_core", "rg_idr",
                  "total_hel", "helix1", "helix2", "helix3"]
CONTACT_PANELS = list(HELICES)          # h1_dna, h2_dna, h3_dna  (only for protein+DNA systems)
PER_COPY_PANELS = PROTEIN_PANELS + CONTACT_PANELS   # panels evaluated once per protein copy
YLABEL = {
    "de2e":        "end-to-end (nm)",
    "bend":        "DNA bend angle (deg)",
    "contacts":    "DNA contacts (frac. native)",
    "q_core":      r"$Q_{\rm core}$ (native frac.)",
    "rmsd_core":   "RMSD core (nm)",
    "n_tail_core": "IDR-core contacts",
    "rg_idr":      "Rg IDR (nm)",
    "total_hel":   r"total $S_\alpha$ (H1+H2+H3)",
    "helix1":      r"$S_\alpha$ helix 1 (27-42)",
    "helix2":      r"$S_\alpha$ helix 2 (48-61)",
    "helix3":      r"$S_\alpha$ helix 3 (64-93)",
    "h1_dna":      "H1-DNA contacts (<0.45 nm)",
    "h2_dna":      "H2-DNA contacts (<0.45 nm)",
    "h3_dna":      "H3-DNA contacts (<0.45 nm)",
}
REP_COLOR = ["C0", "C1", "C2", "C3"]
CACHE = "equil_cvs{tag}_skip{skip}.dat"     # per-rep (per-copy) protein-CV cache
CCACHE = "helix_dna_contacts_skip{skip}.dat"   # per-rep helix-DNA contact cache (all copies)

# --- steered pulls (m-state probe) ---
PULL_PREFIX = "1Nhp6A_mstate_pull_c{k}"     # traceable prefix of every file in a pull dir
PULL_PANELS = ["d_core", "pull_force", "pull_work", "n_core", "n_idr", "wall_bias",
               "core_dna_mind", "bend", "de2e", "contacts"]
PULL_COLOR = {0: "#0072B2", 1: "#E69F00", 3: "#009E73", 4: "#CC79A7"}  # Okabe-Ito, fixed per config
PULL_CORE = (27, 93)            # core for the min-distance gate (PLAN.md decision 2)
# reference lines / bands (PLAN.md "Reference values measured")
PULL_HLINES = {"n_core": (4.85, "survival threshold"), "n_idr": (11.6, "IDR wall"),
               "core_dna_mind": (0.45, "contact cut-off"), "contacts": (0.9, "Q$_{DNA}$ gate")}
PULL_BANDS = {"n_idr": (11.6, 28.9, "bound p5-p95"), "n_core": (16.3, 21.9, "bound p5-p95")}
# --- unbiased m-state production: no restraint, no wall ---
MD_PREFIX = "{system}_{run}"                # run = c<k>_rep<r>
MD_PANELS = ["n_core", "n_idr", "d_core", "core_dna_mind", "contacts", "bend", "de2e"]
MD_COPY_PANELS = ["n_core", "n_idr", "d_core"]   # per-copy monitor CVs (2:1: suffix _A/_B)
MD2_COLOR = {"c1_rep0": "#E69F00", "c1_rep1": "#D55E00",   # 2:1: colour = run (Okabe-Ito)
             "c3_rep0": "#009E73", "c3_rep1": "#0072B2"}
SURVIVAL_TH_B = 5.31            # copy B: 25 % of its own 2:1 bound n_core median (PLAN.md B1)
CORE_A_P5 = 15.9                # copy A: its own 2:1 bound n_core p5 (PLAN.md B1)
MD_HLINES = {p: v for p, v in PULL_HLINES.items() if p != "n_idr"}   # the IDR wall is gone
SURVIVAL_TH = 4.85              # 25 % of the bound core-DNA coordination median (decision 3)
SURVIVAL_HOLD = 1.0             # ns the threshold must be held
# bound-mode labels (PLAN.md "Bound-mode classification", 2026-10-09), all over the last 10 ns:
BOUND_FRAC = 0.9                # bound = core-DNA heavy pair < CONTACT_CUTOFF in >= this frame fraction
REGISTER_Q = 0.5                # best-register native Q >= this -> (shifted-)native footprint; the
                                # observed values fall in two groups (<= 0.13 / >= 0.54) around it
REGISTER_TIE = 0.05             # registers within this of the best are all listed (e.g. "0/-1")
HELIX_DOM = 0.5                 # helices with >= this fraction of the top helix's contacts are listed
QNATIVE_TABLE = "1Nhp6A_mstate_md_qnative.dat"   # written by mstate_contact_pca.py (run it first)
YLABEL.update({
    "d_core":        "core-DNA COM dist. (nm)",
    "pull_force":    "pull force (kJ/mol/nm)",
    "pull_work":     "pull work (kJ/mol)",
    "n_core":        "core-DNA coordination",
    "n_idr":         "IDR-DNA coordination",
    "wall_bias":     "IDR wall bias (kJ/mol)",
    "core_dna_mind": "min core-DNA heavy dist. (nm)",
})


def mstate(system):
    """None for the equilibrium campaign, 'pull' or 'md' for the m-state probe runs."""
    return SYSTEMS[system].get("mstate")


def is_pull(system):
    return mstate(system) == "pull"


def is_series_label(label):
    """True for a replica/config line label ('rep0', 'c3', 'rep1 B', 'c3 rep1'), not a
    reference key."""
    import re
    return re.fullmatch(r"(rep|c)\d+( rep\d+)?( [AB])?", label) is not None


def md_config(rep):
    """'c3_rep1' -> (3, 1)."""
    k, r = rep.split("_")
    return int(k[1:]), int(r[3:])


def run_prefix(system, rep):
    """Traceable file prefix of an m-state run dir."""
    return PULL_PREFIX.format(k=rep) if is_pull(system) else MD_PREFIX.format(system=system, run=rep)


def run_label(system, rep):
    if is_pull(system):
        return f"c{rep}"
    if mstate(system) == "md":
        return rep.replace("_", " ")
    return f"rep{rep}"


def run_dir(system, rep):
    if is_pull(system):
        return os.path.join(ROOT, "data", "mstate", "pull", f"c{rep}")
    if mstate(system) == "md":
        k, r = md_config(rep)
        return os.path.join(ROOT, "data", SYSTEMS[system].get("datadir", "mstate"), "md",
                            f"c{k}", f"rep{r}")
    return os.path.join(ROOT, "data", "equil", system, f"rep{rep}")


def run_src(system, rep):
    """The raw trajectory whose mtime invalidates the caches."""
    if mstate(system):
        return os.path.join(run_dir(system, rep), run_prefix(system, rep) + ".xtc")
    return os.path.join(run_dir(system, rep), "md.xtc")


def cache_path(system, rep, kind, skip, tag=""):
    """kind: 'protein' or 'contacts' (or 'mindist', m-state runs only)."""
    if mstate(system):
        name = {"protein": "protein_cvs", "contacts": "helix_dna_contacts",
                "mindist": "core_dna_mindist"}[kind]
        return os.path.join(run_dir(system, rep),
                            f"{run_prefix(system, rep)}_{name}_skip{skip}.dat")
    fmt = CACHE if kind == "protein" else CCACHE
    return os.path.join(run_dir(system, rep), fmt.format(skip=skip, tag=tag))


def load_run(system, rep, skip):
    """pbc-whole non-water trajectory (.time in ns).  Equil runs go through
    equil_monitor.load_whole; m-state runs get the same trjconv -pbc whole treatment under
    prefixed names (<P>_solute_whole_skip<N>.xtc, topology <P>_solute_end.gro)."""
    if not mstate(system):
        return load_whole(system, rep, skip)
    import mdtraj as md

    os.environ.setdefault("HWLOC_COMPONENTS", "-gl")   # rubisco gmx_mpi hangs without it
    d, P = run_dir(system, rep), run_prefix(system, rep)
    tpr, src = os.path.join(d, P + ".tpr"), run_src(system, rep)
    top = os.path.join(d, f"{P}_solute_end.gro")
    xtc = os.path.join(d, f"{P}_solute_whole_skip{skip}.xtc")
    if not os.path.exists(top):
        gmx(["trjconv", "-f", os.path.join(d, P + ".gro"), "-s", tpr, "-o", top,
             "-pbc", "whole"], stdin="non-Water\n")
    if not os.path.exists(xtc) or os.path.getmtime(xtc) < os.path.getmtime(src):
        r = gmx(["trjconv", "-f", src, "-s", tpr, "-o", xtc, "-pbc", "whole",
                 "-skip", str(skip)], stdin="non-Water\n")
        if not os.path.exists(xtc):
            sys.exit(f"trjconv failed for {P}:\n{r.stderr[-2000:]}")
    t = md.load(xtc, top=top)
    t.time = np.arange(t.n_frames) * 0.01 * skip     # xtc written every 10 ps
    return t


def read_pull_colvar(rep):
    """Pull colvar -> dict of time series (time in ns).  The moving-restraint target and
    force are rebuilt from AT0/AT1/KAPPA0/STEP1 in the as-run plumed input."""
    d, P = run_dir("1Nhp6A_mstate_pull", rep), PULL_PREFIX.format(k=rep)
    path = os.path.join(d, f"colvar_{P}.dat")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        names = fh.readline().split()[2:]
    a = np.genfromtxt(path, invalid_raise=False, usecols=range(len(names)))
    a = a[~np.isnan(a).any(axis=1)]
    col = {n: a[:, i] for i, n in enumerate(names)}

    par = {}
    for tok in open(os.path.join(d, f"plumed_{P}.dat")).read().split():
        for key in ("AT0", "AT1", "KAPPA0", "STEP1"):
            if tok.startswith(key + "="):
                par[key] = float(tok.split("=")[1])
    t_ps = col["time"]
    frac = np.clip(t_ps / (par["STEP1"] * 0.002), 0, 1)       # dt = 2 fs
    target = par["AT0"] + (par["AT1"] - par["AT0"]) * frac
    out = {"time": t_ps / 1000.0,
           "d_core": col["d_core"], "d_target": target,
           "pull_force": par["KAPPA0"] * (target - col["d_core"]),
           "pull_work": col["pull.work"],
           "n_core": col["coord_core_dna"], "n_idr": col["coord_tail_dna"],
           "wall_bias": col["idr_wall.bias"],
           "bend": 180.0 - np.degrees(col["theta"]), "de2e": col["de2e"]}
    q = os.path.join(d, f"{P}_qdna.dat")
    if os.path.exists(q):
        qa = np.genfromtxt(q)
        out["contacts_time"], out["contacts"] = qa[:, 0] / 1000.0, qa[:, 1]
    return out


def read_md_colvar(rep, system="1Nhp6A_mstate_md"):
    """Force-free production colvar -> dict of time series (time in ns), 0.2 ps stride.
    For the 2:1 system the per-copy CVs (MD_COPY_PANELS) are tuples (copy A, copy B)."""
    d, P = run_dir(system, rep), run_prefix(system, rep)
    path = os.path.join(d, f"colvar_{P}.dat")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        names = fh.readline().split()[2:]
    a = np.genfromtxt(path, invalid_raise=False, usecols=range(len(names)))
    a = a[~np.isnan(a).any(axis=1)]
    # a -cpi restart re-appends from the checkpoint time: keep a row only if no later row
    # has an equal or earlier time (the restarted segment supersedes the killed tail)
    later_min = np.minimum.accumulate(a[::-1, 0])[::-1]
    a = a[np.append(a[:-1, 0] < later_min[1:], True)]
    col = {n: a[:, i] for i, n in enumerate(names)}
    if SYSTEMS[system]["ncopies"] > 1:
        per = lambda n: tuple(col[f"{n}_{c}"] for c in COPIES)
    else:
        per = lambda n: col[n]
    return {"time": col["time"] / 1000.0,
            "n_core": per("coord_core_dna"), "n_idr": per("coord_tail_dna"),
            "d_core": per("d_core"), "contacts": col["cmap"],
            "bend": 180.0 - np.degrees(col["theta"]), "de2e": col["de2e"]}


def first_sustained(t, above, hold=SURVIVAL_HOLD):
    """Start time of the first contiguous stretch of `above` frames spanning >= hold ns."""
    dt = np.median(np.diff(t))
    need = int(round(hold / dt))
    run = 0
    for i, x in enumerate(above):
        run = run + 1 if x else 0
        if run >= need:
            return t[i - run + 1]
    return np.nan


def read_qnative_registers():
    """{run: (register labels, last-10-ns Q per register)} from mstate_contact_pca.py's table,
    or {} if it has not been written yet."""
    path = os.path.join(ROOT, "analysis", "mstate_probe", QNATIVE_TABLE)
    if not os.path.exists(path):
        return {}
    hdr = [l for l in open(path) if l.startswith("# run")][-1].split()[2:]
    regs = [h.replace("_bp", "").replace("native", "0") for h in hdr]
    return {l.split()[0]: (regs, np.array(l.split()[1:], float))
            for l in open(path) if l.strip() and not l.startswith("#")}


def bound_mode(qreg, helix_last10):
    """(best register, Q_best, mode, contacting helices) for one run's last 10 ns.
    mode: native | shifted | non-native (REGISTER_Q on the best register-variant Q);
    the register field lists every variant within REGISTER_TIE of the best ('-' if non-native,
    where all variants are near zero and the ranking is meaningless)."""
    hel = "+".join(h for h, v in helix_last10.items()
                   if v >= HELIX_DOM * max(helix_last10.values()) and v > 0) or "none"
    if qreg is None:
        return "nan", np.nan, "nan", hel
    regs, q = qreg
    best = q.max()
    tied = [r for r, v in zip(regs, q) if v >= best - REGISTER_TIE]
    tied.sort(key=lambda r: r != "0")                      # identity register first
    mode = ("non-native" if best < REGISTER_Q else "native" if tied[0] == "0" else "shifted")
    return ("/".join(tied) if mode != "non-native" else "-"), best, mode, hel


def rebinding_times(t, nc, threshold):
    """Re-binding of one core (PLAN.md decision 3) under the four readings, no primary:
    (t_strict_0.2ps, t_strict_10ps, t_runmean_1ns, t_first_above, frac_above,
    frac_above_after_strict); nan = not within the run."""
    above = nc > threshold
    dt = np.median(np.diff(t))
    s10 = max(1, int(round(0.01 / dt)))              # 10 ps = the xtc stride
    w = int(round(SURVIVAL_HOLD / dt))
    rm = np.convolve(nc, np.ones(w) / w, "valid")      # rm[i] = mean over [t_i, t_i + 1 ns)
    t_rm = t[np.argmax(rm > threshold)] if (rm > threshold).any() else np.nan
    t_strict = first_sustained(t, above)
    return (t_strict, first_sustained(t[::s10], above[::s10]), t_rm,
            t[np.argmax(above)] if above.any() else np.nan, above.mean(),
            above[t >= t_strict].mean() if np.isfinite(t_strict) else np.nan)


def survival_table_2to1(series, reps):
    """Stage B (2:1): copy-B re-binding (threshold SURVIVAL_TH_B) under the four readings,
    copy-A retention and the DNA state, from the monitor colvar alone."""
    cols = ["t_end", "t_strict_0.2ps", "t_strict_10ps", "t_runmean_1ns", "t_first_above",
            "frac_above", "frac_above_after_strict", "n_core_B_last10", "d_core_B_last10",
            "n_idr_B_median", "n_idr_B_min", "n_idr_B_last10", "n_core_A_min_rm1ns",
            "n_core_A_last10", "d_core_A_last10", "n_idr_A_last10", "q_dna_median",
            "bend_last10", "de2e_last10"]
    rows = []
    for rep in reps:
        if (rep, 1) not in series.get("n_core", {}):
            continue
        t, ncb = series["n_core"][(rep, 1)]
        _, nca = series["n_core"][(rep, 0)]
        _, dcb = series["d_core"][(rep, 1)]
        _, dca = series["d_core"][(rep, 0)]
        _, nib = series["n_idr"][(rep, 1)]
        _, nia = series["n_idr"][(rep, 0)]
        _, q = series["contacts"][(rep, 0)]
        _, bend = series["bend"][(rep, 0)]
        _, de2e = series["de2e"][(rep, 0)]
        late = t >= t[-1] - 10.0
        w = int(round(SURVIVAL_HOLD / np.median(np.diff(t))))
        rows.append([rep, t[-1], *rebinding_times(t, ncb, SURVIVAL_TH_B),
                     ncb[late].mean(), dcb[late].mean(), np.median(nib), nib.min(),
                     nib[late].mean(), np.convolve(nca, np.ones(w) / w, "valid").min(),
                     nca[late].mean(), dca[late].mean(), nia[late].mean(), np.median(q),
                     bend[late].mean(), de2e[late].mean()])
    out = os.path.join(ROOT, "analysis", "mstate_probe", "2Nhp6A_mstate_md_survival.dat")
    with open(out, "w") as fh:
        fh.write(f"# Stage B 2:1 m-state survival (PLAN.md B1), monitor colvar only (0.2 ps): "
                 f"copy-B re-binding = n_core_B > {SURVIVAL_TH_B} sustained >= {SURVIVAL_HOLD} ns; "
                 "times in ns, nan = not within the run (t_end = colvar end)\n"
                 "# four readings side by side, no primary, as in 1Nhp6A_mstate_md_survival.dat. "
                 "*_last10 = mean over the last 10 ns\n"
                 f"# copy A retention: n_core_A_min_rm1ns = min of its 1-ns running mean "
                 f"(its 2:1 bound p5 = {CORE_A_P5}).  q_dna = monitor cmap (DNA duplex fraction)\n")
        fh.write("# run  " + "  ".join(cols) + "\n")
        for r in rows:
            fh.write(f"{r[0]:8s} " + " ".join(f"{v:8.3f}" for v in r[1:]) + "\n")
    print(open(out).read())
    print(f"  wrote {out}")


def survival_table(series, reps, skip):
    """Re-binding time per run under the four readings of PLAN.md decision 3 (no primary),
    plus the structural first-contact time, end-state descriptors and the bound-mode label."""
    cols = ["t_strict_0.2ps", "t_strict_10ps", "t_runmean_1ns", "t_first_above",
            "t_first_pair_0.45nm", "frac_above", "frac_above_after_strict",
            "n_core_last10", "d_core_last10", "n_idr_median", "n_idr_min", "q_dna_median",
            "frac_contact_last10", "bend_last10", "h1_last10", "h2_last10", "h3_last10",
            "q_best_last10", "bound", "mode", "register", "helices"]
    qtab = read_qnative_registers()
    if not qtab:
        print(f"  WARNING: {QNATIVE_TABLE} missing -> register/mode columns are nan "
              "(run mstate_contact_pca.py first)")
    rows = []
    for rep in reps:
        if (rep, 0) not in series.get("n_core", {}):
            continue
        t, nc = series["n_core"][(rep, 0)]
        t_strict, t_s10, t_rm, t_first, f_above, f_after = rebinding_times(t, nc, SURVIVAL_TH)
        tm, mind = series["core_dna_mind"][(rep, 0)]
        late = t >= t[-1] - 10.0
        _, nidr = series["n_idr"][(rep, 0)]
        _, q = series["contacts"][(rep, 0)]
        _, dc = series["d_core"][(rep, 0)]
        rows.append([rep, t_strict, t_s10, t_rm, t_first,
                     tm[np.argmax(mind < CONTACT_CUTOFF)] if (mind < CONTACT_CUTOFF).any() else np.nan,
                     f_above, f_after,
                     nc[late].mean(), dc[late].mean(), np.median(nidr), nidr.min(), np.median(q)])
        mlate = tm >= tm[-1] - 10.0
        fcont = np.mean(mind[mlate] < CONTACT_CUTOFF)
        tb, bend = series["bend"][(rep, 0)]
        hl = {}
        for h in HELICES:
            th, y = series[h][(rep, 0)]
            hl[h[:2].upper()] = y[th >= th[-1] - 10.0].mean()
        reg, qbest, mode, hel = bound_mode(qtab.get(rep), hl)
        rows[-1] += [fcont, bend[tb >= tb[-1] - 10.0].mean(), hl["H1"], hl["H2"], hl["H3"], qbest,
                     "bound" if fcont >= BOUND_FRAC else "unbound", mode, reg, hel]
    out = os.path.join(ROOT, "analysis", "mstate_probe", "1Nhp6A_mstate_md_survival.dat")
    with open(out, "w") as fh:
        fh.write(f"# m-state survival (PLAN.md decision 3): re-binding = n_core > {SURVIVAL_TH} "
                 f"sustained >= {SURVIVAL_HOLD} ns; times in ns, nan = not within the run\n"
                 "# t_strict_0.2ps: every colvar frame above; t_strict_10ps: every 10 ps frame "
                 "above; t_runmean_1ns: first 1-ns window whose mean is above; t_first_above: "
                 "first single frame (no hold).  All four reported, no primary (user, 2026-10-08)\n"
                 f"# t_first_pair_0.45nm: first frame with a core(27-93)-DNA heavy pair < "
                 f"{CONTACT_CUTOFF} nm, at {skip * 10} ps stride.  frac_above_after_strict: "
                 "fraction of frames above threshold after t_strict_0.2ps.  *_last10 = mean "
                 "over the last 10 ns\n"
                 f"# Bound-mode labels (PLAN.md, 2026-10-09), last 10 ns: bound = core-DNA heavy pair "
                 f"< {CONTACT_CUTOFF} nm in >= {BOUND_FRAC:g} of frames (frac_contact); mode = native / "
                 f"shifted / non-native from the best register-variant native Q (q_best >= {REGISTER_Q:g}; "
                 f"from {QNATIVE_TABLE}); register = variants within {REGISTER_TIE:g} of the best "
                 f"(flip_ = pseudo-dyad flip); helices = those with >= {HELIX_DOM:g} x the top helix's "
                 "contacts (h1-h3 = residue-pair contacts)\n")
        fh.write("# run  " + "  ".join(cols) + "\n")
        for r in rows:
            fh.write(f"{r[0]:8s} " + " ".join(f"{v:8.3f}" if isinstance(v, (float, np.floating))
                                               else f"{v:>8s}" for v in r[1:]) + "\n")
    print(open(out).read())
    print(f"  wrote {out}")


def core_dna_mindist(system, rep, skip):
    """Minimum heavy-atom distance between the core (27-93) and the DNA, per frame
    (closest-heavy over residue pairs = global heavy-atom minimum; min-image)."""
    cache = cache_path(system, rep, "mindist", skip)
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(run_src(system, rep)):
        d = np.genfromtxt(cache)
        return d[:, 0], d[:, 1]
    import mdtraj as md
    t = load_run(system, rep, skip)
    r0, _ = protein_copies(t)[0]
    core = [r.index for r in t.top.residues
            if r.is_protein and r.index >= r0 and PULL_CORE[0] <= r.resSeq <= PULL_CORE[1]]
    dna = [r.index for r in t.top.residues if r.name[:2] in DNA_PREFIXES]
    dists, _ = md.compute_contacts(t, np.array([(c, n) for c in core for n in dna]),
                                   scheme="closest-heavy", periodic=True,
                                   ignore_nonprotein=False)
    mind = dists.min(axis=1)
    np.savetxt(cache, np.column_stack([t.time, mind]),
               header=f"time_ns  min_core_dna_heavy_nm  (core {PULL_CORE[0]}-{PULL_CORE[1]}, "
                      "closest-heavy, min-image)")
    return t.time, mind


def read_colvar(repdir):
    """dict time_ns -> {de2e, bend, contacts} from a force-free colvar.dat."""
    path = os.path.join(repdir, "colvar.dat")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        names = fh.readline().split()[2:]  # drop "#!" and "FIELDS"
    d = np.genfromtxt(path, invalid_raise=False, usecols=range(len(names)))
    d = d[~np.isnan(d).any(axis=1)]        # a live run can leave a half-written last row
    col = {n: d[:, i] for i, n in enumerate(names)}
    out = {"time": col["time"] / 1000.0}   # ps -> ns
    if "de2e" in col:
        out["de2e"] = col["de2e"]
    if "theta" in col:
        out["bend"] = 180.0 - np.degrees(col["theta"])   # 0 deg = straight DNA
    if "cmap" in col:
        out["contacts"] = col["cmap"]
    if "coord_tail_dna" in col:
        out["n_idr"] = col["coord_tail_dna"]
    return out


def pmf_overlay(system, reps, version):
    """Equilibrium replicas (colvar n_IDR-DNA, bend) drawn on the MetaD F(n_IDR-DNA, bend):
    faint raw points every 0.2 ns + a line of OVL_BLOCK_NS block means, one colour per
    replica; circle = start, square = end.  Axes as article Panel E (bend x, n_IDR y)."""
    import article_plot as f5
    msys = PMF_OVERLAY[system]
    nc, bend, mean, std, cnt, cov = f5.load2d(f"{msys}_pmf2d_idrdna_bend.dat",
                                              data=os.path.join(ROOT, "analysis"), with_cov=True)
    F = f5.reliable_2d(mean, std, cnt, OVL_STD_CEIL, OVL_MIN_VISITS,
                       walker_cov=cov, min_walkers=OVL_MIN_WALKERS)
    with open(os.path.join(ROOT, "analysis", f"{msys}_pmf2d_idrdna_bend.dat")) as fh:
        src = fh.readline().strip("# \n")
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    Fm = np.ma.masked_invalid(F)
    im = ax.contourf(bend, nc, Fm, levels=np.linspace(0, 50, 21), cmap="RdYlBu_r", extend="max")
    ax.contour(bend, nc, Fm, levels=np.arange(5, 50, 5), colors="k", linewidths=0.35, alpha=0.45)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\Delta F$ (kJ mol$^{-1}$)")
    for k, r in enumerate(reps):
        cv = read_colvar(run_dir(system, r))
        if cv is None:
            print(f"  rep{r}: no colvar")
            continue
        t, x, y = cv["time"], cv["bend"], cv["n_idr"]
        dt = np.median(np.diff(t))
        raw = max(1, int(round(0.2 / dt)))
        nb = max(1, int(round(OVL_BLOCK_NS / dt)))
        m = len(t) // nb * nb
        bx, by = x[:m].reshape(-1, nb).mean(1), y[:m].reshape(-1, nb).mean(1)
        c = OVL_COLOR[k % len(OVL_COLOR)]
        ax.scatter(x[::raw], y[::raw], s=1, color=c, alpha=0.06, lw=0, rasterized=True)
        ax.plot(bx, by, color=c, lw=0.8, alpha=0.9,
                label=f"rep{r} ({t[-1]:.0f} ns): bend {np.median(x):.0f}°, "
                      f"n$_{{IDR}}$ {np.median(y):.1f}")
        ax.plot(bx[0], by[0], "o", ms=6, mfc=c, mec="k", mew=0.8)
        ax.plot(bx[-1], by[-1], "s", ms=6, mfc=c, mec="k", mew=0.8)
        print(f"  rep{r}: {t[-1]:.1f} ns, bend median {np.median(x):.1f} deg "
              f"(p5-p95 {np.percentile(x, 5):.1f}-{np.percentile(x, 95):.1f}), "
              f"n_IDR median {np.median(y):.1f} "
              f"(p5-p95 {np.percentile(y, 5):.1f}-{np.percentile(y, 95):.1f})")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 40)
    ax.set_box_aspect(1)
    ax.set_xlabel("DNA bend angle (deg)")
    ax.set_ylabel(r"$n_\mathrm{IDR-DNA}$")
    ax.set_title(f"{SYSTEMS[system]['title']}: unbiased MD on MetaD PMF\n"
                 f"{src.split(',')[-1].strip()}; lines = {OVL_BLOCK_NS:g} ns means", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    out = figpath(f"{system}_equil_on_pmf2d_idrdna_bend_{version}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def wt_atom_map(system):
    """WT PLUMED serial -> this system's serial, matched on (residue, atom name) with the
    system's residue k = WT k + wt_offset (WT residues inside the deletion have no entry).
    Residue names are checked so a wrong offset fails loudly instead of mis-mapping."""
    import mdtraj as md
    off = wt_offset(system)
    wt = md.load(os.path.join(ROOT, WT_TOPOL)).top
    tg = md.load(os.path.join(ROOT, TOPOL[system])).top
    tgt = {(a.residue.resSeq, a.name): (a.index + 1, a.residue.name)
           for a in tg.atoms if a.residue.is_protein}
    amap = {}
    for a in wt.atoms:
        if not a.residue.is_protein or a.residue.resSeq <= off + 1:
            continue
        key = (a.residue.resSeq - off, a.name)
        if key in tgt:
            serial, rname = tgt[key]
            if rname[-3:] != a.residue.name[-3:]:   # tolerate N/C-terminal prefixes
                sys.exit(f"{system}: WT {a.residue} maps onto {rname}{key[0]} -- wrong offset?")
            amap[a.index + 1] = serial
    return amap


def protein_cv_block(system="1Nhp6A_holo", workdir=None):
    """The tail/core/rmsd_core/q_core/n_tail_core/helix block from the apo MetaD input,
    with the rmsd reference made absolute and H3 extended 64-91 -> 64-93.

    For a deletion system (wt_offset > 0) every atom index is mapped onto its own
    numbering (wt_atom_map), the RMSD reference is rewritten into `workdir`, the helix
    RESIDUES are shifted, and the tail group is taken from the system's own MetaD input.
    """
    import re
    lines = open(PROTEIN_CV_SRC, encoding="latin-1").read().splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith("tail:"))
    i1 = next(i for i, l in enumerate(lines) if l.startswith("wall_core:"))
    block = "\n".join(lines[i0:i1])
    block = block.replace("RESIDUES=64-91", "RESIDUES=64-93")
    off = wt_offset(system)
    if not off:
        return block.replace("REFERENCE=../../../ref_core_ca_1Nhp6A_large.pdb",
                             f"REFERENCE={REF_CORE}")

    amap = wt_atom_map(system)
    remap = lambda m: m.group(1) + ",".join(str(amap[int(i)]) for i in m.group(2).split(","))
    src = open(os.path.join(ROOT, SYSTEMS[system]["tail_src"]), encoding="latin-1").read()
    tail = next(l for l in src.splitlines() if l.startswith("tail:"))
    out = []
    for l in block.splitlines():
        if l.startswith("tail:"):
            out.append(tail)
        else:
            l = re.sub(r"(ATOMS\d*=)([\d,]+)", remap, l)
            l = re.sub(r"RESIDUES=(\d+)-(\d+)",
                       lambda m: f"RESIDUES={int(m.group(1)) - off}-{int(m.group(2)) - off}", l)
            out.append(l)
    # RMSD reference: same CA atoms, renumbered to this system's serials / residues
    ref = os.path.join(workdir, f"ref_core_ca_{system}.pdb")
    with open(REF_CORE) as fi, open(ref, "w") as fo:
        for l in fi:
            if l.startswith(("ATOM", "HETATM")):
                l = (l[:6] + f"{amap[int(l[6:11])]:5d}" + l[11:22]
                     + f"{int(l[22:26]) - off:4d}" + l[26:])
            fo.write(l)
    return "\n".join(out).replace("REFERENCE=../../../ref_core_ca_1Nhp6A_large.pdb",
                                  f"REFERENCE={ref}")


PROTEIN_KEYS = ["q_core", "n_tail_core", "rmsd_core", "helix1", "helix2", "helix3"]


def protein_observables(system, rep, skip, copy=0):
    """Protein folding/helicity time series for one replica (one protein copy).

    plumed driver (over the pbc-whole protein slice) reproduces the biased-run CVs
    q_core, n_tail_core, rmsd_core, helix1/2/3; Rg(IDR) and total helicity are added
    from mdtraj.  Cached to data/equil/<system>/rep<rep>/equil_cvs[_copy<X>]_skip<skip>.dat.
    Returns dict with 'time' (ns) and one array per PROTEIN_PANELS entry.

    The CV block uses absolute PLUMED indices <= 1539, i.e. it is written for ONE copy.
    For the 2:1 system each copy is sliced out separately so that those indices land on
    the intended copy; the copies are never pooled (a pooled "residue 23 to 93" would
    measure the distance between the two boxes, not the size of either).
    """
    tag = "" if SYSTEMS[system]["ncopies"] < 2 else f"_copy{COPIES[copy]}"
    cache = cache_path(system, rep, "protein", skip, tag)
    src = run_src(system, rep)
    cols = ["time"] + PROTEIN_PANELS
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        d = np.genfromtxt(cache); d = d[~np.isnan(d).any(axis=1)]
        return {c: d[:, i] for i, c in enumerate(cols)}

    t = load_run(system, rep, skip)                   # non-water solute, pbc-whole, .time in ns
    # Split on the resSeq reset: the .gro topology has no chain records, so both copies
    # otherwise land in one chain and a residue-range selection would pool them.
    bounds = protein_copies(t)
    if len(bounds) != max(1, SYSTEMS[system]["ncopies"]):
        sys.exit(f"{system} rep{rep}: found {len(bounds)} protein copies, expected "
                 f"{SYSTEMS[system]['ncopies']}")
    r0, r1 = bounds[copy]
    prot = t.atom_slice(t.top.select(f"resid {r0} to {r1}"))  # 0-based index, per protein_copies
    rs = sorted({r.resSeq for r in prot.top.residues})
    entity = f"1-{SYSTEMS[system].get('prot_atoms', int(PROTEIN_ENTITY.split('-')[1]))}"
    if rs[0] != 1 or prot.n_atoms != int(entity.split("-")[1]):
        sys.exit(f"{system} rep{rep} copy {COPIES[copy]}: protein slice {prot.n_atoms} atoms, "
                 f"resSeq {rs[0]}..{rs[-1]} does not match expected {entity}")

    import mdtraj as md
    idr = SYSTEMS[system].get("idr_resid", IDR_RESID)
    rg_idr = md.compute_rg(prot.atom_slice(prot.top.select(f"resid {idr[0]} to {idr[1]}")))

    with tempfile.TemporaryDirectory() as tmp:
        pdb = os.path.join(tmp, "ref.pdb")
        xtc = os.path.join(tmp, "traj.xtc")
        dat = os.path.join(tmp, "cv.dat")
        out = os.path.join(tmp, "cv_out.dat")
        # mdtraj keeps the ORIGINAL atom serials through atom_slice, so a copy that does not
        # start at atom 1 (copy B here: serials 1540-3078) would be written with those serials
        # and MOLINFO would resolve the CV residues to indices past the end of the 1539-atom
        # slice ("atom 1992 out of range"). Renumber 1..N so serial == index in the sliced traj.
        for i, a in enumerate(prot.top.atoms):
            a.serial = i + 1
        prot[0].save_pdb(pdb)
        prot.save_xtc(xtc)
        with open(dat, "w") as fh:
            fh.write(f"MOLINFO STRUCTURE={pdb}\n")
            fh.write(f"WHOLEMOLECULES ENTITY0={entity}\n\n")
            fh.write(protein_cv_block(system, tmp) + "\n\n")
            fh.write(f"PRINT ARG={','.join(PROTEIN_KEYS)} FILE={out} STRIDE=1\n")
        r = subprocess.run([PLUMED, "driver", "--mf_xtc", xtc, "--plumed", dat],
                           cwd=tmp, capture_output=True, text=True)
        if not os.path.exists(out):
            sys.exit(f"plumed driver failed for {system} rep{rep}:\n{r.stderr[-2500:]}")
        d = np.genfromtxt(out); d = d[~np.isnan(d).any(axis=1)]

    pk = {k: d[:, 1 + i] for i, k in enumerate(PROTEIN_KEYS)}  # col0 = frame index
    res = {"time": prot.time,
           "q_core": pk["q_core"], "rmsd_core": pk["rmsd_core"],
           "n_tail_core": pk["n_tail_core"], "rg_idr": rg_idr,
           "total_hel": pk["helix1"] + pk["helix2"] + pk["helix3"],
           "helix1": pk["helix1"], "helix2": pk["helix2"], "helix3": pk["helix3"]}
    np.savetxt(cache, np.column_stack([res[c] for c in cols]),
               header="  ".join(cols) + "  (q_core/n_tail_core/rmsd_core/ALPHARMSD via "
                      "plumed driver on protein slice; helices 27-42,48-61,64-93)")
    return res


def helix_dna_contacts(system, rep, skip):
    """Per-frame count of (helix residue x DNA residue) closest-heavy-atom contacts
    below CONTACT_CUTOFF, for every protein copy and every helix (H1/H2/H3).

    Residue-pair count (matches medoid_contacts.py): a contact is a (helix residue,
    DNA residue) pair whose closest heavy-atom distance < 0.45 nm, so the count can
    exceed the number of helix residues.  Every copy is scored against the WHOLE
    duplex (both strands).

    Distances use minimum image (periodic=True), which is exact at 0.45 nm no matter
    how the duplex or the two copies are wrapped across periodic images, so this does
    NOT rely on a whole-molecule PBC reconstruction (unlike the cross-strand COM CVs).

    Returns {(copy, panel): (time_ns, counts)}.  Cached to
    data/equil/<system>/rep<rep>/helix_dna_contacts_skip<skip>.dat.
    """
    import mdtraj as md
    ncopies = SYSTEMS[system]["ncopies"]
    cache = cache_path(system, rep, "contacts", skip)
    src = run_src(system, rep)
    cols = ["time"] + [f"{COPIES[cp]}_{panel}"
                       for cp in range(ncopies) for panel in CONTACT_PANELS]
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        d = np.atleast_2d(np.genfromtxt(cache))
        d = d[~np.isnan(d).any(axis=1)]
        col = {c: d[:, i] for i, c in enumerate(cols)}
        return {(cp, panel): (col["time"], col[f"{COPIES[cp]}_{panel}"])
                for cp in range(ncopies) for panel in CONTACT_PANELS}

    t = load_run(system, rep, skip)                   # non-water solute, box preserved
    top = t.top
    bounds = protein_copies(t)
    if len(bounds) != ncopies:
        sys.exit(f"{system} rep{rep}: found {len(bounds)} protein copies, expected "
                 f"{ncopies} (helix-DNA contacts)")
    dna_res = [r.index for r in top.residues if r.name[:2] in DNA_PREFIXES]
    if not dna_res:
        sys.exit(f"{system} rep{rep}: no DNA residues found for helix-DNA contacts")

    out = {}
    for cp, (r0, r1) in enumerate(bounds):
        # helix residues of THIS copy: split by resSeq WITHIN the copy's residue-index span,
        # never by a bare resSeq range (that would pool both copies -- see protein_copies).
        copy_res = [r for r in top.residues if r0 <= r.index <= r1]
        pairs, segs = [], []
        for panel in CONTACT_PANELS:
            lo, hi = (x - wt_offset(system) for x in HELICES[panel])
            hres = [r.index for r in copy_res if lo <= r.resSeq <= hi]
            start = len(pairs)
            pairs.extend((h, dn) for h in hres for dn in dna_res)
            segs.append((panel, slice(start, len(pairs))))
        dists, _ = md.compute_contacts(t, np.array(pairs), scheme="closest-heavy",
                                       periodic=True, ignore_nonprotein=False)
        within = dists < CONTACT_CUTOFF
        for panel, sl in segs:
            out[(cp, panel)] = (t.time, within[:, sl].sum(axis=1).astype(float))

    stack = [t.time] + [out[(cp, panel)][1]
                        for cp in range(ncopies) for panel in CONTACT_PANELS]
    np.savetxt(cache, np.column_stack(stack),
               header="  ".join(cols) +
               f"  (residue-pair helix-DNA contacts, closest-heavy < {CONTACT_CUTOFF} nm, "
               "min-image; helices 27-42/48-61/64-93; copy vs whole duplex)")
    return out


def block(x, y, nblocks=250):
    w = max(1, len(y) // nblocks)
    n = len(y) // w * w
    return x[:n].reshape(-1, w).mean(1), y[:n].reshape(-1, w).mean(1)


def gather(system, cfg, reps, skip):
    """series[panel][(rep, copy)] = (t_ns, y).  copy is 0 for whole-system observables
    (the DNA CVs) and for single-copy systems."""
    series = {}
    for rep in reps:
        repdir = run_dir(system, rep)
        if not os.path.isdir(repdir):
            continue
        if cfg.get("mstate"):
            if is_pull(system):
                cv, panels = read_pull_colvar(rep), PULL_PANELS + ["d_target"]
            else:
                cv, panels = read_md_colvar(rep, system), MD_PANELS
            if cv:
                for p in panels:
                    if p in cv:
                        x = cv.get("contacts_time", cv["time"]) if p == "contacts" else cv["time"]
                        ys = cv[p] if isinstance(cv[p], tuple) else (cv[p],)
                        for copy, y in enumerate(ys):
                            series.setdefault(p, {})[(rep, copy)] = (x, y)
            if cfg["protein"]:   # colvar-only systems (2:1 Stage B) skip every trajectory pass
                print(f"  {system} {run_label(system, rep)}: min core-DNA heavy-atom distance "
                      "via mdtraj ...", flush=True)
                series.setdefault("core_dna_mind", {})[(rep, 0)] = core_dna_mindist(system, rep, skip)
        elif cfg["colvar"]:
            cv = read_colvar(repdir)
            if cv:
                for p in COLVAR_PANELS:
                    if p in cv:
                        series.setdefault(p, {})[(rep, 0)] = (cv["time"], cv[p])
        if cfg["protein"]:
            for copy in range(cfg["ncopies"]):
                lbl = f" copy {COPIES[copy]}" if cfg["ncopies"] > 1 else ""
                print(f"  {system} {run_label(system, rep)}{lbl}: protein CVs via plumed "
                      "driver ...", flush=True)
                po = protein_observables(system, rep, skip, copy)
                for p in PROTEIN_PANELS:
                    series.setdefault(p, {})[(rep, copy)] = (po["time"], po[p])
        if cfg["protein"] and cfg["dna"]:
            print(f"  {system} {run_label(system, rep)}: per-helix / DNA contacts via "
                  "mdtraj ...", flush=True)
            for (copy, panel), (x, y) in helix_dna_contacts(system, rep, skip).items():
                series.setdefault(panel, {})[(rep, copy)] = (x, y)
    return series


def make_figure(system, cfg, series, version):
    kind = cfg.get("mstate")
    pull, md_run = kind == "pull", kind == "md"
    first = PULL_PANELS if pull else MD_PANELS if md_run else COLVAR_PANELS
    hlines = cfg.get("hlines", PULL_HLINES if pull else MD_HLINES if md_run else {})
    bands = cfg.get("bands", PULL_BANDS if kind else {})
    panels = [p for p in first + PROTEIN_PANELS + CONTACT_PANELS if p in series]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.5 * nrow),
                             squeeze=False, sharex=True)
    tmax = 0.0
    multi = cfg["ncopies"] > 1
    for ax, p in zip(axes.flat, panels):
        if p in bands:
            lo, hi, txt = bands[p]
            ax.axhspan(lo, hi, color="0.88", lw=0, zorder=0, label=txt)
        if p in hlines:
            yv, txt = hlines[p]
            ax.axhline(yv, color="0.35", lw=0.8, ls=":", zorder=1, label=txt)
        for (rep, copy), (x, y) in sorted(series[p].items()):
            lab = run_label(system, rep)
            if pull:
                c = PULL_COLOR.get(rep, "0.3")
            elif md_run and multi:
                c = MD2_COLOR.get(rep, "0.3")
            elif md_run:
                c = PULL_COLOR.get(md_config(rep)[0], "0.3")
            else:
                c = REP_COLOR[rep % len(REP_COLOR)]
            # colour = replica/config (2:1 m-state: run), linestyle = protein copy (solid A,
            # dashed B), or for the 1:1 m-state production the seed (solid rep0, dashed rep1)
            ls = "-" if (md_config(rep)[1] if md_run and not multi else copy) == 0 else "--"
            if multi and p in PER_COPY_PANELS + MD_COPY_PANELS:
                lab += f" {COPIES[copy]}"
            ax.plot(x, y, lw=0.3, color=c, alpha=0.20)
            ax.plot(*block(x, y), lw=1.3, color=c, ls=ls, label=lab)
            if p == "d_core" and (rep, copy) in series.get("d_target", {}):
                ax.plot(*series["d_target"][(rep, copy)], lw=0.8, color=c, ls="--")
            tmax = max(tmax, x[-1])
        if p in hlines or p in bands:   # key the reference lines only
            ax.legend(frameon=False, fontsize="x-small",
                      handles=[h for h in ax.get_legend_handles_labels()[0]
                               if not is_series_label(h.get_label())])
        ax.set_ylabel(YLABEL[p])
        ax.set_xlabel("time (ns)")
    # legend on a protein panel when there are copies to distinguish, else the first panel
    leg_ax = axes.flat[panels.index(PROTEIN_PANELS[0])] if multi and PROTEIN_PANELS[0] in panels \
        else axes.flat[0]
    if kind:   # config legend on its own, so it never collides with the reference-line keys
        h, l = zip(*[(h, l) for h, l in zip(*leg_ax.get_legend_handles_labels())
                     if is_series_label(l)])
        fig.legend(h, l, loc="upper right", ncol=(len(l) + 1) // 2 if multi else len(l), frameon=False, fontsize="small")
    else:
        leg_ax.legend(frameon=False, fontsize="small", ncol=2 if multi else len(series[panels[0]]))
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    if pull:
        fig.suptitle(f"{cfg['title']} ({system}), configs overlaid, 0-{tmax:.0f} ns\n"
                     "dashed = moving-restraint target; thin = raw, thick = block mean",
                     x=0.01, ha="left", fontsize="medium")
    elif md_run:
        key = ("colour = run, solid/dashed = copy A/B" if multi
               else "colour = start config, solid/dashed = rep0/rep1")
        fig.suptitle(f"{cfg['title']} ({system}), 0-{tmax:.0f} ns\n"
                     f"{key}; thin = raw, thick = block mean", x=0.01, ha="left",
                     fontsize="medium")
    else:
        fig.suptitle(f"{cfg['title']}  ({system}) — unbiased equilibrium MD, "
                     f"reps overlaid (to {tmax:.0f} ns)")
    fig.tight_layout(rect=(0, 0, 1, 0.93 if multi else 0.975) if kind else None)
    stem = "timeseries" if kind else "equil_timeseries"
    out = figpath(f"{system}_{stem}_{version}.png", root=ROOT)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=sorted(SYSTEMS), help="default: all systems")
    p.add_argument("--reps", default=None,
                   help="comma-separated replica (pull: config) indices; default 0,1,2 "
                        "(pull: 0,1,3,4; mstate_md: run ids c<k>_rep<r>, default all 8)")
    p.add_argument("--skip", type=int, default=10, help="frame stride for the driver pass")
    p.add_argument("--version", default="v2")
    p.add_argument("--pmf-overlay", action="store_true",
                   help="draw the replicas' (bend, n_IDR-DNA) on the MetaD F(n_IDR-DNA, bend) "
                        f"instead of the time series (systems: {', '.join(PMF_OVERLAY)})")
    args = p.parse_args()

    if args.pmf_overlay:
        for s in ([args.system] if args.system else list(PMF_OVERLAY)):
            if s not in PMF_OVERLAY:
                print(f"=== {s}: no MetaD F(n_IDR-DNA, bend) table to overlay on, skipped")
                continue
            print(f"=== {s} on {PMF_OVERLAY[s]} ===")
            reps = [int(r) for r in args.reps.split(",")] if args.reps else [0, 1, 2]
            pmf_overlay(s, reps, args.version)
        return

    # the m-state runs are not part of the equilibrium overview: only on request
    systems = [args.system] if args.system else [s for s in SYSTEMS if not mstate(s)]
    for s in systems:
        if args.reps:
            reps = args.reps.split(",")
            reps = reps if mstate(s) == "md" else [int(r) for r in reps]
        else:
            reps = list(SYSTEMS[s].get("reps", (0, 1, 2)))
        print(f"=== {s} ({SYSTEMS[s]['title']}) ===")
        series = gather(s, SYSTEMS[s], reps, args.skip)
        if not series:
            print("  no data")
            continue
        make_figure(s, SYSTEMS[s], series, args.version)
        if mstate(s) == "md" and SYSTEMS[s]["ncopies"] > 1:
            survival_table_2to1(series, reps)
        elif mstate(s) == "md":
            survival_table(series, reps, args.skip)


if __name__ == "__main__":
    main()
