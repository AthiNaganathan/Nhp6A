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
biased runs / Figure 5:

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
this H3 Sa runs ~1.5 above Figure-5 panel-B helix3 by construction.

Reuses equil_monitor.load_whole for the trjconv -pbc whole + topology machinery.

Usage:
    equil_timeseries.py                       # all three systems
    equil_timeseries.py --system 1Nhp6A_holo  # just one
    equil_timeseries.py --skip 20             # coarser trajectory stride
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
from equil_monitor import ROOT, load_whole, protein_copies  # pbc-whole + topology + copy split

PLUMED = os.environ.get("PLUMED_BIN", "plumed")
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
}
COPIES = "AB"      # copy labels, in topology order (matches equil_monitor.COPIES)

# canonical HMG-box helices (1LWM; the RESIDUES ranges of the apo ALPHARMSD CVs). Used for
# the per-helix / DNA contact panels -- do NOT re-derive helix bounds from DSSP on the traj.
HELICES = {"h1_dna": (27, 42), "h2_dna": (48, 61), "h3_dna": (64, 93)}
CONTACT_CUTOFF = 0.45      # nm, closest-heavy-atom; project convention (medoid_contacts.py)
DNA_PREFIXES = ("DA", "DT", "DG", "DC")   # residue-name prefixes (also matches DA5/DG3 termini)

COLVAR_PANELS = ["de2e", "bend", "contacts"]
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
    return out


def protein_cv_block():
    """The tail/core/rmsd_core/q_core/n_tail_core/helix block from the apo MetaD input,
    with the rmsd reference made absolute and H3 extended 64-91 -> 64-93."""
    lines = open(PROTEIN_CV_SRC, encoding="latin-1").read().splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith("tail:"))
    i1 = next(i for i, l in enumerate(lines) if l.startswith("wall_core:"))
    block = "\n".join(lines[i0:i1])
    block = block.replace("REFERENCE=../../../ref_core_ca_1Nhp6A_large.pdb",
                          f"REFERENCE={REF_CORE}")
    block = block.replace("RESIDUES=64-91", "RESIDUES=64-93")
    return block


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
    repdir = os.path.join(ROOT, "data", "equil", system, f"rep{rep}")
    tag = "" if SYSTEMS[system]["ncopies"] < 2 else f"_copy{COPIES[copy]}"
    cache = os.path.join(repdir, CACHE.format(skip=skip, tag=tag))
    src = os.path.join(repdir, "md.xtc")
    cols = ["time"] + PROTEIN_PANELS
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        d = np.genfromtxt(cache); d = d[~np.isnan(d).any(axis=1)]
        return {c: d[:, i] for i, c in enumerate(cols)}

    t = load_whole(system, rep, skip)                 # non-water solute, pbc-whole, .time in ns
    # Split on the resSeq reset: the .gro topology has no chain records, so both copies
    # otherwise land in one chain and a residue-range selection would pool them.
    bounds = protein_copies(t)
    if len(bounds) != max(1, SYSTEMS[system]["ncopies"]):
        sys.exit(f"{system} rep{rep}: found {len(bounds)} protein copies, expected "
                 f"{SYSTEMS[system]['ncopies']}")
    r0, r1 = bounds[copy]
    prot = t.atom_slice(t.top.select(f"resid {r0} to {r1}"))  # 0-based index, per protein_copies
    rs = sorted({r.resSeq for r in prot.top.residues})
    if rs[0] != 1 or prot.n_atoms != int(PROTEIN_ENTITY.split("-")[1]):
        sys.exit(f"{system} rep{rep} copy {COPIES[copy]}: protein slice {prot.n_atoms} atoms, "
                 f"resSeq {rs[0]}..{rs[-1]} does not match expected {PROTEIN_ENTITY}")

    import mdtraj as md
    rg_idr = md.compute_rg(prot.atom_slice(prot.top.select(
        f"resid {IDR_RESID[0]} to {IDR_RESID[1]}")))

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
            fh.write(f"WHOLEMOLECULES ENTITY0={PROTEIN_ENTITY}\n\n")
            fh.write(protein_cv_block() + "\n\n")
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
    repdir = os.path.join(ROOT, "data", "equil", system, f"rep{rep}")
    cache = os.path.join(repdir, CCACHE.format(skip=skip))
    src = os.path.join(repdir, "md.xtc")
    cols = ["time"] + [f"{COPIES[cp]}_{panel}"
                       for cp in range(ncopies) for panel in CONTACT_PANELS]
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(src):
        d = np.atleast_2d(np.genfromtxt(cache))
        d = d[~np.isnan(d).any(axis=1)]
        col = {c: d[:, i] for i, c in enumerate(cols)}
        return {(cp, panel): (col["time"], col[f"{COPIES[cp]}_{panel}"])
                for cp in range(ncopies) for panel in CONTACT_PANELS}

    t = load_whole(system, rep, skip)                 # non-water solute, box preserved
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
            lo, hi = HELICES[panel]
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
        repdir = os.path.join(ROOT, "data", "equil", system, f"rep{rep}")
        if not os.path.isdir(repdir):
            continue
        if cfg["colvar"]:
            cv = read_colvar(repdir)
            if cv:
                for p in COLVAR_PANELS:
                    if p in cv:
                        series.setdefault(p, {})[(rep, 0)] = (cv["time"], cv[p])
        if cfg["protein"]:
            for copy in range(cfg["ncopies"]):
                lbl = f" copy {COPIES[copy]}" if cfg["ncopies"] > 1 else ""
                print(f"  {system} rep{rep}{lbl}: protein CVs via plumed driver ...", flush=True)
                po = protein_observables(system, rep, skip, copy)
                for p in PROTEIN_PANELS:
                    series.setdefault(p, {})[(rep, copy)] = (po["time"], po[p])
        if cfg["protein"] and cfg["dna"]:
            print(f"  {system} rep{rep}: per-helix / DNA contacts via mdtraj ...", flush=True)
            for (copy, panel), (x, y) in helix_dna_contacts(system, rep, skip).items():
                series.setdefault(panel, {})[(rep, copy)] = (x, y)
    return series


def make_figure(system, cfg, series, version):
    panels = [p for p in COLVAR_PANELS + PROTEIN_PANELS + CONTACT_PANELS if p in series]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.5 * nrow),
                             squeeze=False, sharex=True)
    tmax = 0.0
    multi = cfg["ncopies"] > 1
    for ax, p in zip(axes.flat, panels):
        for (rep, copy), (x, y) in sorted(series[p].items()):
            c = REP_COLOR[rep % len(REP_COLOR)]
            # colour = replica, linestyle = protein copy (solid A, dashed B)
            ls = "-" if copy == 0 else "--"
            lab = f"rep{rep}"
            if multi and p in PER_COPY_PANELS:
                lab += f" {COPIES[copy]}"
            ax.plot(x, y, lw=0.3, color=c, alpha=0.20)
            ax.plot(*block(x, y), lw=1.3, color=c, ls=ls, label=lab)
            tmax = max(tmax, x[-1])
        ax.set_ylabel(YLABEL[p])
        ax.set_xlabel("time (ns)")
    # legend on a protein panel when there are copies to distinguish, else the first panel
    leg_ax = axes.flat[panels.index(PROTEIN_PANELS[0])] if multi and PROTEIN_PANELS[0] in panels \
        else axes.flat[0]
    leg_ax.legend(frameon=False, fontsize="small", ncol=2 if multi else len(series[panels[0]]))
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{cfg['title']}  ({system}) — unbiased equilibrium MD, "
                 f"reps overlaid (to {tmax:.0f} ns)")
    fig.tight_layout()
    figdir = os.path.join(ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, f"{system}_equil_timeseries_{version}.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=sorted(SYSTEMS), help="default: all systems")
    p.add_argument("--reps", default="0,1,2", help="comma-separated replica indices")
    p.add_argument("--skip", type=int, default=10, help="frame stride for the driver pass")
    p.add_argument("--version", default="v2")
    args = p.parse_args()

    systems = [args.system] if args.system else list(SYSTEMS)
    reps = [int(r) for r in args.reps.split(",")]
    for s in systems:
        print(f"=== {s} ({SYSTEMS[s]['title']}) ===")
        series = gather(s, SYSTEMS[s], reps, args.skip)
        if not series:
            print("  no data")
            continue
        make_figure(s, SYSTEMS[s], series, args.version)


if __name__ == "__main__":
    main()
