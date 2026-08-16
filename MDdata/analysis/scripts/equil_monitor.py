#!/usr/bin/env python
"""Health/progress monitor for the unbiased equilibrium MD campaign.

Works for any system/replica under data/equil/<system>/rep<N>/, whether the run
is finished or still writing. Reports thermodynamics from md.edr, the PLUMED
force-free CVs from colvar.dat (DNA-containing systems), and protein/DNA
structural observables computed from the trajectory.

The raw md.xtc has molecules broken across the periodic boundary, so any
trajectory-based observable is computed on a `trjconv -pbc whole` copy; without
this the HMG-box Rg/RMSD come out physically meaningless (Rg ~3 nm for a folded
71-residue domain).

Usage:
    equil_monitor.py --system 1Nhp6A_apo_large --rep 0
    equil_monitor.py --system 2Nhp6A --rep 0 --skip 20   # stride in frames
"""
import argparse
import os
import subprocess
import sys
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GMXRC = os.environ.get("GMXRC", "/opt/gromacs/2024.2-plumed/bin/GMXRC.bash")

# reference coordinate file per system, used as the mdtraj topology
TOPOL = {
    "1Nhp6A_apo_large": "data/1Nhp6A_apo_large/1Nhp6A_parambsc1_tip3p_ions0.15_npt.gro",
    "1Nhp6A_holo": "data/1Nhp6A/1Nhp6A_DNA_parambsc1_tip3p_ions0.15_npt.gro",
    "DNA": "data/DNA/DNA_parambsc1_tip3p_ions0.15_npt.gro",
    "2Nhp6A": "data/2Nhp6A/2Nhp6A_DNA_parambsc1_tip3p_ions0.15_npt.gro",
}
IDR_RESIDUES = "1 to 22"      # N-terminal basic disordered tail of Nhp6A
BOX_RESIDUES = "23 to 93"     # folded HMG box
COPIES = "ABCD"               # labels for the protein copies, in topology order
COPY_COLOR = {"A": "C0", "B": "C3", "C": "C2", "D": "C4"}

# Helix 3 as defined in the reference structure 1LWM: Pro64-Glu-Glu ... Thr91-Leu92-Ala93.
# This is the project's canonical definition -- do NOT re-derive helix bounds from DSSP on
# the trajectory. H3 is reported on its own because the whole-box helicity average dilutes
# it (30 of 71 residues), and because its characteristic motion is a bend about ~75-77 that
# leaves the i->i+4 hydrogen bonds intact and is therefore invisible to helicity alone.
H3 = (64, 93)
H3_KINK = ((64, 74), (78, 93))   # halves either side of the hinge, for the kink angle
CORE_RESIDUES = (23, 63)         # box N-terminal to H3; a fitting frame, not a helix definition

# The three HMG-box helices (1LWM; RESIDUES ranges of the apo ALPHARMSD CVs). Used for the
# per-helix / DNA contact panels. H3 above is the same range, kept separate for the helicity block.
HELICES = {"H1": (27, 42), "H2": (48, 61), "H3": (64, 93)}
CONTACT_CUTOFF = 0.45            # nm, closest-heavy-atom; project convention (medoid_contacts.py)
DNA_PREFIXES = ("DA", "DT", "DG", "DC")   # residue-name prefixes (also matches DA5/DG3 termini)


def gmx(args, stdin=None):
    """Run a gmx_mpi command inside a GMXRC-sourced shell."""
    cmd = f"set +u; source {GMXRC} >/dev/null 2>&1; set -u; gmx_mpi " + " ".join(args)
    return subprocess.run(["bash", "-c", cmd], input=stdin, capture_output=True, text=True)


def thermo(edr, begin):
    """Average T/P/density/volume/potential after `begin` ps."""
    terms = "Temperature\nPressure\nDensity\nVolume\nPotential\n"
    with tempfile.NamedTemporaryFile(suffix=".xvg") as tmp:
        r = gmx(["energy", "-f", edr, "-o", tmp.name, "-b", str(begin)], stdin=terms)
    out = {}
    for line in r.stdout.splitlines():
        f = line.split()
        if len(f) >= 5 and f[0] in ("Temperature", "Pressure", "Density", "Volume", "Potential"):
            out[f[0]] = tuple(float(x) for x in f[1:5])  # average, err.est, rmsd, drift
    return out


def whole_traj(repdir, skip):
    """trjconv -pbc whole on the solute; returns path to the cached xtc."""
    import mdtraj as md

    cache = os.path.join(repdir, f"whole_skip{skip}.xtc")
    if not os.path.exists(cache) or os.path.getmtime(cache) < os.path.getmtime(f"{repdir}/md.xtc"):
        r = gmx(["trjconv", "-f", f"{repdir}/md.xtc", "-s", f"{repdir}/md.tpr",
                 "-o", cache, "-pbc", "whole", "-skip", str(skip)], stdin="non-Water\n")
        if not os.path.exists(cache):
            sys.exit(f"trjconv failed:\n{r.stderr[-2000:]}")
    return cache


def block_means(t, y, width=25.0):
    edges = np.arange(0, t[-1] + width, width)
    return [y[(t >= a) & (t < a + width)].mean() for a in edges if ((t >= a) & (t < a + width)).any()]


def load_whole(system, rep, skip=10, root=ROOT):
    """Load the pbc-whole solute trajectory for data/equil/<system>/rep<rep>.

    Returns an mdtraj Trajectory whose .time is in NANOSECONDS (the raw md.xtc
    writes every 10 ps, so a stride of `skip` frames is 0.01*skip ns apart).
    """
    import mdtraj as md

    repdir = os.path.join(root, "data", "equil", system, f"rep{rep}")
    ref = md.load(os.path.join(root, TOPOL[system]))
    sub = ref.atom_slice(ref.top.select("not water"))  # must match trjconv's non-Water group
    t = md.load(whole_traj(repdir, skip), top=sub.top)
    t.time = np.arange(t.n_frames) * 0.01 * skip
    return t


def protein_copies(t):
    """First/last 0-based residue index of each protein copy in `t`.

    2Nhp6A carries two protein copies. The .gro topology has no chain records, so
    mdtraj puts both in one chain with resSeq restarting at 1 -- a bare
    "residue 23 to 93" would pool both HMG boxes and measure the distance between
    the copies, not their size. Split the copies on the resSeq reset and select on
    the 0-based residue index instead.
    """
    pres = [t.top.atom(i).residue for i in t.top.select("protein")]
    pres = sorted({r.index: r for r in pres}.values(), key=lambda r: r.index)
    starts = [k for k, r in enumerate(pres) if k == 0 or r.resSeq <= pres[k - 1].resSeq]
    return [(pres[s].index, pres[e - 1].index)
            for s, e in zip(starts, starts[1:] + [len(pres)])]


def structural_observables(t, r0):
    """Per-copy protein observables for the copy whose first residue index is r0.

    H3 (1LWM: Pro64...Ala93) is reported on its own because the whole-box helicity
    average dilutes it (30 of 71 residues), and because its characteristic motion is
    a bend about ~75-77 that leaves the i->i+4 hydrogen bonds intact and is therefore
    invisible to helicity alone -- hence the internal RMSD, the kink angle across the
    hinge, and the rigid-body displacement relative to the rest of the box alongside it.
    """
    import mdtraj as md

    lo_idr, hi_idr = (int(x) - 1 for x in IDR_RESIDUES.split(" to "))
    lo_box, hi_box = (int(x) - 1 for x in BOX_RESIDUES.split(" to "))
    box = f"resid {r0 + lo_box} to {r0 + hi_box}"
    idr = f"resid {r0 + lo_idr} to {r0 + hi_idr}"
    box_bb = t.top.select(f"{box} and backbone")
    # DSSP on a protein-only slice: with ions present the residue axis would
    # otherwise be padded with non-protein entries and dilute the fraction.
    prot = t.atom_slice(t.top.select(f"resid {r0 + lo_idr} to {r0 + hi_box}"))
    dssp = md.compute_dssp(prot, simplified=True)
    in_box = np.array([r.resSeq >= lo_box + 1 for r in prot.top.residues])
    h3_bb = t.top.select(f"resid {r0 + H3[0] - 1} to {r0 + H3[1] - 1} and backbone")
    core_bb = t.top.select(
        f"resid {r0 + CORE_RESIDUES[0] - 1} to {r0 + CORE_RESIDUES[1] - 1} and backbone")
    fit = md.Trajectory(t.xyz.copy(), t.top)
    fit.superpose(fit, 0, atom_indices=core_bb)
    h3_disp = np.sqrt(((fit.xyz[:, h3_bb] - fit.xyz[0, h3_bb]) ** 2).sum(-1).mean(-1))

    def helix_axis(lo, hi):
        x = t.xyz[:, t.top.select(f"resid {r0 + lo - 1} to {r0 + hi - 1} and name CA")]
        x = x - x.mean(axis=1, keepdims=True)
        return np.array([np.linalg.svd(f)[2][0] for f in x])

    cosang = (helix_axis(*H3_KINK[0]) * helix_axis(*H3_KINK[1])).sum(-1)
    out = {
        "Rg HMG box (nm)": md.compute_rg(t.atom_slice(t.top.select(box))),
        "RMSD HMG box (nm)": md.rmsd(t, t, 0, atom_indices=box_bb),
        "Rg IDR (nm)": md.compute_rg(t.atom_slice(t.top.select(idr))),
        "HMG helicity": (dssp[:, in_box] == "H").mean(axis=1),
        "H3 helicity": (dssp[:, H3[0] - 1:H3[1]] == "H").mean(axis=1),
        "H3 RMSD (nm)": md.rmsd(t, t, 0, atom_indices=h3_bb),
        "H3 displacement (nm)": h3_disp,
        "H3 kink (deg)": np.degrees(np.arccos(np.clip(np.abs(cosang), 0, 1))),
    }
    out.update(helix_dna_contacts(t, r0))
    return out


def helix_dna_contacts(t, r0):
    """Per-helix / DNA contact count for the protein copy whose first residue index is r0.

    A contact is a (helix residue x DNA residue) pair whose closest heavy-atom distance is
    < CONTACT_CUTOFF (residue-pair count, the medoid_contacts.py convention -- so the count
    can exceed the number of helix residues). The copy is scored against the WHOLE duplex.
    Distances are minimum image, exact at 0.45 nm however the duplex/copies are wrapped, so
    this needs no whole-molecule PBC reconstruction. Returns {} when there is no DNA (apo).
    """
    import mdtraj as md

    dna = [r.index for r in t.top.residues if r.name[:2] in DNA_PREFIXES]
    if not dna:
        return {}
    out = {}
    for name, (lo, hi) in HELICES.items():
        # residue index of resSeq s in this copy = r0 + (s-1) (resSeq restarts at 1 per copy)
        hres = [r0 + s - 1 for s in range(lo, hi + 1)]
        pairs = np.array([(h, dn) for h in hres for dn in dna])
        dists, _ = md.compute_contacts(t, pairs, scheme="closest-heavy",
                                       periodic=True, ignore_nonprotein=False)
        out[f"{name}-DNA contacts (<{CONTACT_CUTOFF:g} nm)"] = \
            (dists < CONTACT_CUTOFF).sum(axis=1).astype(float)
    return out


def main():
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True, choices=sorted(TOPOL))
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--skip", type=int, default=10, help="frame stride for trajectory analysis")
    p.add_argument("--equil", type=float, default=20.0, help="ns discarded from thermo averages")
    p.add_argument("--version", default="v1", help="figure version tag")
    args = p.parse_args()

    repdir = os.path.join(ROOT, "data", "equil", args.system, f"rep{args.rep}")
    if not os.path.isdir(repdir):
        sys.exit(f"no such replica: {repdir}")
    tag = f"{args.system}_rep{args.rep}"

    print(f"=== {tag} ===")
    done = os.path.exists(f"{repdir}/md.gro")
    print(f"status: {'COMPLETE' if done else 'RUNNING/partial'}")

    th = thermo(f"{repdir}/md.edr", args.equil * 1000)
    print(f"\nthermodynamics (t > {args.equil:g} ns)")
    print(f"{'term':12s} {'average':>12s} {'rmsd':>10s} {'drift':>12s}")
    for k, v in th.items():
        print(f"{k:12s} {v[0]:12.3f} {v[2]:10.3f} {v[3]:12.3f}")

    # Panels are keyed by axis label; per-copy observables (…_A/…_B) share an axis so
    # the two protein copies can be compared directly, one colour each.
    panels = {}  # axis label -> [(copy or None, time_ns, values), ...]

    def add_panel(label, copy, x, y):
        panels.setdefault(label, []).append((copy, x, y))

    colvar = f"{repdir}/colvar.dat"
    if os.path.exists(colvar):
        with open(colvar) as fh:
            names = fh.readline().split()[3:]  # drop "#! FIELDS time"
        # PLUMED is still appending to colvar.dat on a live run, so the final line can
        # be a half-written row; drop any row that doesn't have the full field count.
        d = np.genfromtxt(colvar, invalid_raise=False, usecols=range(len(names) + 1))
        d = d[~np.isnan(d).any(axis=1)]
        tc = d[:, 0] / 1000.0
        print(f"\nPLUMED CVs ({len(tc)} frames, 0 -> {tc[-1]:.1f} ns) | 25-ns block means")
        for i, n in enumerate(names):
            y = d[:, i + 1]
            # PLUMED writes theta as an ANGLE in radians (pi = straight DNA). Report it in
            # the project-wide bend convention instead -- bend = 180 - deg(theta), so 0 deg
            # = straight -- matching equil_timeseries.py, dna_bending_wall_compare.py and
            # Figure 5. Converting here covers both the printed block means and the panel.
            if n == "theta":
                n, y = "bend (deg)", 180.0 - np.degrees(y)
            bs = " ".join(f"{b:6.2f}" for b in block_means(tc, y))
            print(f"{n:20s} {y.mean():7.3f} +/- {y.std():5.3f} | {bs}")
            base, _, copy = n.rpartition("_")
            if base and copy in COPIES:
                add_panel(base, copy, tc, y)
            else:
                add_panel(n, None, tc, y)
        if "cmap" in names:
            cm = d[:, names.index("cmap") + 1]
            print(f"\nDNA integrity: cmap frac<0.90 = {100 * (cm < 0.90).mean():.2f}%, "
                  f"frac<0.80 = {100 * (cm < 0.80).mean():.3f}%")

    t = load_whole(args.system, args.rep, args.skip)
    copies = protein_copies(t)
    if copies:
        print(f"\nstructure ({t.n_frames} frames, {len(copies)} protein copy/copies) "
              "| 25-ns block means")
    for ci, (r0, _) in enumerate(copies):
        copy = COPIES[ci] if len(copies) > 1 else None
        suff = f" {copy}" if copy else ""
        for k, y in structural_observables(t, r0).items():
            bs = " ".join(f"{b:6.2f}" for b in block_means(t.time, y))
            print(f"{k + suff:22s} {y.mean():7.3f} +/- {y.std():5.3f} | {bs}")
            add_panel(k, copy, t.time, y)

    n = len(panels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow), squeeze=False)
    for ax, (label, series) in zip(axes.flat, panels.items()):
        for copy, x, y in series:
            color = COPY_COLOR.get(copy, "C0")
            ax.plot(x, y, lw=0.4, color=color, alpha=0.25)          # raw, faint
            w = max(1, len(y) // 200)
            ax.plot(x[: len(y) // w * w].reshape(-1, w).mean(1),    # block-averaged
                    y[: len(y) // w * w].reshape(-1, w).mean(1),
                    lw=1.4, color=color, label=copy)
        ax.set_xlabel("time (ns)")
        ax.set_ylabel(label)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize="small", ncol=len(series))
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{tag} — unbiased equilibrium MD ({t.time[-1]:.0f} ns)")
    fig.tight_layout()
    figdir = os.path.join(ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, f"{args.system}_equil_rep{args.rep}_{args.version}.png")
    fig.savefig(out, dpi=300)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
