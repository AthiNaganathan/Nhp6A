#!/usr/bin/env python
"""Pick start frames for the IDR-only (m-state) probe from the unbiased holo runs (PLAN.md, A1.1).

Frames are chosen so that the IDR *register* differs between them by selection, not by
placement. Each frame gets an IDR footprint: for every base pair k (strand-1 nt k paired with
strand-2 nt 16-k), the number of (IDR residue 1-20, nucleotide of bp k) residue pairs whose
closest heavy-atom distance is < 0.45 nm. Footprints are pooled over the replicas, the first
--equil ns of each is discarded, and they are clustered with k-means. Per cluster we pick the
frame nearest the centroid that is a canonical bound (M-mode) frame:

  * n_IDR-DNA (coord_tail_dna) within the pooled bound p5-p95,
  * core-DNA coordination (27-93) >= its pooled bound p5 (the core is properly engaged),
  * cmap (DNA inter-strand Q, from colvar.dat) >= 0.90.

Bound-state reference values (pooled median/p5/p95 of n_IDR-DNA and core-DNA coordination, and
the 25 % survival threshold) are written to the output as well.

For the 2:1 complex (Stage B, --system 2Nhp6A --copy B) the footprint and the IDR/core criteria
are those of the copy to be detached; the other copy must keep its core engaged (>= its own
pooled p5). Reference percentiles are computed per copy from the 2:1 runs themselves.

Usage:
    mstate_select_frames.py [--k 5] [--skip 10] [--equil 20] [--extract [--only 0 1 3 4]]
    mstate_select_frames.py --pdb 0 1 3 4     # solute PDBs of extracted frames, for viewing
    mstate_select_frames.py --system 2Nhp6A --copy B --reps 1 2 --k 4 [--extract]

--extract dumps the selected full-system frames (with solvent) from md.xtc to
data/mstate/start/c<k>.gro for the steered pulls.
"""
import argparse
import json
import os
import subprocess

import numpy as np

from equil_monitor import (ROOT, CONTACT_CUTOFF, DNA_PREFIXES, PROBE_IDR,
                           coordination, load_whole, probe_groups, protein_copies)

OUTDIR = os.path.join(ROOT, "analysis", "mstate_probe")
# per-system output names; the 1:1 names predate the traceable-prefix convention and are kept
OUTPUTS = {
    "1Nhp6A_holo": dict(json="start_frames.json", startdir=os.path.join("data", "mstate", "start"),
                        gro="c{c}.gro", pdb="start_c{c}.pdb"),
    "2Nhp6A": dict(json="2Nhp6A_mstate_start_frames.json",
                   startdir=os.path.join("data", "mstate_2Nhp6A", "start"),
                   gro="2Nhp6A_mstate_start_c{c}.gro", pdb="2Nhp6A_mstate_start_c{c}.pdb"),
}
GMX_PLAIN = "/usr/local/gromacs/2024.1/bin/gmx"
NBP = 15


def base_pairs(top):
    """Residue indices of each base pair, bp k = (strand-1 nt k, strand-2 nt 16-k), k = 1..15."""
    nts = [r for r in top.residues if r.name[:2] in DNA_PREFIXES]
    assert len(nts) == 2 * NBP, f"expected {2 * NBP} nucleotides, found {len(nts)}"
    s1, s2 = nts[:NBP], nts[NBP:]
    return [(s1[k].index, s2[NBP - 1 - k].index) for k in range(NBP)]


def footprint(t, r0=0):
    """(n_frames, 15) IDR residue-pair contact counts per base pair, for the copy starting at residue r0."""
    import mdtraj as md

    # by 0-based residue index: in 2Nhp6A resSeq restarts at 1 for copy B (see protein_copies)
    idr = list(range(r0 + PROBE_IDR[0] - 1, r0 + PROBE_IDR[1]))
    bps = base_pairs(t.top)
    pairs = np.array([(i, n) for i in idr for bp in bps for n in bp])
    d, _ = md.compute_contacts(t, pairs, scheme="closest-heavy", periodic=True,
                               ignore_nonprotein=False)
    c = (d < CONTACT_CUTOFF).reshape(t.n_frames, len(idr), NBP, 2)
    return c.sum(axis=(1, 3)).astype(float)


def write_view_pdbs(clusters, system, copy):
    """Solute-only PDBs of the extracted start frames, for visual inspection.

    The duplex is two molecules, so a whole-molecule unwrap is not enough: both strands are
    re-imaged next to the protein (image_molecules anchored on it). The reconstruction is then
    checked with NON-periodic distances: every Watson-Crick pair must be at H-bond range and the
    protein-DNA minimum distance must equal the minimum-image one. All frames are superposed on
    the DNA heavy atoms of the first one. B-factor = region (0 DNA, 1 IDR 1-20, 2 linker 21-26,
    3 core 27-93; 4 = the other protein copy in the 2:1) for colouring.
    """
    import mdtraj as md

    o = OUTPUTS[system]
    ref_top = md.load(os.path.join(ROOT, "data", "equil", system, "rep0", "md_proc.pdb")).topology
    n = ref_top.n_atoms
    sdir = os.path.join(ROOT, o["startdir"])
    trajs = []
    for c in clusters:
        g = md.load(os.path.join(sdir, o["gro"].format(c=c)))
        t = md.Trajectory(g.xyz[:, :n], ref_top, unitcell_lengths=g.unitcell_lengths,
                          unitcell_angles=g.unitcell_angles)
        prot = {a.index for a in t.top.atoms if a.residue.is_protein}
        mols = t.top.find_molecules()
        anchor = [m for m in mols if next(iter(m)).index in prot]
        t = t.image_molecules(anchor_molecules=anchor, make_whole=True)
        bps = base_pairs(t.top)
        # N1(purine)-N3(pyrimidine) of a WC pair is ~0.29 nm; take the min over the N1/N3 choices
        wc = []
        for i, j in bps:
            ai, aj = t.top.select(f"resid {i} and (name N1 or name N3)"), t.top.select(f"resid {j} and (name N1 or name N3)")
            wc.append(md.compute_distances(t, [(x, y) for x in ai for y in aj], periodic=False).min())
        heavy_p = t.top.select("protein and not element H")
        heavy_d = np.array([a.index for a in t.top.atoms if a.residue.name[:2] in DNA_PREFIXES and a.element.symbol != "H"])
        pd = np.array([(x, y) for x in heavy_p[::3] for y in heavy_d])
        nonper = md.compute_distances(t, pd, periodic=False).min()
        per = md.compute_distances(t, pd, periodic=True).min()
        print(f"c{c}: WC N1-N3 max {max(wc):.3f} nm (all 15 bp), protein-DNA min nonperiodic {nonper:.3f} "
              f"vs periodic {per:.3f} -> {'OK' if max(wc) < 0.35 and abs(nonper - per) < 1e-3 else 'CHECK'}")
        trajs.append(t)
    dna_heavy = trajs[0].top.select("not protein and not element H")
    region = np.zeros(n)
    r0, r1 = protein_copies(trajs[0])["AB".index(copy)]
    for a in trajs[0].top.atoms:
        if a.residue.is_protein:
            r = a.residue.resSeq
            region[a.index] = (1 if r <= PROBE_IDR[1] else (2 if r < 27 else 3)) if r0 <= a.residue.index <= r1 else 4
    for c, t in zip(clusters, trajs):
        t.superpose(trajs[0], 0, atom_indices=dna_heavy)
        out = os.path.join(OUTDIR, o["pdb"].format(c=c))
        t.save_pdb(out, bfactors=region)
        print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", default="1Nhp6A_holo", choices=sorted(OUTPUTS))
    p.add_argument("--copy", default="A", choices=["A", "B"],
                   help="protein copy to be detached (footprint + IDR/core criteria); 2Nhp6A only for B")
    p.add_argument("--k", type=int, default=5, help="number of start frames / clusters")
    p.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--skip", type=int, default=10, help="stride in 10-ps frames")
    p.add_argument("--equil", type=float, default=20.0, help="ns discarded per replica")
    p.add_argument("--seed", type=int, default=20261002, help="k-means seed")
    p.add_argument("--extract", action="store_true", help="dump full-system .gro of the picks")
    p.add_argument("--only", type=int, nargs="+", help="clusters to extract (default: all)")
    p.add_argument("--pdb", type=int, nargs="+", metavar="C",
                   help="only write view PDBs for already-extracted clusters, then exit")
    a = p.parse_args()
    o = OUTPUTS[a.system]
    if a.pdb:
        return write_view_pdbs(a.pdb, a.system, a.copy)
    from sklearn.cluster import KMeans

    rows = []
    for rep in a.reps:
        t = load_whole(a.system, rep, a.skip)
        copies = protein_copies(t)
        it = "AB".index(a.copy)
        g = probe_groups(t.top, copies[it][0])
        n_core_other = np.full(t.n_frames, np.nan)          # the other copy (2:1 only) must stay core-engaged
        if len(copies) == 2:
            go = probe_groups(t.top, copies[1 - it][0])
            n_core_other = coordination(t, go["core"], go["dna"])
        colvar = os.path.join(ROOT, "data", "equil", a.system, f"rep{rep}", "colvar.dat")
        with open(colvar) as fh:
            fields = fh.readline().split()[2:]                    # "#! FIELDS time ..."
        cv = np.loadtxt(colvar, comments="#")
        cmap = np.interp(t.time * 1000.0, cv[:, 0], cv[:, fields.index("cmap")])
        rows.append(dict(rep=np.full(t.n_frames, rep), time=t.time,
                         n_idr=coordination(t, g["idr"], g["dna"]),
                         n_core=coordination(t, g["core"], g["dna"]),
                         n_core_other=n_core_other, cmap=cmap, fp=footprint(t, copies[it][0])))
        print(f"rep{rep}: {t.n_frames} frames", flush=True)
    D = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    keep = D["time"] >= a.equil
    D = {k: v[keep] for k, v in D.items()}

    ref = {}
    for k in ("n_idr", "n_core"):
        ref[k] = dict(zip(("p5", "median", "p95"), np.percentile(D[k], [5, 50, 95]).round(3).tolist()))
    ref["survival_threshold_n_core"] = round(0.25 * ref["n_core"]["median"], 3)
    two_copies = not np.isnan(D["n_core_other"]).all()
    other_ok = np.ones(len(D["time"]), bool)
    if two_copies:
        ref["n_core_other_copy_p5"] = round(float(np.percentile(D["n_core_other"], 5)), 3)
        other_ok = D["n_core_other"] >= ref["n_core_other_copy_p5"]

    ok = ((D["n_idr"] >= ref["n_idr"]["p5"]) & (D["n_idr"] <= ref["n_idr"]["p95"])
          & (D["n_core"] >= ref["n_core"]["p5"]) & other_ok & (D["cmap"] >= 0.90))
    km = KMeans(n_clusters=a.k, n_init=20, random_state=a.seed).fit(D["fp"])
    picks = []
    for c in range(a.k):
        member = km.labels_ == c
        cand = np.flatnonzero(member & ok)
        dist = np.linalg.norm(D["fp"][cand] - km.cluster_centers_[c], axis=1)
        i = cand[np.argmin(dist)]
        reps, counts = np.unique(D["rep"][member], return_counts=True)
        picks.append(dict(cluster=c, population=round(float(member.mean()), 4),
                          reps_in_cluster={int(r): int(n) for r, n in zip(reps, counts)},
                          rep=int(D["rep"][i]), time_ns=round(float(D["time"][i]), 2),
                          n_idr=round(float(D["n_idr"][i]), 2), n_core=round(float(D["n_core"][i]), 2),
                          n_core_other=round(float(D["n_core_other"][i]), 2),
                          cmap=round(float(D["cmap"][i]), 4),
                          centroid=km.cluster_centers_[c].round(2).tolist(),
                          footprint=D["fp"][i].astype(int).tolist()))

    os.makedirs(OUTDIR, exist_ok=True)
    out = dict(system=a.system, copy=a.copy, reps=a.reps, skip=a.skip, equil_ns=a.equil, seed=a.seed,
               n_frames_pooled=int(len(D["time"])), n_frames_eligible=int(ok.sum()),
               bound_reference=ref, picks=picks)
    with open(os.path.join(OUTDIR, o["json"]), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "picks"}, indent=1))
    print("cluster pop   reps-in-cluster        pick(rep,t ns)  n_idr n_core other  cmap   footprint bp1..15")
    for q in picks:
        print(f"c{q['cluster']}  {q['population']:5.3f}  {str(q['reps_in_cluster']):22s} "
              f"rep{q['rep']} {q['time_ns']:7.1f}   {q['n_idr']:5.1f} {q['n_core']:5.1f}  "
              f"{q['n_core_other']:5.1f}  {q['cmap']:.3f}  {' '.join(f'{x:2d}' for x in q['footprint'])}")

    if a.extract:
        sdir = os.path.join(ROOT, o["startdir"])
        os.makedirs(sdir, exist_ok=True)
        for q in picks:
            if a.only is not None and q["cluster"] not in a.only:
                continue
            rd = os.path.join(ROOT, "data", "equil", a.system, f"rep{q['rep']}")
            out_gro = os.path.join(sdir, o["gro"].format(c=q["cluster"]))
            # plain (non-MPI) gmx: gmx_mpi hangs at MPI init when run without a terminal (setsid),
            # and trjconv needs neither MPI nor PLUMED
            cmd = (f"echo System | {GMX_PLAIN} trjconv -f {rd}/md.xtc "
                   f"-s {rd}/md.tpr -dump {q['time_ns'] * 1000:.0f} -o {out_gro}")
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
            print(f"{out_gro}: {'ok' if os.path.exists(out_gro) else 'FAILED ' + r.stderr[-500:]}")


if __name__ == "__main__":
    main()
