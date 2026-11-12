#!/usr/bin/env python3
"""PBC check for the m-state probe production runs (PLAN.md A2): is the detached / re-binding
core interacting with the IDR's own DNA, or with a periodic image of it?

Why a separate check. In the m-state the protein is stretched between an IDR bound to the DNA and
a core that has swung out (d_core up to ~4.8 nm against a half box vector of 4.6 nm), so the
complex is far less compact than the holo one. Two things can then go wrong, and they look the
same in `gmx mindist -pi`:
  * physics:   the core touches a DNA image while the IDR holds the original copy (bridging);
  * placement: `trjconv -pbc cluster` (and the -pbc mol pass after it) put the protein, or one
               DNA strand, in a different cell from its partner, so ordinary IDR-DNA or
               base-pair contacts are scored as image contacts.
The second is what c3 rep0 showed at 7.2-8.3 ns (2026-10-06). So this script never trusts the
cell trjconv chose. Per frame it (1) welds strand B onto strand A (lattice shift minimising the
inter-strand distance), (2) puts the protein on the lattice shift that minimises IDR-DNA
distance, and only then (3) measures the complex-to-own-image heavy-atom distance and (4) the
core's distance to the IDR's DNA copy versus the nearest other DNA image.

Inputs (all on rubisco after syncing from the run host):
  data/mstate/md/c<k>/rep<r>/colvar_1Nhp6A_mstate_md_c<k>_rep<r>.dat     PLUMED monitor
  data/mstate/md/c<k>/rep<r>/1Nhp6A_mstate_md_c<k>_rep<r>_vmd_ref.pdb    Protein_DNA topology
  analysis/pbc/mstate_c<k>/mindist_pi_r<r>_dt10ps.xvg                   gmx mindist -pi (ns)
  analysis/pbc/mstate_c<k>/mindist_pi_r<r>_dt10ps_proc.xtc              clustered solute
both written by `pbc_mindist_check.sh -m equil -d data/mstate/md/c<k> -o analysis/pbc/mstate_c<k>
-s Protein_DNA -r <r> -D 10 -k` on the run host.

Outputs:
  figures/1Nhp6A/1Nhp6A_mstate_md_pbc_check_v<N>.png       one row per run, three panels
  analysis/pbc/1Nhp6A_mstate_md_pbc_summary.dat     per-run numbers quoted in the notebook

Usage:
  mstate_pbc_check.py --runs c0/rep0 c0/rep1 c1/rep0 c1/rep1 c3/rep0 --version v1
  # faster: fill the per-run caches in parallel first, then build the figure from them
  for x in c0/rep0 c0/rep1 c1/rep0 c1/rep1 c3/rep0; do mstate_pbc_check.py --runs $x --cache-only & done; wait

2:1 copy-B pulls (--system 2Nhp6A_pull, PLAN.md B1). Stage B runs in the existing, already
marginal 2:1 box (8.52 nm octahedron), so the PBC check is a gate before re-solvation. The solute
is four molecules (copy A 1-1539, copy B 1540-3078, DNA strand A, strand B). Same idea as above,
one more placement step: weld the strands, put copy A on the shift minimising its distance to the
DNA (its core is bound), put copy B on the shift minimising ITS IDR-DNA distance (the IDR is held
on by the wall), then measure the rebuilt complex against its own images and ask whether copy B's
core prefers a DNA image or an image of copy A. Reads the raw pull xtc directly (solute atoms
only, every 10 ps frame) after `trjconv -pbc mol` has made each molecule whole: that pass picks
no cell between molecules, so there is no placement to second-guess, and no gmx xvg.
  mstate_pbc_check.py --system 2Nhp6A_pull --runs c0 c1 c2 c3 --version v1
  mstate_pbc_check.py --system 2Nhp6A_pull --runs c0 c1 c2 c3 --write-vmd   # placed viewing xtc
  mstate_pbc_check.py --system 2Nhp6A_mstate_md --runs c1/rep0 c3/rep0 --write-vmd   # Stage B production
"""
import argparse
import itertools
import os

import numpy as np
import mdtraj as md
from scipy.spatial import cKDTree
from figpaths import figpath
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P = "1Nhp6A_mstate_md_c{k}_rep{r}"
N_PROT = 1539             # protein atoms 1-1539, DNA 1540-2489 (both strands, 15 nt each)
IDR_MAX, CORE_MIN = 20, 27   # PLAN.md decision 2: IDR 1-20, linker 21-26, core 27-93
CONTACT = 0.45            # nm, heavy-atom contact (as in the gate)
D_BOUND = 1.56            # nm, d_core at the start of the pulls (bound reference)
SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
ZERO = int(np.where(~SHIFTS.any(1))[0][0])


def read_colvar(path):
    with open(path) as fh:
        names = fh.readline().split()[2:]
    d = np.genfromtxt(path, invalid_raise=False, usecols=range(len(names)))
    d = d[~np.isnan(d).any(axis=1)]          # live run: half-written last row
    return {n: d[:, i] for i, n in enumerate(names)}


def selections(top):
    """Heavy-atom index arrays for IDR, core, DNA strand A and strand B (by residue order: the
    trjconv pdb has no usable chain records)."""
    prot, dna = [], []
    for res in top.residues:
        (prot if res.atom(0).index < N_PROT else dna).append(res)
    heavy = lambda rs: np.array([a.index for r in rs for a in r.atoms if a.element.symbol != "H"])
    half = len(dna) // 2
    return dict(idr=heavy(prot[:IDR_MAX]), core=heavy(prot[CORE_MIN - 1:]),
                prot=heavy(prot), A=heavy(dna[:half]), B=heavy(dna[half:]))


CAP = 2.5                 # nm, neighbour-search cap: only short distances matter; farther -> CAP


def _bounded(tree, pts, cap=CAP):
    d = tree.query(pts, distance_upper_bound=cap)[0]
    return min(d.min(), cap)


def frame_geometry(X, box, sel):
    """Placement-independent per-frame numbers (see module docstring). Distances beyond CAP are
    returned as CAP."""
    lat = SHIFTS @ box                                       # 27 lattice translations
    # (1) weld strand B onto strand A: the paired strands' COMs sit < 1 nm apart, so the lattice
    #     shift that brings B's centroid nearest A's is the welding shift
    gA, gB = X[sel["A"]].mean(0), X[sel["B"]].mean(0)
    sB = int(np.argmin(np.linalg.norm(gB + lat - gA, axis=1)))
    dna = np.vstack([X[sel["A"]], X[sel["B"]] + lat[sB]])
    trD = cKDTree(dna)
    weld = _bounded(cKDTree(X[sel["A"]]), X[sel["B"]] + lat[sB])
    # (2) the IDR's DNA copy = lattice shift of the protein minimising IDR-DNA distance
    dI = [_bounded(trD, X[sel["idr"]] + v, 1.5) for v in lat]
    b = int(np.argmin(dI))
    prot = X[sel["prot"]] + lat[b]
    core = X[sel["core"]] + lat[b]
    # (4) core -> IDR's copy vs nearest other DNA image
    dC = np.array([_bounded(trD, core - v) for v in lat])
    same, other = dC[ZERO], np.delete(dC, ZERO).min()
    # (3) true complex-to-own-image distance on the rebuilt complex
    cplx = np.vstack([prot, dna])
    trC = cKDTree(cplx)
    img = min(_bounded(trC, cplx + v) for k, v in enumerate(lat) if k != ZERO)
    return dI[b], same, other, img, weld


def selections_2to1(top):
    """2:1 heavy-atom selections by atom index (copy A, copy B, then the two strands)."""
    def heavy(lo, hi, rmin=1, rmax=10**6):
        return np.array([a.index for a in top.atoms if lo <= a.index < hi and a.element.symbol != "H"
                         and rmin <= a.residue.resSeq <= rmax])
    n = N_PROT
    dna = [r for r in top.residues if r.atom(0).index >= 2 * n]
    s2 = dna[len(dna) // 2].atom(0).index
    return dict(protA=heavy(0, n), idrB=heavy(n, 2 * n, 1, IDR_MAX), coreB=heavy(n, 2 * n, CORE_MIN),
                protB=heavy(n, 2 * n), A=heavy(2 * n, s2), B=heavy(s2, dna[-1].atom(-1).index + 1))


def place_2to1(X, box, sel):
    """The 2:1 placement shared by the PBC check and the viewing trajectories: weld strand B onto
    strand A, then copy A on the shift minimising its distance to the DNA, copy B on the shift
    minimising its IDR-DNA distance. Returns (lattice shifts, strand-B / copy-A / copy-B shift
    indices, welded DNA heavy atoms, its tree, copy-A and IDR-B distances per shift)."""
    lat = SHIFTS @ box
    gA, gB = X[sel["A"]].mean(0), X[sel["B"]].mean(0)
    sB = int(np.argmin(np.linalg.norm(gB + lat - gA, axis=1)))
    dna = np.vstack([X[sel["A"]], X[sel["B"]] + lat[sB]])
    trD = cKDTree(dna)
    dA = [_bounded(trD, X[sel["protA"]] + v, 1.5) for v in lat]
    dI = [_bounded(trD, X[sel["idrB"]] + v, 1.5) for v in lat]
    return lat, sB, int(np.argmin(dA)), int(np.argmin(dI)), dna, trD, dA, dI


def write_vmd_2to1(spec, system="2Nhp6A_pull", stride_ps=100.0):
    """Viewing trajectory for one 2:1 run with the PBC check's explicit placement applied to
    every atom of each molecule (`-pbc cluster` misplaces c0/c2): every stride_ps, plus the
    placed first and last frames as PDBs, next to the run outputs in data/.
    spec: "c<k>" (2Nhp6A_pull) or "c<k>/rep<r>" (2Nhp6A_mstate_md, Stage B production; solute
    copies from run_solute_whole.sh on the run host, -pbc mol every 100 ps, synced to
    analysis/pbc/mstate_2Nhp6A_md/)."""
    if system == "2Nhp6A_pull":
        run = f"2Nhp6A_mstate_pull_{spec}"
        pdir = os.path.join(ROOT, "analysis", "pbc", "mstate_2Nhp6A_pull")
        rdir = os.path.join(ROOT, "data", "mstate_2Nhp6A", "pull", spec)
        ref = os.path.join(pdir, "2Nhp6A_mstate_pull_solute_ref.pdb")
    else:
        k, r = spec.split("/")
        run = f"2Nhp6A_mstate_md_{k}_{r}"
        pdir = os.path.join(ROOT, "analysis", "pbc", "mstate_2Nhp6A_md")
        rdir = os.path.join(ROOT, "data", "mstate_2Nhp6A", "md", k, r)
        ref = os.path.join(pdir, f"2Nhp6A_mstate_md_{k}_solute_ref.pdb")
    t = md.load(os.path.join(pdir, f"{run}_solute_whole.xtc"), top=ref)
    step = int(round(stride_ps / (t.time[1] - t.time[0])))
    t = t[::step]
    sel = selections_2to1(t.topology)
    n = N_PROT
    s2 = sel["B"].min()      # first atom of strand B (heavy selections start at a heavy atom)
    s2 = next(r.atom(0).index for r in t.topology.residues if r.atom(0).index <= s2 <= r.atom(-1).index)
    assert (t.n_atoms, s2) == (4028, 2 * n + 484), "2:1 solute layout (copy A, copy B, strands 484+466)"
    mol = dict(protA=slice(0, n), protB=slice(n, 2 * n), B=slice(s2, t.n_atoms))
    for f in range(t.n_frames):
        lat, sB, a, b = place_2to1(t.xyz[f], t.unitcell_vectors[f], sel)[:4]
        t.xyz[f, mol["protA"]] += lat[a]
        t.xyz[f, mol["protB"]] += lat[b]
        t.xyz[f, mol["B"]] += lat[sB]
        # -pbc mol leaves strand A in whichever cell its COM fell in, so the whole complex can
        # jump by a box vector between frames: recentre the welded duplex on the box centre
        dna = np.vstack([t.xyz[f, sel["A"]], t.xyz[f, sel["B"]]])
        t.xyz[f] += 0.5 * t.unitcell_vectors[f].sum(0) - dna.mean(0)
    out = os.path.join(rdir, f"{run}_vmd")
    t.save_xtc(out + ".xtc")
    t[0].save_pdb(out + "_ref.pdb")
    t[-1].save_pdb(out + "_end.pdb")
    print(f"  wrote {out}.xtc ({t.n_frames} frames, {t.time[0]/1000:.1f}-{t.time[-1]/1000:.1f} ns), "
          f"{os.path.basename(out)}_ref.pdb, _end.pdb", flush=True)


def img_allatom_2to1(k):
    """(time_ns, complex->own-image distance over ALL solute atoms incl. H) per 10 ps frame, with
    the place_2to1 placement; cached. The heavy-atom 'img' of analyse_pull_2to1 can sit up to
    ~0.2 nm above it (H on both sides), so the 0.9 nm cut-off test must use this one."""
    run = f"2Nhp6A_mstate_pull_c{k}"
    pdir = os.path.join(ROOT, "analysis", "pbc", "mstate_2Nhp6A_pull")
    cache = os.path.join(pdir, f"{run}_img_allatom.npy")
    if not os.path.exists(cache):
        t = md.load(os.path.join(pdir, f"{run}_solute_whole.xtc"),
                    top=os.path.join(pdir, "2Nhp6A_mstate_pull_solute_ref.pdb"))
        sel = selections_2to1(t.topology)
        n, s2 = N_PROT, 2 * N_PROT + 484
        d = []
        for f in range(t.n_frames):
            X = t.xyz[f].copy()
            lat, sB, a, b = place_2to1(X, t.unitcell_vectors[f], sel)[:4]
            X[:n] += lat[a]; X[n:2 * n] += lat[b]; X[s2:] += lat[sB]
            tr = cKDTree(X)
            d.append(min(_bounded(tr, X + v) for j, v in enumerate(lat) if j != ZERO))
        np.save(cache, np.array([t.time / 1000.0, d]))
    return np.load(cache)


def write_start_2to1(k, cutoff=0.9):
    """Stage B start structure (user, 2026-10-12): the last frame BEFORE the complex first comes
    within `cutoff` (= rcoulomb = rvdw) of its own periodic image, so no direct image interaction
    has ever shaped it (the last clean frame overall can follow an image contact). Placed with
    place_2to1 and recentred on the duplex; written as a solute-only gro for re-solvation."""
    run = f"2Nhp6A_mstate_pull_c{k}"
    pdir = os.path.join(ROOT, "analysis", "pbc", "mstate_2Nhp6A_pull")
    R = analyse_pull_2to1(k)                       # cached per-frame geometry (every 10 ps frame)
    ta, img = img_allatom_2to1(k)
    assert len(ta) == len(R["t"]) and np.allclose(ta, R["t"]), "all-atom scan / cache frames differ"
    bad = np.flatnonzero(img < cutoff)
    i = bad[0] - 1 if len(bad) else len(img) - 1
    assert i >= 0, f"c{k}: image contact in the first frame"
    # standard_names=False: keep the GROMACS atom names (mdtraj otherwise renames ~400 atoms to
    # its PDB conventions and grompp flags them as non-matching against the topology)
    top = md.load_pdb(os.path.join(pdir, "2Nhp6A_mstate_pull_solute_ref.pdb"),
                      standard_names=False).topology
    t = md.load_frame(os.path.join(pdir, f"{run}_solute_whole.xtc"), i, top=top)
    assert abs(t.time[0] - 1000.0 * R["t"][i]) < 0.1, f"frame/time mismatch: xtc {t.time[0]} ps vs cache {1000.0 * R['t'][i]} ps"
    sel = selections_2to1(t.topology)
    n = N_PROT
    s2 = next(r.atom(0).index for r in t.topology.residues if r.atom(0).index <= sel["B"].min() <= r.atom(-1).index)
    assert (t.n_atoms, s2) == (4028, 2 * n + 484), "2:1 solute layout (copy A, copy B, strands 484+466)"
    X, box = t.xyz[0], t.unitcell_vectors[0]
    lat, sB, a, b = place_2to1(X, box, sel)[:4]
    X[0:n] += lat[a]
    X[n:2 * n] += lat[b]
    X[s2:] += lat[sB]
    X += 0.5 * box.sum(0) - np.vstack([X[sel["A"]], X[sel["B"]]]).mean(0)
    g = frame_geometry_2to1(X, box, sel)
    trX = cKDTree(X)
    g_all = min(_bounded(trX, X + v) for j, v in enumerate(SHIFTS @ box) if j != ZERO)
    out = os.path.join(ROOT, "data", "mstate_2Nhp6A", "pull", f"c{k}", f"{run}_start.gro")
    t.save_gro(out)
    first = f"{ta[bad[0]]:.2f} ns" if len(bad) else "never"
    print(f"c{k}: first all-atom image contact < {cutoff} nm at {first}; start frame t = {R['t'][i]:.2f} ns -> {out}\n"
          f"     re-measured: complex->own image all-atom {g_all:.3f} / heavy {g[3]:.3f} nm, coreB->DNA {g[1]:.3f}, "
          f"coreB->copy A {g[5]:.3f}, IDR_B->DNA {g[0]:.3f}, copy A->DNA {g[7]:.3f}, weld {g[4]:.3f}", flush=True)


def frame_geometry_2to1(X, box, sel):
    """2:1 version of frame_geometry: returns (IDR_B-DNA, coreB->own DNA, coreB->other DNA image,
    complex->own image, strand weld, coreB->copy A as placed, coreB->other copy-A image,
    copy A->DNA). Distances beyond CAP are returned as CAP."""
    lat, sB, a, b, dna, trD, dA, dI = place_2to1(X, box, sel)
    weld = _bounded(cKDTree(X[sel["A"]]), X[sel["B"]] + lat[sB])
    protA = X[sel["protA"]] + lat[a]
    protB, coreB = X[sel["protB"]] + lat[b], X[sel["coreB"]] + lat[b]
    dC = np.array([_bounded(trD, coreB - v) for v in lat])
    trA = cKDTree(protA)
    dP = np.array([_bounded(trA, coreB - v) for v in lat])
    cplx = np.vstack([protA, protB, dna])
    trC = cKDTree(cplx)
    img = min(_bounded(trC, cplx + v) for k, v in enumerate(lat) if k != ZERO)
    return (dI[b], dC[ZERO], np.delete(dC, ZERO).min(), img, weld,
            dP[ZERO], np.delete(dP, ZERO).min(), dA[a])


def analyse_pull_2to1(k):
    """One 2:1 copy-B pull, read straight from the raw xtc (solute atoms only)."""
    run = f"2Nhp6A_mstate_pull_c{k}"
    rdir = os.path.join(ROOT, "data", "mstate_2Nhp6A", "pull", f"c{k}")
    pdir = os.path.join(ROOT, "analysis", "pbc", "mstate_2Nhp6A_pull")
    os.makedirs(pdir, exist_ok=True)
    cv = read_colvar(os.path.join(rdir, f"colvar_{run}.dat"))
    cache = os.path.join(pdir, f"{run}_pbc_geometry.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        g, time, box = z["g"], z["time"], z["box"]
    else:
        # the raw mdrun xtc has molecules split across the box, so read the solute-only
        # `trjconv -pbc mol` copy (each molecule whole via the tpr bonds, no cell chosen between
        # molecules): <pdir>/<run>_solute_whole.xtc + 2Nhp6A_mstate_pull_solute_ref.pdb
        t = md.load(os.path.join(pdir, f"{run}_solute_whole.xtc"),
                    top=os.path.join(pdir, "2Nhp6A_mstate_pull_solute_ref.pdb"))
        sel = selections_2to1(t.topology)
        g = np.array([frame_geometry_2to1(t.xyz[f], t.unitcell_vectors[f], sel) for f in range(t.n_frames)])
        time, box = t.time, t.unitcell_vectors[0]
        np.savez(cache, g=g, time=time, box=box)
    names = ("idr", "same", "other", "img", "strand", "coreA", "coreA_other", "protA_dna")
    R = dict(run=run, k=k, cv=cv, t=time / 1000.0, half=0.5 * np.linalg.norm(box[0]))
    R.update({n: g[:, i] for i, n in enumerate(names)})
    return R


def summarise_2to1(R):
    tc = R["cv"]["time"] / 1000.0
    late = R["t"] >= R["t"][-1] - 5.0
    return dict(run=R["run"], t_ns=tc[-1], dcore_max=R["cv"]["d_core"].max(), half_box=R["half"],
                dcore_maxstep=np.abs(np.diff(R["cv"]["d_core"])).max(),
                true_pi_min=R["img"].min(), true_pi_lt09=int((R["img"] < 0.9).sum()),
                true_pi_lt1=int((R["img"] < 1.0).sum()),
                other_min=R["other"].min(), core_closer_other=int((R["other"] < R["same"]).sum()),
                coreA_other_min=R["coreA_other"].min(),
                coreB_closer_Aimage=int((R["coreA_other"] < R["coreA"]).sum()),
                coreB_dna_last5=np.median(R["same"][late]), coreB_A_last5=np.median(R["coreA"][late]),
                idr_unbound=int((R["idr"] > CONTACT).sum()), protA_dna_max=R["protA_dna"].max(),
                strand_max=R["strand"].max(), nframes=len(R["t"]))


def make_figure_2to1(runs, out):
    fig, ax = plt.subplots(len(runs), 3, figsize=(16, 2.9 * len(runs)), squeeze=False)
    for i, R in enumerate(runs):
        tc = R["cv"]["time"] / 1000.0
        a = ax[i, 0]
        a.plot(tc[::50], R["cv"]["d_core"][::50], color="#2a78d6", lw=0.6, label="d_core copy B (PLUMED)")
        a.axhline(R["half"], color="#e34948", ls="--", lw=1, label=f"half box vector {R['half']:.2f} nm")
        a.set_ylabel(f"c{R['k']}\nd_core (nm)")
        a = ax[i, 1]
        a.plot(R["t"], R["img"], color="k", lw=0.6, label="2:1 complex rebuilt → own image")
        a.axhline(0.9, color="#e34948", ls="--", lw=1, label="cut-off 0.9 nm")
        a.set_ylabel(f"solute–own image\nmin dist (nm, cap {CAP})")
        a.set_ylim(0, None)
        a = ax[i, 2]
        a.plot(R["t"], R["same"], color="#2a78d6", lw=0.6, label="core B → own DNA")
        a.plot(R["t"], R["other"], color="#eb6834", lw=0.6, label="core B → other DNA image")
        a.plot(R["t"], R["coreA"], color="#3a9e5f", lw=0.6, label="core B → copy A")
        a.plot(R["t"], R["coreA_other"], color="#8e5bc2", lw=0.6, label="core B → other copy-A image")
        a.axhline(CONTACT, color="0.4", ls=":", lw=1, label=f"contact {CONTACT} nm")
        a.set_ylabel(f"core B heavy\nmin dist (nm, cap {CAP})")
        a.set_ylim(0, None)
        for a in ax[i]:
            a.grid(alpha=0.25, lw=0.5)
            a.set_xlim(0, tc[-1])
        if i == 0:
            ax[0, 0].set_title("a  copy-B core–DNA COM distance (pulled)", loc="left", fontsize=10)
            ax[0, 1].set_title("b  periodic-image distance of the rebuilt complex", loc="left", fontsize=10)
            ax[0, 2].set_title("c  does core B prefer an image (DNA or copy A)?", loc="left", fontsize=10)
            for a in ax[0]:
                a.legend(fontsize=7, loc="upper right")
    for a in ax[-1]:
        a.set_xlabel("time (ns)")
    fig.suptitle("2Nhp6A copy-B steered pulls (BIASED construction) — PBC check (10 ps frames)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def analyse(k, r):
    run = P.format(k=k, r=r)
    rdir = os.path.join(ROOT, "data", "mstate", "md", f"c{k}", f"rep{r}")
    pdir = os.path.join(ROOT, "analysis", "pbc", f"mstate_c{k}")
    cv = read_colvar(os.path.join(rdir, f"colvar_{run}.dat"))
    xvg = np.loadtxt(os.path.join(pdir, f"mindist_pi_r{r}_dt10ps.xvg"), comments=("#", "@"))
    xtc = os.path.join(pdir, f"mindist_pi_r{r}_dt10ps_proc.xtc")
    # per-frame geometry is the slow part (~65 ms/frame): cache it next to the trajectory, keyed
    # on the frame count so a re-scan of a longer run is recomputed rather than silently reused
    cache = os.path.join(pdir, f"{run}_pbc_geometry.npz")
    t = md.load(xtc, top=os.path.join(rdir, f"{run}_vmd_ref.pdb"))
    if os.path.exists(cache) and np.load(cache)["g"].shape[0] == t.n_frames:
        g = np.load(cache)["g"]
    else:
        sel = selections(t.topology)
        g = np.array([frame_geometry(t.xyz[f], t.unitcell_vectors[f], sel) for f in range(t.n_frames)])
        np.savez(cache, g=g, time=t.time)
    return dict(run=run, k=k, r=r, cv=cv, xvg=xvg, t=t.time / 1000.0, idr=g[:, 0], same=g[:, 1],
                other=g[:, 2], img=g[:, 3], strand=g[:, 4],
                half=0.5 * np.linalg.norm(t.unitcell_vectors[0][0]))


def summarise(R):
    tc = R["cv"]["time"] / 1000.0
    dstep = np.abs(np.diff(R["cv"]["d_core"])).max()
    gmx_low = R["xvg"][:, 1] < 1.0
    return dict(run=R["run"], t_ns=tc[-1], dcore_max=R["cv"]["d_core"].max(), half_box=R["half"],
                dcore_maxstep=dstep, gmx_pi_min=R["xvg"][:, 1].min(), gmx_pi_lt1=int(gmx_low.sum()),
                true_pi_min=R["img"].min(), true_pi_lt1=int((R["img"] < 1.0).sum()),
                other_min=R["other"].min(), core_closer_other=int((R["other"] < R["same"]).sum()),
                idr_unbound=int((R["idr"] > CONTACT).sum()), strand_max=R["strand"].max(),
                nframes=len(R["t"]))


def make_figure(runs, out):
    fig, ax = plt.subplots(len(runs), 3, figsize=(16, 2.9 * len(runs)), squeeze=False)
    for i, R in enumerate(runs):
        tc = R["cv"]["time"] / 1000.0
        a = ax[i, 0]
        a.plot(tc[::50], R["cv"]["d_core"][::50], color="#2a78d6", lw=0.6, label="d_core (PLUMED)")
        a.axhline(R["half"], color="#e34948", ls="--", lw=1, label=f"half box vector {R['half']:.2f} nm")
        a.axhline(D_BOUND, color="0.4", ls=":", lw=1, label=f"bound {D_BOUND} nm")
        a.set_ylabel(f"{R['run'].split('md_')[1]}\nd_core (nm)")
        a = ax[i, 1]
        a.plot(R["xvg"][:, 0], R["xvg"][:, 1], color="0.7", lw=0.6, label="gmx mindist -pi (as clustered)")
        a.plot(R["t"], R["img"], color="k", lw=0.7, label="complex rebuilt (strands welded, protein on IDR's copy)")
        a.axhline(1.0, color="#e34948", ls="--", lw=1, label="1.0 nm")
        a.set_ylabel(f"solute–own image\nmin dist (nm, cap {CAP})")
        a.set_ylim(0, None)
        a = ax[i, 2]
        a.plot(R["t"], R["same"], color="#2a78d6", lw=0.7, label="core → IDR's DNA copy")
        a.plot(R["t"], R["other"], color="#eb6834", lw=0.7, label="core → nearest other DNA image")
        a.axhline(CONTACT, color="0.4", ls=":", lw=1, label=f"contact {CONTACT} nm")
        a.set_ylabel(f"core–DNA heavy\nmin dist (nm, cap {CAP})")
        a.set_ylim(0, None)
        for a in ax[i]:
            a.grid(alpha=0.25, lw=0.5)
            a.set_xlim(0, tc[-1])
        if i == 0:
            ax[0, 0].set_title("a  core–DNA COM distance (min image)", loc="left", fontsize=10)
            ax[0, 1].set_title("b  periodic-image distance: raw vs rebuilt complex", loc="left", fontsize=10)
            ax[0, 2].set_title("c  does the core ever prefer a DNA image?", loc="left", fontsize=10)
            for a in ax[0]:
                a.legend(fontsize=7, loc="upper right")
    for a in ax[-1]:
        a.set_xlabel("time (ns)")
    fig.suptitle("1Nhp6A m-state unbiased production — PBC check (10 ps frames)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="c<k>/rep<r> ... (2Nhp6A_pull: c<k> ...)")
    p.add_argument("--system", default="1Nhp6A_mstate_md",
                   choices=("1Nhp6A_mstate_md", "2Nhp6A_pull", "2Nhp6A_mstate_md"),
                   help="2Nhp6A_mstate_md (Stage B production) supports --write-vmd only")
    p.add_argument("--version", default="v1")
    p.add_argument("--cache-only", action="store_true",
                   help="only compute/cache the per-frame geometry (run one process per run in parallel)")
    p.add_argument("--write-vmd", action="store_true",
                   help="2:1 only: write explicitly placed viewing trajectories "
                        "(data/mstate_2Nhp6A/{pull/c<k>,md/c<k>/rep<r>}/<run>_vmd.xtc, 100 ps) and exit")
    p.add_argument("--write-start", action="store_true",
                   help="2Nhp6A_pull only: write the Stage B start structure per pull (last frame "
                        "before the first complex-image contact < 0.9 nm), "
                        "data/mstate_2Nhp6A/pull/c<k>/<run>_start.gro, and exit")
    a = p.parse_args()
    if a.write_start:
        assert a.system == "2Nhp6A_pull", "--write-start is for --system 2Nhp6A_pull"
        for spec in a.runs:
            write_start_2to1(int(spec.replace("c", "")))
        return
    if a.write_vmd:
        assert a.system.startswith("2Nhp6A"), "--write-vmd is for the 2:1 systems"
        for spec in a.runs:
            write_vmd_2to1(spec, a.system)
        return
    if a.system == "2Nhp6A_pull":
        runs = []
        for spec in a.runs:
            print(f"analysing {spec} ...", flush=True)
            runs.append(analyse_pull_2to1(int(spec.replace("c", ""))))
        if a.cache_only:
            return
        out = figpath(f"2Nhp6A_mstate_pull_pbc_check_{a.version}.png", root=ROOT)
        make_figure_2to1(runs, out)
        rows = [summarise_2to1(R) for R in runs]
        dat = os.path.join(ROOT, "analysis", "pbc", "2Nhp6A_mstate_pull_pbc_summary.dat")
        with open(dat, "w") as fh:
            fh.write("# " + " ".join(rows[0]) + "\n")
            for row in rows:
                fh.write(" ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")
        for row in rows:
            print(row)
        print(out, dat)
        return
    runs = []
    for spec in a.runs:
        k, r = spec.replace("c", "").replace("rep", "").split("/")
        print(f"analysing c{k} rep{r} ...", flush=True)
        runs.append(analyse(int(k), int(r)))
    if a.cache_only:
        return
    out = figpath(f"1Nhp6A_mstate_md_pbc_check_{a.version}.png", root=ROOT)
    make_figure(runs, out)
    rows = [summarise(R) for R in runs]
    keys = list(rows[0])
    dat = os.path.join(ROOT, "analysis", "pbc", "1Nhp6A_mstate_md_pbc_summary.dat")
    with open(dat, "w") as fh:
        fh.write("# " + " ".join(keys) + "\n")
        for row in rows:
            fh.write(" ".join(f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k]) for k in keys) + "\n")
    for row in rows:
        print(row)
    print(out, dat)


if __name__ == "__main__":
    main()
