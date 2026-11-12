"""
select_snapshots.py
-------------------
Extract representative structural snapshots of the metastable STATES (free-energy
basins) of each HMG-box/DNA binary system from its converged 2D WT-MetaD FES.

Pipeline (per system):
  1. Load the 2D FES grid (CV1 = coord_tail_dna, CV2 = de2e) — orientation-proof
     cell assignment, no reshape assumptions.
  2. Segment the FES into basins by connectivity in (cv1, cv2) space: each local
     minimum grows an energy-tight core {F < F_min + DELTA_F} restricted to the
     connected component containing it. Two equal-depth but spatially separate
     basins therefore get distinct labels (energy AND CV proximity).
  3. Pool all NWALKERS replicas' colvar, sub-sampled to trajectory frame times
     (multiples of DT_TRAJ) so each candidate maps exactly to a proc.xtc frame.
  4. Assign each candidate frame to a basin by its (cv1, cv2) grid cell.
  5. Per basin, draw N_PER_STATE frames by Boltzmann-weighted sampling and extract
     them from the replica proc.xtc into one xtc per state.
  6. Overlay the selected frame positions on the 2D FES to confirm placement.

Run `--plot-only` first to review/confirm the basin segmentation before extraction.

Mapping:  colvar time (ps) -> proc.xtc frame index = round(t_ps / dt_traj)
          (dt_traj is per-system: 20 ps for the bound binaries, 100 ps for the
          apo run; see process_replica.sh.  Each replica's candidates are also
          capped to its own proc.xtc length, so every candidate maps to a real
          frame even while the production job is still appending.)
"""

import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "scripts"))
from figpaths import figpath  # noqa: E402

# -- constants ---------------------------------------------------------------
kBT          = 2.494339      # kJ/mol at 300 K (consistent with prior analysis)
NWALKERS     = 6
DT_TRAJ      = 20.0          # ps per regenerated proc.xtc frame (per-system default)
COLVAR_DT    = 0.2           # ps per colvar row
N_PER_STATE  = 15            # representative frames per basin
F_CEIL       = 12.0          # kJ/mol: ignore local minima above this

# Per-system colvar/trajectory defaults (overridable per SYSTEMS entry).
# Bound binaries: colvar cols = time, coord_tail_dna, de2e, theta, cmap @ 20 ps.
DEFAULT_DT_TRAJ  = 20.0
DEFAULT_CV_COLS  = (1, 2)                      # 0-based colvar cols for (cv1, cv2)
DEFAULT_CV_NAMES = ("coord_tail_dna", "de2e (nm)")
DEFAULT_EXTRA    = {"theta": 3, "cmap": 4}     # extra colvar cols to carry through
DELTA_F      = 2.0           # kJ/mol: basin-core depth window
BARRIER      = 0.0           # kJ/mol: min saddle barrier to keep basins distinct
                             # (0 = raw watershed, no merge; >0 merges shallow sub-basins)
RNG_SEED     = 42
ANALYSIS_DIR = "analysis"
DATA_DIR     = "data"
SNAP_DIR     = os.path.join(ANALYSIS_DIR, "snapshots")
FIG_VERSION  = 1             # figure filename suffix _v<N>; set from --fig-version.
                             # Bump only when the underlying run grew (project convention);
                             # default 1 keeps every previously written filename unchanged.

# -- system definitions ------------------------------------------------------
SYSTEMS = {
    "1Nhp6A": {
        "fes":    f"{ANALYSIS_DIR}/1Nhp6A_fes_2D_all.dat",
        "colvar": f"{DATA_DIR}/1Nhp6A/replica_{{i}}/colvar_r{{i}}.dat",
        "xtc":    f"{DATA_DIR}/1Nhp6A/replica_{{i}}/1Nhp6A_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D_r{{i}}_proc.xtc",
        "top":    f"{DATA_DIR}/1Nhp6A/1Nhp6A_protdna.gro",
    },
    "1T63D": {
        "fes":    f"{ANALYSIS_DIR}/1T63D_fes_2D_all.dat",
        "colvar": f"{DATA_DIR}/1T63D/replica_{{i}}/colvar_r{{i}}.dat",
        "xtc":    f"{DATA_DIR}/1T63D/replica_{{i}}/1T63D_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D_r{{i}}_proc.xtc",
        "top":    f"{DATA_DIR}/1T63D/1T63D_protdna.gro",
    },
    "1S26D_large": {
        "fes":    f"{ANALYSIS_DIR}/1S26D_large_fes_2D_all.dat",
        "colvar": f"{DATA_DIR}/1S26D_large/replica_{{i}}/colvar_r{{i}}.dat",
        "xtc":    f"{DATA_DIR}/1S26D_large/replica_{{i}}/1S26D_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D_r{{i}}_proc.xtc",
        # 2489-atom Protein_DNA topology derived from the md.tpr (must match proc.xtc)
        "top":    f"{DATA_DIR}/1S26D_large/1S26D_large_protdna.gro",
    },
    # soft-wall variant of 1Nhp6A: LOWER_WALLS AT=0.9 on cmap instead of the stiff
    # RESTRAINT AT=0.96 (same coord_tail_dna x de2e CVs); solute atom indices are
    # unchanged from 1Nhp6A, so the topology is reused as-is.
    "1Nhp6A_wall": {
        "fes":    f"{ANALYSIS_DIR}/1Nhp6A_wall_fes_2D_all.dat",
        "colvar": f"{DATA_DIR}/1Nhp6A_wall/replica_{{i}}/colvar_r{{i}}.dat",
        "xtc":    f"{DATA_DIR}/1Nhp6A_wall/replica_{{i}}/1Nhp6A_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D_r{{i}}_proc.xtc",
        "top":    f"{DATA_DIR}/1Nhp6A/1Nhp6A_protdna.gro",
    },
    # NOTE: the four holo `<SYS>_wall_bend` entries (WT + the three phosphomimetic
    # mutants) are generated below by _holo_bend_cfg -- they differ only in system
    # name, so a literal block each would be four near-duplicates.
    # apo protein (no DNA): CVs are folding/helicity descriptors, 100 ps frames.
    "1Nhp6A_apo_large": {
        "fes":    f"{ANALYSIS_DIR}/1Nhp6A_apo_large_fes_sumhills.dat",
        "colvar": f"{DATA_DIR}/1Nhp6A_apo_large/replica_{{i}}/colvar_r{{i}}.dat",
        "xtc":    f"{DATA_DIR}/1Nhp6A_apo_large/replica_{{i}}/1Nhp6A_parambsc1_tip3p_ions0.15_meta_2D_r{{i}}_proc.xtc",
        # protein-only 1539-atom topology (matches the protein-only proc.xtc)
        "top":    f"{DATA_DIR}/1Nhp6A_apo_large/1Nhp6A_parambsc1_tip3p_ed.gro",
        "dt_traj":  100.0,
        "cv_cols":  (1, 2),                    # q_core, n_tail_core
        "cv_names": ("q_core", "n_tail_core"),
        "extra_cols": {"rmsd_core": 3, "helix1": 4, "helix2": 5, "helix3": 6},
        # axis window of the populated region (matches the apo PMF plot)
        "zoom":     ((0.7, 1.0), (0.0, 60.0)),
    },
    # apo protein, helicity projection: the REWEIGHTED c(t) FES
    # F(total_helicity, q_core) (helicity is a spectator, not biased, so its FES
    # comes from reweighting, NOT sum_hills; grid written by write_fes_dat).
    # cv1 = total_helicity = helix1+helix2+helix3 (summed colvar cols 4,5,6);
    # cv2 = q_core (col 1).  Same proc.xtc/topology as 1Nhp6A_apo_large.
    "1Nhp6A_apo_large_helicity": {
        # Panel C's PLOTTED surface (article_data.py --export-seg-fes): coarser and
        # snapshot-averaged, i.e. the same estimator the figure shows. Segmenting on the finer
        # single-shot FES (1Nhp6A_apo_large_fes_helicity_qcore.dat) splits the shallow second well
        # into sub-minima separated by less than its own +/-4-6 kJ/mol uncertainty, and the medoid
        # then lands off the basin the figure draws.
        "fes":    f"{ANALYSIS_DIR}/1Nhp6A_apo_large_fes_helicity_qcore_panelC.dat",
        "colvar": f"{DATA_DIR}/1Nhp6A_apo_large/replica_{{i}}/colvar_r{{i}}.dat",
        # protein-only, whole-molecule trajectories built from the .trr by pbc_mindist_check.sh
        # (200 ps stride, full run). The trr carries the same coordinates as the 102 GB raw xtc
        # ~30x cheaper.
        #
        # 2026-08-07: repointed from the replica-dir `_proc.xtc` to analysis/pbc/solute_r*.xtc.
        # The `_proc.xtc` copies were written 07-31, BEFORE job 3822 finished, so they stop at
        # 1000 ns (5001 frames) while the panel-C surface is built on the full 1250 ns colvar --
        # the markers were representatives of the first 900 ns of a 1250 ns surface. The
        # solute_r*.xtc were rebuilt 08-06 after the run completed and cover 0-1250 ns
        # (6251 frames, 200 ps, 1539 atoms); they are bit-identical to `_proc.xtc` over the
        # 0-1000 ns overlap (max |dx| = 0.0 nm, times match), so this only ADDS the last 250 ns.
        # They are also the exact files reweight_qbh.compute_q uses for the 1250 ns Q_BH cache.
        "xtc":    f"{ANALYSIS_DIR}/pbc/solute_r{{i}}.xtc",
        "top":    f"{DATA_DIR}/1Nhp6A_apo_large/1Nhp6A_parambsc1_tip3p_ed.gro",
        "dt_traj":  200.0,
        # medoid over the FOLDED CORE only (res 21-93 = CORE_LO/CORE_HI in reweight_qbh.py,
        # 596 heavy atoms). The disordered IDR otherwise decides the medoid outright -- see
        # the note at the medoid block in process_system().
        "medoid_sel": "not element H and resSeq 21 to 93",
        "cv_cols":  ((4, 5, 6), 1),            # cv1=total_helicity(sum), cv2=q_core
        "cv_names": ("total_helicity", "q_core"),
        "extra_cols": {"n_tail_core": 2, "rmsd_core": 3,
                       "helix1": 4, "helix2": 5, "helix3": 6},
        # trim to the populated region: helicity>=10, 0.5<=q_core<=1.0
        "zoom":     ((10.0, 44.0), (0.5, 1.0)),
    },
    # apo protein, helicity projection segmented on the CANONICAL Best-Hummer Q
    # instead of the biasing q_core, so the panel-C markers are Q_BH-derived end to
    # end (state membership AND coordinates) rather than inheriting q_core basins.
    # Both inputs come from reweight_qbh.py --emit-seg-inputs:
    #   fes    : the panel-C Q_BH surface with axes swapped to (total_helicity, Q_BH),
    #            built from the 2 ps `plumed driver` pass (1.8 M frames, 0-1000 ns).
    #   colvar : colvar-SHAPED tables whose col 1 is Q_BH where the real colvar has
    #            q_core, so the layout below is identical to its q_core twin; q_core
    #            rides along as col 7 and is reported in the medoid table.
    # 2026-08-07: everything here is now 0-1250 ns, matching the q_core twin. The
    # 2 ps driver pass was re-run over the completed run (analysis/qbh_1250 on
    # insulin -> local analysis/qbh), so the surface and the qbh_seg colvars both
    # reach 1250 ns, and the structural pool is the same 6251-frame solute_r*.xtc.
    # Before that the surface stopped at 1000 ns (the driver read while 3822 was
    # still extending) -- do NOT mix a 1250 ns pool with a 1000 ns surface.
    "1Nhp6A_apo_large_helicity_qbh": {
        "fes":    f"{ANALYSIS_DIR}/1Nhp6A_apo_large_fes_helicity_qbh_panelC.dat",
        "colvar": f"{ANALYSIS_DIR}/qbh_seg/colvar_qbh_r{{i}}.dat",
        "xtc":    f"{ANALYSIS_DIR}/pbc/solute_r{{i}}.xtc",
        "top":    f"{DATA_DIR}/1Nhp6A_apo_large/1Nhp6A_parambsc1_tip3p_ed.gro",
        "dt_traj":  200.0,
        # medoid over the folded core only -- same reason as the q_core twin
        "medoid_sel": "not element H and resSeq 21 to 93",
        "cv_cols":  ((4, 5, 6), 1),            # cv1=total_helicity(sum), cv2=q_bh
        "cv_names": ("total_helicity", "q_bh"),
        "extra_cols": {"n_tail_core": 2, "rmsd_core": 3,
                       "helix1": 4, "helix2": 5, "helix3": 6, "q_core": 7},
        # the Q_BH surface only spans Q=0.782-0.998; helicity as for the q_core twin
        "zoom":     ((10.0, 44.0), (0.78, 1.0)),
    },
}


# -- holo (DNA-bound) systems on the F(n_IDR-DNA, bend) projection ------------
# WT and the three phosphomimetic mutants, all on the SAME reweighted surface
# family `analysis/<SYS>_wall_pmf2d_idrdna_bend.dat`, so their representative
# structures are directly comparable. coord_tail_dna is biased but the bending
# angle is a spectator, so the medoid must come off the REWEIGHTED grid, never a
# sum_hills one. cv2 = bend = 180 - degrees(theta) (theta is radians in col 3).
#
# The colvar layout is identical for all four (time, coord_tail_dna, de2e, theta,
# cmap), so only the file naming and the trajectory stride differ:
#   WT      proc.xtc @  20 ps (15001 frames), topology = the 1Nhp6A .gro
#   mutants proc.xtc @ 200 ps ( ~1501 frames), topology = each system's own
#           <SYS>_proc_ref.pdb (atom counts differ: 2490/2487/2489)
# Every one of these trajectories covers 0-300 ns only, so candidates cap there
# even where the surface now runs to 350 ns.
_HOLO_BEND = {
    #  system   xtc infix          topology                       dt_traj
    "1Nhp6A": ("1Nhp6A", "",       f"{DATA_DIR}/1Nhp6A/1Nhp6A_protdna.gro",   20.0),
    "1S26D":  ("1S26D", "_wall",   f"{DATA_DIR}/1S26D_wall/1S26D_proc_ref.pdb", 200.0),
    "1T63D":  ("1T63D", "_wall",   f"{DATA_DIR}/1T63D_wall/1T63D_proc_ref.pdb", 200.0),
    "1TM":    ("1TM",   "_wall",   f"{DATA_DIR}/1TM_wall/1TM_proc_ref.pdb",     200.0),
    # revision: IDR-only peptide (res 1-23) + DNA; proc.xtc = peptide+DNA, 1365 atoms
    "1IDR":   ("1IDR",  "_wall",   f"{DATA_DIR}/1IDR_wall/1IDR_DNA_parambsc1_tip3p.pdb", 20.0),
    # revision: Δ2-10 + DNA (solute, 2324 atoms). Canonical _proc.xtc (20 ps, 0-350 ns,
    # -pbc cluster on the solute group), written 2026-11-05 after job 3918.
    "1D2_10": ("1D2_10", "_wall",  f"{DATA_DIR}/1D2_10/1D2_10_DNA_parambsc1_tip3p.gro", 20.0),
}


def _holo_bend_cfg(sys_name, prefix, infix, top, dt_traj):
    return {
        "fes":    f"{ANALYSIS_DIR}/{sys_name}_wall_pmf2d_idrdna_bend.dat",
        "colvar": f"{DATA_DIR}/{sys_name}_wall/replica_{{i}}/colvar_r{{i}}.dat",
        "xtc":    (f"{DATA_DIR}/{sys_name}_wall/replica_{{i}}/{prefix}"
                   f"_DNA_parambsc1_tip3p_ions0.15_cmap_meta_2D{infix}_r{{i}}_proc.xtc"),
        "top":    top,
        "dt_traj":  dt_traj,
        "cv_cols":  (1, 3),                    # coord_tail_dna, theta(rad)
        "cv_names": ("n_IDR-DNA", "bend (deg)"),
        "cv_transform": (None, lambda th: 180.0 - np.degrees(th)),
        "extra_cols": {"de2e": 2, "theta": 3, "cmap": 4},
        # cols of the reweighted grid: 0 cv1, 1 cv2, 2 F_mean, 3 F_std,
        # 4 frame_counts, 5 n_walkers
        "mask": {"std_col": 3, "std_max": 5.0,
                 "nwalk_col": 5, "min_walkers": 3},
    }


for _s, (_p, _in, _top, _dt) in _HOLO_BEND.items():
    SYSTEMS[f"{_s}_wall_bend"] = _holo_bend_cfg(_s, _p, _in, _top, _dt)

# 1IDR: the peptide is disordered and ~1/4 of the heavy atoms, so an all-heavy-atom medoid is
# decided by the DNA. Superpose on the DNA, then rank by the peptide's RMSD WITHOUT refitting:
# the medoid is the most typical IDR placement on the DNA, not the most typical DNA.
SYSTEMS["1IDR_wall_bend"].update(medoid_fit="chainid 1 2 and not element H",
                                 medoid_sel="chainid 0 and not element H")

# Free DNA (corrected dna_end2 run): 1D MetaD on de2e only, so the surface is the
# REWEIGHTED F(de2e, bend) (dna_bending_wall_compare.py --de2e-bend-2d); bend is a
# spectator. Colvar = time, de2e, theta, cmap. proc.xtc = DNA duplex only (950 atoms,
# -pbc cluster so both strands stay together), 20 ps, 0-100 ns.
SYSTEMS["DNA_wall2_de2e_bend"] = {
    "fes":    f"{ANALYSIS_DIR}/DNA_wall2_pmf2d_de2e_bend.dat",
    "colvar": f"{DATA_DIR}/DNA_wall2/replica_{{i}}/colvar_r{{i}}.dat",
    "xtc":    (f"{DATA_DIR}/DNA_wall2/replica_{{i}}/DNA_parambsc1_tip3p_ions0.15"
               f"_cmap_meta_e2e_wall_r{{i}}_proc.xtc"),
    "top":    f"{DATA_DIR}/DNA_wall2/DNA_wall2_proc_ref.pdb",
    "dt_traj":  20.0,
    "cv_cols":  (1, 2),                    # de2e, theta(rad)
    "cv_names": ("de2e (nm)", "bend (deg)"),
    "cv_transform": (None, lambda th: 180.0 - np.degrees(th)),
    "extra_cols": {"de2e": 1, "theta": 2, "cmap": 3},
    "mask": {"std_col": 3, "std_max": 5.0, "nwalk_col": 5, "min_walkers": 3},
}


# -- FES loading (orientation-proof) -----------------------------------------
def load_fes_grid(path, mask=None):
    """Load a 2D free-energy grid into a regular array.

    Columns: cv1, cv2, F, then either the sum_hills derivatives (dF/dcv1,
    dF/dcv2) or, for a reweighted mean+/-std surface, F_std, frame_counts and
    optionally n_walkers.  Returns (cv1_vals, cv2_vals, F) where F has shape
    (n_cv2, n_cv1) and F[j, i] = FES(cv1_vals[i], cv2_vals[j]).  Cells are
    placed by value lookup, so the result is independent of row ordering.

    `mask` (per-system, from the SYSTEMS entry) blanks unreliable cells to NaN
    BEFORE segmentation, so they can neither seed a basin nor accept members:
      std_col/std_max     : drop cells whose cross-snapshot std exceeds std_max
                            (project convention: 5 kJ/mol)
      nwalk_col/min_walkers: drop cells visited by fewer than min_walkers
                            walkers -- the ergodicity mask that exposed S26D's
                            low-bend/high-contact corner as a walker artifact.
    Column indices are taken from the config rather than sniffed, because a
    sum_hills file and a reweighted file can both be 5 columns wide.
    """
    d = np.loadtxt(path, comments="#")
    cv1_vals = np.unique(d[:, 0])
    cv2_vals = np.unique(d[:, 1])
    F = np.full((len(cv2_vals), len(cv1_vals)), np.nan)
    i1 = np.searchsorted(cv1_vals, d[:, 0])
    i2 = np.searchsorted(cv2_vals, d[:, 1])
    F[i2, i1] = d[:, 2]
    if mask:
        keep = np.ones(len(d), dtype=bool)
        if mask.get("std_col") is not None and mask.get("std_max") is not None:
            std = d[:, mask["std_col"]]
            keep &= np.isfinite(std) & (std <= mask["std_max"])
        if mask.get("nwalk_col") is not None and mask.get("min_walkers"):
            keep &= d[:, mask["nwalk_col"]] >= mask["min_walkers"]
        drop = ~keep
        n_before = int(np.isfinite(F).sum())
        F[i2[drop], i1[drop]] = np.nan
        n_after = int(np.isfinite(F).sum())
        print(f"  Reliability mask: {n_before} -> {n_after} cells "
              f"(std<={mask.get('std_max')}, n_walkers>={mask.get('min_walkers')})")
    return cv1_vals, cv2_vals, F


def _nearest_index(vals, x):
    """Nearest grid index for value x on a regular axis, or -1 if out of range."""
    n = len(vals)
    step = vals[1] - vals[0]
    idx = int(round((x - vals[0]) / step))
    if idx < 0 or idx >= n:
        return -1
    return idx


# -- basin segmentation ------------------------------------------------------
def _watershed_labels(F, Ffill, finite, f_ceil):
    """Steepest-descent (watershed) partition. Returns (labels, minid).

    Each finite cell drains downhill (8-connectivity) to the local minimum it
    reaches; cells draining to a minimum below f_ceil inherit that minimum's id.
    """
    ny, nx = F.shape
    is_min = (ndimage.minimum_filter(Ffill, size=3) == Ffill) & finite
    min_cells = {(int(j), int(i)) for j, i in np.argwhere(is_min)}
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def drain_target(j, i):
        while (j, i) not in min_cells:
            bv = Ffill[j, i]
            best = None
            for dj, di in neigh:
                nj, ni = j + dj, i + di
                if 0 <= nj < ny and 0 <= ni < nx and Ffill[nj, ni] < bv:
                    bv = Ffill[nj, ni]
                    best = (nj, ni)
            if best is None:
                return (j, i)
            j, i = best
        return (j, i)

    kept = sorted((m for m in min_cells if F[m] < f_ceil), key=lambda m: F[m])
    minid = {m: k + 1 for k, m in enumerate(kept)}
    labels = np.zeros(F.shape, dtype=int)
    cache = {}
    for j in range(ny):
        for i in range(nx):
            if not finite[j, i]:
                continue
            tgt = cache.get((j, i))
            if tgt is None:
                tgt = drain_target(j, i)
                cache[(j, i)] = tgt
            labels[j, i] = minid.get(tgt, 0)
    return labels, minid


def _merge_shallow(labels, Ffill, fmin, barrier):
    """Prominence merge: fold each basin whose separating saddle is < `barrier`
    above its own minimum into the neighbour across that lowest saddle.

    Iterates until every surviving basin is separated from all others by a
    barrier >= `barrier`. Returns a {raw_id -> merged_id} mapping.
    """
    ny, nx = labels.shape
    parent = {b: b for b in fmin}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    while True:
        # lowest saddle between each pair of current roots
        saddle = {}
        for j in range(ny):
            for i in range(nx):
                la = labels[j, i]
                if la <= 0:
                    continue
                ra = find(la)
                for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    nj, ni = j + dj, i + di
                    if 0 <= nj < ny and 0 <= ni < nx and labels[nj, ni] > 0:
                        rb = find(labels[nj, ni])
                        if rb == ra:
                            continue
                        h = max(Ffill[j, i], Ffill[nj, ni])
                        key = (min(ra, rb), max(ra, rb))
                        if key not in saddle or h < saddle[key]:
                            saddle[key] = h
        if not saddle:
            break
        # root minima
        rmin = {}
        for raw, f in fmin.items():
            r = find(raw)
            rmin[r] = min(f, rmin.get(r, np.inf))
        # find the merge with smallest prominence of the shallower partner
        best = None
        for (a, b), h in saddle.items():
            prom = h - max(rmin[a], rmin[b])      # depth of shallower basin
            if best is None or prom < best[0]:
                best = (prom, a, b)
        prom, a, b = best
        if prom >= barrier:
            break
        # merge shallower into deeper
        lo, hi = (a, b) if rmin[a] <= rmin[b] else (b, a)  # lo = deeper
        parent[find(hi)] = find(lo)

    return {raw: find(raw) for raw in fmin}


def segment_basins(cv1_vals, cv2_vals, F, f_ceil=F_CEIL, delta_f=DELTA_F,
                   barrier=BARRIER, min_cells=1):
    """Partition the FES into metastable basins and merge spurious sub-basins.

    1. Watershed (steepest-descent) partition into basins of attraction — a true
       non-overlapping partition that separates adjacent wells at their dividing
       saddle (couples energy AND CV proximity).
    2. Prominence merge: basins separated from a deeper neighbour by a saddle
       barrier < `barrier` are folded in, so flat valley-floor ripples do not
       masquerade as distinct states.

    3. Drop basins whose energy-tight core is smaller than `min_cells`. On a
       reliability/ergodicity-masked surface the prominence merge cannot fold a
       fragment into its neighbour when the two are separated by NaN rather than
       by a saddle, so a handful of cells stranded against the mask edge would
       otherwise be reported as a state. Such a fragment carries no basin volume
       and its medoid would be meaningless.

    Only minima below f_ceil seed basins. `ncells` reports the energy-tight core
    {label == k AND F < F_min + delta_f}, used downstream for membership.

    Returns (labels, basins):
      labels : int grid; 0 = unassigned, k = basin k.
      basins : list of dicts {id, i1, i2, cv1, cv2, fmin, ncells}.
    """
    finite = np.isfinite(F)
    Ffill = np.where(finite, F, np.inf)
    raw_labels, minid = _watershed_labels(F, Ffill, finite, f_ceil)
    fmin_raw = {bid: float(F[m]) for m, bid in minid.items()}

    mapping = _merge_shallow(raw_labels, Ffill, fmin_raw, barrier)
    # deepest minimum defines each merged basin; relabel sequentially by depth
    merged_min = {}                               # merged_id -> (cell, fmin)
    for m, bid in minid.items():
        mg = mapping[bid]
        if mg not in merged_min or F[m] < merged_min[mg][1]:
            merged_min[mg] = (m, float(F[m]))
    order = sorted(merged_min, key=lambda mg: merged_min[mg][1])
    relabel = {mg: k + 1 for k, mg in enumerate(order)}

    labels = np.zeros(F.shape, dtype=int)
    for j in range(F.shape[0]):
        for i in range(F.shape[1]):
            if raw_labels[j, i] > 0:
                labels[j, i] = relabel[mapping[raw_labels[j, i]]]

    basins = []
    for mg in order:
        (j, i), fmin = merged_min[mg]
        bid = relabel[mg]
        core = (labels == bid) & finite & (F < fmin + delta_f)
        basins.append({
            "id": bid, "i1": int(i), "i2": int(j),
            "cv1": float(cv1_vals[i]), "cv2": float(cv2_vals[j]),
            "fmin": fmin, "ncells": int(core.sum()),
        })

    # drop mask-edge fragments, then renumber the survivors by depth so the ids
    # stay contiguous (state_1 = deepest)
    dropped = [b for b in basins if b["ncells"] < min_cells]
    if dropped:
        for b in dropped:
            print(f"    dropped fragment id {b['id']} "
                  f"({cv1_vals[b['i1']]:.2f}, {cv2_vals[b['i2']]:.2f}): "
                  f"core {b['ncells']} < --min-cells {min_cells}")
        keep = [b for b in basins if b["ncells"] >= min_cells]
        renum = {b["id"]: k + 1 for k, b in enumerate(keep)}
        new_labels = np.zeros(F.shape, dtype=int)
        for old, new in renum.items():
            new_labels[labels == old] = new
        labels = new_labels
        for b in keep:
            b["id"] = renum[b["id"]]
        basins = keep
    return labels, basins


# -- colvar loading ----------------------------------------------------------
def _flatten_cols(spec):
    """0-based column index or a tuple/list of indices -> flat list of indices."""
    return list(spec) if isinstance(spec, (tuple, list)) else [spec]


def _col_values(d, spec):
    """Column values for a CV spec: a single 0-based index, or a tuple/list of
    indices whose columns are SUMMED (e.g. total_helicity = helix1+helix2+helix3)."""
    if isinstance(spec, (tuple, list)):
        return d[:, list(spec)].sum(axis=1)
    return d[:, spec]


def load_colvar(path, cv_cols, extra_cols):
    """Load a PLUMED COLVAR file -> dict of arrays (time in ps).

    `cv_cols` are the (cv1, cv2) column specs; each is either a single 0-based
    column index or a tuple/list of indices to SUM (derived CV, e.g. a total
    helicity from per-helix columns).  `extra_cols` maps extra observable names to
    their column index. Tolerates truncated lines at restart/crash boundaries
    (rows with a non-finite time or CV are dropped)."""
    d = np.genfromtxt(path, comments="#", invalid_raise=False, filling_values=np.nan)
    check = ([0] + _flatten_cols(cv_cols[0]) + _flatten_cols(cv_cols[1])
             + list(extra_cols.values()))
    valid = np.all(np.isfinite(d[:, check]), axis=1)
    d = d[valid]
    out = {"time": d[:, 0], "cv1": _col_values(d, cv_cols[0]),
           "cv2": _col_values(d, cv_cols[1])}
    for name, col in extra_cols.items():
        out[name] = d[:, col]
    return out


def xtc_nframes(xtc_path, top_path):
    """Number of frames in an xtc, read without loading all coordinates."""
    import mdtraj as md
    n = 0
    for chunk in md.iterload(xtc_path, top=top_path, chunk=1000):
        n += chunk.n_frames
    return n


def pool_candidates(cfg, t_max_ps):
    """Pool all replicas' colvar, sub-sampled to trajectory frame times.

    Keeps rows whose time is (within tolerance) a multiple of the per-system
    dt_traj and <= t_max_ps, AND that fall within the replica's own proc.xtc
    length (the production job may still be appending, so colvar can run past the
    last extracted frame). This guarantees each candidate maps to a real frame.
    """
    dt_traj = cfg.get("dt_traj", DEFAULT_DT_TRAJ)
    cv_cols = cfg.get("cv_cols", DEFAULT_CV_COLS)
    extra   = cfg.get("extra_cols", DEFAULT_EXTRA)
    transform = cfg.get("cv_transform", (None, None))
    keys = ["replica", "time", "cv1", "cv2"] + list(extra)
    cols = {k: [] for k in keys}
    for i in range(NWALKERS):
        cv = load_colvar(cfg["colvar"].format(i=i), cv_cols, extra)
        # optional per-CV transform (e.g. bend = 180 - degrees(theta)), applied so
        # the candidate CV values match the units/convention of the FES grid axes.
        if transform[0] is not None:
            cv["cv1"] = transform[0](cv["cv1"])
        if transform[1] is not None:
            cv["cv2"] = transform[1](cv["cv2"])
        nfr = xtc_nframes(cfg["xtc"].format(i=i), cfg["top"])
        cap_ps = min(t_max_ps, (nfr - 1) * dt_traj)   # last extractable frame
        frac = cv["time"] / dt_traj
        on_grid = (np.abs(frac - np.round(frac)) < 1e-3) & (cv["time"] <= cap_ps + 1e-6)
        sel = np.where(on_grid)[0]
        cols["replica"].append(np.full(len(sel), i, dtype=int))
        cols["time"].append(cv["time"][sel])
        cols["cv1"].append(cv["cv1"][sel])
        cols["cv2"].append(cv["cv2"][sel])
        for name in extra:
            cols[name].append(cv[name][sel])
        print(f"  Replica {i}: {len(sel):,} on-grid frames "
              f"(colvar to {cv['time'][-1]/1000:.1f} ns, xtc {nfr} fr, "
              f"cap {cap_ps/1000:.1f} ns)")
    return {k: np.concatenate(v) for k, v in cols.items()}


# -- frame extraction --------------------------------------------------------
def extract_frame(xtc_path, top_path, frame_idx):
    import mdtraj as md
    try:
        return md.load_frame(xtc_path, frame_idx, top=top_path)
    except Exception as e:
        print(f"    Warning: failed frame {frame_idx} in {os.path.basename(xtc_path)}: {e}")
        return None


# -- per-system driver -------------------------------------------------------
def add_shoulder_region(cv1_vals, cv2_vals, F, labels, basins, spec, delta_f):
    """Label an explicit CV window as an extra state, on top of the watershed basins.

    A `--barrier` above the noise floor correctly merges away any feature whose saddle
    prominence is not resolved by the data, so a high-free-energy *shoulder* leaves no
    basin to draw a structure from. This paints a user-given CV window as its own label
    so a representative can still be extracted. It is a region of the surface, NOT a
    detected basin, and must be reported that way: `shoulder=True` marks it, and its
    fmin is the depth of the window relative to the global minimum.

    spec = (cv1_c, cv2_c, r1, r2): window centre and half-widths, in CV units.
    """
    c1, c2, r1, r2 = spec
    G1, G2 = np.meshgrid(cv1_vals, cv2_vals)          # (ny, nx)
    win = (np.abs(G1 - c1) <= r1) & (np.abs(G2 - c2) <= r2) & np.isfinite(F)
    if not win.any():
        raise SystemExit(f"--shoulder window {spec} covers no reliable FES cells")
    fmin = float(np.nanmin(np.where(win, F, np.inf)))
    j, i = np.unravel_index(int(np.argmin(np.where(win, F, np.inf))), F.shape)
    sid = max(b["id"] for b in basins) + 1
    labels = labels.copy()
    labels[win] = sid                                  # only steals cells far above any
                                                       # basin core, so no basin loses members
    core = win & (F < fmin + delta_f)
    basins = basins + [{"id": sid, "i1": int(i), "i2": int(j),
                        "cv1": float(cv1_vals[i]), "cv2": float(cv2_vals[j]),
                        "fmin": fmin, "ncells": int(core.sum()), "shoulder": True}]
    return labels, basins


def process_system(name, cfg, args, rng):
    print(f"\n{'='*64}\n  {name}\n{'='*64}")
    cv_names = cfg.get("cv_names", DEFAULT_CV_NAMES)
    dt_traj  = cfg.get("dt_traj", DEFAULT_DT_TRAJ)
    extra    = cfg.get("extra_cols", DEFAULT_EXTRA)
    meta_keys = ["replica", "time", "cv1", "cv2"] + list(extra)
    cv1_name = cv_names[0].split()[0]
    cv2_name = cv_names[1].split()[0]
    mask = cfg.get("mask")
    if mask and (args.std_max is not None or args.min_walkers is not None):
        mask = dict(mask)                      # CLI override, for sensitivity checks
        if args.std_max is not None:
            mask["std_max"] = args.std_max
        if args.min_walkers is not None:
            mask["min_walkers"] = args.min_walkers
    cv1_vals, cv2_vals, F = load_fes_grid(cfg["fes"], mask=mask)
    labels, basins = segment_basins(cv1_vals, cv2_vals, F,
                                    f_ceil=args.f_ceil, delta_f=args.delta_f,
                                    barrier=args.barrier, min_cells=args.min_cells)
    print(f"  FES grid: {len(cv1_vals)} x {len(cv2_vals)} "
          f"({cv1_name} {cv1_vals[0]:.2f}-{cv1_vals[-1]:.2f}, "
          f"{cv2_name} {cv2_vals[0]:.2f}-{cv2_vals[-1]:.2f})")
    print(f"  Detected {len(basins)} basin(s):")
    if args.shoulder:
        labels, basins = add_shoulder_region(cv1_vals, cv2_vals, F, labels, basins,
                                             args.shoulder, args.delta_f)
    print(f"    id   {cv1_name:>7s} {cv2_name:>7s}   Fmin   ncells  kind")
    for b in basins:
        kind = "SHOULDER (not a basin)" if b.get("shoulder") else "basin"
        print(f"    {b['id']:2d}  {b['cv1']:7.3f} {b['cv2']:7.2f} "
              f"{b['fmin']:6.2f}   {b['ncells']:5d}  {kind}")

    if args.plot_only:
        plot_overlay(name, cv1_vals, cv2_vals, F, labels, basins, None,
                     suffix="basins", cv_names=cv_names, zoom=cfg.get("zoom"))
        return

    # pool candidate frames (trajectory-resolution)
    t_max = args.t_max_ns * 1000.0
    print(f"  Pooling candidates (<= {args.t_max_ns:.0f} ns, dt={dt_traj:.0f} ps):")
    cand = pool_candidates(cfg, t_max)

    # assign each candidate to a basin via its grid cell, restricted to the
    # energy-tight core {F < F_min + delta_f} so representatives stay near the well
    fmin_of = {b["id"]: b["fmin"] for b in basins}
    basin_of = np.zeros(len(cand["time"]), dtype=int)
    f_of = np.full(len(cand["time"]), np.nan)
    for k in range(len(cand["time"])):
        i = _nearest_index(cv1_vals, cand["cv1"][k])
        j = _nearest_index(cv2_vals, cand["cv2"][k])
        if i >= 0 and j >= 0:
            lbl = labels[j, i]
            f_of[k] = F[j, i]
            if lbl > 0 and F[j, i] < fmin_of[lbl] + args.delta_f:
                basin_of[k] = lbl

    # select representatives per basin and extract
    import mdtraj as md
    os.makedirs(os.path.join(SNAP_DIR, name), exist_ok=True)
    chosen_all = {k: [] for k in (["state"] + meta_keys)}
    medoid_frames, medoid_meta = [], []   # one structural medoid per state
    for b in basins:
        members = np.where(basin_of == b["id"])[0]
        if len(members) == 0:
            print(f"  State {b['id']}: no frames sampled — skipped")
            continue
        # Boltzmann weights relative to basin minimum
        w = np.exp(-(f_of[members] - b["fmin"]) / kBT)
        w /= w.sum()
        nsel = min(args.n_per_state, len(members))
        pick = rng.choice(members, size=nsel, replace=False, p=w)
        pick = pick[np.argsort(cand["time"][pick])]
        print(f"  State {b['id']}: {len(members):,} member frames -> {nsel} picked")

        # extract from proc.xtc, keeping frame metadata aligned with the traj
        frames, fmeta = [], []
        for r in range(NWALKERS):
            sub = pick[cand["replica"][pick] == r]
            if len(sub) == 0:
                continue
            xtc = cfg["xtc"].format(i=r)
            for idx in sub:
                fidx = int(round(cand["time"][idx] / dt_traj))
                fr = extract_frame(xtc, cfg["top"], fidx)
                if fr is not None:
                    frames.append(fr)
                    fmeta.append({k: cand[k][idx] for k in meta_keys})
        if not frames:
            continue
        traj = md.join(frames)
        out = os.path.join(SNAP_DIR, name, f"state_{b['id']}.xtc")
        traj.save_xtc(out)
        print(f"    wrote {len(frames)} frames -> {out}")

        # structural medoid = frame minimising mean RMSD to the rest, over
        # `medoid_sel` (default: all heavy atoms).
        #
        # 2026-08-07: the apo systems MUST restrict this to the folded core. With all
        # 761 protein heavy atoms the criterion is decided by the disordered IDR --
        # mean RMSD 5.8-11.3 A against 2.2-3.0 A over the core -- so the winner beat
        # the runner-up by 0.06 A out of 5.83 A and the marker landed at q_core 0.9674
        # while the basin minimum sits at 0.998. On core heavy atoms the ranking
        # inverts and the medoid returns to q_core 0.9990, on the minimum.
        heavy = traj.topology.select(cfg.get("medoid_sel", "not element H"))
        if len(heavy) == 0:
            heavy = np.arange(traj.n_atoms)
        print(f"    medoid RMSD over {len(heavy)} atoms "
              f"({cfg.get('medoid_sel', 'not element H')})")
        fit_sel = cfg.get("medoid_fit")
        if fit_sel:
            fit = traj.topology.select(fit_sel)
            print(f"    superposed on {len(fit)} atoms ({fit_sel}); RMSD not refitted")
        if traj.n_frames == 1:
            mi = 0
        else:
            sums = np.zeros(traj.n_frames)
            for i in range(traj.n_frames):
                if fit_sel:
                    # copy: the saved state/medoid coordinates must stay as extracted
                    al = traj[:].superpose(traj, i, atom_indices=fit)
                    d = al.xyz[:, heavy] - traj.xyz[i, heavy]
                    sums += np.sqrt((d ** 2).sum(axis=2).mean(axis=1))
                else:
                    sums += md.rmsd(traj, traj, i, atom_indices=heavy)
            mi = int(np.argmin(sums))
            # The argmin is never decisive -- with 200 frames the best and second-best
            # mean RMSD differ by ~0.01 A against a ~1 A spread, so "the medoid" is one
            # arbitrary draw from a set of structurally equivalent frames. With
            # --medoid-cv-tiebreak, take that set explicitly (the best
            # `--medoid-tol-pct` percent of the ranking) and among its members keep the
            # one closest to the region's own free-energy minimum, measured in grid
            # cells so the two CVs are commensurate. This changes only WHICH of several
            # equivalent structures is shown -- it does not relax the RMSD criterion --
            # and it stops the marker being drawn on the flank of the feature it labels.
            if args.medoid_cv_tiebreak:
                k = max(1, int(np.ceil(args.medoid_tol_pct / 100.0 * traj.n_frames)))
                near = np.argsort(sums)[:k]
                d1 = np.median(np.diff(cv1_vals)) or 1.0
                d2 = np.median(np.diff(cv2_vals)) or 1.0
                dist = np.hypot([(fmeta[i]["cv1"] - b["cv1"]) / d1 for i in near],
                                [(fmeta[i]["cv2"] - b["cv2"]) / d2 for i in near])
                mi_cv = int(near[int(np.argmin(dist))])
                rm = sums / (traj.n_frames - 1)
                print(f"    CV tiebreak: {k} frames within the top {args.medoid_tol_pct:g}% "
                      f"of the RMSD ranking ({rm[near].min()*10:.2f}-{rm[near].max()*10:.2f} A); "
                      f"kept the one nearest ({cv1_name} {b['cv1']:.2f}, {cv2_name} {b['cv2']:.3f})")
                print(f"      pure medoid : {cv1_name} {fmeta[mi]['cv1']:7.3f}  "
                      f"{cv2_name} {fmeta[mi]['cv2']:7.4f}  meanRMSD {rm[mi]*10:.2f} A")
                print(f"      CV-nearest  : {cv1_name} {fmeta[mi_cv]['cv1']:7.3f}  "
                      f"{cv2_name} {fmeta[mi_cv]['cv2']:7.4f}  meanRMSD {rm[mi_cv]*10:.2f} A")
                mi = mi_cv
        medoid_frames.append(traj[mi])
        medoid_meta.append({"state": b["id"], **fmeta[mi]})

        for mt in fmeta:
            chosen_all["state"].append(b["id"])
            for key in meta_keys:
                chosen_all[key].append(mt[key])

    # column header: state, replica, time, the two CVs (by name), then extras
    extra_names = list(extra)
    hdr = ["state", "replica", "time_ps", cv1_name, cv2_name] + extra_names

    def _fmt_row(src, k):
        vals = [f"{int(src['state'][k]):3d}", f"{int(src['replica'][k]):2d}",
                f"{src['time'][k]:12.2f}",
                f"{src['cv1'][k]:9.4f}", f"{src['cv2'][k]:8.4f}"]
        vals += [f"{src[e][k]:8.4f}" for e in extra_names]
        return " ".join(vals)

    # CV companion table for all selected frames
    cv_out = os.path.join(SNAP_DIR, f"{name}_states_cvs.dat")
    with open(cv_out, "w") as fh:
        fh.write("# " + " ".join(hdr) + "\n")
        order = np.lexsort((chosen_all["time"], chosen_all["replica"], chosen_all["state"]))
        for k in order:
            fh.write(_fmt_row(chosen_all, k) + "\n")
    print(f"  Wrote CV table -> {cv_out}")

    # medoids: one representative structure per state + a positions file
    if medoid_frames:
        mtraj = md.join(medoid_frames)
        mxtc = os.path.join(SNAP_DIR, f"{name}_medoids.xtc")
        mtraj.save_xtc(mxtc)
        for k, mt in enumerate(medoid_meta):
            mtraj[k].save_pdb(os.path.join(SNAP_DIR, name,
                                           f"state_{mt['state']}_medoid.pdb"))
        mcv = os.path.join(SNAP_DIR, f"{name}_medoids_cvs.dat")
        with open(mcv, "w") as fh:
            fh.write("# medoid (min mean heavy-atom RMSD per state). "
                     "frame_in_xtc = row order below (0-based)\n")
            fh.write("# " + " ".join(hdr) + "\n")
            for mt in medoid_meta:
                vals = [f"{int(mt['state']):3d}", f"{int(mt['replica']):2d}",
                        f"{mt['time']:12.2f}",
                        f"{mt['cv1']:9.4f}", f"{mt['cv2']:8.4f}"]
                vals += [f"{mt[e]:8.4f}" for e in extra_names]
                fh.write(" ".join(vals) + "\n")
        print(f"  Wrote {len(medoid_frames)} medoids -> {mxtc}")
        print(f"  Wrote medoid positions -> {mcv}")

    med = (np.array([mt["cv1"] for mt in medoid_meta]),
           np.array([mt["cv2"] for mt in medoid_meta]),
           np.array([mt["state"] for mt in medoid_meta])) if medoid_meta else None
    plot_overlay(name, cv1_vals, cv2_vals, F, labels, basins,
                 (np.array(chosen_all["cv1"]), np.array(chosen_all["cv2"]),
                  np.array(chosen_all["state"])), suffix="states", medoids=med,
                 delta_f=args.delta_f, cv_names=cv_names, zoom=cfg.get("zoom"))


# -- plotting ----------------------------------------------------------------
def plot_overlay(name, cv1_vals, cv2_vals, F, labels, basins, picks, suffix,
                 medoids=None, delta_f=DELTA_F, cv_names=DEFAULT_CV_NAMES, zoom=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    Fm = np.ma.masked_invalid(F)
    levels = np.linspace(0, np.nanmin([F[np.isfinite(F)].max(), 40]), 21)
    cf = ax.contourf(cv1_vals, cv2_vals, Fm, levels=levels,
                     cmap="RdYlBu_r", extend="max")
    ax.contour(cv1_vals, cv2_vals, Fm, levels=levels[::4],
               colors="k", linewidths=0.4, alpha=0.5)
    # outline the per-state selection windows: watershed cell AND F < F_min + delta_f
    # (this is the region snapshots were actually drawn from)
    finite = np.isfinite(F)
    for b in basins:
        core = ((labels == b["id"]) & finite &
                (F < b["fmin"] + delta_f)).astype(float)
        if core.any():
            ax.contour(cv1_vals, cv2_vals, core, levels=[0.5],
                       colors="k", linewidths=1.0)
    fig.colorbar(cf, ax=ax, label=r"$\Delta F$ (kJ/mol)")
    cmap = plt.get_cmap("tab10")
    # faint cloud of all selected snapshots, to show the spread within each basin
    if picks is not None:
        pc1, pc2, _ = picks
        ax.scatter(pc1, pc2, c="0.4", s=8, alpha=0.30, linewidths=0, zorder=4)
    # medoids: one circle per state, distinct colour, black edge
    if medoids is not None:
        mc1, mc2, mst = medoids
        for cv1, cv2, st in zip(mc1, mc2, mst):
            ax.scatter(cv1, cv2, s=150, facecolor=cmap((int(st) - 1) % 10),
                       edgecolors="black", linewidths=1.5, zorder=6)
            ax.annotate(str(int(st)), (cv1, cv2), textcoords="offset points",
                        xytext=(6, 6), fontsize=10, fontweight="bold")
    if zoom is not None:
        ax.set_xlim(*zoom[0])
        ax.set_ylim(*zoom[1])
    ax.set_xlabel(cv_names[0])
    ax.set_ylabel(cv_names[1])
    ax.set_title(f"{name}: FES basins + state medoids")
    out = figpath(f"{name}_fes2d_{suffix}_v{FIG_VERSION}.png")   # figures/<system>/
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Wrote figure -> {out}")


def main():
    global FIG_VERSION
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", choices=list(SYSTEMS) + ["all"], default="all")
    ap.add_argument("--plot-only", action="store_true",
                    help="segment + plot basins for confirmation; no extraction")
    ap.add_argument("--n-per-state", type=int, default=N_PER_STATE)
    ap.add_argument("--medoid-cv-tiebreak", action="store_true",
                    help="among the structurally equivalent frames at the top of the "
                         "mean-RMSD ranking (see --medoid-tol-pct), keep the one nearest "
                         "the region's free-energy minimum in grid-cell units, so a "
                         "marker is not drawn on the flank of the feature it labels. The "
                         "RMSD criterion itself is unchanged.")
    ap.add_argument("--medoid-tol-pct", type=float, default=10.0, metavar="PCT",
                    help="size of that equivalent set, as a %% of the frames in the "
                         "state (default %(default)s). 10 was chosen on the apo shoulder: "
                         "at 5%% the set is too narrow to contain a frame near the "
                         "free-energy minimum and the pick lands on the tail of the helix2 "
                         "distribution (2nd percentile); at 10%% it sits at the minimum with "
                         "helix2 at the 24th, for 0.11 A of mean RMSD.")
    ap.add_argument("--f-ceil", type=float, default=F_CEIL)
    ap.add_argument("--delta-f", type=float, default=DELTA_F)
    ap.add_argument("--barrier", type=float, default=BARRIER,
                    help="min saddle barrier (kJ/mol) for a basin to count as distinct")
    ap.add_argument("--min-cells", type=int, default=1, metavar="N",
                    help="discard basins whose energy-tight core is smaller than N grid "
                         "cells (default %(default)s = keep all). On a masked surface, "
                         "fragments stranded against the mask edge cannot be merged away "
                         "by --barrier; use e.g. 5 to drop them.")
    ap.add_argument("--t-max-ns", type=float, default=150.0)
    ap.add_argument("--std-max", type=float, default=None, metavar="KJMOL",
                    help="override the per-system reliability mask: blank FES cells whose "
                         "cross-snapshot std exceeds this (default: the system's own value, "
                         "5 kJ/mol for the reweighted holo surfaces)")
    ap.add_argument("--min-walkers", type=int, default=None, metavar="N",
                    help="override the per-system ergodicity mask: blank FES cells visited "
                         "by fewer than N walkers (default: the system's own value, 3 for "
                         "the reweighted holo surfaces). 0 disables it.")
    ap.add_argument("--fig-version", type=int, default=FIG_VERSION, metavar="N",
                    help="figure filename suffix _v<N> in figures/<system>/ (default %(default)s). "
                         "Bump when the underlying run grew, so earlier versions survive.")
    ap.add_argument("--shoulder", type=str, default=None, metavar="CV1,CV2,R1,R2",
                    help="Additionally extract a representative from an explicit CV window "
                         "(centre CV1,CV2; half-widths R1,R2). For a feature that is NOT a "
                         "metastable basin -- e.g. a shoulder whose saddle prominence is below "
                         "the noise floor, which --barrier correctly merges away. Reported as "
                         "SHOULDER, never as a state.")
    args = ap.parse_args()
    FIG_VERSION = args.fig_version
    if args.shoulder:
        try:
            args.shoulder = tuple(float(v) for v in args.shoulder.split(","))
            if len(args.shoulder) != 4:
                raise ValueError
        except ValueError:
            raise SystemExit("--shoulder needs 4 comma-separated numbers: CV1,CV2,R1,R2")

    rng = np.random.default_rng(RNG_SEED)
    names = list(SYSTEMS) if args.system == "all" else [args.system]
    for nm in names:
        process_system(nm, SYSTEMS[nm], args, rng)
    print("\nDone.")


if __name__ == "__main__":
    main()
