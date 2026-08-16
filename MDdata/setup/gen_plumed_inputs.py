#!/usr/bin/env python3
"""
gen_plumed_inputs.py — generate PLUMED inputs for protein-only 2D WT-MetaD.

Usage:
    python3 gen_plumed_inputs.py <reference.pdb> <system_name> [options]

<reference.pdb>  Protein-only PDB from the last NPT frame (GROMACS output).
                 Atoms numbered 1..N_prot; residues 1-20 = tail, 21-93 = core.
<system_name>    e.g. 1Nhp6A, 1S26D, 1T63D

Options (defaults reproduce the original 3-walker, 0.25 nm-wall, no-helicity output):
    --walkers N        number of multi-walkers (WALKERS_N), default 3
    --rmsd-wall X      upper-wall position on rmsd_core in nm, default 0.25
    --helices RANGES   comma-separated residue ranges, e.g. 27-42,48-61,64-91;
                       emits monitored-only ALPHARMSD helixK CVs added to PRINT
    --out-tag TAG      suffix for outputs and MOLINFO pdb; e.g. TAG=large gives
                       plumed_meta_2D_<sys>_large.dat, ref_core_ca_<sys>_large.pdb,
                       MOLINFO ../../../<sys>_parambsc1_large.pdb, run dir <sys>_apo_large

Outputs:
    plumed_meta_2D_<system_name>[_<tag>].dat   PLUMED input with REPID placeholder
    ref_core_ca_<system_name>[_<tag>].pdb      Core CA reference for rmsd_core
"""

import argparse
import numpy as np


def parse_pdb(pdbfile):
    atoms = []
    with open(pdbfile) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            serial = int(line[6:11])
            name   = line[12:16].strip()
            resnum = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            atoms.append((serial, name, resnum, x, y, z))
    return atoms


def get_serials(atoms, res_start, res_end, names):
    return sorted(s for s, n, r, x, y, z in atoms
                  if res_start <= r <= res_end and n in names)


def get_ca_coords(atoms, res_start, res_end):
    return [(s, r, np.array([x, y, z]))
            for s, n, r, x, y, z in atoms
            if res_start <= r <= res_end and n == 'CA']


def native_contacts(ca_list, cutoff_nm=0.65, min_sep=4):
    cutoff_A = cutoff_nm * 10.0
    contacts = []
    for i in range(len(ca_list)):
        for j in range(i + 1, len(ca_list)):
            si, ri, ci = ca_list[i]
            sj, rj, cj = ca_list[j]
            if abs(ri - rj) < min_sep:
                continue
            d = np.linalg.norm(ci - cj)
            if d < cutoff_A:
                contacts.append((si, sj, d / 10.0))
    return contacts


def write_core_ca_pdb(atoms, outfile, res_start=21, res_end=93):
    ca = [(s, r, x, y, z) for s, n, r, x, y, z in atoms
          if res_start <= r <= res_end and n == 'CA']
    with open(outfile, 'w') as f:
        for s, r, x, y, z in ca:
            f.write(f"ATOM  {s:5d}  CA  ALA A{r:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
        f.write("END\n")


def fmt(serials):
    return ','.join(map(str, serials))


def parse_helices(spec):
    """Parse '27-42,48-61,64-91' -> [(27,42),(48,61),(64,91)]."""
    ranges = []
    for chunk in spec.split(','):
        a, b = chunk.split('-')
        ranges.append((int(a), int(b)))
    return ranges


def main():
    p = argparse.ArgumentParser(
        usage="python3 gen_plumed_inputs.py <reference.pdb> <system_name> [options]",
        description="Generate PLUMED inputs for protein-only 2D WT-MetaD.")
    p.add_argument("pdbfile")
    p.add_argument("sysname")
    p.add_argument("--walkers", type=int, default=3,
                   help="number of multi-walkers (WALKERS_N), default 3")
    p.add_argument("--rmsd-wall", type=float, default=0.25,
                   help="upper-wall position on rmsd_core in nm, default 0.25")
    p.add_argument("--helices", default=None,
                   help="comma-separated residue ranges, e.g. 27-42,48-61,64-91")
    p.add_argument("--out-tag", default=None,
                   help="output/MOLINFO filename suffix, e.g. large")
    args = p.parse_args()

    pdbfile, sysname = args.pdbfile, args.sysname
    tag      = args.out_tag
    suffix   = f"_{tag}" if tag else ""
    helices  = parse_helices(args.helices) if args.helices else []

    atoms = parse_pdb(pdbfile)
    if not atoms:
        p.error(f"no ATOM records in {pdbfile}")

    n_prot = max(s for s, *_ in atoms)

    tail_ca   = get_serials(atoms,  1, 20, {'CA'})
    tail_cb   = get_serials(atoms,  1, 20, {'CB'})
    tail_cacb = sorted(tail_ca + tail_cb)

    core_ca   = get_serials(atoms, 21, 93, {'CA'})
    core_cb   = get_serials(atoms, 21, 93, {'CB'})
    core_cacb = sorted(core_ca + core_cb)

    ca_list  = get_ca_coords(atoms, 21, 93)
    contacts = native_contacts(ca_list)
    nc       = len(contacts)
    w        = 1.0 / nc if nc else 1.0

    ref_pdb = f"ref_core_ca_{sysname}{suffix}.pdb"
    write_core_ca_pdb(atoms, ref_pdb)

    helix_labels = [f"helix{k}" for k in range(1, len(helices) + 1)]
    print_args = ",".join(["q_core", "n_tail_core", "rmsd_core"] + helix_labels)

    outdat = f"plumed_meta_2D_{sysname}{suffix}.dat"
    with open(outdat, 'w') as f:
        f.write(f"# Protein-only 2D well-tempered metadynamics: {sysname}\n")
        f.write(f"# CV1: q_core       fraction native contacts, core (res 21-93)\n")
        f.write(f"# CV2: n_tail_core  tail-core coordination (analogous to coord_tail_dna)\n")
        f.write(f"# Wall: rmsd_core   upper wall at {args.rmsd_wall:g} nm prevents core unfolding\n")
        if helices:
            f.write(f"# Monitored: helix1..{len(helices)}  per-helix ALPHARMSD (not biased)\n")
        f.write(f"# Run from data/{sysname}_apo{suffix}/replica_REPID/\n")
        f.write("\n")
        f.write(f"MOLINFO STRUCTURE=../../../{sysname}_parambsc1{suffix}.pdb\n")
        f.write(f"WHOLEMOLECULES ENTITY0=1-{n_prot}\n")
        f.write("\n")
        f.write(f"tail: GROUP ATOMS={fmt(tail_cacb)}\n")
        f.write(f"core: GROUP ATOMS={fmt(core_cacb)}\n")
        f.write("\n")
        f.write(f"rmsd_core: RMSD REFERENCE=../../../{ref_pdb} TYPE=OPTIMAL\n")
        f.write("\n")
        f.write(f"# {nc} CA-CA contacts within 0.65 nm, |i-j| >= 4, for res 21-93\n")
        f.write(f"q_core: CONTACTMAP ...\n")
        for k, (si, sj, d) in enumerate(contacts, 1):
            f.write(f"   ATOMS{k}={si},{sj} SWITCH{k}={{Q R_0=0.1 BETA=50.0 LAMBDA=1.4 REF={d:.3f}}} WEIGHT{k}={w:.8f}\n")
        f.write(f"   SUM\n")
        f.write(f"...\n")
        f.write("\n")
        f.write(f"n_tail_core: COORDINATION GROUPA=tail GROUPB=core R_0=0.5 NN=6 MM=12\n")
        f.write("\n")
        if helices:
            f.write(f"# Per-helix helicity (monitored only): ALPHARMSD over DSSP-derived ranges\n")
            for label, (r0, r1) in zip(helix_labels, helices):
                f.write(f"{label}: ALPHARMSD RESIDUES={r0}-{r1}\n")
            f.write("\n")
        f.write(f"wall_core: UPPER_WALLS ARG=rmsd_core AT={args.rmsd_wall:g} KAPPA=5000 EXP=2 OFFSET=0\n")
        f.write("\n")
        f.write(f"metad: METAD ARG=q_core,n_tail_core ...\n")
        f.write(f"   PACE=500 HEIGHT=1.2\n")
        f.write(f"   SIGMA=0.025,1.5\n")
        f.write(f"   BIASFACTOR=10 TEMP=298\n")
        f.write(f"   FILE=hills.dat\n")
        f.write(f"   GRID_MIN=0.3,0 GRID_MAX=1.0,150 GRID_BIN=100,300\n")
        f.write(f"   WALKERS_N={args.walkers} WALKERS_ID=REPID WALKERS_DIR=../\n")
        f.write(f"   WALKERS_RSTRIDE=500\n")
        f.write(f"...\n")
        f.write("\n")
        # NOTE: no periodic REWEIGHT_BIAS->HISTOGRAM->CONVERT_TO_FES->DUMPGRID block.
        # On a *fresh* run that DUMPGRID writes analysis.<n>.fes and aborts at the
        # 101-backup cap (~10 ns). FES is computed post-hoc from sum_hills, and the
        # colvar does not print metad.bias, so the block is unnecessary. Removed
        # 2026-06-18 from all apo plumed files; keep it out of the generator too.
        f.write("\n")
        f.write(f"PRINT ARG={print_args} STRIDE=100 FILE=colvar_rREPID.dat\n")

    print(f"Written: {outdat}")
    print(f"  {n_prot} protein atoms | {nc} native contacts (q_core)")
    print(f"  tail: {len(tail_cacb)} atoms (CA+CB res 1-20)")
    print(f"  core: {len(core_cacb)} atoms (CA+CB res 21-93)")
    print(f"  walkers: {args.walkers} | rmsd wall: {args.rmsd_wall:g} nm"
          + (f" | helices: {len(helices)}" if helices else ""))
    print(f"Written: {ref_pdb}")


if __name__ == '__main__':
    main()
