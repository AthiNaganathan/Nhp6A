#!/usr/bin/env python3
"""
Mutate 1Nhp6A WT binary complex to triple mutant (S26D + S41D + T63D).
Strategy: keep backbone unchanged; replace SER/THR side chain with ASP
using idealized tetrahedral CB and sp2 carboxylate geometry.

Usage: python3 make_1TM_mutant.py
"""
import numpy as np
import sys

IN  = "AF_Nhp6A/1Nhp6A_DNA_AF_fix.pdb"
OUT = "AF_Nhp6A/1TM_DNA_AF_fix.pdb"

MUTATIONS = {26: 'ASP', 41: 'ASP', 63: 'ASP'}  # residue number → new name

# Atoms to KEEP from original SER/THR residue (backbone + CB)
KEEP_ATOMS = {'N', 'CA', 'C', 'O', 'CB', 'OXT', 'H', 'HA', 'HB2', 'HB3', 'HB'}

# Bond lengths and angles for ASP side chain (Å, degrees)
CG_CB_BOND  = 1.52   # CB–CG
OD_CG_BOND  = 1.25   # CG–OD1/OD2
OD_CG_OD_ANGLE = 120.0  # OD1–CG–OD2 (sp2)
CA_CB_CG_ANGLE  = 113.0  # CA–CB–CG (tetrahedral-ish)

BOND_BACK_CA_CB = 1.53


def parse_pdb(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(line)
    return records


def get_atoms(records, chain, resnum):
    """Return dict of atom_name → (serial, xyz, full_line) for a residue."""
    atoms = {}
    for line in records:
        if not line.startswith('ATOM'):
            continue
        ch   = line[21]
        rnum = int(line[22:26])
        name = line[12:16].strip()
        if ch == chain and rnum == resnum:
            serial = int(line[6:11])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            atoms[name] = (serial, xyz, line)
    return atoms


def unit(v):
    return v / np.linalg.norm(v)


def place_asp_sidechain(ca_xyz, cb_xyz, n_xyz):
    """
    Place CG, OD1, OD2 for ASP given backbone CA, CB, N positions.
    CG is placed by extending the CA→CB bond; carboxylate plane is defined
    by the N–CA–CB plane normal so the side chain avoids the backbone.
    Returns dict: name → xyz
    """
    cb_ca = unit(cb_xyz - ca_xyz)

    # CG along CA→CB direction
    cg_xyz = cb_xyz + CG_CB_BOND * cb_ca

    # Build a local frame: x along CB→CG, z = N–CA–CB plane normal
    x_ax = cb_ca
    v_ca_n = unit(n_xyz - ca_xyz)
    z_ax = unit(np.cross(v_ca_n, x_ax))
    y_ax = np.cross(z_ax, x_ax)

    # OD1 and OD2 in sp2 plane (CG–OD1 and CG–OD2 at ±60° from x_ax in xy-plane)
    half = np.radians(OD_CG_OD_ANGLE / 2)
    od1_xyz = cg_xyz + OD_CG_BOND * (np.cos(np.pi - half) * x_ax + np.sin(np.pi - half) * y_ax)
    od2_xyz = cg_xyz + OD_CG_BOND * (np.cos(np.pi - half) * x_ax - np.sin(np.pi - half) * y_ax)

    return {'CG': cg_xyz, 'OD1': od1_xyz, 'OD2': od2_xyz}


def fmt_atom(serial, name, resname, chain, resnum, xyz, bfac=0.0):
    name_field = f' {name:<3s}' if len(name) < 4 else name
    return (f"ATOM  {serial:5d} {name_field} {resname:3s} {chain}{resnum:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00{bfac:6.2f}          "
            f"  {name[0]:>2s}\n")


def main():
    records = parse_pdb(IN)

    # Collect max serial number for new atoms
    max_serial = max(int(l[6:11]) for l in records if l.startswith(('ATOM','HETATM')))

    out_lines = []
    skip_resids = set()  # (chain, resnum) whose side chains we've replaced

    # First pass: mark mutations and gather backbone coords
    new_atoms = {}  # (chain, resnum) → list of new ATOM lines (CG, OD1, OD2)
    for resnum, newname in MUTATIONS.items():
        atoms = get_atoms(records, 'A', resnum)
        ca = atoms['CA'][1]
        cb = atoms['CB'][1]
        n  = atoms['N'][1]
        sc = place_asp_sidechain(ca, cb, n)
        lines = []
        for aname, xyz in sc.items():
            max_serial += 1
            lines.append(fmt_atom(max_serial, aname, newname, 'A', resnum, xyz))
        new_atoms[('A', resnum)] = lines
        skip_resids.add(('A', resnum))

    # Second pass: write output
    for line in records:
        if line.startswith('ATOM'):
            ch   = line[21]
            rnum = int(line[22:26])
            name = line[12:16].strip()
            if (ch, rnum) in skip_resids:
                resnum = rnum
                newname = MUTATIONS[resnum]
                # Skip old side-chain atoms (keep backbone + CB)
                if name not in KEEP_ATOMS:
                    continue
                # Rename residue
                line = line[:17] + f'{newname:3s}' + line[20:]
                out_lines.append(line)
                # Append new side-chain atoms after CB
                if name == 'CB':
                    out_lines.extend(new_atoms[(ch, rnum)])
                continue
        out_lines.append(line)

    with open(OUT, 'w') as f:
        f.writelines(out_lines)

    print(f"Written: {OUT}")

    # Verify
    for resnum in MUTATIONS:
        atoms = get_atoms(out_lines if False else parse_pdb(OUT), 'A', resnum)
        resname = list(atoms.values())[0][2][17:20].strip()
        print(f"  Res {resnum}: {resname}  atoms = {sorted(atoms.keys())}")


if __name__ == '__main__':
    main()
