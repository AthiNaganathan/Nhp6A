#!/usr/bin/env python3
"""Extract PLUMED atom group indices from a GROMACS GRO file.

Usage:
    python get_group_indices.py 1Nhp6A_DNA_parambsc1.gro

Outputs two GROUP lines ready to paste into the PLUMED file:
  - tail: CA + Cbeta of residues 1-20 (protein N-terminal tail)
  - dna_backbone: P + C1' of all DNA nucleotides
"""
import sys

if len(sys.argv) < 2:
    sys.exit("Usage: get_group_indices.py <file.gro>")

gro = sys.argv[1]
tail_names = {'CA', 'CB'}
dna_names  = {'P', "C1'"}
dna_residues = {'DA', 'DT', 'DG', 'DC',
                'DA5', 'DT5', 'DG5', 'DC5',
                'DA3', 'DT3', 'DG3', 'DC3'}
tail_res_range = range(1, 21)  # residues 1-20

tail_idx, dna_idx = [], []
with open(gro) as f:
    next(f)  # title
    next(f)  # atom count
    for line in f:
        if len(line) < 20:
            continue
        try:
            resnum  = int(line[0:5])
            resname = line[5:10].strip()
            atname  = line[10:15].strip()
            atidx   = int(line[15:20])
        except ValueError:
            continue
        if resnum in tail_res_range and atname in tail_names:
            tail_idx.append(atidx)
        if resname in dna_residues and atname in dna_names:
            dna_idx.append(atidx)

print("tail: GROUP ATOMS=" + ",".join(map(str, tail_idx)))
print("dna_backbone: GROUP ATOMS=" + ",".join(map(str, dna_idx)))
print(f"\n# tail atoms: {len(tail_idx)}, dna_backbone atoms: {len(dna_idx)}", file=sys.stderr)
