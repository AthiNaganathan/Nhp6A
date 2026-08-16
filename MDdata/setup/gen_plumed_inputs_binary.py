#!/usr/bin/env python3
"""
gen_plumed_inputs_binary.py — derive PLUMED CV atom-index groups for protein+DNA
2D (or higher-D) WT-MetaD and emit a plumed.dat template.

Usage:
    python3 gen_plumed_inputs_binary.py <binary.pdb> <system_name> \
        [--protein-chains A[,B,...]] [--dna-chains B,C] [--walkers 6]

<binary.pdb>   Protein+DNA PDB. Default chain layout: chain A = protein,
               chain B = DNA strand 1, chain C = DNA strand 2. Pass
               --protein-chains / --dna-chains to override for systems with
               more than one protein copy (e.g. 2Nhp6A: protein A,B + DNA C,D).
               Atom serials must match the GROMACS topology (1-indexed, sequential).
<system_name>  e.g. 1TM, 2Nhp6A

Outputs:
    plumed_meta_2D_<system_name>_DNA.dat  PLUMED input template (REPID placeholder)

With a single protein chain (default), CVs are named `tail`/`coord_tail_dna`
(unchanged, for backward compatibility). With multiple protein chains, one
tail/coord_tail_dna CV is generated per chain, suffixed by chain id
(coord_tail_dna_A, coord_tail_dna_B, ...), and all of them plus de2e are
biased together in the METAD ARG list.

CONTACT MAP — NOT generated here:
    The DNA inter-strand CONTACTMAP block must come from contact_map_setup.ipynb,
    which is the canonical, consistent protocol (all DNA atoms incl. H, residue
    separation > 3, cutoff 0.30 nm, LAMBDA=1.4). It produces
    <system_name>_DNA_parambsc1_cmap.{txt,dat}. Paste the `cmap: CONTACTMAP ...`
    block (without its trailing PRINT line) into the template where marked.
    Do NOT generate the contact map with a different definition — doing so yields
    an inconsistent map (e.g. ~65 vs the correct ~140-175 for a 15-bp duplex).
"""

import argparse

TAIL_RES_START = 1
TAIL_RES_END   = 20


def parse_pdb(pdbfile):
    atoms = []
    with open(pdbfile) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            serial  = int(line[6:11])
            name    = line[12:16].strip()
            resname = line[17:20].strip()
            chain   = line[21]
            resnum  = int(line[22:26])
            atoms.append({'serial': serial, 'name': name, 'resname': resname,
                          'chain': chain, 'resnum': resnum})
    return atoms


def chain(atoms, ch):
    return [a for a in atoms if a['chain'] == ch]


def tail_atoms(atoms, ch):
    """CA+CB serials from residues 1-20 of the given protein chain."""
    return sorted(
        a['serial'] for a in atoms
        if a['chain'] == ch
        and TAIL_RES_START <= a['resnum'] <= TAIL_RES_END
        and a['name'] in ('CA', 'CB')
    )


def dna_backbone_atoms(atoms, dna_chains):
    """P and C1' serials from the given DNA chains (both strands)."""
    return sorted(
        a['serial'] for a in atoms
        if a['chain'] in dna_chains and a['name'] in ('P', "C1'")
    )


def terminal_residue_serials(chain_atoms, end='first'):
    """Serials of the first or last residue in a chain."""
    residues = sorted(set(a['resnum'] for a in chain_atoms))
    target = residues[0] if end == 'first' else residues[-1]
    return sorted(a['serial'] for a in chain_atoms if a['resnum'] == target)


def fmt_range(serials):
    """Format a contiguous serial list as 'first-last', else comma-separated."""
    if not serials:
        return ''
    serials = sorted(serials)
    if serials[-1] - serials[0] == len(serials) - 1:
        return f"{serials[0]}-{serials[-1]}"
    return ','.join(map(str, serials))


def fmt_two_ranges(serials_a, serials_b):
    """Format two serial groups as 'range_a,range_b'."""
    return f"{fmt_range(serials_a)},{fmt_range(serials_b)}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pdbfile')
    p.add_argument('sysname')
    p.add_argument('--protein-chains', default='A',
                    help="comma-separated protein chain ids (default: A)")
    p.add_argument('--dna-chains', default='B,C',
                    help="comma-separated pair of DNA strand chain ids (default: B,C)")
    p.add_argument('--walkers', type=int, default=6,
                    help="WALKERS_N for the multi-walker METAD block (default: 6)")
    args = p.parse_args()

    pdbfile, sysname = args.pdbfile, args.sysname
    protein_chains = args.protein_chains.split(',')
    dna_chains = args.dna_chains.split(',')
    if len(dna_chains) != 2:
        raise SystemExit("ERROR: --dna-chains must name exactly two chain ids")

    atoms = parse_pdb(pdbfile)
    if not atoms:
        raise SystemExit(f"ERROR: no ATOM records in {pdbfile}")

    ch_b = chain(atoms, dna_chains[0])
    ch_c = chain(atoms, dna_chains[1])
    if not ch_b or not ch_c:
        raise SystemExit(f"ERROR: DNA chains {dna_chains} not found in {pdbfile}")

    for pc in protein_chains:
        if not chain(atoms, pc):
            raise SystemExit(f"ERROR: protein chain {pc} not found in {pdbfile}")

    # --- Atom groups ---
    multi_protein = len(protein_chains) > 1
    tails = {pc: tail_atoms(atoms, pc) for pc in protein_chains}
    dna_bb = dna_backbone_atoms(atoms, dna_chains)

    dna_first = min(a['serial'] for a in ch_b + ch_c)
    dna_last  = max(a['serial'] for a in ch_b + ch_c)

    # Terminal nucleotide atoms for end-to-end distance
    # dna_end1 = first nt of strand B  +  last nt of strand C  (one end of helix)
    # dna_end2 = last nt of strand B   +  first nt of strand C (other end)
    end1_b = terminal_residue_serials(ch_b, 'first')
    end1_c = terminal_residue_serials(ch_c, 'last')
    end2_b = terminal_residue_serials(ch_b, 'last')
    end2_c = terminal_residue_serials(ch_c, 'first')

    # --- Reference PDB name (symlinked into each replica dir by SLURM script) ---
    ref_pdb = f"{sysname}_DNA_parambsc1.pdb"
    cmapdat = f"{sysname}_DNA_parambsc1_cmap.dat"

    def cv_name(base, pc):
        return base if not multi_protein else f"{base}_{pc}"

    coord_cvs = [cv_name('coord_tail_dna', pc) for pc in protein_chains]
    metad_args = coord_cvs + ['de2e']
    n_cv = len(metad_args)

    # --- Write PLUMED template ---
    outdat = f"plumed_meta_2D_{sysname}_DNA.dat"
    with open(outdat, 'w') as f:
        f.write(f"MOLINFO STRUCTURE={ref_pdb}\n")
        f.write(f"WHOLEMOLECULES ENTITY0={dna_first}-{dna_last}\n")
        f.write(f"FIT_TO_TEMPLATE REFERENCE={ref_pdb}\n")
        f.write("\n")
        f.write("## geometric centers\n")
        f.write(f"dna_center: COM ATOMS={dna_first}-{dna_last}\n")
        f.write(f"dna_end1: COM ATOMS={fmt_two_ranges(end1_b, end1_c)}\n")
        f.write(f"dna_end2: COM ATOMS={fmt_two_ranges(end2_b, end2_c)}\n")  # note: end2_b then end2_c
        f.write("\n")
        f.write("# angle\n")
        f.write("theta: ANGLE ATOMS=dna_end1,dna_center,dna_end2\n")
        f.write("\n")
        f.write("## de2e\n")
        f.write("de2e: DISTANCE ATOMS=dna_end1,dna_end2 NOPBC\n")
        f.write("\n")
        f.write("# --- Tail and DNA backbone groups ---\n")
        f.write("# P + C1' of all nucleotides (both strands, ~2 atoms/nt)\n")
        f.write(f"dna_backbone: GROUP ATOMS={','.join(map(str, dna_bb))}\n")
        for pc in protein_chains:
            tail_name = cv_name('tail', pc)
            f.write(f"# CA + Cbeta of residues 1-20, chain {pc} (N-terminal disordered tail)\n")
            f.write(f"{tail_name}: GROUP ATOMS={','.join(map(str, tails[pc]))}\n")
        f.write("\n")
        for pc in protein_chains:
            tail_name = cv_name('tail', pc)
            coord_name = cv_name('coord_tail_dna', pc)
            f.write(f"# Coordination between chain {pc} tail and DNA backbone\n")
            f.write(f"{coord_name}: COORDINATION GROUPA={tail_name} GROUPB=dna_backbone R_0=0.5 NN=6 MM=12\n")
        f.write("\n")
        f.write("## restraint on DNA double-helix integrity (inter-strand contacts)\n")
        f.write("## >>> PASTE the `cmap: CONTACTMAP ... SUM ...` block here <<<\n")
        f.write(f"## Generate it with contact_map_setup.ipynb (canonical protocol):\n")
        f.write(f"##   all DNA atoms incl. H, residue separation > 3, cutoff 0.30 nm, LAMBDA=1.4\n")
        f.write(f"## Source file: {cmapdat}  (omit its trailing PRINT line when pasting).\n")
        f.write("\n")
        f.write("dbias: RESTRAINT ARG=cmap KAPPA=100000 AT=0.96\n")
        f.write("\n")
        f.write(f"# --- {n_cv}D well-tempered metadynamics, file-based multiple walkers ---\n")
        f.write("# REPID is replaced by sed in slurm script\n")
        f.write(f"metad: METAD ARG={','.join(metad_args)} ...\n")
        f.write("   PACE=500 HEIGHT=2.0\n")
        f.write("   SIGMA=" + ','.join(['3.0'] * len(coord_cvs) + ['0.25']) + "\n")
        f.write("   BIASFACTOR=5 TEMP=298\n")
        f.write("   FILE=hills.dat\n")
        f.write("   GRID_MIN=" + ','.join(['0'] * len(coord_cvs) + ['0']) +
                "  GRID_MAX=" + ','.join(['150'] * len(coord_cvs) + ['8']) +
                "  GRID_BIN=" + ','.join(['100'] * len(coord_cvs) + ['200']) + "\n")
        f.write(f"   WALKERS_N={args.walkers} WALKERS_ID=REPID WALKERS_DIR=../\n")
        f.write("   WALKERS_RSTRIDE=500\n")
        f.write("...\n")
        f.write("\n")
        f.write("bias: REWEIGHT_BIAS ARG=metad.bias\n")
        f.write("hh_de2e: HISTOGRAM ARG=de2e STRIDE=100 GRID_MIN=0 GRID_MAX=8 "
                "GRID_BIN=200 BANDWIDTH=0.05 LOGWEIGHTS=bias\n")
        f.write("ff_de2e: CONVERT_TO_FES GRID=hh_de2e\n")
        f.write("\n")
        f.write("DUMPGRID GRID=ff_de2e FILE=fes_de2e_rREPID.dat\n")
        f.write("\n")
        f.write(f"PRINT ARG={','.join(metad_args)},theta,cmap STRIDE=100 FILE=colvar_rREPID.dat\n")

    print(f"Written: {outdat}")
    for pc in protein_chains:
        n_prot = max(a['serial'] for a in chain(atoms, pc))
        print(f"  protein chain {pc}: last serial {n_prot}, tail: {len(tails[pc])} atoms (CA+CB res 1-{TAIL_RES_END})")
    print(f"  DNA atoms: {dna_first}-{dna_last} ({dna_last - dna_first + 1} total, chains {dna_chains[0]}+{dna_chains[1]})")
    print(f"  dna_backbone: {len(dna_bb)} atoms (P+C1', chains {dna_chains[0]}+{dna_chains[1]})")
    print(f"  dna_end1: {len(end1_b)} atoms (first nt chain {dna_chains[0]}) + {len(end1_c)} (last nt chain {dna_chains[1]})")
    print(f"  dna_end2: {len(end2_b)} atoms (last nt chain {dna_chains[0]}) + {len(end2_c)} (first nt chain {dna_chains[1]})")
    print(f"  METAD ARG ({n_cv}D): {','.join(metad_args)}")
    print()
    print(f"Contact map NOT generated here — use contact_map_setup.ipynb to produce")
    print(f"  {cmapdat}, then paste its `cmap: CONTACTMAP ... SUM ...` block into {outdat}.")
    print(f"Next: symlink {ref_pdb} into each replica directory before running metadynamics.")


if __name__ == '__main__':
    main()
