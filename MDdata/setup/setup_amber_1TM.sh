#!/bin/bash
# Setup 1TM (triple mutant S26D+S41D+T63D) + DNA binary complex for 2D WT-MetaD.
# Follows the same protocol as setup_amber_1S26D.sh.
# Run from the project root.
set -e

# ---------------------------------------------------------------------------
# Preparation steps (run once, manually, before uncommenting gmx commands):
#
#   1. Convert CIF to PDB with biopython:
#        python3 -c "
#        from Bio.PDB import MMCIFParser, PDBIO
#        p = MMCIFParser(QUIET=True)
#        s = p.get_structure('1TM', 'AF_Nhp6A/Alphafold_1TM_1DNA.cif')
#        io = PDBIO(); io.set_structure(s); io.save('AF_Nhp6A/1TM_DNA_AF.pdb')
#        "
#
#   2. Rename AlphaFold phosphate oxygens (OP1/OP2/OP3 -> O1P/O2P/O3P):
#        sed -e 's/ OP1 / O1P /g' -e 's/ OP2 / O2P /g' -e 's/ OP3 / O3P /g' \
#            AF_Nhp6A/1TM_DNA_AF.pdb > AF_Nhp6A/1TM_DNA_AF_fix.pdb
#
#   3. Remove 5' terminal phosphate atoms (O3P, P, O1P, O2P from residue 1 of
#      chains B and C) — required by GROMACS parambsc1 (DG5/DC5 rtp entries):
#        python3 -c "
#        remove = {('B','1'), ('C','1')}; drop_atoms = {'O3P','P','O1P','O2P'}
#        with open('AF_Nhp6A/1TM_DNA_AF_fix.pdb') as f, \
#             open('AF_Nhp6A/1TM_DNA_AF_fix2.pdb','w') as out:
#            for line in f:
#                if line.startswith(('ATOM','HETATM')):
#                    chain=line[21]; resnum=line[22:26].strip(); atname=line[12:16].strip()
#                    if (chain,resnum) in remove and atname in drop_atoms: continue
#                out.write(line)
#        import os; os.rename('AF_Nhp6A/1TM_DNA_AF_fix2.pdb','AF_Nhp6A/1TM_DNA_AF_fix.pdb')
#        "
#   Steps 1-3 must be completed before running the gmx commands below.
# ---------------------------------------------------------------------------

source /opt/gromacs/2024.2-plumed/bin/GMXRC.bash

pdb="1TM_DNA"
pdbfile="AF_Nhp6A/1TM_DNA_AF_fix.pdb"

ff=parambsc1
wat=tip3p
ions=0.15

top="${pdb}_${ff}_${wat}"
out="data/1TM/${pdb}_${ff}_${wat}"

mkdir -p data/1TM

# --- Topology and initial structure ---
#gmx pdb2gmx -f $pdbfile -ignh -p $top -o $out -i $top -n $out <<EOF
#1
#1
#EOF

inp=$out
out="data/1TM/${pdb}_${ff}_${wat}_ed"
#gmx editconf -f $inp -o $out -d 1 -bt octahedron -c

top="${pdb}_${ff}_${wat}.top"
topnew="${pdb}_${ff}_${wat}_ions.top"
#cp $top $topnew
top=$topnew

inp=$out
out="data/1TM/${pdb}_${ff}_${wat}_solv"
#gmx solvate -p $top -cp $inp -o $out

mdp="data/mdp/em.mdp"
inp=$out
out="data/1TM/${pdb}_${ff}_${wat}_genion"
#gmx grompp -p $top -c $inp -f $mdp -o $out -maxwarn 3

inp=$out
out="data/1TM/${pdb}_${ff}_${wat}_ions${ions}"
#gmx genion -s $inp -p $top -o $out -conc $ions -neutral -pname K <<EOF
#14
#EOF

# --- Energy minimisation ---
mdp="data/mdp/em.mdp"
inp=$out
out="data/1TM/${pdb}_${ff}_${wat}_ions${ions}_mini"
#gmx grompp -p $top -c $inp -f $mdp -o $out -maxwarn 2
#gmx mdrun -v -ntmpi 2 -ntomp 1 -s $out -deffnm $out

# --- NVT equilibration ---
mdp="data/mdp/nvt.mdp"
inp=$out
out="data/1TM/${pdb}_${ff}_${wat}_ions${ions}_nvt"
#gmx grompp -p $top -c $inp -f $mdp -o $out -r $inp -maxwarn 2
#gmx mdrun -v -ntmpi 2 -ntomp 1 -s $out -deffnm $out

# --- NPT equilibration ---
mdp="data/mdp/npt.mdp"
inp=$out
out="data/1TM/${pdb}_${ff}_${wat}_ions${ions}_npt"
#gmx grompp -f $mdp -c $inp -p $top -r $inp -o $out -maxwarn 1
#gmx mdrun -v -ntmpi 2 -ntomp 1 -s $out -deffnm $out

# --- Production TPR (metadynamics launched by SLURM script) ---
mdp="data/mdp/md.mdp"
inp=$out
out="data/1TM/${pdb}_${ff}_${wat}_ions${ions}_md"
gmx grompp -f $mdp -c $inp -p $top -r $inp -o $out -maxwarn 2

# --- Extract protein+DNA reference PDB for PLUMED ---
# Requires a protein+DNA index group; create one if not present:
#   echo -e "q\n" | gmx make_ndx -f data/1TM/${pdb}_${ff}_${wat}_ions${ions}_npt.gro \
#       -n data/1TM/${pdb}_${ff}_${wat}.ndx -o data/1TM/1TM_protdna.ndx
# Then select "Protein" | "DNA" (check group numbers with make_ndx first):
#   echo "Protein_DNA" | gmx trjconv \
#       -f data/1TM/${pdb}_${ff}_${wat}_ions${ions}_npt.gro \
#       -s data/1TM/${pdb}_${ff}_${wat}_ions${ions}_npt.tpr \
#       -n data/1TM/1TM_protdna.ndx \
#       -o 1TM_DNA_parambsc1.pdb \
#       -pbc whole -dump 0

# --- Generate PLUMED inputs from equilibrated binary PDB ---
# Run AFTER extracting 1TM_DNA_parambsc1.pdb above.
# Generates: plumed_meta_2D_1TM_DNA.dat, 1TM_DNA_parambsc1_cmap.dat, 1TM_DNA_parambsc1_cmap.txt
#
# python3 gen_plumed_inputs_binary.py 1TM_DNA_parambsc1.pdb 1TM
#
# CRITICAL: inspect the cmap contact count (expect ~130-160 for 14-bp duplex)
#           and verify cmap value = 1.0 on the reference structure before
#           submitting production runs.

rm -f \#* data/1TM/\#*

echo "Setup complete. Production TPR: data/1TM/${pdb}_${ff}_${wat}_ions${ions}_md.tpr"
echo "Next: extract reference PDB, run gen_plumed_inputs_binary.py, then submit slurm_meta_2D_1TM_large.sh"
