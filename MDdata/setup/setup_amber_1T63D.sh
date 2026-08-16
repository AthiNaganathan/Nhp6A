#!/bin/bash
set -e

# Preparation steps (run once before this script):
#   1. Convert CIF to PDB with biopython:
#        python3 -c "from Bio.PDB import MMCIFParser,PDBIO; p=MMCIFParser(QUIET=True); s=p.get_structure('1T63D','AF_Nhp6A/Alphafold_1T63D_1DNA.cif'); io=PDBIO(); io.set_structure(s); io.save('AF_Nhp6A/1T63D_DNA_AF.pdb')"
#   2. Rename AlphaFold phosphate oxygens (OP1/OP2/OP3 → O1P/O2P/O3P):
#        sed -e 's/ OP1 / O1P /g' -e 's/ OP2 / O2P /g' -e 's/ OP3 / O3P /g' \
#            AF_Nhp6A/1T63D_DNA_AF.pdb > AF_Nhp6A/1T63D_DNA_AF_fix.pdb
#   3. Remove 5' terminal phosphate atoms (O3P, P, O1P, O2P from residue 1 of chains B and C)
#      — required by GROMACS parambsc1 (DG5/DC5 rtp entries do not include these atoms)
#      Use the Python snippet in the project notes or edit AF_Nhp6A/1T63D_DNA_AF_fix.pdb manually.

pdb="1T63D_DNA"
pdbfile="AF_Nhp6A/1T63D_DNA_AF_fix.pdb"

ff=parambsc1
wat=tip3p

top="${pdb}_${ff}_${wat}"
out="data/${pdb}_${ff}_${wat}"

#gmx pdb2gmx -f $pdbfile -ignh -p $top -o $out -i $top -n $out <<EOF
#1
#1
#EOF

inp=$out
out="data/${pdb}_${ff}_${wat}_ed"
#gmx editconf -f $inp -o $out -d 1 -bt octahedron -c

top="${pdb}_${ff}_${wat}.top"
topnew="${pdb}_${ff}_${wat}_ions.top"
#cp $top $topnew
top=$topnew

inp=$out
out="data/${pdb}_${ff}_${wat}_solv"
#gmx solvate -p $top -cp $inp -o $out

mdp="data/mdp/em.mdp"
inp=$out
out="data/${pdb}_${ff}_${wat}_genion"
#gmx grompp -p $top -c $inp -f $mdp -o $out -maxwarn 3

inp=$out
ions="0.15"
out="data/${pdb}_${ff}_${wat}_ions${ions}"
#gmx genion -s $inp -p $top -o $out -conc $ions -neutral -pname K <<EOF
#14
#EOF

mdp="data/mdp/em.mdp"
inp=$out
out="data/${pdb}_${ff}_${wat}_ions${ions}_mini"
#gmx grompp -p $top -c $inp -f $mdp -o $out -maxwarn 2

inp=$out
out=$inp
#gmx mdrun -v -ntmpi 2 -ntomp 1 -s $inp -deffnm $out

mdp="data/mdp/nvt.mdp"
inp=$out
out="data/${pdb}_${ff}_${wat}_ions${ions}_nvt"
#gmx grompp -p $top -c $inp -f $mdp -o $out -r $inp -maxwarn 2

inp=$out
out=$inp
#gmx mdrun -v -ntmpi 2 -ntomp 1 -s $inp -deffnm $out

mdp="data/mdp/npt.mdp"
inp=$out
out="data/${pdb}_${ff}_${wat}_ions${ions}_npt"
#gmx grompp -f $mdp -c $inp -p $top -r $inp -o $out -maxwarn 1

inp=$out
out=$inp
#gmx mdrun -v -ntmpi 2 -ntomp 1 -s $inp -deffnm $out

mdp="data/mdp/md.mdp"
inp=$out
out="data/${pdb}_${ff}_${wat}_ions${ions}_md"
gmx grompp -f $mdp -c $inp -p $top -r $inp -o $out -maxwarn 2

inp=$out
out=$inp
#gmx mdrun -v -ntmpi 2 -ntomp 1 -s $inp -deffnm $out -nsteps 500000000

rm \#* */\#* ../*/\#*
exit
