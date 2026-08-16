#!/bin/bash
set -e

pdb="DNA"
pdbfile="AF_Nhp6A/${pdb}_AF_fix.pdb"

ff=parambsc1
wat=tip3p

top="${pdb}_${ff}_${wat}"
out="data/${pdb}_${ff}_${wat}"

##gmx pdb2gmx -f $pdbfile -ignh -p $top -o $out -i $top -n $out <<EOF
#1
#1
#EOF

inp=$out
out="data/${pdb}_${ff}_${wat}_ed"
#gmx editconf -f $inp -o $out -d 1.5 -bt octahedron -c

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
#3
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
##gmx mdrun -v -ntmpi 4 -ntomp 1 -s $inp -deffnm $out -nsteps 500000000

rm \#* */\#* ../*/\#*
exit
