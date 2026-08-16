#!/bin/bash
# Setup protein-only (apo) 1TM for 2D WT-MetaD.
# Strips DNA from the binary ions0.15 GRO, builds a new octahedral box,
# solvates, adds ions, runs EM/NVT/NPT, and generates the PLUMED input.
# Follows the same protocol as setup_1S26D.sh.
# Run from the project root AFTER setup_amber_1TM.sh completes.
set -e

source /opt/gromacs/2024.2-plumed/bin/GMXRC.bash

sys="1TM"
ff=parambsc1
wat=tip3p
ions=0.15
base="data/${sys}_apo"

mkdir -p "${base}"

# Fresh working topology (protein-only; copy from binary top, remove DNA itp includes)
top_orig="${sys}_${ff}_${wat}.top"
top="${sys}_${ff}_${wat}_ions.top"
#cp "${top_orig}" "${top}"
# NOTE: edit ${top} to remove the DNA .itp includes before proceeding.

# --- Extract protein chain from binary complex ---
#src_gro="data/1TM/1TM_DNA_${ff}_${wat}_ions${ions}.gro"
#src_tpr="data/1TM/1TM_DNA_${ff}_${wat}_ions${ions}_md.tpr"
#prot_gro="${base}/${sys}_${ff}_${wat}.gro"
#echo "Protein" | gmx trjconv -f "${src_gro}" -s "${src_tpr}" -o "${prot_gro}" -pbc whole

# --- New octahedral box (1.2 nm clearance) ---
#inp="${prot_gro}"
#out="${base}/${sys}_${ff}_${wat}_ed"
#gmx editconf -f "${inp}" -o "${out}" -d 1.2 -bt octahedron -c

# --- Solvate ---
#inp="${out}"
#out="${base}/${sys}_${ff}_${wat}_solv"
#gmx solvate -cp "${inp}" -p "${top}" -o "${out}"

# --- Genion ---
#mdp="data/mdp/em.mdp"
#inp="${out}"
#out="${base}/${sys}_${ff}_${wat}_genion"
#gmx grompp -p "${top}" -c "${inp}" -f "${mdp}" -o "${out}" -maxwarn 3
#inp="${out}"
#out="${base}/${sys}_${ff}_${wat}_ions${ions}"
#echo "SOL" | gmx genion -s "${inp}" -p "${top}" -o "${out}" -conc ${ions} -neutral -pname K -nname CL

# --- Energy minimisation ---
#mdp="data/mdp/em.mdp"
#inp="${out}"
#out="${base}/${sys}_${ff}_${wat}_ions${ions}_mini"
#gmx grompp -p "${top}" -c "${inp}" -f "${mdp}" -o "${out}" -maxwarn 2
#gmx mdrun -v -ntmpi 2 -ntomp 1 -s "${out}" -deffnm "${out}"

# --- NVT equilibration ---
#mdp="data/mdp/nvt.mdp"
#inp="${out}"
#out="${base}/${sys}_${ff}_${wat}_ions${ions}_nvt"
#gmx grompp -p "${top}" -c "${inp}" -f "${mdp}" -o "${out}" -r "${inp}" -maxwarn 2
#gmx mdrun -v -ntmpi 2 -ntomp 1 -s "${out}" -deffnm "${out}"

# --- NPT equilibration ---
mdp="data/mdp/npt.mdp"
inp="${base}/${sys}_${ff}_${wat}_ions${ions}_nvt"
out="${base}/${sys}_${ff}_${wat}_ions${ions}_npt"
gmx grompp -f "${mdp}" -c "${inp}" -p "${top}" -r "${inp}" -o "${out}" -maxwarn 1
gmx mdrun -v -ntmpi 2 -ntomp 1 -s "${out}" -deffnm "${out}"

# --- Production TPR (metadynamics launched by SLURM script) ---
mdp="data/mdp/md.mdp"
inp="${out}"
out="${base}/${sys}_${ff}_${wat}_ions${ions}_md"
gmx grompp -f "${mdp}" -c "${inp}" -p "${top}" -r "${inp}" -o "${out}" -maxwarn 2

echo ""
echo "Setup complete. Production TPR: ${out}.tpr"

# --- Reference PDB for PLUMED (protein-only, last NPT frame) ---
echo "Protein" | gmx trjconv \
    -f "${base}/${sys}_${ff}_${wat}_ions${ions}_npt.gro" \
    -s "${base}/${sys}_${ff}_${wat}_ions${ions}_npt.tpr" \
    -o "${sys}_${ff}.pdb" \
    -pbc whole -dump 0

# --- Generate PLUMED input (q_core + n_tail_core CVs) ---
python3 gen_plumed_inputs.py "${sys}_${ff}.pdb" ${sys}

echo "Next: submit metadynamics with slurm_meta_2D_${sys}.sh"

rm -f \#* "${base}/"\#*
