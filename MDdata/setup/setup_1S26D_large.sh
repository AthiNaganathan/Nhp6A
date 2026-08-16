#!/bin/bash
# Setup S26D with a larger simulation box (-d 1.5 nm instead of -d 1 nm).
# Starts from the existing pdb2gmx output (data/1S26D/1S26D_DNA_parambsc1_tip3p.gro)
# and reruns editconf onward. Outputs go to data/1S26D_large/.
#
# Run from the project root.
set -e

source /opt/gromacs/2024.2-plumed/bin/GMXRC.bash

pdb="1S26D_DNA"
ff=parambsc1
wat=tip3p
ions=0.15
base="data/1S26D_large"

mkdir -p "${base}"

# Symlink forcefield and ITP files so topology #include paths resolve
ln -sf "$(pwd)/amber14sb_parmbsc1.ff" "${base}/amber14sb_parmbsc1.ff"
for itp in ${pdb}_${ff}_${wat}_*.itp; do
    ln -sf "$(pwd)/${itp}" "${base}/${itp}"
done

# Topology: copy clean pdb2gmx output (protein+DNA only, no solvent/ions)
# The _ions.top has counts from a previous run — do not use it as source
top_orig="${pdb}_${ff}_${wat}.top"
top="${base}/${pdb}_${ff}_${wat}_ions.top"
cp "${top_orig}" "${top}"

# --- editconf: larger box (-d 1.5) ---
pdb2gmx_gro="data/1S26D/${pdb}_${ff}_${wat}.gro"
out="${base}/${pdb}_${ff}_${wat}_ed"
gmx editconf -f "${pdb2gmx_gro}" -o "${out}" -d 1.5 -bt octahedron -c

# --- solvate ---
inp="${out}"
out="${base}/${pdb}_${ff}_${wat}_solv"
gmx solvate -p "${top}" -cp "${inp}" -o "${out}"

# --- genion ---
mdp="data/mdp/em.mdp"
inp="${out}"
out="${base}/${pdb}_${ff}_${wat}_genion"
gmx grompp -p "${top}" -c "${inp}" -f "${mdp}" -o "${out}" -maxwarn 3

inp="${out}"
out="${base}/${pdb}_${ff}_${wat}_ions${ions}"
gmx genion -s "${inp}" -p "${top}" -o "${out}" -conc ${ions} -neutral -pname K <<EOF
14
EOF

# --- energy minimisation ---
mdp="data/mdp/em.mdp"
inp="${out}"
out="${base}/${pdb}_${ff}_${wat}_ions${ions}_mini"
gmx grompp -p "${top}" -c "${inp}" -f "${mdp}" -o "${out}" -maxwarn 2
gmx mdrun -v -ntmpi 2 -ntomp 1 -s "${out}" -deffnm "${out}"

# --- NVT equilibration ---
mdp="data/mdp/nvt.mdp"
inp="${out}"
out="${base}/${pdb}_${ff}_${wat}_ions${ions}_nvt"
gmx grompp -p "${top}" -c "${inp}" -f "${mdp}" -o "${out}" -r "${inp}" -maxwarn 2
gmx mdrun -v -ntmpi 2 -ntomp 1 -s "${out}" -deffnm "${out}"

# --- NPT equilibration ---
mdp="data/mdp/npt.mdp"
inp="${out}"
out="${base}/${pdb}_${ff}_${wat}_ions${ions}_npt"
gmx grompp -f "${mdp}" -c "${inp}" -p "${top}" -r "${inp}" -o "${out}" -maxwarn 1
gmx mdrun -v -ntmpi 2 -ntomp 1 -s "${out}" -deffnm "${out}"

# --- production TPR ---
mdp="data/mdp/md.mdp"
inp="${out}"
out="${base}/${pdb}_${ff}_${wat}_ions${ions}_md"
gmx grompp -f "${mdp}" -c "${inp}" -p "${top}" -r "${inp}" -o "${out}" -maxwarn 2

echo ""
echo "Setup complete. Production TPR: ${out}.tpr"
echo "Next: submit metadynamics with slurm_meta_2D_1S26D_large.sh"

rm -f \#* "${base}/\#"*
