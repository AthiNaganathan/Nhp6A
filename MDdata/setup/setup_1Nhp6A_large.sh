#!/bin/bash
# Setup protein-only (apo) 1Nhp6A in a LARGER octahedral box for 2D WT-MetaD.
# Mirrors setup_1Nhp6A.sh but: box -d 1.7 (was 1.2, +1 nm/dim), outputs to
# data/1Nhp6A_apo_large/, and generates a 6-walker PLUMED input with the
# rmsd_core wall at 0.5 nm and per-helix ALPHARMSD (monitored only).
#
# EM/NVT/NPT and the production grompp need no PLUMED, so an un-patched GROMACS
# is sufficient here; the metadynamics itself is launched separately with the
# PLUMED-patched build.
#
# Run from the project root.
set -e

source /usr/local/gromacs/2024.1/bin/GMXRC.bash

sys="1Nhp6A"
ff=parambsc1
wat=tip3p
ions=0.15
tag=large
base="data/${sys}_apo_${tag}"

mkdir -p "${base}"

# Symlink forcefield and protein ITP so topology #include paths resolve
# whether grompp searches cwd or the top-file directory.
ln -sf "$(pwd)/amber14sb_parmbsc1.ff" "${base}/amber14sb_parmbsc1.ff"
for itp in 1Nhp6A_DNA_${ff}_${wat}_Protein_chain_A*.itp; do
    ln -sf "$(pwd)/${itp}" "${base}/${itp}"
done

# Working topology in base/ (solvate/genion update it in place) — do NOT reuse
# the repo-root _ions.top, whose solvent/ion counts belong to the small box.
top_orig="${sys}_${ff}_${wat}.top"
top="${base}/${sys}_${ff}_${wat}_ions.top"
cp "${top_orig}" "${top}"

# --- New, larger octahedral box (1.7 nm clearance; +0.5 nm/side vs 1.2) ---
# Reuse the existing protein-only gro (same solute as the small-box apo run).
inp="data/${sys}_apo/${sys}_${ff}_${wat}.gro"
out="${base}/${sys}_${ff}_${wat}_ed"
gmx editconf -f "${inp}" -o "${out}" -d 1.7 -bt octahedron -c

# --- Solvate ---
inp="${out}"
out="${base}/${sys}_${ff}_${wat}_solv"
gmx solvate -cp "${inp}" -p "${top}" -o "${out}"

# --- Genion ---
mdp="data/mdp/em.mdp"
inp="${out}"
out="${base}/${sys}_${ff}_${wat}_genion"
gmx grompp -p "${top}" -c "${inp}" -f "${mdp}" -o "${out}" -maxwarn 3
inp="${out}"
out="${base}/${sys}_${ff}_${wat}_ions${ions}"
echo "SOL" | gmx genion -s "${inp}" -p "${top}" -o "${out}" -conc ${ions} -neutral -pname K -nname CL

# --- Energy minimisation ---
mdp="data/mdp/em.mdp"
inp="${out}"
out="${base}/${sys}_${ff}_${wat}_ions${ions}_mini"
gmx grompp -p "${top}" -c "${inp}" -f "${mdp}" -o "${out}" -maxwarn 2
gmx mdrun -v -ntmpi 2 -ntomp 1 -s "${out}" -deffnm "${out}"

# --- NVT equilibration ---
mdp="data/mdp/nvt.mdp"
inp="${out}"
out="${base}/${sys}_${ff}_${wat}_ions${ions}_nvt"
gmx grompp -p "${top}" -c "${inp}" -f "${mdp}" -o "${out}" -r "${inp}" -maxwarn 2
gmx mdrun -v -ntmpi 2 -ntomp 1 -s "${out}" -deffnm "${out}"

# --- NPT equilibration ---
mdp="data/mdp/npt.mdp"
inp="${out}"
out="${base}/${sys}_${ff}_${wat}_ions${ions}_npt"
gmx grompp -f "${mdp}" -c "${inp}" -p "${top}" -r "${inp}" -o "${out}" -maxwarn 1
gmx mdrun -v -ntmpi 2 -ntomp 1 -s "${out}" -deffnm "${out}"

# --- Production TPR (metadynamics launched separately) ---
mdp="data/mdp/md.mdp"
inp="${out}"
out="${base}/${sys}_${ff}_${wat}_ions${ions}_md"
gmx grompp -f "${mdp}" -c "${inp}" -p "${top}" -r "${inp}" -o "${out}" -maxwarn 2

echo ""
echo "Setup complete. Production TPR: ${out}.tpr"

# --- Reference PDB for PLUMED (protein-only, last NPT frame), tagged _large ---
echo "Protein" | gmx trjconv \
    -f "${base}/${sys}_${ff}_${wat}_ions${ions}_npt.gro" \
    -s "${base}/${sys}_${ff}_${wat}_ions${ions}_npt.tpr" \
    -o "${sys}_${ff}_${tag}.pdb" \
    -pbc whole -dump 0

# --- Generate PLUMED input: 6 walkers, rmsd wall 0.5 nm, per-helix helicity ---
# Helix ranges are DSSP-derived (fraction >= 0.5): H1 27-42, H2 48-61, H3 64-91.
python3 gen_plumed_inputs.py "${sys}_${ff}_${tag}.pdb" ${sys} \
    --walkers 6 --rmsd-wall 0.5 --helices 27-42,48-61,64-91 --out-tag ${tag}

echo "Next: launch the 6 walkers with plumed_meta_2D_${sys}_${tag}.dat, keeping"
echo "      data/${sys}_apo_${tag}/, ref_core_ca_${sys}_${tag}.pdb and"
echo "      ${sys}_${ff}_${tag}.pdb alongside the production TPR."

rm -f \#* "${base}/"\#*
