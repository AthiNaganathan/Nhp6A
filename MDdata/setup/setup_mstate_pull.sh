#!/bin/bash
# Prepare the steered core-detachment pull for one m-state probe start frame (PLAN.md, A1.2).
#
#   setup_mstate_pull.sh <cluster> [system]     e.g. setup_mstate_pull.sh 0
#                                               setup_mstate_pull.sh 2 2Nhp6A
#
# system = 1Nhp6A (default; Stage A, 1:1) or 2Nhp6A (Stage B, mM: copy B's core is pulled,
# copy A untouched; PLUMED template plumed_pull_mstate_2Nhp6A.dat, start frames
# data/mstate_2Nhp6A/start/2Nhp6A_mstate_start_c<k>.gro, run dir data/mstate_2Nhp6A/pull/c<k>/).
#
# 1. grompp with the protocol of record (data/mdp/md_equil.mdp: C-rescale, gen_vel with a
#    recorded seed), nsteps = 40 ns, -maxwarn 0, on the full-system start frame
#    data/mstate/start/c<k>.gro (dumped from the unbiased holo/2:1 runs by mstate_select_frames.py).
# 2. 0-step mdrun with the pull input stripped of both biases -> D0 = d_core as PLUMED sees it.
# 3. Fill __D0__/__D1__ (= D0 + 2.0 nm) into the PLUMED template
#    -> <run dir>/plumed_<P>.dat
# Every file in c<k>/ carries the traceable prefix P = <system>_mstate_pull_c<k>.
# 4. 0-step mdrun with the full input: the pull bias must be ~0 at step 0, and the IDR wall
#    must be inactive (n_IDR-DNA >= 11.6 (1:1) / 5.3 (2:1, copy B) in every start frame).
# Production of the pull (40 ns) is launched separately, after review.
#
# Run from the project root on CALMODULIN (PLUMED build /opt/gromacs/2024.2). On rubisco every
# gmx_mpi launch (singleton or mpirun) hangs at OpenMPI init -- do not use it there.
set -euo pipefail

k=${1:?usage: setup_mstate_pull.sh <cluster> [1Nhp6A|2Nhp6A]}
SYS=${2:-1Nhp6A}
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
GMXRC=${GMXRC:-/opt/gromacs/2024.2/bin/GMXRC.bash}     # calmodulin's PLUMED-patched build
set +u; source "$GMXRC"; set -u
GMX=gmx_mpi
$GMX --version < /dev/null 2>/dev/null | grep -q "plumed_2.9.2" || { echo "not the PLUMED 2.9.2 build"; exit 1; }

NSTEPS=20000000                         # 40 ns @ 2 fs, matches STEP1 in the PLUMED template
case $SYS in
  1Nhp6A) SEED=$((48190 + k))           # recorded; distinct from the equil seeds 48170-48182
          TOP=1Nhp6A_DNA_parambsc1_tip3p_ions.top      # same topology as the holo equil runs
          CONF=data/mstate/start/c${k}.gro
          D=data/mstate/pull/c${k}
          TPL=plumed_pull_mstate.dat ;;
  2Nhp6A) SEED=$((48300 + k))           # recorded; distinct from all 1:1 m-state seeds (48190-48241)
          TOP=2Nhp6A_DNA_parambsc1_tip3p_ions.top      # same topology as the 2:1 equil runs
          CONF=data/mstate_2Nhp6A/start/2Nhp6A_mstate_start_c${k}.gro
          D=data/mstate_2Nhp6A/pull/c${k}
          TPL=plumed_pull_mstate_2Nhp6A.dat ;;
  *) echo "unknown system $SYS (1Nhp6A|2Nhp6A)"; exit 1 ;;
esac
P=${SYS}_mstate_pull_c${k}             # traceable prefix for every file in $D
mkdir -p "$D"

# --- 1. tpr ---
sed -e "s/__SEED__/${SEED}/" -e "s/^nsteps .*/nsteps                   = ${NSTEPS}   ; 40 ns pull/" \
    data/mdp/md_equil.mdp > "$D/${P}_grompp.mdp"
$GMX grompp -f "$D/${P}_grompp.mdp" -c "$CONF" -p "$TOP" -o "$D/${P}.tpr" -po "$D/${P}_mdout.mdp" -maxwarn 0 \
    > "$D/${P}_grompp.log" 2>&1

# --- 2. D0 from a bias-free 0-step run ---
awk '/^pull: MOVINGRESTRAINT/{skip=1} skip&&/^\.\.\.$/{skip=0; next} skip{next}
     /^idr_wall:/{next} {gsub(/,pull\.bias,pull\.work,idr_wall\.bias/,""); print}' \
    "$TPL" | sed "s#FILE=colvar_pull.dat#FILE=colvar_${P}_d0.dat#" > "$D/plumed_${P}_d0.dat"
( cd "$D" && $GMX mdrun -s ${P}.tpr -deffnm ${P}_d0 -nsteps 0 -plumed plumed_${P}_d0.dat -ntomp 4 \
    > ${P}_d0.out 2>&1 )
D0=$(awk '!/^#/{printf "%.4f", $2; exit}' "$D/colvar_${P}_d0.dat")
D1=$(awk -v d="$D0" 'BEGIN{printf "%.4f", d + 2.0}')
echo "${SYS} c${k}: D0 = ${D0} nm  ->  D1 = ${D1} nm"

# --- 3. full pull input ---
sed -e "s/__D0__/${D0}/" -e "s/__D1__/${D1}/" "$TPL" \
    | sed "s#FILE=colvar_pull.dat#FILE=colvar_${P}.dat#" > "$D/plumed_${P}.dat"

# --- 4. 0-step check with both biases ---
( cd "$D" && $GMX mdrun -s ${P}.tpr -deffnm ${P}_t0 -nsteps 0 -plumed plumed_${P}.dat -ntomp 4 \
    > ${P}_t0.out 2>&1 )
echo "${SYS} c${k}: step-0 colvar_${P}.dat:"
cat "$D/colvar_${P}.dat"
rm -f "$D"/${P}_{d0,t0}.{gro,cpt,edr,trr,xtc} "$D"/\#*
