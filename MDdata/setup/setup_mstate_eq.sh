#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-06:00:00
#SBATCH --job-name=mstate_eq
#SBATCH --output=data/mstate/eq/slurm_%x_%j.out
#
# Re-solvate and equilibrate one m-state probe configuration after its steered pull
# (PLAN.md A1.3), reproducing the holo protocol of record (setup_amber_eq.sh chain):
#
#   build : pull end frame -> strip water/ions -> reassemble complex (pbc cluster) ->
#           octahedron, 1.0 nm buffer -> TIP3P -> 0.15 M KCl (neutral) -> EM
#   nvt   : 1 ns NVT, heavy-atom POSRES 1000 (data/mdp/nvt.mdp, -r = EM structure)
#   npt   : 1 ns NPT Berendsen, POSRES      (data/mdp/npt.mdp, -r = NVT structure)
#
#   em2   : optional second pass of the same em.mdp from the EM output (nvt then uses it)
#   md    : UNBIASED production (PLAN.md A2), replica <rep>: data/mdp/md_equil.mdp from the NPT
#           end structure, gen_vel seed 48200+10k+rep, 100 ns (nsteps 50e6; extend later with
#           convert-tpr -until + rerun, which resumes from the checkpoint), force-free monitor
#           plumed_monitor_mstate.dat. grompp needs the topology (calmodulin); on a host without
#           it (insulin) copy the run dir over and the stage only runs mdrun.
#   sbatch setup_mstate_eq.sh <k> <build|em2|nvt|npt>    (calmodulin, from the project root)
#   sbatch --time=3-00:00:00 --job-name=mstate_md --output=data/mstate/md/slurm_%x_%j.out \
#          setup_mstate_eq.sh <k> md <rep>               (calmodulin or insulin)
#   bash setup_mstate_eq.sh <k> md <rep> prep           (grompp + staging only, no mdrun)
#   MD_NSTEPS=<n> overrides the md length (default 50000000 = 100 ns), applied when the tpr is built
#
# Run the stages one at a time and inspect each before the next. No PLUMED and no IDR wall:
# the construction bias ends with the pull. grompp warning counts must match the A0 audit
# (genion 1 net charge / EM 0 / NVT 2 obsolete keys / NPT 1 Berendsen) or the stage aborts.
# Every file carries the prefix P = 1Nhp6A_mstate_eq_c<k>.
#
# Stage B 2:1 (copy B detached): MSTATE_SYS=2Nhp6A, files 2Nhp6A_mstate_eq_c<k>_* in
# data/mstate_2Nhp6A/eq/c<k>/, start = placed pre-image-contact pull frame, buffer 1.5 nm:
#   MSTATE_SYS=2Nhp6A sbatch --output=data/mstate_2Nhp6A/eq/slurm_%x_%j.out setup_mstate_eq.sh <k> build
# md: monitor plumed_monitor_mstate_2Nhp6A.dat, seed 48400+10k+rep, data/mstate_2Nhp6A/md/c<k>/rep<r>
set -euo pipefail

k=${1:?usage: setup_mstate_eq.sh <k> <build|em2|nvt|npt|md> [rep]}
stage=${2:?usage: setup_mstate_eq.sh <k> <build|em2|nvt|npt|md> [rep]}
ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
cd "$ROOT"
case $(hostname -s) in      # PLUMED 2.9.2-patched gmx_mpi 2024.2 per host (project CLAUDE.md)
    insulin) GMXRC=/home/daviddesancho/opt/gromacs-2024.2/bin/GMXRC.bash
             # this build loads PLUMED at runtime (as in every slurm_meta_*_insulin*.sh)
             export PLUMED_KERNEL=/home/daviddesancho/opt/plumed-2.9.2/lib/libplumedKernel.so ;;
    rubisco) GMXRC=/opt/gromacs/2024.2-plumed/bin/GMXRC.bash       # PLUMED 2.9.2 linked in
             export HWLOC_COMPONENTS=-gl ;;   # else gmx_mpi hangs at OpenMPI init (global CLAUDE.md)
    *)       GMXRC=/opt/gromacs/2024.2/bin/GMXRC.bash ;;           # calmodulin
esac
set +u; source "$GMXRC"; set -u   # (GMXRC trips set -u)
GMX="gmx_mpi"
NT=${NT:-${SLURM_CPUS_PER_TASK:-8}}   # NT=<n> for non-SLURM hosts (rubisco)

SYS=${MSTATE_SYS:-1Nhp6A}        # 2Nhp6A = Stage B, copy B detached (PLAN.md B1), see below
MD_NSTEPS=${MD_NSTEPS:-50000000}   # md length (2 fs steps); default 100 ns. Stage B production: MD_NSTEPS=125000000 (250 ns)
case $SYS in
1Nhp6A)
    P=1Nhp6A_mstate_eq_c${k}
    D=data/mstate/eq/c${k}
    PULL=data/mstate/pull/c${k}/1Nhp6A_mstate_pull_c${k}
    HOLO_TOP=1Nhp6A_DNA_parambsc1_tip3p_ions.top
    NSOLUTE=2489                # protein 1-1539 + DNA 1540-2489 (holo numbering)
    BUFFER=1.0
    MOLS='Protein_chain_A     1\nDNA_chain_B         1\nDNA_chain_C         1\n'
    MD_DIR=data/mstate/md; MONITOR=plumed_monitor_mstate.dat; SEED0=48200 ;;
2Nhp6A)
    # Stage B (user, 2026-10-12): the copy-B pull is only a structure generator. The start is the
    # last pull frame before the complex first came within 0.9 nm (= rcoulomb = rvdw) of its own
    # image, already placed (strands welded, both copies on the DNA) by
    # `mstate_pbc_check.py --system 2Nhp6A_pull --write-start`, so no trjconv/cluster step.
    # Buffer 1.5 nm (as 1IDR; 2:1 boxes at 1.0 nm were marginal), not the 1.0 nm of record.
    P=2Nhp6A_mstate_eq_c${k}
    D=data/mstate_2Nhp6A/eq/c${k}
    START=data/mstate_2Nhp6A/pull/c${k}/2Nhp6A_mstate_pull_c${k}_start.gro
    HOLO_TOP=2Nhp6A_DNA_parambsc1_tip3p_ions.top
    BUFFER=1.5
    MOLS='Protein_chain_A     1\nProtein_chain_B     1\nDNA_chain_C         1\nDNA_chain_D         1\n'
    # production from c1/c3 only: c0/c2 failed the NPT detachment gate (user, 2026-10-14)
    MD_DIR=data/mstate_2Nhp6A/md; MONITOR=plumed_monitor_mstate_2Nhp6A.dat; SEED0=48400 ;;
*) echo "unknown MSTATE_SYS=$SYS"; exit 1 ;;
esac
mkdir -p "$D"

# grompp with an exact expected warning count (A0 audit), never a blanket -maxwarn
grompp_checked() {    # <expected_warnings> <log> <grompp args...>
    local want=$1 log=$2; shift 2
    $GMX grompp "$@" -maxwarn "$want" > "$log" 2>&1 || { tail -30 "$log"; exit 1; }
    local got; got=$(grep -c "^WARNING [0-9]" "$log" || true)
    [[ $got == "$want" ]] || { echo "grompp: $got warnings, expected $want ($log)"; exit 1; }
    grep -A3 "^WARNING [0-9]" "$log" || true
}

case $stage in
build)
    if [[ $SYS == 2Nhp6A ]]; then
        [[ -f $START ]] || { echo "missing $START (mstate_pbc_check.py --write-start)"; exit 1; }
        cp "$START" "$D/${P}_complex.gro"
    else
    [[ -f ${PULL}.gro && -f ${PULL}.tpr ]] || { echo "missing pull output ${PULL}.{gro,tpr}"; exit 1; }
    # 1. complex only, reassembled across PBC (duplex = 2 molecules: -pbc whole is not enough)
    printf '[ complex ]\n' > "$D/${P}_complex.ndx"
    seq 1 $NSOLUTE | xargs -n 15 >> "$D/${P}_complex.ndx"
    printf 'complex\ncomplex\n' | $GMX trjconv -s "${PULL}.tpr" -f "${PULL}.gro" \
        -n "$D/${P}_complex.ndx" -pbc cluster -o "$D/${P}_complex.gro" > "$D/${P}_trjconv.log" 2>&1
    fi
    # 2. box, solvent, ions
    $GMX editconf -f "$D/${P}_complex.gro" -o "$D/${P}_box.gro" -d $BUFFER -bt octahedron -c \
        > "$D/${P}_editconf.log" 2>&1
    # topology: holo molecule definitions, includes made absolute so it can live in $D
    sed -n '1,/^\[ *molecules *\]/p' "$HOLO_TOP" \
        | sed -E "s#^\#include \"(\./)?#\#include \"${ROOT}/#" > "$D/${P}.top"
    printf "$MOLS" >> "$D/${P}.top"
    $GMX solvate -cp "$D/${P}_box.gro" -p "$D/${P}.top" -o "$D/${P}_solv.gro" \
        > "$D/${P}_solvate.log" 2>&1
    grompp_checked 1 "$D/${P}_genion_grompp.log" -f data/mdp/em.mdp -c "$D/${P}_solv.gro" \
        -p "$D/${P}.top" -o "$D/${P}_genion.tpr" -po "$D/${P}_genion_mdout.mdp"
    echo SOL | $GMX genion -s "$D/${P}_genion.tpr" -p "$D/${P}.top" -o "$D/${P}_ions0.15.gro" \
        -conc 0.15 -neutral -pname K -nname CL > "$D/${P}_genion.log" 2>&1
    # 3. EM
    grompp_checked 0 "$D/${P}_em_grompp.log" -f data/mdp/em.mdp -c "$D/${P}_ions0.15.gro" \
        -p "$D/${P}.top" -o "$D/${P}_em.tpr" -po "$D/${P}_em_mdout.mdp"
    ( cd "$D" && $GMX mdrun -s ${P}_em.tpr -deffnm ${P}_em -ntomp $NT < /dev/null )
    ;;
em2)
    # second pass of the SAME em.mdp from the first EM's output (user, 2026-10-05): the
    # post-pull structures stopped at machine precision with Fmax 650-2100 on single
    # carboxylate carbons, where every earlier project EM reached Fmax < 100
    grompp_checked 0 "$D/${P}_em2_grompp.log" -f data/mdp/em.mdp -c "$D/${P}_em.gro" \
        -p "$D/${P}.top" -o "$D/${P}_em2.tpr" -po "$D/${P}_em2_mdout.mdp"
    ( cd "$D" && $GMX mdrun -s ${P}_em2.tpr -deffnm ${P}_em2 -ntomp $NT < /dev/null )
    ;;
nvt)
    EMOUT="$D/${P}_em.gro"; [[ -f $D/${P}_em2.gro ]] && EMOUT="$D/${P}_em2.gro"
    echo "NVT starts from (and is restrained to) $EMOUT"
    grompp_checked 2 "$D/${P}_nvt_grompp.log" -f data/mdp/nvt.mdp -c "$EMOUT" \
        -r "$EMOUT" -p "$D/${P}.top" -o "$D/${P}_nvt.tpr" -po "$D/${P}_nvt_mdout.mdp"
    ( cd "$D" && $GMX mdrun -s ${P}_nvt.tpr -deffnm ${P}_nvt -ntomp $NT < /dev/null )
    ;;
npt)
    grompp_checked 1 "$D/${P}_npt_grompp.log" -f data/mdp/npt.mdp -c "$D/${P}_nvt.gro" \
        -r "$D/${P}_nvt.gro" -p "$D/${P}.top" -o "$D/${P}_npt.tpr" -po "$D/${P}_npt_mdout.mdp"
    ( cd "$D" && $GMX mdrun -s ${P}_npt.tpr -deffnm ${P}_npt -ntomp $NT < /dev/null )
    ;;
md)
    rep=${3:?usage: setup_mstate_eq.sh <k> md <rep>}
    Q=${SYS}_mstate_md_c${k}_rep${rep}
    M=${MD_DIR}/c${k}/rep${rep}
    SEED=$((SEED0 + 10 * k + rep))          # recorded; distinct from equil 48170-82, pulls 48190-94/48300-03
    mkdir -p "$M"
    if [[ ! -f $M/$Q.tpr ]]; then
        [[ -f $D/${P}.top ]] || { echo "no topology here and no $M/$Q.tpr: grompp on calmodulin"; exit 1; }
        sed -e "s/__SEED__/${SEED}/" \
            -e "s/^nsteps .*/nsteps                   = ${MD_NSTEPS}   ; $((MD_NSTEPS / 500000)) ns @ 2 fs (extendable)/" \
            data/mdp/md_equil.mdp > "$M/${Q}_grompp.mdp"
        grompp_checked 0 "$M/${Q}_grompp.log" -f "$M/${Q}_grompp.mdp" -c "$D/${P}_npt.gro" \
            -p "$D/${P}.top" -o "$M/$Q.tpr" -po "$M/${Q}_mdout.mdp"
        sed "s#FILE=colvar.dat#FILE=colvar_${Q}.dat#" "$MONITOR" > "$M/plumed_${Q}.dat"
        cp ${SYS}_DNA_parambsc1_tip3p.pdb "$M/"      # MOLINFO / FIT_TO_TEMPLATE reference
        echo "c${k} rep${rep}: tpr built (seed ${SEED})"
        [[ ${4:-} == prep ]] && exit 0               # build only (e.g. before copying to insulin)
    fi
    cpi=(); [[ -f $M/$Q.cpt ]] && cpi=(-cpi $Q.cpt -append)
    # MDRUN_EXTRA: extra production-only mdrun flags, e.g. "-bonded gpu" (Stage B continuations
    # from 2026-10-23, user: PLUMED keeps update on the CPU, which is the bottleneck)
    extra=(); [[ -n ${MDRUN_EXTRA:-} ]] && read -ra extra <<< "$MDRUN_EXTRA"
    ( cd "$M" && $GMX mdrun -s $Q.tpr -deffnm $Q -plumed plumed_${Q}.dat -ntomp $NT \
        "${cpi[@]}" "${extra[@]}" < /dev/null )
    rm -f "$M"/\#*
    ;;
*) echo "unknown stage: $stage"; exit 1 ;;
esac
rm -f "$D"/\#*
echo "[$(date '+%F %T')] c${k} ${stage} done"
