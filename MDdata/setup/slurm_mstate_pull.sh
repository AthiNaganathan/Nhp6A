#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=mstate_pull
#SBATCH --output=data/mstate/pull/slurm_%x_%j.out
#
# Steered core-detachment pull, 40 ns, for one m-state probe start frame (PLAN.md, A1.2).
# CONSTRUCTION RUN: biased (MOVINGRESTRAINT on core-DNA COM distance + IDR lower wall).
# Prepared by setup_mstate_pull.sh <k> [system] (tpr + plumed with D0 filled, 0-step checked).
# Files in c<k>/ carry the prefix P = <system>_mstate_pull_c<k>.
#
#   sbatch slurm_mstate_pull.sh <k>                (1:1; calmodulin, from the project root)
#   sbatch --output=data/mstate_2Nhp6A/pull/slurm_%x_%j.out slurm_mstate_pull.sh <k> 2Nhp6A
#
# AGAMEDE (EHU HPC; detected by /eb/x86_64): no PLUMED-patched 2024.x exists there, so it runs the
# GROMACS 2025.2 CUDA module (built-in PLUMED interface) with the PLUMED 2.9.4 kernel, on the tprs
# made by the 2024.2 grompp. Build chosen by GPU compute capability (A100 8.0 / H100 9.0).
# The #SBATCH defaults below are calmodulin's; on agamede override them:
#   sbatch -p p_fast -A poxlp --time=2-00:00:00 --output=data/mstate_2Nhp6A/pull/slurm_%x_%j.out \
#          slurm_mstate_pull.sh <k> 2Nhp6A
# TEST=<nsteps> (env) runs a short validation instead, under the prefix <P>_test_<host>
# (no -cpi, never touches the production files).
#
# 4 configs on 3 GPUs: three run at once, the fourth queues. -cpi makes a resubmission resume.
set -euo pipefail

k=${1:?usage: sbatch slurm_mstate_pull.sh <cluster> [1Nhp6A|2Nhp6A]}
SYS=${2:-1Nhp6A}
if [[ -d /eb/x86_64 ]]; then                                  # agamede
    set +u
    type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)   # 8.0 / 9.0
    module purge
    module load GROMACS/2025.2-foss-2025a-CUDA-12.8.0-capability-${cc} PLUMED/2.9.4-foss-2025a
    export PLUMED_KERNEL=$EBROOTPLUMED/lib/libplumedKernel.so
    set -u
    # The patched 2024.2 runs on calmodulin/insulin update+constrain on the CPU (the patch forbids GPU
    # update); 2025.2's built-in PLUMED interface would put update on the GPU. Pin it to the CPU so the
    # integration path matches the reference runs. PLUMED writes its log to PLUMED.OUT here.
    EXTRA="-update cpu"
else
    set +u; source /opt/gromacs/2024.2/bin/GMXRC.bash; set -u    # 2024.2-plumed_2.9.2 (GMXRC trips set -u)
    EXTRA=""
fi
case $SYS in
  1Nhp6A) d=data/mstate/pull/c${k} ;;
  2Nhp6A) d=data/mstate_2Nhp6A/pull/c${k} ;;
  *) echo "unknown system $SYS"; exit 1 ;;
esac
P=${SYS}_mstate_pull_c${k}
[[ -f $d/${P}.tpr && -f $d/plumed_${P}.dat ]] || { echo "run setup_mstate_pull.sh $k $SYS first"; exit 1; }
grep -q "__D0__" "$d/plumed_${P}.dat" && { echo "plumed_${P}.dat still has the __D0__ placeholder"; exit 1; }

cd "$d"
echo "[$(date '+%F %T')] ${SYS} c${k} pull start on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, $(command -v gmx_mpi)"
if [[ -n ${TEST:-} ]]; then
    T=${P}_test_$(hostname -s)${TESTTAG:-}
    sed "s#FILE=colvar_${P}.dat#FILE=colvar_${T}.dat#" plumed_${P}.dat > plumed_${T}.dat
    gmx_mpi mdrun -s ${P}.tpr -deffnm ${T} -nsteps ${TEST} -plumed plumed_${T}.dat ${EXTRA} \
        -ntomp ${SLURM_CPUS_PER_TASK:-8} < /dev/null
else
    gmx_mpi mdrun -s ${P}.tpr -deffnm ${P} -cpi ${P} -plumed plumed_${P}.dat ${EXTRA} \
        -ntomp ${SLURM_CPUS_PER_TASK:-8} < /dev/null
fi
echo "[$(date '+%F %T')] ${SYS} c${k} pull done, exit $?"
