#!/bin/bash
# Build and equilibrate a protein/DNA complex: the holo protocol of record (project CLAUDE.md).
#
#   build : AF_Nhp6A/<SYS>_DNA_AF_fix.pdb -> pdb2gmx (amber14sb_parmbsc1, TIP3P; default termini)
#           -> truncated octahedron, <buffer> nm -> solvate -> 0.15 M KCl (neutral) -> EM
#   nvt   : 1 ns NVT, heavy-atom POSRES 1000   (data/mdp/nvt.mdp, -r = EM structure)
#   npt   : 1 ns NPT Berendsen, POSRES         (data/mdp/npt.mdp, -r = NVT structure)
#   md    : production tpr from the NPT end    ($MDP, default data/mdp/md.mdp: the tpr every holo MetaD runs from)
#
#   bash setup_amber_eq.sh <SYS> <build|nvt|npt|md> [buffer_nm, build only, default 1.0]
#
# Run the stages one at a time and inspect each before the next. Topology files land in the
# project root (<SYS>_DNA_parambsc1_tip3p*.{top,itp}, as for every system); coordinates and
# logs in data/<SYS>/. grompp must give exactly the audited warning count (genion 1 net charge /
# EM 0 / NVT 2 obsolete keys / NPT 1 Berendsen / md 0), never a blanket -maxwarn.
#
# Systems built with it: 1IDR (Nhp6A residues 1-23, C-amidated via an NHE cap, on the 1Nhp6A
# AF DNA; buffer 1.5 nm, 2026-10-08). Before 2026-10-08 this script was a hard-coded, partly
# commented-out record of the 2Nhp6A build (git f72cce7); that run used the same commands.
set -euo pipefail

SYS=${1:?usage: setup_amber_eq.sh <SYS> <build|nvt|npt|md> [buffer_nm]}
stage=${2:?usage: setup_amber_eq.sh <SYS> <build|nvt|npt|md> [buffer_nm]}
buffer=${3:-1.0}
MD_WARN=${MD_WARN:-0}          # production grompp warning count (audited: 0 for md.mdp, 2026-10-08)
MDP=${MDP:-data/mdp/md.mdp}    # production mdp (1IDR: data/mdp/md_1IDR.mdp = md.mdp + non-Water xtc)

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
set +u; source /usr/local/gromacs/2024.1/bin/GMXRC; set -u   # rubisco plain gmx (tools + EM/NVT/NPT)
GMX=gmx
NT=${NT:-12}

ff=parambsc1; wat=tip3p; ions=0.15
T=${SYS}_DNA_${ff}_${wat}                  # topology prefix (root)
D=data/${SYS}; P=$D/${T}                   # coordinate prefix
TOP=${T}_ions.top
mkdir -p "$D"

grompp_checked() {    # <expected_warnings> <log> <grompp args...>
    local want=$1 log=$2; shift 2
    $GMX grompp "$@" -maxwarn "$want" > "$log" 2>&1 || { tail -30 "$log"; exit 1; }
    local got; got=$(grep -c "^WARNING [0-9]" "$log" || true)
    [[ $got == "$want" ]] || { echo "grompp: $got warnings, expected $want ($log)"; exit 1; }
    grep -A4 "^WARNING [0-9]" "$log" || true
}

case $stage in
build)
    pdb=AF_Nhp6A/${SYS}_DNA_AF_fix.pdb
    [[ -f $pdb ]] || { echo "missing $pdb"; exit 1; }
    [[ -e $TOP ]] && { echo "$TOP exists: refusing to overwrite a built system"; exit 1; }
    printf '1\n1\n' | $GMX pdb2gmx -f "$pdb" -ignh -p "$T" -o "$P" -i "$T" -n "$P" \
        > "$P"_pdb2gmx.log 2>&1
    grep -E "Total charge|terminus" "$P"_pdb2gmx.log
    $GMX editconf -f "$P".gro -o "$P"_ed.gro -d "$buffer" -bt octahedron -c > "$P"_editconf.log 2>&1
    cp "$T".top "$TOP"
    $GMX solvate -cp "$P"_ed.gro -p "$TOP" -o "$P"_solv.gro > "$P"_solvate.log 2>&1
    grompp_checked 1 "$P"_genion_grompp.log -f data/mdp/em.mdp -c "$P"_solv.gro -p "$TOP" \
        -o "$P"_genion.tpr -po "$P"_genion_mdout.mdp
    echo SOL | $GMX genion -s "$P"_genion.tpr -p "$TOP" -o "$P"_ions${ions}.gro \
        -conc $ions -neutral -pname K -nname CL > "$P"_genion.log 2>&1
    grompp_checked 0 "$P"_em_grompp.log -f data/mdp/em.mdp -c "$P"_ions${ions}.gro -p "$TOP" \
        -o "$P"_ions${ions}_mini.tpr -po "$P"_em_mdout.mdp
    $GMX mdrun -s "$P"_ions${ions}_mini.tpr -deffnm "$P"_ions${ions}_mini -ntmpi 1 -ntomp "$NT" \
        < /dev/null > "$P"_em_mdrun.log 2>&1
    ;;
nvt)
    grompp_checked 2 "$P"_nvt_grompp.log -f data/mdp/nvt.mdp -c "$P"_ions${ions}_mini.gro \
        -r "$P"_ions${ions}_mini.gro -p "$TOP" -o "$P"_ions${ions}_nvt.tpr -po "$P"_nvt_mdout.mdp
    $GMX mdrun -s "$P"_ions${ions}_nvt.tpr -deffnm "$P"_ions${ions}_nvt -ntmpi 1 -ntomp "$NT" \
        < /dev/null > "$P"_nvt_mdrun.log 2>&1
    ;;
npt)
    grompp_checked 1 "$P"_npt_grompp.log -f data/mdp/npt.mdp -c "$P"_ions${ions}_nvt.gro \
        -r "$P"_ions${ions}_nvt.gro -p "$TOP" -o "$P"_ions${ions}_npt.tpr -po "$P"_npt_mdout.mdp
    $GMX mdrun -s "$P"_ions${ions}_npt.tpr -deffnm "$P"_ions${ions}_npt -ntmpi 1 -ntomp "$NT" \
        < /dev/null > "$P"_npt_mdrun.log 2>&1
    ;;
md)
    grompp_checked "$MD_WARN" "$P"_md_grompp.log -f "$MDP" -c "$P"_ions${ions}_npt.gro \
        -r "$P"_ions${ions}_npt.gro -p "$TOP" -o "$P"_ions${ions}_md.tpr -po "$P"_md_mdout.mdp
    ;;
*) echo "unknown stage: $stage"; exit 1 ;;
esac
rm -f "$D"/\#* \#*
echo "[$(date '+%F %T')] ${SYS} ${stage} done"
