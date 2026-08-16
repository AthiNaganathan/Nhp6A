#!/bin/bash
# Minimum-image (PBC artefact) check for a multi-walker MetaD run (mode "metad", the default)
# or for the unbiased equilibrium campaign (mode "equil").
#
# Runs ON THE HOST THAT HOLDS THE TRAJECTORIES (they are too big to move): builds a
# solute-only, whole-molecule trajectory and runs `gmx mindist -pi` on it.
#
# -pi reports, per frame: the minimum distance between the solute and its own periodic images,
# the solute's maximum internal distance, and the box dimensions. A solute that never comes
# within the cut-off of its own image has no minimum-image artefact. Per project convention,
# transient few-frame brushes below the cut-off are NOT artefacts -- what matters is whether the
# solute is trapped against its image (see the mindist -pi interpretation note).
#
# metad mode: reads data/<sys>/replica_<i>/<PREFIX>_r<i>.trr (full system + box, already strided,
#   ~30x cheaper to read than the raw .xtc) and strips it to $SEL with -pbc mol.
#
# equil mode: reads data/equil/<sys>/rep<i>/{md.tpr,md.xtc} and additionally leaves behind the
#   PBC-clean viewing trajectory rep<i>/md_proc.{xtc,pdb,gro} (-dt $DT, solute only).
#   Multi-molecule solutes (a DNA duplex is 2 molecules; the 2:1 complex is 4) are NOT made whole
#   by -pbc mol alone -- the strands/chains stay in different periodic images and every
#   inter-molecular distance comes out wrong. So equil mode welds the complex with -pbc cluster
#   FIRST, then applies -pbc mol -center -ur compact. Single-molecule solutes are unaffected by
#   the extra pass (verified: identical max-internal distance either way).
#
# FINE mode (equil only): the default 1 ns stride badly undersamples image approaches -- on the
#   2:1 complex it reported 0.83 nm where a 10 ps stride finds 0.55 nm. Give -D (a stride other
#   than the 1000 ps default) and/or a -b/-e window to re-scan finely. A fine run is diagnostic,
#   not a viewing trajectory: it tags its xvg (mindist_pi_r<i>_dt<DT>ps[_<b>-<e>ns].xvg) and
#   leaves rep<i>/md_proc.* untouched, so the provenance of the 1 ns viewing trajectories stands.
#
# Usage:
#   metad: pbc_mindist_check.sh -d DATADIR -t TPR -p PREFIX [-n NREP] [-g GMX] [-o OUTDIR]
#   equil: pbc_mindist_check.sh -m equil -d data/equil/2Nhp6A -o analysis/pbc/equil_2nhp6a \
#                               -s Protein_DNA -n 3 [-D DT_PS] [-g GMX]
#   fine:  pbc_mindist_check.sh -m equil -d data/equil/2Nhp6A -o analysis/pbc/equil_2nhp6a \
#                               -s Protein_DNA -r 0 -D 10 [-b 170000 -e 185000] [-k]
set -u

MODE="metad"
DATADIR="data/1Nhp6A_apo_large"
TPR="data/1Nhp6A_apo_large/1Nhp6A_parambsc1_tip3p_ions0.15_md_ext1000.tpr"
PREFIX="1Nhp6A_parambsc1_tip3p_ions0.15_meta_2D"
NREP=6
GMX="/opt/gromacs/2024.1/bin/gmx_mpi"
OUTDIR="analysis/pbc"
SEL="Protein"
DT=1000          # equil mode: output stride in ps (250 ns -> 251 frames)
BEG=""           # equil mode: window start in ps (empty = from the beginning)
END=""           # equil mode: window end in ps   (empty = to the end)
REPS=""          # explicit replica list, e.g. "0" or "0,2"; empty = 0..NREP-1
KEEP=0           # fine runs: 1 = keep the clustered solute trajectory for further analysis

while getopts "m:d:t:p:n:g:o:s:D:b:e:r:k" opt; do
    case $opt in
        m) MODE="$OPTARG" ;;
        d) DATADIR="$OPTARG" ;;
        t) TPR="$OPTARG" ;;
        p) PREFIX="$OPTARG" ;;
        n) NREP="$OPTARG" ;;
        g) GMX="$OPTARG" ;;
        o) OUTDIR="$OPTARG" ;;
        s) SEL="$OPTARG" ;;
        D) DT="$OPTARG" ;;
        b) BEG="$OPTARG" ;;
        e) END="$OPTARG" ;;
        r) REPS="$OPTARG" ;;
        k) KEEP=1 ;;
        *) echo "bad flag"; exit 1 ;;
    esac
done

# A fine run is any equil run that is not the canonical 1 ns full-length pass.
FINE=0
[ "$DT" != "1000" ] && FINE=1
[ -n "$BEG$END" ] && FINE=1
TAG=""
WINOPT=""
if [ "$FINE" = "1" ]; then
    TAG="_dt${DT}ps"
    if [ -n "$BEG$END" ]; then
        b_ns=$(awk -v x="${BEG:-0}" 'BEGIN{printf "%g", x/1000}')
        e_ns=$(awk -v x="${END:-0}" 'BEGIN{printf "%g", x/1000}')
        TAG="${TAG}_${b_ns}-${e_ns}ns"
    fi
    [ -n "$BEG" ] && WINOPT="$WINOPT -b $BEG"
    [ -n "$END" ] && WINOPT="$WINOPT -e $END"
fi

if [ -n "$REPS" ]; then
    REPLIST=$(echo "$REPS" | tr ',' ' ')
else
    REPLIST=$(seq 0 $((NREP - 1)))
fi

mkdir -p "$OUTDIR"

if [ "$MODE" = "equil" ]; then
    TPR="$DATADIR/rep0/md.tpr"
fi
[ -f "$TPR" ] || { echo "ERROR: no tpr at $TPR"; exit 1; }

# A composite solute (e.g. Protein_DNA) is not a default group, so build it with make_ndx.
# Merging by NAME rather than by group number: the numbering shifts between builds, the names
# do not.
NDX=""
if [[ "$SEL" == *_* ]] && [[ "$SEL" != "Protein-H" ]]; then
    NDX="$OUTDIR/full.ndx"
    if [ ! -f "$NDX" ]; then
        echo ">> make_ndx: merging $SEL" >>"$OUTDIR/gmx.log"
        printf '"%s" | "%s"\nq\n' "${SEL%%_*}" "${SEL#*_}" \
            | $GMX make_ndx -f "$TPR" -o "$NDX" -quiet 2>>"$OUTDIR/gmx.log" \
            || { echo "ERROR: make_ndx could not merge $SEL"; exit 1; }
    fi
fi
NDXOPT=""
[ -n "$NDX" ] && NDXOPT="-n $NDX"

# Solute-only reference for mindist's -s, so it matches the stripped trajectory atom-for-atom.
# It must be a .tpr, not a .gro: mindist -pi needs the pbc *type*, and a .gro carries only box
# vectors ("Fatal error: pbc = unset is not supported by g_mindist"). convert-tpr subsets the
# full tpr down to $SEL, keeping the pbc type.
REF="$OUTDIR/solute.tpr"
if [ ! -f "$REF" ]; then
    echo "$SEL" | $GMX convert-tpr -s "$TPR" $NDXOPT -o "$REF" -quiet 2>>"$OUTDIR/gmx.log" \
        || { echo "ERROR: could not write $REF"; exit 1; }
fi

if [ "$MODE" = "equil" ]; then
    for i in $REPLIST; do
        rep="$DATADIR/rep$i"
        [ -f "$rep/md.xtc" ] || { echo "rep $i: no md.xtc, skipped"; continue; }
        clus="$OUTDIR/.cluster_r${i}.xtc"
        xvg="$OUTDIR/mindist_pi_r${i}${TAG}.xvg"
        # A fine run must not overwrite the 1 ns viewing trajectory, so it works on a scratch copy.
        if [ "$FINE" = "1" ]; then
            proc="$OUTDIR/.proc_r${i}${TAG}.xtc"
        else
            proc="$rep/md_proc.xtc"
        fi

        echo "rep $i: $rep/md.xtc -> $proc  (dt $DT ps${WINOPT:+, window$WINOPT ps})"
        # -pbc cluster asks for two groups: the one to cluster, then the one to write.
        echo ">> rep$i: trjconv -pbc cluster -dt $DT $WINOPT ($SEL)" >>"$OUTDIR/gmx.log"
        printf '%s\n%s\n' "$SEL" "$SEL" \
            | $GMX trjconv -s "$TPR" -f "$rep/md.xtc" $NDXOPT -pbc cluster -dt "$DT" $WINOPT \
                   -o "$clus" -quiet 2>>"$OUTDIR/gmx.log" \
            || { echo "  trjconv -pbc cluster FAILED on rep $i"; continue; }

        # Now a solute-only trajectory, so all selections come from the reduced tpr's System.
        echo ">> rep$i: trjconv -pbc mol -center -ur compact" >>"$OUTDIR/gmx.log"
        printf 'System\nSystem\n' \
            | $GMX trjconv -s "$REF" -f "$clus" -pbc mol -center -ur compact \
                   -o "$proc" -quiet 2>>"$OUTDIR/gmx.log" \
            || { echo "  trjconv -pbc mol FAILED on rep $i"; continue; }

        if [ "$FINE" = "0" ]; then
            for ext in pdb gro; do
                echo "System" | $GMX trjconv -s "$REF" -f "$proc" -dump 0 \
                       -o "$rep/md_proc.$ext" -quiet 2>>"$OUTDIR/gmx.log"
            done
        fi
        rm -f "$clus"

        # -tu ns so the xvg time column is in ns: pbc_summary.py reads it as ns to time the dips.
        # NOTE: -pi writes only -od. mindist's -on/-o/-or (per-frame contact counts, atom pairs,
        # per-residue distances) belong to its group mode and are silently ignored under -pi, so
        # the identity of the contacting residues has to come from elsewhere -- keep the clustered
        # trajectory with -k and analyse it (the tail of the mindist stdout does name the single
        # closest pair over the whole run, and it is echoed into gmx.log).
        echo "System" | $GMX mindist -s "$REF" -f "$proc" -pi -tu ns -od "$xvg" -quiet \
               2>>"$OUTDIR/gmx.log" \
            || { echo "  mindist FAILED on rep $i"; continue; }
        if [ "$FINE" = "1" ]; then
            if [ "$KEEP" = "1" ]; then
                keep="${xvg%.xvg}_proc.xtc"
                mv "$proc" "$keep"
                echo "  wrote $xvg and $keep (md_proc left untouched)"
            else
                rm -f "$proc"
                echo "  wrote $xvg (fine run; md_proc left untouched)"
            fi
        else
            echo "  wrote $proc and $xvg"
        fi
    done
    echo "DONE. xvg columns: time  min_periodic_image_dist  max_internal_dist  box_x box_y box_z"
    exit 0
fi

for i in $REPLIST; do
    trr="$DATADIR/replica_$i/${PREFIX}_r${i}.trr"
    [ -f "$trr" ] || { echo "replica $i: no trr, skipped"; continue; }
    whole="$OUTDIR/solute_r${i}.xtc"
    xvg="$OUTDIR/mindist_pi_r${i}.xvg"

    # Stripping re-reads the whole multi-GB trr, so keep an existing result. Only a trjconv that
    # exited 0 is kept (see the .ok stamp) -- a truncated xtc from a killed run is never reused.
    if [ -f "$whole" ] && [ -f "$whole.ok" ]; then
        echo "replica $i: $whole already stripped, reusing"
    else
        echo "replica $i: $trr -> $whole"
        # -pbc mol makes each molecule whole: mandatory before -pi, or the solute can be split
        # across the boundary and the image distance comes out meaninglessly small.
        rm -f "$whole.ok"
        echo "$SEL" | $GMX trjconv -s "$TPR" -f "$trr" -o "$whole" -pbc mol -quiet \
            2>>"$OUTDIR/gmx.log" || { echo "  trjconv FAILED on replica $i"; continue; }
        touch "$whole.ok"
    fi

    echo "System" | $GMX mindist -s "$REF" -f "$whole" -pi -od "$xvg" -tu ns -quiet \
        2>>"$OUTDIR/gmx.log" || { echo "  mindist FAILED on replica $i"; continue; }
    echo "  wrote $xvg"
done

echo "DONE. xvg columns: time  min_periodic_image_dist  max_internal_dist  box_x box_y box_z"
