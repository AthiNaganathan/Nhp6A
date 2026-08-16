#!/usr/bin/env python3
"""Summarise `gmx mindist -pi` output: is there a minimum-image (PBC) artefact?

Reads the mindist xvg files written by pbc_mindist_check.sh and reports, per walker, how close
the solute ever came to its own periodic image, relative to the interaction cut-off.

By default only the canonical one-per-replica files (mindist_pi_r<i>.xvg) are read, so the
fine-stride diagnostic re-scans (mindist_pi_r<i>_dt<DT>ps[...].xvg) and their -on/-o companions
do not silently end up as extra traces in the figure. Point --glob at them to analyse a fine
scan instead, e.g. --glob 'mindist_pi_r*_dt10ps.xvg'. Note that the coarse 1 ns pass
UNDERSAMPLES the approaches (on the 2:1 complex it reported 0.83 nm where the 10 ps scan of the
same replica finds 0.55 nm), so a coarse result close to the cut-off must be re-scanned finely
before it is called clean.

Project convention on interpretation (do not re-litigate it here): a transient, few-frame brush
below the cut-off is NOT a PBC artefact. What would be one is the solute sitting against its own
image -- i.e. sustained residence below the cut-off, or a min-image distance that tracks the box
size. So this reports both the extremes AND the *time spent* below the cut-off, and flags only
sustained excursions.
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUTOFF = 1.2        # nm -- conservative margin over the 0.9 nm nonbonded cut-off (rvdw/rcoulomb);
                    # the project treats >=~1.29 nm as clean (see the mindist -pi interpretation note)
SUSTAINED_NS = 1.0  # a dip below the cut-off lasting longer than this is worth a look
FIGDIR = "figures"
PBCDIR = "analysis/pbc"
GLOB = "mindist_pi_r[0-9].xvg"   # canonical per-replica pass only; see --glob for fine re-scans
DEF_TITLE = "1Nhp6A apo (large box): solute-to-periodic-image distance"
DEF_OUT = "figures/1Nhp6A_apo_large_mindist_pi_v1"


def read_xvg(path):
    rows = [l.split() for l in open(path) if l[0] not in "#@"]
    return np.array(rows, dtype=float)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pbcdir", default=PBCDIR, help="dir of mindist_pi_r*.xvg (default %(default)s)")
    p.add_argument("--glob", default=GLOB,
                   help="filename pattern inside --pbcdir (default %(default)s; use e.g. "
                        "'mindist_pi_r*_dt10ps.xvg' for a fine re-scan)")
    p.add_argument("--cutoff", type=float, default=CUTOFF, help="nm (default %(default)s)")
    p.add_argument("--sustained", type=float, default=SUSTAINED_NS,
                   help="ns; dips longer than this are flagged (default %(default)s)")
    p.add_argument("--title", default=DEF_TITLE)
    p.add_argument("--out", default=DEF_OUT, help="figure path without extension")
    p.add_argument("--replabel", default="walker",
                   help="what a trajectory is called here: 'walker' (MetaD) or 'rep' (equil)")
    p.add_argument("--solute", default="protein",
                   help="what the solute is called in the verdict, e.g. 'protein:DNA complex'")
    args = p.parse_args()
    cutoff, sustained_ns = args.cutoff, args.sustained
    replabel, solute = args.replabel, args.solute

    files = sorted(glob.glob(f"{args.pbcdir}/{args.glob}"))
    if not files:
        raise SystemExit(f"no {args.glob} in {args.pbcdir}/ -- run pbc_mindist_check.sh first")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    print(f"{replabel:>7} {'frames':>8} {'min image dist (nm)':>21} {'median':>8} "
          f"{'frames < cutoff':>16} {'longest dip':>12}")
    worst, sustained_any = np.inf, False

    for f in files:
        d = read_xvg(f)
        t, dmin = d[:, 0], d[:, 1]          # ns, nm
        below = dmin < cutoff
        # longest contiguous run below the cut-off, in ns
        dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
        longest = 0
        run = 0
        for b in below:
            run = run + 1 if b else 0
            longest = max(longest, run)
        longest_ns = longest * dt
        sustained_any |= longest_ns > sustained_ns
        worst = min(worst, dmin.min())

        # mindist_pi_r0.xvg -> "0"; mindist_pi_r0_dt10ps_170-185ns.xvg -> "0 (dt10ps 170-185ns)"
        m = re.match(r"mindist_pi_r(\d+)_?(.*)\.xvg$", os.path.basename(f))
        r = m.group(1) + (f" ({m.group(2).replace('_', ' ')})" if m and m.group(2) else "")
        print(f"{r:>7} {len(t):8,d} {dmin.min():13.3f} (min) {np.median(dmin):8.3f} "
              f"{below.sum():10d} ({100 * below.mean():4.1f}%) {longest_ns:9.2f} ns")
        ax.plot(t, dmin, lw=0.6, alpha=0.8, label=f"{replabel} {r}")

    ax.axhline(cutoff, color="crimson", ls="--", lw=1.2,
               label=f"cut-off {cutoff} nm")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("min. periodic image distance (nm)")
    ax.legend(frameon=False, fontsize=7, ncol=4)
    ax.set_title(args.title, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=300)
    print(f"\nwrote {args.out}.png")

    print(f"\nclosest approach over all {replabel}s: {worst:.3f} nm  (cut-off {cutoff} nm)")
    if worst >= cutoff:
        print(f"VERDICT: no minimum-image artefact -- the {solute} never comes within the cut-off "
              "of its own image.")
    elif not sustained_any:
        print(f"VERDICT: dips below the cut-off exist but none lasts > {sustained_ns} ns -- "
              "transient brushes, not a trapped image contact. Per project convention this is "
              "NOT an artefact, but eyeball the figure.")
    else:
        print(f"VERDICT: SUSTAINED sub-cut-off contact with the periodic image. The {solute} needs "
              "a human look before anything derived from these trajectories is trusted.")


if __name__ == "__main__":
    main()
