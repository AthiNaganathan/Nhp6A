#!/usr/bin/env python3
"""1D marginal PMF convergence for multiple-walker well-tempered metadynamics.

Produces, for any of the project's MetaD systems (1D or 2D), the standard pair
of convergence figures:

  figures/<label>_fes_convergence_v<N>.png
      the 1D marginal PMF of each BIASED CV drawn at a series of increasing
      cumulative-time snapshots, lines coloured by time with a colourbar.
  figures/<label>_mad_convergence_v<N>.png
      mean absolute deviation between CONSECUTIVE snapshot marginals vs time,
      semilogy, with a dashed 0.5*kBT reference line.

and archives the numbers so they can be re-plotted without recomputation:

  analysis/<label>_fes_convergence[_<cv>].dat   CV grid + one column per snapshot
  analysis/<label>_mad_convergence.dat          time + one MAD column per CV

Method (identical to the per-system notebooks, e.g. analysis_1TM_2D.ipynb
cells 22-23, and to analysis_1Nhp6A_apo_large_2D.ipynb cell 22):

  1. Merge the walker HILLS time-sorted (metad_convergence.merge_hills) and get
     cumulative FES snapshots in ONE `plumed sum_hills --stride` pass
     (metad_convergence.run_sum_hills) on the grid taken from the run's
     GRID_MIN/GRID_MAX/GRID_BIN.
  2. For ndim=2, Boltzmann-marginalise each 2D snapshot onto each biased CV
     (fes_marginal_meanstd.marginal_fes, summing the other CV over a window).
     For ndim=1 the sum_hills output IS the marginal.
  3. MAD between consecutive snapshots on the common grid, restricted to the
     reliable region (see --fmax).

Only BIASED CVs are ever marginalised here: a plain (non-reweighted) FES of an
unbiased spectator is meaningless and is a standing rule against in this
project.

kBT
---
`plumed sum_hills` was run WITHOUT --kt throughout this project, i.e. PLUMED's
default kBT = 2.494339 kJ/mol (= kB * 300 K), and every notebook uses that same
number for the Boltzmann marginalisation.  It is kept as the default here so
the marginals -- and the 0.5*kBT reference line, which is 1.247 kJ/mol -- match
the existing analyses.  (The runs themselves used TEMP=298; the 0.7% mismatch
is inherited from the existing protocol, not introduced here.)

Reliable region
---------------
The .dat archives always carry the FULL grid.  Truncation is applied at plot
time only: the plotted/MAD region is the contiguous stretch around the minimum
of the FINAL marginal where F <= --fmax (default 30 kJ/mol ~ 12 kBT), or an
explicit --plot-cv1/--plot-cv2 window if given.

Usage (see the three project systems at the bottom of this docstring):

  fes_convergence.py data/DNA_wall2 --label DNA_wall2 --ndim 1 \
      --cv1 de2e --cv1-min 0 --cv1-max 8 --cv1-bin 200

  fes_convergence.py data/1Nhp6A_wall --label 1Nhp6A_wall --ndim 2 \
      --cv1 coord_tail_dna --cv1-min 0 --cv1-max 150 --cv1-bin 100 \
      --cv2 de2e --cv2-min 0 --cv2-max 8 --cv2-bin 200

  fes_convergence.py data/1Nhp6A_apo_large --label 1Nhp6A_apo_large --ndim 2 \
      --cv1 q_core --cv1-min 0 --cv1-max 1.0 --cv1-bin 100 \
      --cv2 n_tail_core --cv2-min 0 --cv2-max 150 --cv2-bin 300
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metad_convergence import find_hills, merge_hills, run_sum_hills  # noqa: E402
from fes_marginal_meanstd import load_fes, marginal_fes               # noqa: E402

# PLUMED's default kt used by sum_hills when --kt is not given (kB * 300 K).
KBT_PLUMED_DEFAULT = 2.494339  # kJ/mol

# pretty axis labels for the CVs used in this project
CV_LABELS = {
    "de2e": r"$d_\mathrm{e-e}$ (nm)",
    "coord_tail_dna": r"$n_\mathrm{IDR-DNA}$",
    "q_core": r"$Q_\mathrm{core}$",
    "n_tail_core": r"$n_\mathrm{IDR-core}$",
}


def cv_label(name):
    return CV_LABELS.get(name, name)


def reliable_region(cv, f, fmax, explicit=None):
    """Contiguous stretch around the minimum of `f` with f <= fmax.

    Returns a boolean mask. If `explicit` (lo, hi) is given it wins.
    """
    if explicit is not None:
        return (cv >= explicit[0]) & (cv <= explicit[1]) & np.isfinite(f)
    ok = np.isfinite(f) & (f <= fmax)
    if not ok.any():
        return np.isfinite(f)
    i0 = int(np.nanargmin(np.where(np.isfinite(f), f, np.inf)))
    lo = i0
    while lo - 1 >= 0 and ok[lo - 1]:
        lo -= 1
    hi = i0
    while hi + 1 < len(f) and ok[hi + 1]:
        hi += 1
    mask = np.zeros(len(f), dtype=bool)
    mask[lo:hi + 1] = True
    return mask


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("system_dir")
    ap.add_argument("--label", required=True, help="system tag for file names")
    ap.add_argument("--ndim", type=int, choices=(1, 2), required=True)
    # CV1 is the FIRST METAD ARG, CV2 the second (ndim=2 only)
    ap.add_argument("--cv1", required=True, help="name of the 1st biased CV")
    ap.add_argument("--cv1-min", type=float, required=True)
    ap.add_argument("--cv1-max", type=float, required=True)
    ap.add_argument("--cv1-bin", type=int, required=True)
    ap.add_argument("--cv2", help="name of the 2nd biased CV (ndim=2)")
    ap.add_argument("--cv2-min", type=float)
    ap.add_argument("--cv2-max", type=float)
    ap.add_argument("--cv2-bin", type=int)
    ap.add_argument("--win-cv1", type=float, nargs=2, default=None,
                    help="marginalisation window on cv1 when projecting onto "
                         "cv2 (default: full grid)")
    ap.add_argument("--win-cv2", type=float, nargs=2, default=None,
                    help="marginalisation window on cv2 when projecting onto "
                         "cv1 (default: full grid)")
    ap.add_argument("--plot-cv1", type=float, nargs=2, default=None,
                    help="explicit plot/MAD window on cv1 (overrides --fmax)")
    ap.add_argument("--plot-cv2", type=float, nargs=2, default=None)
    ap.add_argument("--fmax", type=float, default=30.0,
                    help="reliable-region cutoff on the FINAL marginal "
                         "(kJ/mol, default 30 ~ 12 kBT)")
    ap.add_argument("--nsnap", type=int, default=10,
                    help="number of cumulative-time snapshots (default 10). "
                         "NOTE: the MAD magnitude depends on this, since each "
                         "consecutive pair differs by 1/nsnap of the run.")
    ap.add_argument("--kbt", type=float, default=KBT_PLUMED_DEFAULT,
                    help="kBT (kJ/mol) for the Boltzmann marginal and the "
                         "0.5 kBT line; default = PLUMED sum_hills default")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--datadir", default="analysis")
    ap.add_argument("--version", type=int, default=1, help="figure _v<N> suffix")
    ap.add_argument("--workdir", default=None,
                    help="scratch dir for the merged HILLS (needs ~1 GB for "
                         "the apo system); default: system tempdir")
    args = ap.parse_args()

    if args.ndim == 2 and (args.cv2 is None or args.cv2_min is None
                           or args.cv2_max is None or args.cv2_bin is None):
        sys.exit("--cv2/--cv2-min/--cv2-max/--cv2-bin are required for --ndim 2")

    kBT = args.kbt
    cvnames = [args.cv1] if args.ndim == 1 else [args.cv1, args.cv2]
    mins = [args.cv1_min] + ([args.cv2_min] if args.ndim == 2 else [])
    maxs = [args.cv1_max] + ([args.cv2_max] if args.ndim == 2 else [])
    bins = [args.cv1_bin] + ([args.cv2_bin] if args.ndim == 2 else [])

    hills_files = find_hills(args.system_dir)
    workdir = tempfile.mkdtemp(prefix=f"fesconv_{args.label}_", dir=args.workdir)
    try:
        merged = os.path.join(workdir, "hills_merged.dat")
        print(f"=== {args.label} (ndim={args.ndim}) ===")
        print(f"walkers={len(hills_files)}  merging HILLS ...", flush=True)
        times = merge_hills(hills_files, merged)
        print(f"  per-walker length {times.max()/1000:.1f} ns, "
              f"{len(times)} hills total")
        print(f"sum_hills --stride on grid "
              f"min={mins} max={maxs} bin={bins} ...", flush=True)
        fes_files, stride, n_hills = run_sum_hills(
            merged, args.nsnap, workdir, grid=(mins, maxs, bins))
        snap_idx = [min((k + 1) * stride - 1, len(times) - 1)
                    for k in range(len(fes_files))]
        # sum_hills --stride emits a final complete-set dump that duplicates the
        # last strided one when n_hills is an exact multiple of stride; drop it,
        # otherwise the last MAD is trivially 0.
        keep = [0] + [k for k in range(1, len(snap_idx))
                      if snap_idx[k] != snap_idx[k - 1]]
        if len(keep) != len(snap_idx):
            print(f"  dropped {len(snap_idx)-len(keep)} duplicate final "
                  f"sum_hills dump(s)")
        fes_files = [fes_files[k] for k in keep]
        snap_idx = [snap_idx[k] for k in keep]
        snap_t_ns = times[snap_idx] / 1000.0
        print(f"  {len(fes_files)} snapshots, stride={stride} hills, "
              f"times (ns) = {np.round(snap_t_ns, 1)}")

        # ---- marginals: one stack (n_snap, n_grid) per biased CV ----
        stacks = {c: [] for c in cvnames}
        grids = {}
        wins = {args.cv1: (args.win_cv1 if args.win_cv1 is not None
                           else (args.cv1_min, args.cv1_max))}
        if args.ndim == 2:
            wins[args.cv2] = (args.win_cv2 if args.win_cv2 is not None
                              else (args.cv2_min, args.cv2_max))
        for fname in fes_files:
            if args.ndim == 1:
                cv, _, f = load_fes(fname, ndim=1)
                f = f - np.nanmin(f)
                grids[args.cv1] = cv
                stacks[args.cv1].append(f)
            else:
                cv1, cv2, fes2d = load_fes(fname, ndim=2)
                # project onto cv2, summing cv1 over its window
                g2, f2 = marginal_fes(cv1, cv2, fes2d, wins[args.cv1], kBT, axis=0)
                # project onto cv1, summing cv2 over its window
                g1, f1 = marginal_fes(cv1, cv2, fes2d, wins[args.cv2], kBT, axis=1)
                grids[args.cv1], grids[args.cv2] = g1, g2
                stacks[args.cv1].append(f1)
                stacks[args.cv2].append(f2)
        stacks = {c: np.array(v) for c, v in stacks.items()}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    plot_win = {args.cv1: args.plot_cv1}
    if args.ndim == 2:
        plot_win[args.cv2] = args.plot_cv2
    masks = {c: reliable_region(grids[c], stacks[c][-1], args.fmax, plot_win[c])
             for c in cvnames}

    # ---- MAD between consecutive snapshots, on the reliable region ----
    mads = {}
    for c in cvnames:
        m = masks[c]
        vals = []
        for k in range(1, len(stacks[c])):
            v = m & np.isfinite(stacks[c][k]) & np.isfinite(stacks[c][k - 1])
            vals.append(np.mean(np.abs(stacks[c][k][v] - stacks[c][k - 1][v]))
                        if v.sum() > 3 else np.nan)
        mads[c] = np.array(vals)
    t_mad = 0.5 * (snap_t_ns[1:] + snap_t_ns[:-1])   # midpoint of each interval

    # ---- archive the numbers (FULL grid; truncation is plot-time only) ----
    os.makedirs(args.datadir, exist_ok=True)
    for c in cvnames:
        suffix = "" if args.ndim == 1 else f"_{c}"
        out = os.path.join(args.datadir,
                           f"{args.label}_fes_convergence{suffix}.dat")
        hdr = (f"{args.label}: 1D marginal PMF (kJ/mol) of biased CV '{c}' at "
               f"cumulative-time snapshots\nkBT={kBT} kJ/mol   "
               f"marginalisation window="
               f"{wins[args.cv2 if c == args.cv1 else args.cv1] if args.ndim == 2 else 'n/a'}"
               f"\nreliable plot region: "
               f"{grids[c][masks[c]].min():.4g} to {grids[c][masks[c]].max():.4g} "
               f"(final marginal <= {args.fmax} kJ/mol)\n"
               f"col1 = {c}; cols 2.. = F at t(ns) = "
               + " ".join(f"{t:.3f}" for t in snap_t_ns))
        np.savetxt(out, np.column_stack([grids[c], stacks[c].T]), header=hdr,
                   fmt="%14.6g")
        print(f"saved {out}")

    out = os.path.join(args.datadir, f"{args.label}_mad_convergence.dat")
    np.savetxt(out, np.column_stack([t_mad] + [mads[c] for c in cvnames]),
               header=("t_mid(ns)  " + "  ".join(f"MAD_{c}(kJ/mol)" for c in cvnames)
                       + f"\n0.5*kBT = {0.5*kBT:.4f} kJ/mol; "
                         f"snapshot interval = {snap_t_ns[-1]/len(snap_t_ns):.1f} ns"),
               fmt="%14.6g")
    print(f"saved {out}")

    # ---- figure 1: time-coloured marginal overlay ----
    os.makedirs(args.figdir, exist_ok=True)
    cmap_t = matplotlib.colormaps["plasma"]
    norm_t = matplotlib.colors.Normalize(vmin=0, vmax=snap_t_ns[-1])
    n = len(cvnames)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.4), squeeze=False)
    axes = axes[0]
    for ax, c in zip(axes, cvnames):
        m = masks[c]
        for t_ns, f in zip(snap_t_ns, stacks[c]):
            ax.plot(grids[c][m], f[m], lw=1.4, color=cmap_t(norm_t(t_ns)))
        sm = plt.cm.ScalarMappable(cmap=cmap_t, norm=norm_t)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="time (ns)")
        ax.set_xlabel(cv_label(c))
        ax.set_ylabel(r"$\Delta F$ (kJ/mol)")
        ax.set_title(f"{args.label}: {c} marginal PMF vs time")
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig1 = os.path.join(args.figdir,
                        f"{args.label}_fes_convergence_v{args.version}.png")
    fig.savefig(fig1, dpi=300, facecolor="white", transparent=False,
                bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: MAD between consecutive snapshots ----
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.0), squeeze=False)
    axes = axes[0]
    for ax, c in zip(axes, cvnames):
        ax.semilogy(t_mad, mads[c], "-o", ms=5, color="steelblue")
        ax.axhline(0.5 * kBT, ls="--", color="firebrick", lw=1.2,
                   label=r"0.5 $k_BT$ = %.2f kJ/mol" % (0.5 * kBT))
        ax.axhline(kBT, ls=":", color="grey", lw=1.2,
                   label=r"$k_BT$ = %.2f kJ/mol" % kBT)
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("MAD vs previous snapshot (kJ/mol)")
        ax.set_title(f"{args.label}: {c} marginal PMF")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig2 = os.path.join(args.figdir,
                        f"{args.label}_mad_convergence_v{args.version}.png")
    fig.savefig(fig2, dpi=300, facecolor="white", transparent=False,
                bbox_inches="tight")
    plt.close(fig)

    # ---- summary ----
    print(f"\nkBT = {kBT:.4f} kJ/mol  ->  0.5 kBT = {0.5*kBT:.4f}, "
          f"snapshot interval = {snap_t_ns[-1]/len(snap_t_ns):.2f} ns")
    for c in cvnames:
        m = masks[c]
        g, S = grids[c], stacks[c]
        lo, hi = g[m].min(), g[m].max()
        mins_t = [g[m][np.nanargmin(S[k][m])] for k in range(len(S))]
        k3 = int(np.ceil(2 * len(S) / 3)) - 1   # start of the final third
        print(f"\n-- {c} --")
        print(f"  reliable region: [{lo:.4g}, {hi:.4g}] ({int(m.sum())} bins, "
              f"final marginal <= {args.fmax} kJ/mol)")
        print(f"  MAD (kJ/mol): " + " ".join(f"{v:.3f}" for v in mads[c]))
        print(f"  final MAD = {mads[c][-1]:.4f}  "
              f"({mads[c][-1]/kBT:.3f} kBT)  "
              f"[< 0.5 kBT: {mads[c][-1] < 0.5*kBT}; "
              f"< kBT: {mads[c][-1] < kBT}]")
        d = np.diff(mads[c])
        print(f"  monotonic decay: {bool(np.all(d <= 0))}; "
              f"n_upticks = {int(np.sum(d > 0))}; "
              f"last step {'UP' if d[-1] > 0 else 'down'} "
              f"({mads[c][-2]:.3f} -> {mads[c][-1]:.3f})")
        print(f"  minimum position per snapshot: "
              + " ".join(f"{v:.4g}" for v in mins_t))
        print(f"  minimum drift over the final third "
              f"(t={snap_t_ns[k3]:.0f}->{snap_t_ns[-1]:.0f} ns): "
              f"{mins_t[k3]:.4g} -> {mins_t[-1]:.4g} "
              f"(|d| = {abs(mins_t[-1]-mins_t[k3]):.4g})")
    print(f"\nfigures -> {fig1}\n            {fig2}")


if __name__ == "__main__":
    main()
