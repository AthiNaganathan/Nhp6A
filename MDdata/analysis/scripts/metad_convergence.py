#!/usr/bin/env python3
"""Convergence assessment for multiple-walker well-tempered metadynamics.

For one system it:
  1. Merges all walker HILLS files (time-sorted) into a single HILLS.
  2. Runs `plumed sum_hills --stride` to get FES snapshots vs time.
  3. Quantifies FES drift over time (mean |dF| over the well-sampled region),
     reporting the residual drift in the final third of the run.
  4. Tracks well-tempered hill-height decay.
  5. Reports CV exploration (range visited + recrossings of the principal CV).

It is parameterised so the binary (2D), apo (2D) and DNA (1D) datasets all use
the same code path. Read-only on the run data; writes a merged HILLS, temporary
FES snapshots, and one convergence figure.

Usage:
    metad_convergence.py SYSTEM_DIR --label LABEL --ndim {1,2} [options]
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KJ = "kJ/mol"


def find_hills(system_dir):
    files = sorted(glob.glob(os.path.join(system_dir, "hills.dat.*")))
    if not files:
        sys.exit(f"No hills.dat.* found in {system_dir}")
    return files


def merge_hills(hills_files, out_path):
    """Concatenate walker HILLS, drop comment lines, sort by time (col 0).
    Keep the FIELDS/SET header from the first file so plumed can read it."""
    header, rows = [], []
    with open(hills_files[0]) as fh:
        for line in fh:
            if line.startswith("#"):
                header.append(line)
            else:
                break
    for f in hills_files:
        with open(f) as fh:
            for line in fh:
                if not line.startswith("#") and line.strip():
                    rows.append(line)
    # stable sort by time column
    rows.sort(key=lambda ln: float(ln.split()[0]))
    with open(out_path, "w") as out:
        out.writelines(header)
        out.writelines(rows)
    times = np.array([float(r.split()[0]) for r in rows])
    return times


def run_sum_hills(merged, nsnap, workdir, grid=None, prefix="fes_conv_"):
    """Cumulative FES snapshots every `stride` hills, in a single sum_hills pass.

    grid: optional (mins, maxs, bins) sequences -> passed as --min/--max/--bin.
          Without it PLUMED picks its own grid from the hills extent.
    Returns (fes_files_in_time_order, stride, n_hills).
    """
    n_hills = sum(1 for ln in open(merged) if not ln.startswith("#") and ln.strip())
    stride = max(1, n_hills // nsnap)
    cmd = [
        "plumed", "sum_hills", "--hills", os.path.abspath(merged),
        "--stride", str(stride), "--mintozero",
        "--outfile", prefix,
    ]
    if grid is not None:
        mins, maxs, bins = grid
        cmd += ["--min", ",".join(map(str, mins)),
                "--max", ",".join(map(str, maxs)),
                "--bin", ",".join(map(str, bins))]
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"plumed sum_hills failed:\n{res.stderr}")
    fes_files = sorted(
        glob.glob(os.path.join(workdir, f"{prefix}*.dat")),
        key=lambda f: int(f.split("_")[-1].split(".")[0]),
    )
    return fes_files, stride, n_hills


def load_fes(path, ndim):
    d = np.loadtxt(path)
    if ndim == 1:
        return d[:, 0], None, d[:, 1]
    return d[:, 0], d[:, 1], d[:, 2]


def fes_drift(fes_files, ndim, thresh):
    """Mean |F_i - F_final| over cells where F_final < thresh, per snapshot."""
    _, _, f_final = load_fes(fes_files[-1], ndim)
    mask = np.isfinite(f_final) & (f_final < thresh)
    drift = []
    for f in fes_files:
        _, _, fi = load_fes(f, ndim)
        d = np.abs(fi[mask] - f_final[mask])
        drift.append(np.nanmean(d))
    return np.array(drift), int(mask.sum())


def hill_heights(merged, ndim):
    """Return (time, height) from merged HILLS. height col after the sigmas."""
    hcol = 1 + 2 * ndim + 1  # time, ndim cvs, ndim sigmas, then height (1-indexed)
    data = np.loadtxt(merged, comments="#", usecols=(0, hcol - 1))
    return data[:, 0], data[:, 1]


def _hysteretic_roundtrips(x, lo_th, hi_th):
    """Count low->high->low round trips with a dead-band to reject fluctuations."""
    state, trips, last = 0, 0, None  # state: -1 low, +1 high
    for v in x:
        if v <= lo_th:
            state = -1
        elif v >= hi_th:
            state = 1
        if state != 0 and state != last:
            if last is not None:
                trips += 1
            last = state
    return trips // 2  # a full round trip is two state switches


def cv_exploration(system_dir, ndim, princ_col):
    """Report range and basin round-trips of the principal CV across walkers.
    Thresholds (33rd/66th percentile of pooled data) give a dead-band so only
    genuine basin-to-basin excursions are counted, not fast fluctuations."""
    cfiles = sorted(glob.glob(os.path.join(system_dir, "replica_*", "colvar*.dat")))
    if not cfiles:
        return None
    series, skipped = [], 0
    for c in cfiles:
        vals = []
        with open(c) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                try:                       # tolerate truncated restart-boundary lines
                    vals.append(float(parts[princ_col]))
                except (IndexError, ValueError):
                    skipped += 1
        series.append(np.array(vals))
    if skipped:
        print(f"  [warning] skipped {skipped} malformed colvar line(s) "
              f"(restart/crash boundary)")
    pooled = np.concatenate(series)
    lo_th, hi_th = np.percentile(pooled, [33, 66])
    trips = sum(_hysteretic_roundtrips(x, lo_th, hi_th) for x in series)
    return pooled.min(), pooled.max(), trips, len(cfiles), lo_th, hi_th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system_dir")
    ap.add_argument("--label", required=True, help="system tag for figure name")
    ap.add_argument("--ndim", type=int, choices=(1, 2), required=True)
    ap.add_argument("--nsnap", type=int, default=20)
    ap.add_argument("--thresh", type=float, default=25.0,
                    help="FES cutoff (kJ/mol) defining the well-sampled region")
    ap.add_argument("--princ-col", type=int, default=1,
                    help="colvar column index of principal CV for recrossings")
    ap.add_argument("--figdir", default="analysis/figures")
    ap.add_argument("--version", type=int, default=1,
                    help="figure version suffix _v<N>; bump when the underlying data changed "
                         "(the default silently overwrote _v1 on every re-run)")
    args = ap.parse_args()

    hills_files = find_hills(args.system_dir)
    n_walk = len(hills_files)
    workdir = tempfile.mkdtemp(prefix="sumhills_")
    merged = os.path.join(workdir, "hills_merged.dat")
    try:
        times = merge_hills(hills_files, merged)
        fes_files, stride, n_hills = run_sum_hills(merged, args.nsnap, workdir)

        # x-axis: per-walker time (ns) at each snapshot boundary
        snap_idx = [min((k + 1) * stride - 1, len(times) - 1)
                    for k in range(len(fes_files))]
        snap_t_ns = times[snap_idx] / 1000.0

        drift, ncells = fes_drift(fes_files, args.ndim, args.thresh)
        ht_t, ht = hill_heights(merged, args.ndim)
        expl = cv_exploration(args.system_dir, args.ndim, args.princ_col)

        # ---- figure ----
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(snap_t_ns, drift, "o-", color="tab:blue")
        ax[0].set_xlabel("per-walker time (ns)")
        ax[0].set_ylabel(f"mean |F - F_final|  ({KJ})")
        ax[0].set_title(f"{args.label}: FES drift (region < {args.thresh:g} {KJ})")

        # bin hill heights into the same time grid for a clean decay curve
        nb = 100
        edges = np.linspace(ht_t.min(), ht_t.max(), nb + 1)
        idx = np.clip(np.digitize(ht_t, edges) - 1, 0, nb - 1)
        mean_h = np.array([ht[idx == b].mean() if np.any(idx == b) else np.nan
                           for b in range(nb)])
        ctr = 0.5 * (edges[:-1] + edges[1:]) / 1000.0
        ax[1].semilogy(ctr, mean_h, "-", color="tab:red")
        ax[1].set_xlabel("per-walker time (ns)")
        ax[1].set_ylabel(f"mean deposited height ({KJ})")
        ax[1].set_title(f"{args.label}: well-tempered height decay")
        fig.tight_layout()

        os.makedirs(args.figdir, exist_ok=True)
        figpath = os.path.join(args.figdir,
                               f"{args.label}_metad_convergence_v{args.version}.png")
        fig.savefig(figpath, dpi=300)

        # ---- summary ----
        final_third = drift[len(drift) * 2 // 3:]
        h0 = float(np.nanmax(ht))           # true initial deposited height
        h_last = np.nanmean(mean_h[-3:])
        print(f"\n=== {args.label} ===")
        print(f"walkers={n_walk}  hills(total)={n_hills}  "
              f"per-walker length={times.max()/1000:.0f} ns  "
              f"well-sampled cells={ncells}")
        print(f"FES drift: start={drift[0]:.2f}  mid={drift[len(drift)//2]:.2f}  "
              f"end={drift[-1]:.3f} {KJ}")
        print(f"FES residual drift (final third, mean)={final_third.mean():.3f} {KJ}  "
              f"max={final_third.max():.3f}")
        print(f"hill height: initial={h0:.3f} -> final~{h_last:.2e} {KJ}  "
              f"(ratio {h_last/h0:.2e})")
        if expl:
            lo, hi, rc, ncv, lt, ht_ = expl
            print(f"principal CV (col {args.princ_col}): range visited "
                  f"[{lo:.3f}, {hi:.3f}]  (low<{lt:.2f} / high>{ht_:.2f})  "
                  f"basin round-trips={rc} over {ncv} walkers")
        print(f"figure -> {figpath}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
