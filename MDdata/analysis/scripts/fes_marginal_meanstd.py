#!/usr/bin/env python3
"""de2e marginal FES with a mean +/- std error band, from time-snapshots of the
pooled-walker MetaD hills.

Generalises two existing notebook protocols into one parameterised script so
new systems (e.g. soft-wall variants) don't need a new notebook to feed
plot_pmf_comparison.py:

  ndim=2 (analysis_1Nhp6A_2D.ipynb / 1T63D / 1S26D / 1TM): hills deposited on
      (spectator_cv, de2e); Boltzmann-marginalise over the spectator CV within
      --win-cv1 to get the de2e marginal at each snapshot.
  ndim=1 (analysis_replicas.ipynb, DNA): hills deposited on de2e directly;
      sum_hills output IS the marginal.

Two snapshot-selection modes:
  fixed     - n_snaps evenly spaced cumulative snapshots over the whole run
              (matches the 2D protein-system notebooks).
  converged - find t_conv from the running-mean deposited-hill-height decay,
              then average n_snaps snapshots within [t_conv, T_MAX]
              (matches analysis_replicas.ipynb). Falls back to 'fixed' over
              the full (non-converged) run if the bias hasn't decayed below
              threshold yet, printing a PRELIMINARY warning.

Output: <datadir>/<label>_fes_de2e_mean_std.dat (de2e, F_mean, F_std),
        same 3-column format plot_pmf_comparison.py already reads.

Usage:
    fes_marginal_meanstd.py SYSTEM_DIR --label LABEL --ndim {1,2}
        [--snapshot-mode {fixed,converged}] [--n-snaps N] [--win-cv1 LO HI]
        [--win-cv2 LO HI]
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.interpolate import interp1d

KB = 0.00831446261815324  # kJ/mol/K


def find_hills(system_dir):
    files = sorted(glob.glob(os.path.join(system_dir, "hills.dat.*")))
    if not files:
        sys.exit(f"No hills.dat.* in {system_dir}")
    return files


def hills_header(path):
    lines = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                lines.append(line)
            else:
                break
    return "".join(lines)


def load_hills(path):
    return np.loadtxt(path, comments=["#", "!"])


def run_sumhills(hills_csv, outfile, mins, maxs, bins):
    cmd = (f"plumed sum_hills --hills {hills_csv} --outfile {outfile} "
           f"--min {','.join(map(str, mins))} --max {','.join(map(str, maxs))} "
           f"--bin {','.join(map(str, bins))} --mintozero")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"plumed sum_hills failed:\n{r.stderr.strip()[:800]}")


def run_sumhills_upto(hills, header, t_ps, tag, workdir, mins, maxs, bins):
    tmp_files = []
    for i, h in enumerate(hills):
        rows = h[h[:, 0] <= t_ps]
        if len(rows) == 0:
            continue
        tmp = os.path.join(workdir, f"hills_w{i}_{tag}.dat")
        with open(tmp, "w") as f:
            f.write(header)
            np.savetxt(f, rows, fmt="%20.15g")
        tmp_files.append(tmp)
    out = os.path.join(workdir, f"fes_{tag}.dat")
    run_sumhills(",".join(tmp_files), out, mins, maxs, bins)
    for f in tmp_files:
        os.remove(f)
    return out


def load_fes(path, ndim):
    d = np.genfromtxt(path, comments="#", invalid_raise=False, filling_values=np.nan)
    d = d[np.isfinite(d[:, 0])]
    if ndim == 1:
        return d[:, 0], None, d[:, 1]
    cv1u, cv2u = np.unique(d[:, 0]), np.unique(d[:, 1])
    fes = d[:, 2].reshape(len(cv2u), len(cv1u)).T   # PLUMED: cv2 outer, cv1 inner
    return cv1u, cv2u, fes


def marginal_fes(cv1, cv2, fes2d, window, kBT, axis=0):
    """Boltzmann-weighted marginal of a 2D FES (shape (n_cv1, n_cv2)).

    axis=0 (default, unchanged behaviour): project onto cv2, summing over cv1
           restricted to `window` (a window on cv1).
    axis=1: project onto cv1, summing over cv2 restricted to `window`
           (a window on cv2).

    Pass the raw (unmasked) FES -- unsampled bins carry large F, hence
    negligible Boltzmann weight, so no explicit masking is needed.
    """
    if axis == 0:
        m = (cv1 >= window[0]) & (cv1 <= window[1])
        sub = fes2d[m, :].copy()
        cv_out = cv2
    else:
        m = (cv2 >= window[0]) & (cv2 <= window[1])
        sub = fes2d[:, m].copy()
        cv_out = cv1
    sub[~np.isfinite(sub)] = np.inf
    with np.errstate(over="ignore"):
        boltz = np.exp(-sub / kBT)
    boltz[~np.isfinite(boltz)] = 0.0
    z = boltz.sum(axis=axis)
    z[z == 0] = np.nan
    f = -kBT * np.log(z)
    f -= np.nanmin(f)
    return cv_out, f


def find_t_conv(hills, height_col, thresh_frac, smooth_win):
    pool = np.vstack(hills)
    pool = pool[np.argsort(pool[:, 0])]
    h0 = float(np.max(pool[:, height_col]))
    thresh = thresh_frac * h0
    if len(pool) < smooth_win:
        return None, thresh
    running = np.convolve(pool[:, height_col], np.ones(smooth_win) / smooth_win, mode="valid")
    t_smooth_ns = pool[smooth_win // 2: smooth_win // 2 + len(running), 0] / 1000.0
    below = np.where(running < thresh)[0]
    if len(below) == 0:
        return None, thresh
    return float(t_smooth_ns[below[0]]), thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system_dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--ndim", type=int, choices=(1, 2), required=True)
    ap.add_argument("--cv1-min", type=float, default=0.0, help="spectator CV grid (ndim=2)")
    ap.add_argument("--cv1-max", type=float, default=40.0)
    ap.add_argument("--cv1-bin", type=int, default=200)
    ap.add_argument("--cv2-min", type=float, default=2.0, help="de2e grid")
    ap.add_argument("--cv2-max", type=float, default=6.0)
    ap.add_argument("--cv2-bin", type=int, default=200)
    ap.add_argument("--win-cv1", type=float, nargs=2, default=(0.0, 40.0),
                    help="marginalisation window on spectator CV (ndim=2 only)")
    ap.add_argument("--win-cv2", type=float, nargs=2, default=(2.4, 5.6),
                    help="output window on de2e")
    ap.add_argument("--n-snaps", type=int, default=5)
    ap.add_argument("--snapshot-mode", choices=("fixed", "converged"), default="fixed")
    ap.add_argument("--height-thresh-frac", type=float, default=0.02,
                    help="convergence mode: fraction of initial hill height")
    ap.add_argument("--smooth-win", type=int, default=2000,
                    help="convergence mode: running-mean window (hills)")
    ap.add_argument("--temp", type=float, default=298.15)
    ap.add_argument("--datadir", default="analysis")
    args = ap.parse_args()

    kBT = KB * args.temp
    hills_files = find_hills(args.system_dir)
    header = hills_header(hills_files[0])
    hills = [load_hills(f) for f in hills_files]
    t_max_ns = max(h[:, 0].max() for h in hills) / 1000.0
    height_col = 1 + 2 * args.ndim  # time, ndim cv, ndim sigma, [height]

    print(f"=== {args.label} (ndim={args.ndim}) ===")
    print(f"walkers={len(hills_files)}  per-walker length up to {t_max_ns:.2f} ns")

    mins = [args.cv1_min, args.cv2_min] if args.ndim == 2 else [args.cv2_min]
    maxs = [args.cv1_max, args.cv2_max] if args.ndim == 2 else [args.cv2_max]
    bins = [args.cv1_bin, args.cv2_bin] if args.ndim == 2 else [args.cv2_bin]

    preliminary = False
    if args.snapshot_mode == "converged":
        t_conv, thresh = find_t_conv(hills, height_col, args.height_thresh_frac, args.smooth_win)
        if t_conv is None:
            print(f"[WARNING] bias not yet decayed below {thresh:.4f} kJ/mol -- "
                  f"falling back to 'fixed' snapshots over the full (non-converged) "
                  f"run. Result is PRELIMINARY.")
            preliminary = True
            snap_t_ns = np.linspace(t_max_ns / args.n_snaps, t_max_ns, args.n_snaps)
        else:
            print(f"t_conv = {t_conv:.2f} ns (threshold {thresh:.4f} kJ/mol)")
            snap_t_ns = np.linspace(t_conv, t_max_ns, args.n_snaps + 1)[1:]
    else:
        snap_t_ns = np.linspace(t_max_ns / args.n_snaps, t_max_ns, args.n_snaps)

    print(f"Snapshot times (ns): {np.round(snap_t_ns, 2)}")

    cv2_common = np.linspace(args.win_cv2[0], args.win_cv2[1], 300)
    stack = []
    with tempfile.TemporaryDirectory(prefix=f"sumhills_{args.label}_") as workdir:
        for t_ns in snap_t_ns:
            tag = f"t{int(t_ns * 1000):06d}ps"
            print(f"  FES up to {t_ns:.2f} ns ...", end=" ", flush=True)
            fname = run_sumhills_upto(hills, header, t_ns * 1000, tag, workdir, mins, maxs, bins)
            if args.ndim == 1:
                cv2, _, fes = load_fes(fname, ndim=1)
            else:
                cv1, cv2raw, fes2d = load_fes(fname, ndim=2)
                cv2, fes = marginal_fes(cv1, cv2raw, fes2d, args.win_cv1, kBT)
            ok = (cv2 >= args.win_cv2[0]) & (cv2 <= args.win_cv2[1]) & np.isfinite(fes)
            if ok.sum() > 3:
                f_interp = interp1d(cv2[ok], fes[ok], kind="linear",
                                    bounds_error=False, fill_value=np.nan)
                stack.append(f_interp(cv2_common))
            print("done")

    if len(stack) < 2:
        sys.exit("Fewer than 2 usable snapshots -- cannot estimate a std band.")

    stack = np.array(stack)
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    mean -= np.nanmin(mean)

    os.makedirs(args.datadir, exist_ok=True)
    out = os.path.join(args.datadir, f"{args.label}_fes_de2e_mean_std.dat")
    header_note = "PRELIMINARY (not converged) " if preliminary else ""
    np.savetxt(out, np.column_stack([cv2_common, mean, std]),
              header=(f"{header_note}T_MAX={t_max_ns:.2f}ns "
                      f"de2e(nm)  F_mean(kJ/mol)  F_std(kJ/mol)"))
    print(f"Saved: {out}")
    print(f"de2e min at {cv2_common[np.nanargmin(mean)]:.3f} nm | "
          f"mean sigma = {np.nanmean(std):.3f} kJ/mol | max sigma = {np.nanmax(std):.3f} kJ/mol")
    if preliminary:
        print("*** PRELIMINARY: bias has not converged -- re-run once sampling extends. ***")


if __name__ == "__main__":
    main()
