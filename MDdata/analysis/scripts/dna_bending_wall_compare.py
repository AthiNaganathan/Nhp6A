"""
PMFs for the three soft-LOWER_WALLS MetaD systems: DNA-only, 1Nhp6A:DNA, and
2Nhp6A:DNA -- projected on de2e and on the DNA bend angle, plus per-system
de2e-vs-bend 2D maps.

Reuses the final-bias reweighting machinery in reweight_angle.py (same method
as analysis_dna_bending.ipynb): each frame is weighted by the converged WT
bias evaluated on the system's own biased CV(s) (1D de2e for DNA_wall, 2D
coord_tail_dna+de2e for 1Nhp6A_wall, 3D coord_tail_dna_A+coord_tail_dna_B+de2e
for 2Nhp6A). theta and de2e are then histogrammed under those weights --
this works whether or not the observable itself is one of the biased CVs.

Bend angle convention: bend_deg = 180 - theta_deg, so 0 deg = straight DNA and
increasing values = more bending (theta itself keeps the original convention,
180 deg = straight, used elsewhere in the project).

Error bars: mean +/- std F from n_snaps evenly-spaced cumulative-time
snapshots (same time-snapshot-ensemble method as fes_marginal_meanstd.py's
'fixed' mode), per project convention that every FES/PMF plot carries an
uncertainty band.
"""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# The bend angle and the 2D (de2e, bend) grid are reweighted *spectators* --
# only de2e / the coordination CVs are biased -- so their sparsely visited
# bins are legitimately empty in some time-snapshots, and numpy's nan-reductions
# emit "empty slice"/"All-NaN" RuntimeWarnings there. This is understood and
# expected: the unreliable tail is trimmed at plot time (see STD_CEIL), and
# the full profiles are still written to the .dat archives. Silence just these
# two so the run log stays readable; any other warning still surfaces.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

sys.path.insert(0, os.path.dirname(__file__))
import reweight_angle as rw

T = 298.0
kBT = rw.kBT(T)
SCRATCH = "analysis"
os.makedirs("figures", exist_ok=True)

DE2E_RANGE = (2.4, 6.0)
BEND_RANGE = (0, 90)   # 180 - theta; 0 deg = straight
N_SNAPS = 5
# Reliability ceiling for truncating reweighted PMFs. The bend angle is a
# spectator (only de2e / the coordination CVs are biased) so it is only
# trustworthy where those CVs' sampling covers it; de2e is biased but its
# extreme edges (highly compressed / stretched DNA) are still thinly sampled.
# In both, poorly sampled bins hold a handful of frames that flicker between
# time-snapshots, giving spurious non-monotonic spikes with huge cross-snapshot
# std. Truncate each PMF to the contiguous run of bins around its minimum whose
# std stays <= this ceiling (see _reliable_run).
STD_CEIL = 5.0    # kJ/mol (~2 kBT at 298 K)

COLOR_DNA = "#0072B2"
COLOR_HMG = "#D55E00"
COLOR_TRIMER = "#009E73"

# Free-DNA reference: prefer the corrected-CV re-run (DNA_wall2) whenever it is
# present, falling back to the superseded DNA_wall (4-atom dna_end2, which biased BOTH
# de2e and theta -- the two quantities this script compares). This is the same 'auto' rule
# figure5_data.py applies, so the two pipelines can never disagree on which run is the reference.
DNA_DIR = "data/DNA_wall2" if os.path.isdir("data/DNA_wall2") else "data/DNA_wall"
DNA_KEY = os.path.basename(DNA_DIR)

SYSTEMS = {
    DNA_KEY: dict(
        label="DNA (wall)",
        colvar=[f"{DNA_DIR}/replica_{i}/colvar_r{i}.dat" for i in range(6)],
        hills=[f"{DNA_DIR}/hills.dat.{i}" for i in range(6)],
        cols={"de2e": 1, "theta": 2, "cmap": 3},
        biased=["de2e"], gamma_col=4,
        sh_min=[0], sh_max=[8], sh_bin=[200]),
    "1Nhp6A_wall": dict(
        label="1Nhp6A:DNA (wall)",
        colvar=[f"data/1Nhp6A_wall/replica_{i}/colvar_r{i}.dat" for i in range(6)],
        hills=[f"data/1Nhp6A_wall/hills.dat.{i}" for i in range(6)],
        cols={"coord_tail_dna": 1, "de2e": 2, "theta": 3, "cmap": 4},
        biased=["coord_tail_dna", "de2e"], gamma_col=6,
        sh_min=[0, 0], sh_max=[150, 8], sh_bin=[150, 200]),
    "2Nhp6A": dict(
        label="2Nhp6A:DNA (wall)",
        colvar=[f"data/2Nhp6A/replica_{i}/colvar_r{i}.dat" for i in range(6)],
        hills=[f"data/2Nhp6A/hills.dat.{i}" for i in range(6)],
        cols={"coord_tail_dna_A": 1, "coord_tail_dna_B": 2, "de2e": 3,
              "theta": 4, "cmap": 5},
        biased=["coord_tail_dna_A", "coord_tail_dna_B", "de2e"], gamma_col=8,
        # actual sampled range of coord_tail_dna_{A,B} is <=~33 (GRID_MAX=150
        # in the METAD input is a generous safety cap, not the sampled
        # range) -- restrict the sum_hills grid so the 3D marginalization
        # stays tractable while keeping the same de2e resolution used above
        sh_min=[0, 0, 0], sh_max=[40, 40, 8], sh_bin=[40, 40, 200]),
    # Point mutants: byte-for-byte the same 2D wall protocol/CVs/colvar layout as
    # 1Nhp6A_wall (only the protein-atom indices differ), so they clone its config
    # verbatim apart from dir/label. All three converged (2026-08-02).
    "1S26D_wall": dict(
        label="1S26D:DNA (wall)",
        colvar=[f"data/1S26D_wall/replica_{i}/colvar_r{i}.dat" for i in range(6)],
        hills=[f"data/1S26D_wall/hills.dat.{i}" for i in range(6)],
        cols={"coord_tail_dna": 1, "de2e": 2, "theta": 3, "cmap": 4},
        biased=["coord_tail_dna", "de2e"], gamma_col=6,
        sh_min=[0, 0], sh_max=[150, 8], sh_bin=[150, 200]),
    "1T63D_wall": dict(
        label="1T63D:DNA (wall)",
        colvar=[f"data/1T63D_wall/replica_{i}/colvar_r{i}.dat" for i in range(6)],
        hills=[f"data/1T63D_wall/hills.dat.{i}" for i in range(6)],
        cols={"coord_tail_dna": 1, "de2e": 2, "theta": 3, "cmap": 4},
        biased=["coord_tail_dna", "de2e"], gamma_col=6,
        sh_min=[0, 0], sh_max=[150, 8], sh_bin=[150, 200]),
    "1TM_wall": dict(
        label="1TM:DNA (wall)",
        colvar=[f"data/1TM_wall/replica_{i}/colvar_r{i}.dat" for i in range(6)],
        hills=[f"data/1TM_wall/hills.dat.{i}" for i in range(6)],
        cols={"coord_tail_dna": 1, "de2e": 2, "theta": 3, "cmap": 4},
        biased=["coord_tail_dna", "de2e"], gamma_col=6,
        sh_min=[0, 0], sh_max=[150, 8], sh_bin=[150, 200]),
}
# Wong colourblind-safe palette; WT keeps its established vermillion.
COLOR_1S26D = "#CC79A7"   # reddish purple
COLOR_1T63D = "#E69F00"   # orange
COLOR_1TM = "#56B4E9"     # sky blue
colors = {DNA_KEY: COLOR_DNA, "1Nhp6A_wall": COLOR_HMG, "2Nhp6A": COLOR_TRIMER,
          "1S26D_wall": COLOR_1S26D, "1T63D_wall": COLOR_1T63D, "1TM_wall": COLOR_1TM}


def hills_header(path):
    lines = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("#", "!")):
                lines.append(line)
            else:
                break
    return "".join(lines)


def _load_full(cfg):
    colvar_data = {k: [] for k in cfg["cols"]}
    colvar_data["time_ns"] = []
    for c in cfg["colvar"]:
        d = rw.load_colvar(c, cfg["cols"])
        for k in cfg["cols"]:
            colvar_data[k].append(d[k])
        colvar_data["time_ns"].append(d["time_ns"])
    full = {k: np.concatenate(v) for k, v in colvar_data.items()}
    full["theta_deg"] = np.degrees(full["theta"])
    full["bend_deg"] = 180.0 - full["theta_deg"]
    return full


def _pts(full, cfg, mask=None):
    biased = cfg["biased"]
    if mask is None:
        cols = [full[b] for b in biased]
    else:
        cols = [full[b][mask] for b in biased]
    return cols[0] if len(biased) == 1 else np.column_stack(cols)


def process(name, cfg, full=None):
    """Full-length reweighting: gamma, ESS, and a de2e validation check
    (frame-reweighted histogram vs the direct sum_hills marginal).

    `full` may be supplied by a caller that has already pooled the colvars
    (parsing them is the slowest step); omitted, it is loaded here.
    """
    ndim = len(cfg["biased"])
    gamma = rw.hills_gamma(cfg["hills"][0], cfg["gamma_col"])

    fes = os.path.join(SCRATCH, f"fes_biased_{name}.dat")
    rw.run_sumhills(cfg["hills"], fes, cfg["sh_min"], cfg["sh_max"], cfg["sh_bin"])

    if full is None:
        full = _load_full(cfg)
    pts = _pts(full, cfg)
    Ff = rw.fes_at_points(fes, pts, ndim)
    w = rw.frame_weights(Ff, gamma, T)
    ess = w.sum() ** 2 / np.sum(w ** 2)

    # validation: reweighted F(de2e) vs the direct sum_hills marginal on de2e
    de2e_axis = cfg["biased"].index("de2e")
    d = np.loadtxt(fes, comments="#")
    if ndim == 1:
        mcv, mF = d[:, 0], d[:, 1] - np.nanmin(d[:, 1])
    else:
        cols_grid, Fcol = d[:, :ndim], d[:, ndim]
        keep_vals = cols_grid[:, de2e_axis]
        uniq = np.unique(keep_vals)
        mF = np.full(uniq.shape, np.nan)
        for i, v in enumerate(uniq):
            sel = keep_vals == v
            z = np.nansum(np.exp(-Fcol[sel] / kBT))
            if z > 0:
                mF[i] = -kBT * np.log(z)
        mF -= np.nanmin(mF)
        mcv = uniq
    cde, Fde = rw.fes_1d(full["de2e"], w, DE2E_RANGE, 80, T)
    mi = np.interp(cde, mcv, mF, left=np.nan, right=np.nan)
    ok = np.isfinite(Fde) & np.isfinite(mi)
    a = Fde[ok] - np.nanmin(Fde[ok])
    b = mi[ok] - np.nanmin(mi[ok])
    mad = np.mean(np.abs(a - b)) if ok.any() else np.nan

    return dict(ndim=ndim, gamma=gamma, ess=ess, nframes=len(w), mad=mad,
                time_ns=full["time_ns"])


def snapshot_weights(name, cfg, t_max_ns, n_snaps=N_SNAPS, full=None):
    """Final-bias reweighting at n_snaps cumulative-time cutoffs.

    Returns (full, snaps) where `full` is the pooled colvar dict and `snaps` is
    a list of {t_ns, mask, w}: `mask` selects the frames deposited up to t_ns
    and `w` are their weights under the bias accumulated up to that same time.

    Split out of snapshot_pmf so the same snapshot ensemble can back a PMF along
    ANY observable in the colvar (e.g. F(coord_tail_dna, bend) for the figure-5
    IDR-contact/bending map) without re-running sum_hills per observable.
    """
    ndim = len(cfg["biased"])
    gamma = rw.hills_gamma(cfg["hills"][0], cfg["gamma_col"])
    header = hills_header(cfg["hills"][0])
    hills_arrs = [np.loadtxt(h, comments=["#", "!"]) for h in cfg["hills"]]
    if full is None:
        full = _load_full(cfg)

    snaps = []
    for t_ns in np.linspace(t_max_ns / n_snaps, t_max_ns, n_snaps):
        tmp_files = []
        for i, h in enumerate(hills_arrs):
            rows = h[h[:, 0] <= t_ns * 1000]
            if len(rows) == 0:
                continue
            tmp = os.path.join(SCRATCH, f"hills_{name}_snap_w{i}.dat")
            with open(tmp, "w") as f:
                f.write(header)
                np.savetxt(f, rows, fmt="%20.15g")
            tmp_files.append(tmp)
        fes = os.path.join(SCRATCH, f"fes_biased_{name}_snap.dat")
        rw.run_sumhills(tmp_files, fes, cfg["sh_min"], cfg["sh_max"], cfg["sh_bin"])
        for f in tmp_files:
            os.remove(f)

        m = full["time_ns"] <= t_ns
        Ff = rw.fes_at_points(fes, _pts(full, cfg, mask=m), ndim)
        snaps.append(dict(t_ns=t_ns, mask=m,
                          w=rw.frame_weights(Ff, gamma, T)))
    return full, snaps


def snapshot_pmf(name, cfg, t_max_ns, n_snaps=N_SNAPS):
    """Reweight at n_snaps cumulative-time cutoffs -> mean/std F(de2e),
    F(bend), and F(de2e, bend) across snapshots (one sum_hills + one
    reweighting per snapshot serves all three outputs)."""
    full, snaps = snapshot_weights(name, cfg, t_max_ns, n_snaps)

    de2e_curves, bend_curves, grids2d = [], [], []
    tx = ty = cd = cb = None
    for s in snaps:
        m, w = s["mask"], s["w"]
        cd, Fd = rw.fes_1d(full["de2e"][m], w, DE2E_RANGE, 80, T)
        cb, Fb = rw.fes_1d(full["bend_deg"][m], w, BEND_RANGE, 60, T)
        txi, tyi, F2d = rw.fes_2d(full["de2e"][m], full["bend_deg"][m], w,
                                   DE2E_RANGE, BEND_RANGE, [60, 60], T)
        de2e_curves.append(Fd)
        bend_curves.append(Fb)
        grids2d.append(F2d)
        tx, ty = txi, tyi

    de2e_curves = np.array(de2e_curves)
    bend_curves = np.array(bend_curves)
    grids2d = np.array(grids2d)

    de2e_mean = np.nanmean(de2e_curves, axis=0)
    de2e_mean -= np.nanmin(de2e_mean)
    de2e_std = np.nanstd(de2e_curves, axis=0)

    bend_mean = np.nanmean(bend_curves, axis=0)
    bend_mean -= np.nanmin(bend_mean)
    bend_std = np.nanstd(bend_curves, axis=0)

    grid2d_mean = np.nanmean(grids2d, axis=0)
    grid2d_mean -= np.nanmin(grid2d_mean)
    grid2d_std = np.nanstd(grids2d, axis=0)

    return dict(cd=cd, de2e_mean=de2e_mean, de2e_std=de2e_std,
                cb=cb, bend_mean=bend_mean, bend_std=bend_std,
                tx=tx, ty=ty, grid2d_mean=grid2d_mean, grid2d_std=grid2d_std)


def _reliable_run(mean, std, ceiling):
    """Boolean mask of the contiguous set of bins around the free-energy
    minimum whose cross-snapshot std stays <= ceiling. Truncating outward from
    the minimum (rather than masking every std>ceiling bin globally) keeps a
    single physical branch: it drops the sparsely-sampled spectator tails on
    *both* sides -- including any isolated low-std bin stranded beyond a gap of
    unreliable bins in the far tail -- while never punching holes in the
    well-sampled core."""
    finite = np.isfinite(mean) & np.isfinite(std)
    ok = finite & (std <= ceiling)
    if not ok.any():
        return finite
    i0 = int(np.nanargmin(np.where(ok, mean, np.nan)))  # basin (well-sampled)
    mask = np.zeros_like(ok)
    mask[i0] = True
    i = i0 - 1
    while i >= 0 and ok[i]:
        mask[i] = True
        i -= 1
    i = i0 + 1
    while i < len(ok) and ok[i]:
        mask[i] = True
        i += 1
    return mask


def _reliable_region2d(mean, std, ceiling):
    """2D generalization of _reliable_run: the 4-connected component of bins
    with finite mean and std <= ceiling that contains the global free-energy
    minimum. Drops the sparsely-sampled high-std fringe of the (de2e, bend)
    map and any disconnected low-std island stranded out in the tails, keeping
    the single physical basin+branches actually resolved by the sampling."""
    ok = np.isfinite(mean) & np.isfinite(std) & (std <= ceiling)
    if not ok.any():
        return np.isfinite(mean)
    i0, j0 = np.unravel_index(int(np.nanargmin(np.where(ok, mean, np.nan))),
                              mean.shape)
    keep = np.zeros_like(ok)
    ni, nj = mean.shape
    stack = [(int(i0), int(j0))]
    keep[i0, j0] = True
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < ni and 0 <= b < nj and ok[a, b] and not keep[a, b]:
                keep[a, b] = True
                stack.append((a, b))
    return keep


def _band_plot(ax, x, mean, std, color, label=None, std_ceiling=None, ls="-"):
    m = np.isfinite(mean)
    if std_ceiling is not None:
        m = m & _reliable_run(mean, std, std_ceiling)
    ax.plot(x[m], mean[m], lw=2, ls=ls, color=color, label=label)
    ax.fill_between(x[m], (mean - std)[m], (mean + std)[m], alpha=0.25, color=color)


# ── comparison sets ───────────────────────────────────────────────────────────
# Two deliberately separate overlays (user's decision, 2026-07-31):
#
#   "1to1"  DNA-only vs 1Nhp6A:DNA. Both pass their reweighting gates, and this is the
#           comparison reported with full credibility.
#   "all3"  adds the 2:1 trimer. Its PMF is NOT converged -- the reweighting gate fails
#           (ESS ~ 0%, MAD 4.7 kJ/mol vs a 2.99 threshold), which is why figure5_data.py
#           excludes this same run and builds panel F from unbiased equilibrium MD instead.
#           The 2:1 result of interest is helix destabilisation, and the protein-structure
#           disruption there is too widespread to capture at the sampling produced so far.
#           Shown dashed, for inspection only -- do NOT report it alongside the 1:1
#           comparison as equally credible.
#   "mutants"  WT holo vs the three point mutants (all wall, all 6x300 ns, all
#              converged 2026-08-02). The scientific comparison for the mutant panel:
#              how S26D / T63D / TM shift the bend and end-to-end PMFs relative to WT.
GROUPS = {"1to1": [DNA_KEY, "1Nhp6A_wall"],
          "all3": [DNA_KEY, "1Nhp6A_wall", "2Nhp6A"],
          "mutants": ["1Nhp6A_wall", "1S26D_wall", "1T63D_wall", "1TM_wall"]}
LOW_CREDIBILITY = {"2Nhp6A"}      # drawn dashed and labelled, wherever it appears


def run_all(systems=None):
    """Full pipeline: reweight every system, write its tables/figures immediately,
    then draw the cross-system overlays.

    `systems` restricts which systems are reweighted (default: all in SYSTEMS).
    Each system's per-system outputs are written right after its own reweighting,
    so a run killed mid-way (e.g. the slow, gate-failing 2Nhp6A 3D reweighting)
    still leaves complete tables and figures for every system that finished.
    """
    names = list(systems) if systems else list(SYSTEMS)
    RES, SNAP = {}, {}
    for name in names:
        cfg = SYSTEMS[name]
        print(f"processing {name} ...", flush=True)
        RES[name] = process(name, cfg)
        r = RES[name]
        print(f"   ndim={r['ndim']}  gamma={r['gamma']:.0f}  frames={r['nframes']:,}  "
              f"ESS={r['ess']:,.0f} ({100*r['ess']/r['nframes']:.0f}%)  "
              f"VALIDATION MAD(de2e)={r['mad']:.2f} kJ/mol "
              f"({'OK' if r['mad'] < kBT else 'CHECK'})")

        t_max_ns = r["time_ns"].max()
        print(f"   snapshot PMFs (t_max={t_max_ns:.1f} ns, n_snaps={N_SNAPS}) ...", flush=True)
        SNAP[name] = snapshot_pmf(name, cfg, t_max_ns)
        write_system_outputs(name, cfg, SNAP[name])

    overlay_figures(SNAP)
    print("done")


def write_system_outputs(name, cfg, s):
    """Per-system outputs: F(de2e), F(bend), F(de2e,bend) tables + figures."""
    np.savetxt(f"analysis/{name}_pmf_de2e_mean_std.dat",
               np.column_stack([s["cd"], s["de2e_mean"], s["de2e_std"]]),
               header="de2e_nm  F_kJmol  std_kJmol")
    fig, ax = plt.subplots(figsize=(6, 4))
    _band_plot(ax, s["cd"], s["de2e_mean"], s["de2e_std"], colors[name],
               std_ceiling=STD_CEIL)
    ax.set_xlabel(r"$d_\mathrm{e-e}$ (nm)")
    ax.set_ylabel(r"$\Delta F$ (kJ/mol)")
    ax.set_title(f"{cfg['label']} " + r"-- $F(d_\mathrm{e-e})$", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"figures/{name}_pmf_de2e_v1.png", dpi=300, bbox_inches="tight")
    plt.close()

    np.savetxt(f"analysis/{name}_pmf_bend_mean_std.dat",
               np.column_stack([s["cb"], s["bend_mean"], s["bend_std"]]),
               header="bend_deg(180-theta)  F_kJmol  std_kJmol")
    fig, ax = plt.subplots(figsize=(6, 4))
    _band_plot(ax, s["cb"], s["bend_mean"], s["bend_std"], colors[name],
               std_ceiling=STD_CEIL)
    ax.set_xlabel(r"DNA bend angle  $180-\theta$ (deg)   [0 deg = straight]")
    ax.set_ylabel(r"$\Delta F$ (kJ/mol)")
    ax.set_title(f"{cfg['label']} -- $F$(bend)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"figures/{name}_pmf_bend_v1.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    # mask the mean map to the reliable region (std <= STD_CEIL, connected
    # to the minimum); the std panel is left full-range as the diagnostic
    # that justifies where the cut falls.
    keep2d = _reliable_region2d(s["grid2d_mean"], s["grid2d_std"], STD_CEIL)
    g2d = np.where(keep2d, s["grid2d_mean"], np.nan)
    g2d = g2d - np.nanmin(g2d)
    vmax = np.nanpercentile(g2d, 98)
    im0 = axes[0].pcolormesh(s["tx"], s["ty"], np.ma.masked_invalid(g2d).T,
                             cmap="RdYlBu_r", vmin=0, vmax=vmax, shading="auto")
    axes[0].contour(s["tx"], s["ty"], np.ma.masked_invalid(g2d).T,
                    levels=np.arange(0, vmax, 5), colors="k", linewidths=0.4)
    plt.colorbar(im0, ax=axes[0], label=r"$\Delta F$ (kJ/mol)")
    axes[0].set_title(r"mean (5 snapshots), std $\leq$ "
                      f"{STD_CEIL:.0f} kJ/mol", fontsize=9)

    im1 = axes[1].pcolormesh(s["tx"], s["ty"], np.ma.masked_invalid(s["grid2d_std"]).T,
                             cmap="viridis", shading="auto")
    plt.colorbar(im1, ax=axes[1], label="std (kJ/mol)")
    axes[1].set_title("std across snapshots (full)", fontsize=9)

    for a in axes:
        a.set_xlabel(r"$d_\mathrm{e-e}$ (nm)", fontsize=11)
        a.set_ylabel(r"bend $180-\theta$ (deg)", fontsize=11)
    fig.suptitle(f"{cfg['label']} " + r"-- $F(d_\mathrm{e-e}$, bend), reweighted", y=1.02)
    plt.tight_layout()
    plt.savefig(f"figures/{name}_pmf2d_de2e_bend_v1.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"{name}: saved pmf_de2e, pmf_bend, pmf2d_de2e_bend")


# ── SI panel: Figure-5 Panel-E analog for holo + point mutants ──────────────────
# F(n_IDR-DNA contacts, bend), on the SAME grid figure5_data.py uses for the
# main-text holo panel E (COORD_RANGE x BEND_RANGE, E_BINS), so the supplementary
# mutant panels are drawn on axes directly comparable to the paper's Figure 5E.
COORD_RANGE = (0, 40)
E_BINS = [50, 50]


def _save2d(path, x, y, mean, std, counts, header, walkers=None):
    """Long format read by figure5_plot.load2d: cv1 cv2 F_mean F_std count [n_walkers].
    The optional 6th column (n_walkers) carries the multi-walker ergodicity check."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    stack = [X.ravel(), Y.ravel(), mean.ravel(), std.ravel(), counts.ravel()]
    if walkers is not None:
        stack.append(walkers.ravel())
    np.savetxt(path, np.column_stack(stack), fmt="%14.6g", header=header)
    print(f"  wrote {path}  ({len(x)}x{len(y)} bins)")


def _walker_coverage(cfg, mins, maxs, bins, min_fr=50):
    """Per-bin count of walkers that INDEPENDENTLY sample each (coord_tail_dna, bend) bin
    with at least min_fr frames -- a multi-walker ergodicity check on the E-grid. A bin
    reached by only 1-2 of the 6 walkers is not an equilibrated free-energy estimate even
    when densely populated overall (the S26D low-bend/high-contact corner: 2/6 walkers,
    one entering only at 65 ns). min_fr=50 frames (~10 ps at the 0.2 ps colvar stride)
    filters single-frame fly-throughs so a genuine residence is required."""
    cov = np.zeros([bins[0], bins[1]], dtype=int)
    for c in cfg["colvar"]:
        d = rw.load_colvar(c, cfg["cols"])
        bend = 180.0 - np.degrees(d["theta"])
        h, _, _ = np.histogram2d(d["coord_tail_dna"], bend, bins=bins,
                                 range=[mins, maxs])
        cov += (h >= min_fr).astype(int)
    return cov


def write_idrdna_bend_2d(name, cfg=None, n_snaps=N_SNAPS):
    """SI Panel-E analog: reweighted F(n_IDR-DNA, bend) for a holo/mutant system.

    Reuses the SAME snapshot reweighting as the de2e/bend PMFs (snapshot_weights)
    and the SAME grid as figure5_data.py's panel E, then writes a 5-column table
    -> analysis/<name>_pmf2d_idrdna_bend.dat that figure5_plot.load2d reads. Only
    defined for systems whose biased CVs include coord_tail_dna (WT holo + the three
    point mutants); DNA-only and the 2:1 trimer have no single such CV and are skipped.
    """
    cfg = cfg or SYSTEMS[name]
    if "coord_tail_dna" not in cfg["cols"]:
        print(f"  [skip {name}: no coord_tail_dna CV]")
        return
    full = _load_full(cfg)
    t_max = full["time_ns"].max()
    print(f"processing {name} F(n_IDR-DNA, bend) 2D "
          f"(t_max={t_max:.0f} ns, n_snaps={n_snaps}) ...", flush=True)
    full, snaps = snapshot_weights(name, cfg, t_max, n_snaps=n_snaps, full=full)

    grids, ex, ey = [], None, None
    for s in snaps:
        ex, ey, F = rw.fes_2d(full["coord_tail_dna"][s["mask"]], full["bend_deg"][s["mask"]],
                              s["w"], COORD_RANGE, BEND_RANGE, E_BINS, T)
        grids.append(F)
    G = np.array(grids)
    mean = np.nanmean(G, axis=0)
    mean -= np.nanmin(mean)
    std = np.nanstd(G, axis=0)
    cnt, _, _ = np.histogram2d(full["coord_tail_dna"], full["bend_deg"],
                               bins=E_BINS, range=[COORD_RANGE, BEND_RANGE])
    wcov = _walker_coverage(cfg, COORD_RANGE, BEND_RANGE, E_BINS)
    _save2d(f"analysis/{name}_pmf2d_idrdna_bend.dat", ex, ey, mean, std, cnt,
            f"{cfg['label']} F(n_IDR-DNA, bend), final-bias reweighted, {n_snaps} snapshots, "
            f"{t_max:.0f} ns/walker\n"
            "n_IDR_DNA  bend_deg(180-theta)  F_mean(kJ/mol)  F_std(kJ/mol)  frame_counts  n_walkers",
            walkers=wcov)


def load_snap_from_dat(names=None, scratch=SCRATCH):
    """Rebuild the SNAP dict from the per-system PMF tables written by a previous run.

    The reweighting is the expensive part (~1.5 h for the 3D trimer); the overlays are
    pure plotting. This lets `--plot-only` redraw them -- e.g. after regrouping which
    systems appear together -- without recomputing anything.
    """
    snap = {}
    for name in (names or SYSTEMS):
        fd = f"{scratch}/{name}_pmf_de2e_mean_std.dat"
        fb = f"{scratch}/{name}_pmf_bend_mean_std.dat"
        if not (os.path.exists(fd) and os.path.exists(fb)):
            print(f"  [skip {name}: no saved PMF tables -- run without --plot-only first]")
            continue
        d, b = np.loadtxt(fd), np.loadtxt(fb)
        snap[name] = {"cd": d[:, 0], "de2e_mean": d[:, 1], "de2e_std": d[:, 2],
                      "cb": b[:, 0], "bend_mean": b[:, 1], "bend_std": b[:, 2]}
    return snap


def overlay_figures(SNAP):
    """One overlay figure per comparison set in GROUPS x {de2e, bend}."""
    overlays = [
        ("de2e", "cd", "de2e_mean", "de2e_std",
         r"$d_\mathrm{e-e}$ (nm)", r"$F(d_\mathrm{e-e})$"),
        ("bend", "cb", "bend_mean", "bend_std",
         r"DNA bend angle  $180-\theta$ (deg)   [0 deg = straight]", r"$F$(bend)"),
    ]
    for group, members in GROUPS.items():
        if any(name not in SNAP for name in members):
            missing = [n for n in members if n not in SNAP]
            print(f"  [skip group {group}: not reweighted this run: {', '.join(missing)}]")
            continue
        for obs, xk, mk, sk, xlabel, tlabel in overlays:
            fig, ax = plt.subplots(figsize=(6, 4))
            for name in members:
                if name not in SNAP:
                    continue
                s, cfg = SNAP[name], SYSTEMS[name]
                low = name in LOW_CREDIBILITY
                label = cfg["label"] + (" -- not converged" if low else "")
                _band_plot(ax, s[xk], s[mk], s[sk], colors[name], label,
                           std_ceiling=STD_CEIL, ls="--" if low else "-")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\Delta F$ (kJ/mol)")
            ax.legend(fontsize=9)
            title = f"Wall-simulation comparison: {tlabel}"
            if any(n in LOW_CREDIBILITY for n in members):
                title += "   (dashed = reweighting gate fails)"
            ax.set_title(title, fontsize=10)
            plt.tight_layout()
            out = f"figures/pmf_{obs}_compare_wall_{group}_v1.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  saved {out}")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--plot-only", action="store_true",
                     help="redraw the comparison overlays from the saved per-system PMF "
                          "tables in analysis/, skipping the (expensive) reweighting")
    _ap.add_argument("--idrdna-bend-2d", nargs="*", metavar="SYS", default=None,
                     help="write the SI Figure-5E-analog F(n_IDR-DNA, bend) 2D tables "
                          "(analysis/<sys>_pmf2d_idrdna_bend.dat) for the given systems and "
                          "exit; with no names, defaults to WT holo + the three point mutants.")
    _ap.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=None,
                     help="restrict reweighting to these systems (default: all). "
                          "Groups missing a member are skipped. Use e.g. "
                          f"'--systems {DNA_KEY} 1Nhp6A_wall' for the converged 1to1 overlay, "
                          "avoiding the slow, gate-failing 2Nhp6A reweighting.")
    _args = _ap.parse_args()
    if _args.idrdna_bend_2d is not None:
        targets = _args.idrdna_bend_2d or ["1Nhp6A_wall", "1S26D_wall", "1T63D_wall", "1TM_wall"]
        for _nm in targets:
            write_idrdna_bend_2d(_nm)
        print("done (idrdna-bend-2d)")
    elif _args.plot_only:
        overlay_figures(load_snap_from_dat(_args.systems))
        print("done (plot-only)")
    else:
        run_all(_args.systems)
