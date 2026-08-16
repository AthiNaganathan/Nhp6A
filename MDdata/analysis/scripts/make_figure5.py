#!/usr/bin/env python3
"""Assemble Figure 5 (WT Nhp6A: apo unstructuring + holo DNA bending).

Reuses pre-computed data products (no hills re-summing unless missing):
  Panel A  apo-large 2D FES F(Q_core, n_tail_core)  <- analysis/1Nhp6A_apo_large_fes_sumhills.dat
  Panel B  per-helix helicity vs Q_core (raw 2D density, biased run)
  Panel C  reweighted F(bend angle) and F(d_e2e): free DNA vs WT-bound
           (reweight via analysis/scripts/reweight_angle.py + analysis/fes_biased_*.dat)

NOTE: this is the earlier, single-file assembler. The published panels are produced
by figure5_data.py (data products) + figure5_plot.py, driven from
analysis/notebooks/analysis_figure5.ipynb.

Run from project root:  python analysis/scripts/make_figure5.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "analysis/scripts")
import reweight_angle as rw

T = 298.0
ROOT = os.getcwd()
FIG_OUT = "figures/1Nhp6A_fig5_apo_unfolding_dna_bending_v1.png"

# ── Panel A: apo-large 2D FES ───────────────────────────────────────────────
APO_DIR  = "data/1Nhp6A_apo_large"
APO_FES  = "analysis/1Nhp6A_apo_large_fes_sumhills.dat"
Q_MIN, Q_MAX, Q_BIN       = 0.3, 1.0, 100
NTC_MIN, NTC_MAX, NTC_BIN = 0, 150, 300
NWALK = 6


def load_apo_colvar(r):
    f = f"{APO_DIR}/replica_{r}/colvar_r{r}.dat"
    d = np.genfromtxt(f, comments="#", invalid_raise=False, filling_values=np.nan)
    d = d[np.isfinite(d[:, 0])]
    return dict(q_core=d[:, 1], n_tail_core=d[:, 2],
                helix1=d[:, 4], helix2=d[:, 5], helix3=d[:, 6])


def load_apo_fes2d(fname):
    d = np.genfromtxt(fname, comments="#", usecols=(0, 1, 2),
                      invalid_raise=False, filling_values=np.nan)
    d = d[np.isfinite(d[:, 0])]
    q = np.unique(d[:, 0]); ntc = np.unique(d[:, 1])
    fes = d[:, 2].reshape(len(ntc), len(q))
    return q, ntc, fes - np.nanmin(fes)


def sampling_mask(q, ntc, all_q, all_ntc, min_visits=5):
    dq = q[1] - q[0]; dn = ntc[1] - ntc[0]
    eq = np.append(q - dq / 2, q[-1] + dq / 2)
    en = np.append(ntc - dn / 2, ntc[-1] + dn / 2)
    counts, _, _ = np.histogram2d(all_ntc, all_q, bins=[en, eq])
    return counts >= min_visits


apo = [load_apo_colvar(r) for r in range(NWALK)]
all_q   = np.concatenate([c["q_core"] for c in apo])
all_ntc = np.concatenate([c["n_tail_core"] for c in apo])

if not os.path.exists(APO_FES):
    hills = ",".join(f"{APO_DIR}/hills.dat.{i}" for i in range(NWALK))
    rw.run_sumhills([f"{APO_DIR}/hills.dat.{i}" for i in range(NWALK)], APO_FES,
                    [Q_MIN, NTC_MIN], [Q_MAX, NTC_MAX], [Q_BIN, NTC_BIN])
q, ntc, fes2d_raw = load_apo_fes2d(APO_FES)
mask = sampling_mask(q, ntc, all_q, all_ntc, min_visits=5)
fes2d = fes2d_raw.copy(); fes2d[~mask] = np.nan
print(f"[A] apo FES min at Q={q[np.unravel_index(np.nanargmin(fes2d), fes2d.shape)[1]]:.3f}")

# ── Panel C: reweighted bending / end-to-end, free DNA vs WT-bound ───────────
DE2E_RANGE  = (2.4, 6.0)     # nm
THETA_RANGE = (90, 180)      # deg (180 = straight)
COORD_RANGE = (0, 40)

SYS_C = {
    "DNA only": dict(kind="dna",
        colvar=["data/DNA/DNA_parambsc1_tip3p_ions0.15_cmap_meta_e2e_colvar.dat"],
        hills=["data/DNA/DNA_parambsc1_tip3p_ions0.15_cmap_meta_e2e_hills.dat"],
        cols={"de2e": 1, "theta": 2, "cmap": 3}, biased=["de2e"], gamma_col=4,
        color="0.45", ls="--"),
    "WT bound (Nhp6A)": dict(kind="bound",
        colvar=[f"data/1Nhp6A/replica_{i}/colvar_r{i}.dat" for i in range(6)],
        hills=[f"data/1Nhp6A/hills.dat.{i}" for i in range(6)],
        cols={"coord_tail_dna": 1, "de2e": 2, "theta": 3, "cmap": 4},
        biased=["coord_tail_dna", "de2e"], gamma_col=6,
        color="#c44e52", ls="-"),
}


def reweight_system(cfg):
    ndim = len(cfg["biased"])
    gamma = rw.hills_gamma(cfg["hills"][0], cfg["gamma_col"])
    name = "1Nhp6A" if cfg["kind"] == "bound" else "DNA"
    fes = f"analysis/fes_biased_{name}.dat"
    if not os.path.exists(fes):
        sh = dict(dna=([0], [8], [200]),
                  bound=([0, 0], [150, 8], [150, 200]))[cfg["kind"]]
        rw.run_sumhills(cfg["hills"], fes, *sh)
    acc = {k: [] for k in cfg["cols"]}
    for c in cfg["colvar"]:
        d = rw.load_colvar(c, cfg["cols"])
        for k in cfg["cols"]:
            acc[k].append(d[k])
    data = {k: np.concatenate(v) for k, v in acc.items()}
    if ndim == 1:
        pts = data["de2e"]
    else:
        pts = np.column_stack([data["coord_tail_dna"], data["de2e"]])
    F = rw.fes_at_points(fes, pts, ndim)
    w = rw.frame_weights(F, gamma, T)
    ess = w.sum() ** 2 / np.sum(w ** 2)
    return dict(theta_deg=np.degrees(data["theta"]), de2e=data["de2e"],
                coord=data.get("coord_tail_dna"),
                w=w, gamma=gamma, ess=ess, n=len(w))


RC = {}
for name, cfg in SYS_C.items():
    print(f"[C] reweighting {name} ...", flush=True)
    RC[name] = reweight_system(cfg)
    r = RC[name]
    print(f"    gamma={r['gamma']:.0f} frames={r['n']:,} "
          f"ESS={100*r['ess']/r['n']:.0f}%")

# ── Panel D: reweighted 2D F(IDR-DNA coordination, bend angle), WT-bound ─────
rb = RC["WT bound (Nhp6A)"]
BEND_RANGE = (0, 90)            # deg (= 180 - theta)
COORD_D_RANGE = (0, 40)        # n_tail-DNA contacts
bend_b = 180.0 - rb["theta_deg"]
cx, by, Fcb = rw.fes_2d(rb["coord"], bend_b, rb["w"],
                        COORD_D_RANGE, BEND_RANGE, [50, 50], T)

# ════════════════════════════════════════════════════════════════════════════
# Layout: apo (A,B) on top, holo (C,D) on bottom.
plt.rcParams.update({"font.size": 11, "axes.grid": False})
mosaic = [["A",  "A",  "B1", "B1"],
          ["A",  "A",  "B3", "B3"],
          ["C1", "C1", "D",  "D"],
          ["C2", "C2", "D",  "D"]]
fig, ax = plt.subplot_mosaic(mosaic, figsize=(12.5, 9.0),
                             gridspec_kw=dict(wspace=0.5, hspace=0.55))

# ---- Panel A: apo 2D FES -------------------------------------------------
vmax = 40
im = ax["A"].contourf(q, ntc, np.ma.masked_invalid(fes2d),
                      levels=np.linspace(0, vmax, 21), cmap="viridis", extend="max")
ax["A"].contour(q, ntc, np.ma.masked_invalid(fes2d), levels=np.arange(5, vmax, 5),
                colors="k", linewidths=0.4, alpha=0.4)
cb = fig.colorbar(im, ax=ax["A"], pad=0.02)
cb.set_label(r"$\Delta F$ (kJ/mol)")
ax["A"].set_xlabel(r"$Q_\mathrm{core}$")
ax["A"].set_ylabel(r"$n_\mathrm{IDR-core}$")
ax["A"].set_xlim(0.7, 1.0); ax["A"].set_ylim(0, 60)
ax["A"].set_title("apo: core unstructuring", fontsize=11)

# ---- Panel B: helix 1 & helix 3 helicity vs Q_core -----------------------
hlabs = {"B1": ("helix1", r"$S_\alpha$ helix 1"),
         "B3": ("helix3", r"$S_\alpha$ helix 3")}
for key, (col, lab) in hlabs.items():
    hv = np.concatenate([c[col] for c in apo])
    ax[key].hexbin(all_q, hv, gridsize=40, cmap="viridis", mincnt=1, bins="log")
    ax[key].set_ylabel(lab, fontsize=10)
    ax[key].set_xlim(0.7, 1.0)
ax["B1"].set_xticklabels([])
ax["B3"].set_xlabel(r"$Q_\mathrm{core}$")
ax["B1"].set_title("apo: helicity vs core order", fontsize=11)

# ---- Panel C: bending + end-to-end, free DNA vs WT-bound -----------------
for name, r in RC.items():
    cfg = SYS_C[name]
    c_t, F_t = rw.fes_1d(r["theta_deg"], r["w"], THETA_RANGE, 60, T)
    bend = 180.0 - c_t
    m = np.isfinite(F_t)
    ax["C1"].plot(bend[m], F_t[m], cfg["ls"], lw=2, color=cfg["color"], label=name)
    c_d, F_d = rw.fes_1d(r["de2e"], r["w"], DE2E_RANGE, 60, T)
    md = np.isfinite(F_d)
    ax["C2"].plot(c_d[md], F_d[md], cfg["ls"], lw=2, color=cfg["color"], label=name)

ax["C1"].set_xlabel(r"DNA bend angle (deg)")
ax["C1"].set_ylabel(r"$F$ (kJ/mol)")
ax["C1"].set_xlim(0, 90); ax["C1"].set_ylim(0, 25)
ax["C1"].legend(fontsize=8, frameon=False, loc="upper right")
ax["C1"].set_title("holo: DNA bending & end-to-end", fontsize=11)
ax["C2"].set_xlabel(r"$d_\mathrm{e-e}$ (nm)")
ax["C2"].set_ylabel(r"$F$ (kJ/mol)")
ax["C2"].set_ylim(0, 25)

# ---- Panel D: 2D F(IDR-DNA coordination, bend angle), WT-bound -----------
vmaxD = 30
Fm = np.ma.masked_invalid(Fcb.T)
pc = ax["D"].contourf(cx, by, Fm, levels=np.linspace(0, vmaxD, 16),
                      cmap="viridis", extend="max")
ax["D"].contour(cx, by, Fm, levels=np.arange(4, vmaxD, 4), colors="k",
                linewidths=0.4, alpha=0.4)
cbd = fig.colorbar(pc, ax=ax["D"], pad=0.02)
cbd.set_label(r"$F$ (kJ/mol)")
ax["D"].set_xlabel(r"$n_\mathrm{IDR-DNA}$")
ax["D"].set_ylabel(r"DNA bend angle (deg)")
ax["D"].set_title("holo: bending requires IDR–DNA contacts", fontsize=11)

# panel letters
for key, xy in [("A", (-0.14, 1.02)), ("B1", (-0.16, 1.06)),
                ("C1", (-0.14, 1.06)), ("D", (-0.16, 1.02))]:
    ax[key].text(*xy, "(" + key[0] + ")", transform=ax[key].transAxes,
                 fontsize=15, fontweight="bold", va="bottom", ha="right")

os.makedirs("figures", exist_ok=True)
fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
print("saved", FIG_OUT)
