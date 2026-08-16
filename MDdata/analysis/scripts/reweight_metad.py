"""
Rigorous time-dependent (Tiwary-Parrinello c(t)) reweighting of a spectator
observable from well-tempered metadynamics, computed fully locally from the
HILLS + colvar files (no trajectories, no `plumed driver`).

This is the estimator implemented by PLUMED's REWEIGHT_METAD with CALC_RCT.
Unlike the final-bias (Branduardi 2012) approximation in `reweight_angle.py`,
it weights every frame -- including early, non-converged ones -- with the
instantaneous bias minus the time-dependent offset c(t):

    w_i = exp( beta * ( V(s_i, t_i) - c(t_i) ) )

with the WT reweighting offset

    c(t) = (1/beta) * ln[ sum_grid exp( +beta * gamma/(gamma-1) * V(s,t) )
                          / sum_grid exp( +beta * 1/(gamma-1) * V(s,t) ) ].

V(s,t) is the metadynamics bias on the (CV1, CV2) grid, accumulated in
deposition-time order by pooling ALL walkers' gaussians (the multiwalker bias
is shared on a common simulation-time axis).  Each HILLS `height` column is
already the WT-rescaled deposited height, so

    V(s) = sum_k height_k * exp( -sum_d (s_d - s0_{d,k})^2 / (2 sigma_{d,k}^2) ).

Approximation (standard, stated explicitly): PLUMED deposits *stretched/
truncated* gaussians; here we deposit plain (truncated at `trunc`*sigma)
gaussians on the grid.  The difference is negligible for reweighting and the
result is validated by reproducing the sum_hills marginals (see the notebook's
validation cell).

Units: energies in kJ/mol; pass the same kBT used for sum_hills so the
validation comparison is internally consistent.
"""
import numpy as np
from scipy.special import logsumexp


def pool_hills(hills_list):
    """Pool a list of per-walker HILLS arrays into one time-sorted set.

    Each array has PLUMED columns: time q n sigma_q sigma_n height biasf clock.
    Returns (t, q0, n0, height) arrays sorted by deposition time, plus the
    (median) sigmas and the bias factor gamma read from the data.
    """
    H = np.vstack([h[:, :7] for h in hills_list])
    order = np.argsort(H[:, 0], kind="mergesort")
    H = H[order]
    sigma_q = float(np.median(H[:, 3]))
    sigma_n = float(np.median(H[:, 4]))
    gamma = float(np.median(H[:, 6]))
    return H[:, 0], H[:, 1], H[:, 2], H[:, 5], sigma_q, sigma_n, gamma


def metad_ct_weights(t_h, q0, n0, height, sigma_q, sigma_n,
                     frame_t, frame_q, frame_n,
                     q_grid, n_grid, gamma=10.0, kBT=2.494339,
                     checkpoint_dt=1.0, batch_dt=1.0, trunc=4.0):
    """Per-frame WT-MetaD c(t) weights on a 2D (q, n) grid.

    The bias grid is accumulated in deposition-time order.  Frames are processed
    in time order and each frame's instantaneous bias V(s_i,t_i) is read from the
    grid as it stands AT (just below) the frame's own time -- NOT a stale future
    grid.  This is essential: early in the run the bias near the starting basin
    grows by tens of kJ/mol within the first nanosecond, so using an end-of-window
    grid would massively over-weight the first frames.  Frames are evaluated in
    small `batch_dt` windows (default 1 ps) so the residual staleness is at most
    one PACE-block of deposition; c(t) is recomputed on the coarser `checkpoint_dt`
    and linearly interpolated to each frame time.

    Parameters
    ----------
    t_h, q0, n0, height : 1D arrays, pooled hills sorted by time (see pool_hills).
    sigma_q, sigma_n    : gaussian widths (constant).
    frame_t, frame_q, frame_n : 1D arrays, colvar frames (any order; results are
                          returned in this same input order).
    q_grid, n_grid      : 1D grid axes for the bias (e.g. the sum_hills grid).
    gamma, kBT          : WT bias factor and thermal energy (kJ/mol).
    checkpoint_dt       : time (ps) between c(t) re-evaluations (c is smooth).
    batch_dt            : time (ps) window for vectorised frame-bias evaluation.
    trunc               : gaussian truncation radius in units of sigma.

    Returns
    -------
    dict with keys:
      weights : per-frame weights (input order), normalised so max(w)=1.
      rbias   : per-frame V - c(t)   (kJ/mol).
      Vframe  : per-frame instantaneous bias V(s_i,t_i) (kJ/mol).
      ct_t, ct: checkpoint times and c(t) values (kJ/mol).
      ess     : effective sample size (sum w)^2 / sum(w^2).
    """
    beta = 1.0 / kBT
    nq, nn = q_grid.size, n_grid.size
    dq = q_grid[1] - q_grid[0]
    dn = n_grid[1] - n_grid[0]
    wq = int(np.ceil(trunc * sigma_q / dq))   # truncation half-windows (bins)
    wn = int(np.ceil(trunc * sigma_n / dn))

    V = np.zeros((nn, nq))  # bias grid, indexed [n, q] (matches notebook fes2d)

    # ---- precompute bilinear-interp indices/fractions for every frame -------
    fq = np.clip(frame_q, q_grid[0], q_grid[-1])
    fn = np.clip(frame_n, n_grid[0], n_grid[-1])
    iq = np.clip(((fq - q_grid[0]) / dq).astype(int), 0, nq - 2)
    inn = np.clip(((fn - n_grid[0]) / dn).astype(int), 0, nn - 2)
    tq = (fq - q_grid[iq]) / dq
    tn = (fn - n_grid[inn]) / dn

    fo = np.argsort(frame_t, kind="mergesort")   # process frames in time order
    ft_s = frame_t[fo]
    Vframe_s = np.empty(frame_t.size)
    rbias_s = np.empty(frame_t.size)
    nh, nf = t_h.size, frame_t.size

    def add_hill(k):
        qc, nc, hc = q0[k], n0[k], height[k]
        jq = int(round((qc - q_grid[0]) / dq))
        jn = int(round((nc - n_grid[0]) / dn))
        q_lo, q_hi = max(jq - wq, 0), min(jq + wq + 1, nq)
        n_lo, n_hi = max(jn - wn, 0), min(jn + wn + 1, nn)
        if q_lo >= q_hi or n_lo >= n_hi:
            return
        gq = np.exp(-0.5 * ((q_grid[q_lo:q_hi] - qc) / sigma_q) ** 2)
        gn = np.exp(-0.5 * ((n_grid[n_lo:n_hi] - nc) / sigma_n) ** 2)
        V[n_lo:n_hi, q_lo:q_hi] += hc * np.outer(gn, gq)

    def eval_c():
        # c(t) on the current grid (log-sum-exp; constant ds cancels in ratio)
        return (logsumexp(beta * (gamma / (gamma - 1.0)) * V)
                - logsumexp(beta * (1.0 / (gamma - 1.0)) * V)) / beta

    def eval_frames(lo, hi):
        # vectorised bilinear read of current grid for sorted frames [lo:hi)
        s = fo[lo:hi]
        a00 = V[inn[s], iq[s]];      a10 = V[inn[s], iq[s] + 1]
        a01 = V[inn[s] + 1, iq[s]];  a11 = V[inn[s] + 1, iq[s] + 1]
        tqs, tns = tq[s], tn[s]
        Vframe_s[lo:hi] = ((1 - tqs) * (1 - tns) * a00 + tqs * (1 - tns) * a10
                           + (1 - tqs) * tns * a01 + tqs * tns * a11)

    t_end = max(t_h[-1], ft_s[-1])
    ct_t, ct_vals = [], []
    next_cp = -1.0                         # force a c() update on the first batch
    c_cur = 0.0                            # current c(t); grid empty -> 0
    ih = 0                                 # hill pointer
    iff = 0                               # frame pointer
    edges = np.arange(batch_dt, t_end + batch_dt, batch_dt)
    for tb in edges:
        # accumulate every hill deposited up to this batch edge
        added = False
        while ih < nh and t_h[ih] <= tb:
            add_hill(ih); ih += 1; added = True
        # refresh c(t) on the (coarser) checkpoint schedule, from the SAME grid
        if added and tb >= next_cp:
            c_cur = eval_c()
            ct_t.append(tb); ct_vals.append(c_cur)
            next_cp = tb + checkpoint_dt
        # all frames before this edge: read V from this grid, subtract this c
        j0 = iff
        while iff < nf and ft_s[iff] < tb:
            iff += 1
        if iff > j0:
            eval_frames(j0, iff)
            rbias_s[j0:iff] = Vframe_s[j0:iff] - c_cur
    if iff < nf:                           # trailing frames at/after last edge
        eval_frames(iff, nf)
        rbias_s[iff:nf] = Vframe_s[iff:nf] - c_cur

    Vframe = np.empty(nf); rbias = np.empty(nf)
    Vframe[fo] = Vframe_s
    rbias[fo] = rbias_s
    ct_t = np.asarray(ct_t); ct_vals = np.asarray(ct_vals)
    w = np.exp((rbias - rbias.max()) / kBT)
    ess = w.sum() ** 2 / np.sum(w ** 2)
    return {"weights": w, "rbias": rbias, "Vframe": Vframe,
            "ct_t": ct_t, "ct": ct_vals, "ess": ess}
