"""Pneumotach clock CHECK (vendored 2026-10-27 from the PCA session's helper
2026_PCA_registration/workspace/helpers/scripts/_pneumo_clock_fit.py, author: 2026-pca-registration-04 Claude session,
2026-10-26). In the fork the pneumotach clock is MAPPED onto the gas interleaves from the MDH timestamps
(convert_siemens_to_mrd + raw.load_pneumotach_from_arrays); this fit only verifies the mapping: on a correctly
mapped trace a ~ 1, b ~ 0 and >= 60 % of the k0 EE match within 0.3 s. Original docstring follows.
Pneumotach clock alignment to the gas-line clock (PCA session, 2026-10-26; for vendoring into ASAP raw.py with attribution).

The converter zeroes the pneumotach clock at the file's first packet and assumes that is gas interleaf 0. On every session tested
the true relation is t_gas = a * t_pn + b with a ~ 0.989-0.991 and b ~ -26 ... -32 s (pneumotach recording starts ~30 s before the
scan; its sample clock runs ~1 % fast relative to ilvtime = iilv * TR). Fit: end-expiration minima of the pneumotach VOLUME trace are
matched to the SIGNAL (k0) EE list by a coarse grid over (a, b), then a linear least-squares refit on the matched pairs.

    a, b, info = fit_pneumo_clock(pt, pv, ee)
    t_gas = a * pt + b            # apply BEFORE cropping to the acquisition window

pt, pv : pneumotach time (s, first-packet clock) and volume (any monotone units; drift-free enough that minima are breaths)
ee     : SIGNAL end-expiration times on the gas clock (s), as produced by the SIGNAL branch (strict k0 minima)
Returns a, b and info = dict(n_matched, n_pn_minima, n_ee, resid_sd_s, ok). ok = False (and a=1, b=0) when fewer than 60 % of the EE
match within `tol` at the best grid point -> caller should warn loudly and fall back.
Numbers on the four core 2024 sessions (grid a 0.96-1.04 step 0.002, b +-90 s step 0.1 s, tol 0.3 s): 015HK 0.9902/-32.3 (29/29
matched, resid 0.08 s), 014KS 0.9884/-31.7 (23/28, 0.10), 016PG 0.9893/-31.7 (23/24, 0.12), 039CP 0.9913/-26.3 (29/29, 0.09);
rank-binned k0 B/A as coded 1.72/1.60/1.36/2.18 -> aligned 3.98/5.12/4.02/3.86 (= diaphragm rank 3.93/3.70/4.92/4.11).
"""
import numpy as np
from scipy import ndimage as ndi


def pneumo_ee_minima(pt, pv, min_sep_s=1.5, half_window_s=0.8, smooth_n=9):
    """Strict local minima of the (lightly smoothed) volume trace, one per breath (minima closer than min_sep_s collapse)."""
    dt = float(np.median(np.diff(pt))); k = max(int(half_window_s / dt), 1)
    vs = ndi.uniform_filter1d(np.asarray(pv, float), smooth_n)
    idx = np.array([i for i in range(k, vs.size - k) if vs[i] == vs[i - k:i + k + 1].min()], int)
    if idx.size == 0:
        return np.array([])
    tm = pt[idx]
    return tm[np.concatenate([[True], np.diff(tm) > min_sep_s])]


def fit_pneumo_clock(pt, pv, ee, a_range=(0.96, 1.04, 0.002), b_range=(-90.0, 90.0, 0.1), tol_s=0.3, min_frac=0.6):
    pt, pv, ee = np.asarray(pt, float), np.asarray(pv, float), np.sort(np.asarray(ee, float))
    pm = pneumo_ee_minima(pt, pv)
    info = dict(n_matched=0, n_pn_minima=int(pm.size), n_ee=int(ee.size), resid_sd_s=np.nan, ok=False)
    if pm.size < 3 or ee.size < 3:
        return 1.0, 0.0, info
    best = (-1, 1.0, 0.0)
    for a in np.arange(a_range[0], a_range[1] + 1e-9, a_range[2]):
        bs = np.arange(b_range[0], b_range[1] + 1e-9, b_range[2])
        d = np.abs((a * pm)[:, None, None] + bs[None, None, :] - ee[None, :, None]).min(1)  # (n_pm, n_b)
        n = (d < tol_s).sum(0); j = int(np.argmax(n))
        if n[j] > best[0]:
            best = (int(n[j]), float(a), float(b_range[0] + j * b_range[2]))
    n, a, b = best
    tt = a * pm + b; j = np.argmin(np.abs(tt[:, None] - ee[None, :]), 1); g = np.abs(tt - ee[j]) < tol_s
    if g.sum() < max(3, min_frac * ee.size):
        info['n_matched'] = int(g.sum()); return 1.0, 0.0, info
    A = np.polyfit(pm[g], ee[j][g], 1)
    res = ee[j][g] - np.polyval(A, pm[g])
    info.update(n_matched=int(g.sum()), resid_sd_s=float(res.std()), ok=True)
    return float(A[0]), float(A[1]), info

