#!/usr/bin/env python
"""specfit — standalone dissolved-phase 129Xe spectral fit for the cal block (numpy + scipy only).

Written 2026-09-24 (XeCS session, Hooman's go) so that the ASAP RBC/TP re-split CSV and the vendored copy
next to Steve's raw.py (Tyger image) run IDENTICAL code. No Ext cache, no XeCS imports.

Model (time domain, complex, Robertson 2017 / Bier 2019):
    S(t) = sum_n a_n exp(i phi_n) exp(+2 pi i f_n t) exp(-pi wL_n t) [exp(-(pi wG_n t)^2 / (4 ln 2))]
    t = (first_sample + k) * dwell for array index k  -> t = 0 is ADC SAMPLE 0 = TE, whatever was trimmed.
    ADC samples 0 AND 1 are excluded from the residual (drop_before=2): sample 0 is a 3x dip, sample 1 overshoots
    the log-linear decay by ~10 % (ASAP xcheck 045VS + XeCS cache-wide check 2026-09-24); with sample 1 kept the M3
    mem1/mem2 split (F_lump +-15 %) and dphi_k0 (+-6 deg) move. Steve's raw.py drops both (killpts = 2) -> identical.
    RBC = the higher-frequency dissolved line; dphi = phi_RBC - phi_mem, wrapped to (-180, 180].
Ladder: M2 = gas L / RBC L / membrane Voigt (13 p);  M3 = gas L / RBC L / mem1 L + mem2 L (16 p).
Prior (ppm from the FITTED gas line; the chemical-shift prior is NOT optional — without it the fit collapses on
low-RBC blocks): RBC 218 +- 4, FWHM >= 6 ppm; M2 mem 197.5 +- 4, >= 6 ppm; M3 mem1 201.5 +- 3.5 (198-205), >= 8 ppm (broad,
Robertson B1) and mem2 195.75 +- 2.25 (193.5-198), 4-14 ppm (narrow, B2); windows do NOT overlap (overlapping windows let both lines meet at ~196.5 ppm anti-phase, a cancelling local minimum seen 2026-09-24). Gas +- 2 ppm around the located line, FWHM <= 6 ppm.
Solver: scipy least_squares, method 'trf', bounded, x_scale 'jac', residual = [Re, Im], data scaled by max|FID|.

Gas-line location without a cache: candidate peaks of the magnitude spectrum; keep those with a dissolved
partner 185-235 ppm above them; choose the NARROWEST strong one (gas T2* >= 10 ms -> ~1-2 bins; dissolved
~9-12 ppm). Fallback: the strongest peak 150-260 ppm below the carrier.

Lumped membrane at the image k0 time (t_k0 = TE + killpts*dt_img, i.e. offset 10 us for killpts 2, dt 5 us):
    m(t_k0) = a1 e^{i(phi1 + 2 pi f1 t_k0)} + a2 e^{i(phi2 + 2 pi f2 t_k0)}
    dphi_k0 = angle(RBC(t_k0) / m(t_k0));  ratio_lumped = a_RBC / |m|;  ratio_scalar = a_RBC / (a1 + a2);
    F_lump = (a1 + a2) / |m|  (a two-component k0 split returns a_RBC and |m| by construction -> lumped ratio;
    divide a map's RBC/TP by F_lump to put it on the physiological scalar scale).
model_used = M3 unless degenerate (a1/a2 outside [0.3, 3], |f1 - f2| < 2 ppm, or per-rep SD(M3) > 2 SD(M2) + 5) -> M2.

API
    fit_block(fid, dwell, te_s, first_sample=0, t_k0_offset_s=10e-6, hz_per_ppm=17.61, reps=None, gas_hz=None)
        fid  : complex (N,) or (N, nlines) -> averaged over lines (the pooled FID)
        reps : optional list of (N,) or (N, nlines) arrays (one per rep) for the per-rep phase SD
        -> dict of the CSV fields (dphi_k0_deg, ratio_lumped, ..., valid, reason) + 'fits' (raw)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import median_filter

LN2 = np.log(2.0)
MODELS = {
    'M2': [('gas', 'L'), ('rbc', 'L'), ('mem', 'V')],
    'M3': [('gas', 'L'), ('rbc', 'L'), ('mem1', 'L'), ('mem2', 'L')],
}
NPAR = {'L': 4, 'V': 5}
# name -> (centre ppm from gas, half-window ppm, min FWHM ppm, max FWHM ppm)
PRIOR = {'rbc': (218.0, 4.0, 6.0, 25.0), 'mem': (197.5, 4.0, 6.0, 25.0),
         'mem1': (201.5, 3.5, 8.0, 25.0), 'mem2': (195.75, 2.25, 4.0, 14.0)}   # non-overlapping windows: 198-205 / 193.5-198
GAS_WIN_PPM, GAS_WMAX_PPM = 2.0, 6.0


def wrap(deg):
    return float(np.degrees(np.angle(np.exp(1j * np.radians(deg)))))


def circ_sd(deg):
    z = np.mean(np.exp(1j * np.radians(np.asarray(deg, float))))
    return float(np.degrees(np.sqrt(-2 * np.log(max(abs(z), 1e-9)))))


# ----------------------------------------------------------------------------- model
def _unpack(theta, spec):
    out, k = [], 0
    for name, kind in spec:
        n = NPAR[kind]
        out.append((name, kind, theta[k:k + n]))
        k += n
    return out


def model_fid(theta, t, spec):
    s = np.zeros(t.size, complex)
    for name, kind, p in _unpack(theta, spec):
        a, phi, f, wL = p[:4]
        env = np.exp(-np.pi * wL * t)
        if kind == 'V':
            env = env * np.exp(-(np.pi * p[4] * t) ** 2 / (4 * LN2))
        s += a * np.exp(1j * phi) * np.exp(2j * np.pi * f * t) * env
    return s


def _resid(theta, t, y, spec):
    r = model_fid(theta, t, spec) - y
    return np.concatenate([r.real, r.imag])


# ----------------------------------------------------------------------------- gas line
def locate_gas(y, dwell, hz_per_ppm, t=None):
    """Gas-line frequency (Hz from carrier) from the magnitude spectrum of one complex FID."""
    N = y.size
    S = np.abs(np.fft.fftshift(np.fft.fft(y * np.hanning(N) * 2)))
    freq = np.fft.fftshift(np.fft.fftfreq(N, dwell))
    df = freq[1] - freq[0]
    base = median_filter(S, 15, mode='nearest')
    noise = np.median(np.abs(S - base)) * 1.4826 + 1e-12
    cand = [k for k in range(2, N - 2) if S[k] > S[k - 1] and S[k] >= S[k + 1] and S[k] > base[k] + 8 * noise]
    best, best_w = None, np.inf
    for k in cand:
        c = freq[k]
        lo, hi = c + 185 * hz_per_ppm, c + 235 * hz_per_ppm
        sel = (freq > lo) & (freq < hi)
        if not sel.any() or (S[sel] - base[sel]).max() < 5 * noise:
            continue
        half = S[k] / 2
        i0 = k
        while i0 > 0 and S[i0] > half:
            i0 -= 1
        i1 = k
        while i1 < N - 1 and S[i1] > half:
            i1 += 1
        w = (i1 - i0) * df / hz_per_ppm
        if w < best_w and w < 8.0:
            best, best_w = c, w
    if best is None:
        sel = (freq > -260 * hz_per_ppm) & (freq < -150 * hz_per_ppm)
        best = float(freq[sel][np.argmax(S[sel])]) if sel.any() else float(freq[np.argmax(S)])
    return float(best)


# ----------------------------------------------------------------------------- init + fit
def init_guess(y, t, dwell, hz_per_ppm, spec, gas_hz):
    N = y.size
    S = np.abs(np.fft.fftshift(np.fft.fft(y)))
    freq = np.fft.fftshift(np.fft.fftfreq(N, dwell))
    base = median_filter(S, 7, mode='nearest')
    amp_at = lambda f: float(base[np.argmin(np.abs(freq - f))])
    w_gas = 25.0
    theta, lo, hi = [], [], []
    for name, kind in spec:
        if name == 'gas':
            f, w = gas_hz, w_gas
            a = float(S[np.argmin(np.abs(freq - f))]) * np.pi * w / N
            flo, fhi, wmin, wmax = f - GAS_WIN_PPM * hz_per_ppm, f + GAS_WIN_PPM * hz_per_ppm, 2.0, GAS_WMAX_PPM * hz_per_ppm
        else:
            c, win, wmin_ppm, wmax_ppm = PRIOR[name]
            f = gas_hz + c * hz_per_ppm
            w = max(12.0 * hz_per_ppm, 1.2 * wmin_ppm * hz_per_ppm)
            a = amp_at(f) * np.pi * w / N * (0.6 if name == 'mem1' else 0.5 if name == 'mem2' else 1.0)
            flo, fhi, wmin, wmax = f - win * hz_per_ppm, f + win * hz_per_ppm, wmin_ppm * hz_per_ppm, wmax_ppm * hz_per_ppm
        # phase init: project the data on the component frequency
        phi = float(np.angle(np.sum(y * np.exp(-2j * np.pi * f * t))))
        theta += [a, phi, f, w]
        lo += [0.0, -2 * np.pi, flo, wmin]
        hi += [np.inf, 2 * np.pi, fhi, wmax]
        if kind == 'V':
            theta += [0.5 * w]
            lo += [0.0]
            hi += [wmax]
    return np.array(theta), (np.array(lo), np.array(hi))


def fit_fid(y, t, spec, theta0, bounds):
    scale = float(np.abs(y).max()) or 1.0
    th0 = theta0.copy()
    idx_a, k = [], 0
    for _, kind in spec:
        idx_a.append(k)
        k += NPAR[kind]
    th0[idx_a] = th0[idx_a] / scale
    r = least_squares(_resid, th0, bounds=bounds, args=(t, y / scale, spec), method='trf', x_scale='jac', max_nfev=4000)
    theta = r.x.copy()
    theta[idx_a] = theta[idx_a] * scale
    res = _resid(theta, t, y, spec)
    n_obs = 2 * y.size
    rss = float(np.sum(res ** 2))
    return dict(theta=theta, rss=rss, bic=n_obs * np.log(rss / n_obs) + theta.size * np.log(n_obs), nfev=r.nfev,
                status=r.status, spec=spec)


def params_table(theta, spec, hz_per_ppm, te_s=None):
    comps = _unpack(theta, spec)
    gas_f = [p[2] for n, k, p in comps if n == 'gas'][0]
    rows = {}
    for name, kind, p in comps:
        a, phi, f, wL = p[:4]
        wG = p[4] if kind == 'V' else 0.0
        fwhm = 0.5346 * wL + np.sqrt(0.2166 * wL ** 2 + wG ** 2) if kind == 'V' else wL
        row = dict(a=float(a), phi_deg=float(np.degrees(phi)), f_hz=float(f), ppm_from_gas=float((f - gas_f) / hz_per_ppm),
                   fwhm_hz=float(fwhm), fwhm_ppm=float(fwhm / hz_per_ppm), T2s_ms=float(1e3 / (np.pi * fwhm)) if fwhm > 0 else np.nan)
        if te_s is not None:
            row['a_te'] = float(a * np.exp(te_s * np.pi * fwhm))
        rows[name] = row
    return rows


def lumped(tb, model, t_k0):
    """RBC vs lumped-membrane phasors at absolute time t_k0 (s after sample 0 = TE)."""
    ph = lambda r: r['a'] * np.exp(1j * (np.radians(r['phi_deg']) + 2 * np.pi * r['f_hz'] * t_k0))
    rbc = ph(tb['rbc'])
    if model == 'M3':
        m = ph(tb['mem1']) + ph(tb['mem2'])
        a_sum = tb['mem1']['a'] + tb['mem2']['a']
        fc = (tb['mem1']['a'] * tb['mem1']['f_hz'] + tb['mem2']['a'] * tb['mem2']['f_hz']) / a_sum
        a_sum_te = tb['mem1'].get('a_te', np.nan) + tb['mem2'].get('a_te', np.nan)
    else:
        m = ph(tb['mem'])
        a_sum, fc, a_sum_te = tb['mem']['a'], tb['mem']['f_hz'], tb['mem'].get('a_te', np.nan)
    return dict(dphi=wrap(np.degrees(np.angle(rbc / m))), ratio_lumped=tb['rbc']['a'] / abs(m),
                ratio_scalar=tb['rbc']['a'] / a_sum, F_lump=a_sum / abs(m), df_hz=tb['rbc']['f_hz'] - fc,
                ratio_scalar_te=tb['rbc'].get('a_te', np.nan) / a_sum_te)


def dissolved_snr(y, t, hz_per_ppm, gas_hz):
    N = y.size
    S = np.abs(np.fft.fftshift(np.fft.fft(y)))
    freq = np.fft.fftshift(np.fft.fftfreq(N, t[1] - t[0]))
    diss = (freq > gas_hz + 185 * hz_per_ppm) & (freq < gas_hz + 235 * hz_per_ppm)
    noise_sel = (freq > gas_hz + 60 * hz_per_ppm) & (freq < gas_hz + 150 * hz_per_ppm)
    if not noise_sel.any():
        noise_sel = (freq < gas_hz - 60 * hz_per_ppm)
    return float(S[diss].max() / (np.std(S[noise_sel]) + 1e-12))


def _prep(fid, dwell, first_sample, drop_before=1):
    y = np.asarray(fid)
    if y.ndim == 2:
        y = y.mean(axis=1)
    idx = first_sample + np.arange(y.size)
    keep = idx >= drop_before
    return y[keep].astype(complex), idx[keep] * dwell


def fit_block(fid, dwell, te_s, first_sample=0, t_k0_offset_s=10e-6, hz_per_ppm=17.61, reps=None, gas_hz=None,
              drop_before=2):
    """See module docstring. Returns the CSV fields + 'fits'."""
    y, t = _prep(fid, dwell, first_sample, drop_before)
    if gas_hz is None:
        gas_hz = locate_gas(y, dwell, hz_per_ppm)
    snr = dissolved_snr(y, t, hz_per_ppm, gas_hz)
    res = {}
    for m in ('M2', 'M3'):
        th0, bnd = init_guess(y, t, dwell, hz_per_ppm, MODELS[m], gas_hz)
        f = fit_fid(y, t, MODELS[m], th0, bnd)
        tb = params_table(f['theta'], MODELS[m], hz_per_ppm, te_s=te_s)
        res[m] = dict(fit=f, tb=tb, lump=lumped(tb, m, t_k0_offset_s))
    sd = {'M2': np.nan, 'M3': np.nan}
    n_reps = 0
    if reps:
        for m in ('M2', 'M3'):
            d = []
            for rr in reps:
                yr, tr = _prep(rr, dwell, first_sample, drop_before)
                th0, bnd = init_guess(yr, tr, dwell, hz_per_ppm, MODELS[m], gas_hz)
                fr = fit_fid(yr, tr, MODELS[m], th0, bnd)
                d.append(lumped(params_table(fr['theta'], MODELS[m], hz_per_ppm), m, t_k0_offset_s)['dphi'])
            sd[m] = circ_sd(d) if len(d) >= 2 else np.nan
            n_reps = len(d)
    t3 = res['M3']['tb']
    a1, a2 = t3['mem1']['a'], t3['mem2']['a']
    ratio12 = a1 / a2 if a2 > 0 else np.inf
    sep12 = abs(t3['mem1']['ppm_from_gas'] - t3['mem2']['ppm_from_gas'])
    degenerate = (not (0.3 <= ratio12 <= 3.0)) or sep12 < 2.0 or \
        (np.isfinite(sd['M3']) and np.isfinite(sd['M2']) and sd['M3'] > 2 * sd['M2'] + 5)
    model = 'M2' if degenerate else 'M3'
    L, tb = res[model]['lump'], res[model]['tb']
    reasons = []
    if snr < 20:
        reasons.append('snr<20')
    if L['ratio_scalar'] < 0.05:
        reasons.append('rbc/mem<0.05')
    if np.isfinite(sd[model]) and sd[model] > 30:
        reasons.append('rep_sd>30')
    valid = 'no' if reasons else 'yes'
    out = dict(snr_diss=round(snr, 1), gas_hz_from_carrier=round(gas_hz, 1), gas_ppm_from_carrier=round(gas_hz / hz_per_ppm, 1),
               TE_us=round(te_s * 1e6, 1), t_k0_us=round((te_s + t_k0_offset_s) * 1e6, 1), model_used=model,
               dphi_k0_deg=round(L['dphi'], 1), dphi_k0_m3_lumped_deg=round(res['M3']['lump']['dphi'], 1),
               dphi_k0_m2_deg=round(res['M2']['lump']['dphi'], 1),
               dphi_rep_sd_deg=round(sd[model], 1), dphi_rep_sd_m3_deg=round(sd['M3'], 1), dphi_rep_sd_m2_deg=round(sd['M2'], 1),
               n_reps=n_reps,
               ratio_lumped=round(L['ratio_lumped'], 4), ratio_scalar=round(L['ratio_scalar'], 4), F_lump=round(L['F_lump'], 3),
               ratio_m2=round(res['M2']['lump']['ratio_scalar'], 4), ratio_scalar_te=round(L['ratio_scalar_te'], 4),
               df_hz=round(L['df_hz'], 1),
               rbc_ppm=round(tb['rbc']['ppm_from_gas'], 2), rbc_fwhm_ppm=round(tb['rbc']['fwhm_ppm'], 2),
               mem1_ppm=round(t3['mem1']['ppm_from_gas'], 2), mem2_ppm=round(t3['mem2']['ppm_from_gas'], 2),
               mem1_fwhm_ppm=round(t3['mem1']['fwhm_ppm'], 2), mem2_fwhm_ppm=round(t3['mem2']['fwhm_ppm'], 2),
               a1_over_a2=round(ratio12, 3), mem12_dphi_deg=round(wrap(t3['mem1']['phi_deg'] - t3['mem2']['phi_deg']), 1),
               mem_m2_ppm=round(res['M2']['tb']['mem']['ppm_from_gas'], 2), mem_m2_fwhm_ppm=round(res['M2']['tb']['mem']['fwhm_ppm'], 2),
               bic_m2=round(res['M2']['fit']['bic'], 1), bic_m3=round(res['M3']['fit']['bic'], 1),
               valid=valid, reason=';'.join(reasons))
    out['fits'] = res
    return out


CSV_FIELDS = ['snr_diss', 'gas_hz_from_carrier', 'gas_ppm_from_carrier', 'TE_us', 't_k0_us', 'model_used',
              'dphi_k0_deg', 'dphi_k0_m3_lumped_deg', 'dphi_k0_m2_deg', 'dphi_rep_sd_deg', 'dphi_rep_sd_m3_deg',
              'dphi_rep_sd_m2_deg', 'n_reps', 'ratio_lumped', 'ratio_scalar', 'F_lump', 'ratio_m2', 'ratio_scalar_te',
              'df_hz', 'rbc_ppm', 'rbc_fwhm_ppm', 'mem1_ppm', 'mem2_ppm', 'mem1_fwhm_ppm', 'mem2_fwhm_ppm',
              'a1_over_a2', 'mem12_dphi_deg', 'mem_m2_ppm', 'mem_m2_fwhm_ppm', 'bic_m2', 'bic_m3', 'valid', 'reason']
