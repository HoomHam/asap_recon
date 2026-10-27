"""
Loads mrd file populated with siemens .dat file acquisitions and npy trajectory fields
in NdArray field. This script file is used as entrypoint of docker container to run reconstruction on tyger
"""
from typing import BinaryIO
from sys import stderr
import argparse
import re

import matplotlib
matplotlib.use('Agg')  # headless subprocess: results.py calls plt.show()

import numpy as np
from scipy.signal import savgol_filter
from numba import cuda

import mrd

from gtypes import gvar, imgtype, bintype
from raw import traj, raw
from results import results


def _diaphragm_navigator(g, g_raw, g_traj, g_res):
    """Populate g_raw.ilvbin[DIAPHRAGM] via a low-res navigator loop.

    Ported from main.py:calcLVcb (Steve's GUI). Reconstructs each undersampled
    spiral interleave group as one low-res image, extracts the diaphragm z-position
    from the z-dropoff profile (parabola fit on the 25%-75% transition), smooths
    with Savgol, and bins. Caller is responsible for restoring g.MS and recomputing
    the full-res b-matrix afterward.
    """
    bt = bintype.DIAPHRAGM
    saveMS = g.MS
    g.MS = g.IS + 4                       # low-res navigator matrix
    print(f'DIAPHRAGM: computing low-res b (MS={g.MS})...', file=stderr)
    g_res.calcb(g, g_raw, g_traj, g.usegpu)
    # ---- navigator frame grouping: one complete Thomson set per frame ----
    # 26 = one complete Thomson set of the v3 trajectory (832 = 32x26); windows
    # must stay 26-aligned or the varying window PSF injects a period-26 wobble
    # into the navigator (measured 2026-07-17, XeCS
    # workspace/reference/Navigation_Registration_Queue_2026-07-17.md).
    # ilvperusimg is fixed by nusimg (MRD long param, default 32):
    #   ilvperusimg = nuniqueilvs / nusimg = 832 / 32 = 26.
    # killpts trims only the SAMPLE axis, so interleave block phase is preserved:
    # frames are consecutive [iusimg*26 : (iusimg+1)*26], starting at ilv 0.
    THOMSON_SET = 26
    ilvperusimg = int(g_traj.nsmpperusimg / g_raw.npts + .1)
    nusimg = int(g_raw.ntotalilvs * g_raw.npts / g_traj.nsmpperusimg)
    if ilvperusimg != THOMSON_SET:
        print(f'DIAPHRAGM WARNING: navigator window = {ilvperusimg} interleaves, '
              f'expected {THOMSON_SET} (one Thomson set). Set the MRD nusimg param so '
              f'nuniqueilvs/nusimg = {THOMSON_SET}; a misaligned window re-introduces the '
              f'period-{THOMSON_SET} PSF wobble in the navigator edge.', file=stderr)
    print(f'DIAPHRAGM: navigator = {nusimg} frames x {ilvperusimg} interleaves '
          f'(block-aligned from ilv 0; ntotalilvs={g_raw.ntotalilvs}, '
          f'ntotalilvs%{ilvperusimg}={g_raw.ntotalilvs % ilvperusimg})', file=stderr)
    tempvolmeasvol = []
    nav_imgs = []      # per usimg: low-res coronal projection (z, x)
    nav_lines = []     # per usimg: tracked diaphragm z (nan if no fit)
    nav_times = []     # per usimg: mean interleave time
    for iusimg in range(0, nusimg):
        res = g_res.dyn_usimg_recon(g, g_raw, g_traj, g.usegpu, iusimg)
        if not res:
            continue
        # dyn_usimg_recon stores GPDYN z-flipped (np.flip axis 0). For our Siemens-via-
        # MRD data that puts the diaphragm at LOW z, but Steve's edge-finder below walks
        # from the high-z end, so it would lock onto the apex (the less-mobile edge).
        # Flip z back so the diaphragm sits at high z (matching the final recon
        # orientation) — the edge-finder then tracks the diaphragm and the exported
        # projection is already apex-up / diaphragm-down.
        gpdyn = g_res.getimg(imgtype.GPDYN)[::-1, :, :]
        proj = np.abs(gpdyn[:, int(g.IS / 4):int(3 * g.IS / 4), :])
        urfig = np.sum(proj, 1)        # (z, x) coronal projection (navigator image)
        dropoff = np.sum(proj, (1, 2))  # (z,) z-profile
        nav_imgs.append(urfig.astype('float32'))
        nav_times.append(float(np.mean(
            g_raw.ilvtime[(iusimg * ilvperusimg):((iusimg + 1) * ilvperusimg)])))
        urfigline = np.nan
        # walk down from the top to find the 25% and 75% signal crossings
        for p1 in range(g.IS - 1, 0, -1):
            if dropoff[p1] > np.min(dropoff) + 0.25 * (np.max(dropoff) - np.min(dropoff)):
                break
        for p2 in range(p1, 0, -1):
            if dropoff[p2] > np.min(dropoff) + 0.75 * (np.max(dropoff) - np.min(dropoff)):
                break
        if p1 - p2 >= 2:
            p = np.polyfit(np.array(range(p2, p1)), dropoff[p2:p1], 2)
            # solve p[0]x^2 + p[1]x + (p[2] - (max+min)/2) = 0 for the half-max crossing
            det = np.sqrt(p[1] ** 2 - 4 * p[0] * (p[2] - (np.max(dropoff) + np.min(dropoff)) / 2))
            m1 = (-p[1] + det) / (2 * p[0])
            m2 = (-p[1] - det) / (2 * p[0])
            candidate = m1 if (m1 > p2 and m1 < p1) else (m2 if (m2 > p2 and m2 < p1) else None)
            if candidate is not None:
                urfigline = float(candidate)
                tempvolmeasvol.append(candidate)
                g_raw.volmeastime[bt].append(nav_times[-1])
        nav_lines.append(urfigline)
        if len(g_raw.volmeastime[bt]) > 5:
            ftvol = savgol_filter(tempvolmeasvol, 5, 2)
            minftvol = np.min(ftvol)
            g_raw.ilvvol[bt] = np.interp(g_raw.ilvtime, g_raw.volmeastime[bt],
                    (ftvol - minftvol) / (max(ftvol) - minftvol), left=np.nan, right=np.nan)
            g_raw.rescale(g_raw.ilvvol[bt])
            g_raw.ilvbin[bt] = g_raw.bin(g_raw.ilvvol[bt])
    g.MS = saveMS
    print(f'DIAPHRAGM: navigator done, {len(g_raw.volmeastime[bt])} measurements, '
          f'ilvbin populated={bool(len(g_raw.ilvbin[bt]))}', file=stderr)
    return {
        'nav_coronal': np.array(nav_imgs, dtype='float32'),     # (nframes, z, x)
        'nav_diaphragm_z': np.array(nav_lines, dtype='float32'),  # (nframes,)
        'nav_time': np.array(nav_times, dtype='float32'),         # (nframes,)
        'nav_volume': np.asarray(g_raw.ilvvol[bt], dtype='float32'),       # (ntotalilvs,)
        'nav_ilvtime': np.asarray(g_raw.ilvtime, dtype='float32'),         # (ntotalilvs,)
        'nav_volmeastime': np.asarray(g_raw.volmeastime[bt], dtype='float32'),
    }

def _mix_binning(g, g_raw, split_min=0.375, tB_factor=1.25, n_min=4, n_max=10):
    """MIX binning (fork 2026-10-26): inspiration on the absolute clock after the SIGNAL EE, exhale on the navigator
    volume rank. Per session: tau_I = max(split_min * T_med, tB_factor * tB_med) with tB = time of the per-breath k0
    peak; n_insp = round(nbins * tau_I / T_med) clamped [n_min, n_max]. Bins 0..n_insp-1: phi = (n_insp/nbins) *
    (t - t_EE) / tau_I; bins n_insp..nbins-1: phi = n_insp/nbins + (1 - n_insp/nbins) * (rank + 0.5) / N, rank =
    descending navigator volume among all kept exhale interleaves (pooled over breaths). Breaths with T < 1.5 tau_I,
    interleaves outside the EE list, NaN navigator and auto-excluded interleaves -> -1 (navigator volume inside an
    auto-excluded range is set NaN first; the DIAPHRAGM navigator interpolates across such gaps)."""
    nb = g.nbins
    t = np.asarray(g_raw.ilvtime, dtype='float64')
    ee = np.asarray(g_raw.volmeasEEtime[bintype.SIGNAL], dtype='float64')
    k0 = np.asarray(g_raw.volmeasvol[bintype.SIGNAL], dtype='float64')
    nav = np.asarray(g_raw.ilvvol[bintype.DIAPHRAGM], dtype='float64').copy()
    bins = np.full(len(t), -1.0)
    out = {}
    if ee.size < 3:
        print('MIX: fewer than 3 SIGNAL EE minima -- no MIX bins', file=stderr)
        return bins, out
    excl = np.zeros(len(t), dtype=bool)
    for el in g_raw.excluderanges:
        excl |= (t >= el.start) & (t <= el.stop)
    nav[excl] = np.nan
    k = np.searchsorted(ee, t, side='right') - 1
    inside = (k >= 0) & (k < ee.size - 1)
    kk = np.where(inside, k, 0)
    tee = np.where(inside, t - ee[kk], np.nan)
    per = np.where(inside, ee[np.minimum(kk + 1, ee.size - 1)] - ee[kk], np.nan)
    T_med = float(np.median(np.diff(ee)))
    tB = []
    for j in range(ee.size - 1):
        m = inside & (k == j)
        if np.sum(m) >= 10:
            tB.append(float(tee[m][np.argmax(k0[m])]))
    tB_med = float(np.median(tB)) if len(tB) else 0.0
    tau = max(split_min * T_med, tB_factor * tB_med)
    n_insp = int(min(max(round(nb * tau / T_med), n_min), n_max))
    fi = n_insp / nb
    keep = inside & (per >= 1.5 * tau) & ~excl
    insp = keep & (tee < tau)
    bins[insp] = fi * tee[insp] / tau
    exh = keep & (tee >= tau) & np.isfinite(nav)
    if np.any(exh):
        v = nav[exh]
        r = np.empty(len(v))
        r[np.argsort(-v)] = np.arange(len(v))
        bins[exh] = fi + (1.0 - fi) * (r + 0.5) / len(v)
    bins = np.where(bins >= 0, np.clip(bins, 0.0, 1.0 - 1e-9), -1.0)
    print(f'MIX: T_med {T_med:.2f} s, tB_med {tB_med:.2f} s ({len(tB)} breaths), tau_I {tau:.2f} s, n_insp {n_insp}, '
          f'{int(np.sum(insp))} inspiratory + {int(np.sum(exh))} exhale interleaves binned, '
          f'{int(np.sum(~keep & inside))} dropped (short breath / excluded), {int(np.sum(exh | insp))} of {len(t)} total',
          file=stderr)
    out['mix_tau_I'] = np.array([tau], dtype='float32')
    out['mix_n_insp'] = np.array([n_insp], dtype='float32')
    out['mix_tB_med'] = np.array([tB_med], dtype='float32')
    out['mix_T_med'] = np.array([T_med], dtype='float32')
    return bins, out


def _bin_export(g, g_raw, bt):
    """Per-interleave bin coordinate as gridded, interleave times, SIGNAL EE list, effective kernel weight per bin."""
    from recon import bindist0sq
    bins = np.asarray(g_raw.ilvbin[bt], dtype='float64')
    nb = g.nbins
    bd0sq = 2.0 * float(getattr(g, 'binsigma', 1.0)) ** 2 if hasattr(g, 'binsigma') else bindist0sq
    ok = bins >= 0
    c = bins * nb
    w = np.zeros((len(bins), nb))
    for ib in range(nb):
        d = np.minimum(np.minimum(np.abs(ib - c), np.abs(ib + nb - c)), np.abs(ib - nb - c))
        w[:, ib] = np.where(ok, np.exp(-d * d / bd0sq), 0.0)
    if int(getattr(g, 'binpou', 0)):
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    return {'ilvbin': bins.astype('float32'), 'ilvtime': np.asarray(g_raw.ilvtime, dtype='float32'),
            'ee_times': np.asarray(g_raw.volmeasEEtime[bintype.SIGNAL], dtype='float32'),
            'bin_weff': w.sum(0).astype('float32'),
            'bin_mode': np.array([int(bt)], dtype='float32')}


def _write_results_to_mrd(g_res, header, output, nav=None):
    """Write GPDYN and DPDYN reconstructed images as NdArray items to the output MRD stream.
    If `nav` (the DIAPHRAGM navigator dict) is given, also stream its arrays."""
    items = []
    # '1' = images divided by the gridding kernel's rolloff (results.deapod, 2steve/03); absent = older images
    deapod = [mrd.ArrayMetaValue.String('1' if getattr(g_res, 'deapod', False) else '0')]
    if g_res.hasimg(imgtype.GPDYN):
        items.append(mrd.StreamItem.NdArrayFloat(
            mrd.NdArray(data=g_res.getimg(imgtype.GPDYN).astype('float32'),
                        meta={'gas_phase_image': [mrd.ArrayMetaValue.String('1')], 'deapod': deapod})))
        gmag = getattr(g_res, 'gpdyn_magnitude', None)
        if gmag is not None:
            # |F*b| per bin, single-channel only: for videos/QC. real(F*b) above stays the
            # quantitative image (2steve/06: real() attenuates the moving rim outside b's mask).
            items.append(mrd.StreamItem.NdArrayFloat(
                mrd.NdArray(data=np.asarray(gmag).astype('float32'),
                            meta={'gas_phase_magnitude': [mrd.ArrayMetaValue.String('1')], 'deapod': deapod})))
    if g_res.hasimg(imgtype.DPDYN):
        dpmeta = {'dissolved_phase_image': [mrd.ArrayMetaValue.String('1')], 'deapod': deapod,
                  # '1': stored as aRBC + 1j*aTP; '0': unsplit complex (magnitude only)
                  'rbc_tp_separated': [mrd.ArrayMetaValue.String(
                      '1' if getattr(g_res, 'rbc_tp_separated', True) else '0')]}
        split = getattr(g_res, 'rbc_tp_split', None)
        if split:
            # basis angle, conditioning, spectral target and the per-bin solved phase / masked
            # ratio (nan = bin kept unsplit), so the maps can be judged without the log
            dpmeta['rbc_tp_dphi_deg'] = [mrd.ArrayMetaValue.String(f'{split["dphi_deg"]:.2f}')]
            dpmeta['rbc_tp_sin_dphi'] = [mrd.ArrayMetaValue.String(f'{split["sin_dphi"]:.3f}')]
            dpmeta['rbc_tp_df_hz'] = [mrd.ArrayMetaValue.String(f'{split["df_hz"]:.1f}')]
            dpmeta['rbc_tp_target'] = [mrd.ArrayMetaValue.String(f'{split["target"]:.4f}')]
            dpmeta['rbc_tp_ph_rad'] = [mrd.ArrayMetaValue.String(' '.join(f'{v:.3f}' for v in split['ph']))]
            dpmeta['rbc_tp_R'] = [mrd.ArrayMetaValue.String(' '.join(f'{v:.4f}' for v in split['R']))]
            for k in ('model_used', 'ratio_scalar', 'F_lump', 'snr_diss', 'dphi_rep_sd_deg', 'stab_dphi_deg'):
                if k in split:
                    dpmeta[f'rbc_tp_{k}'] = [mrd.ArrayMetaValue.String(str(split[k]))]
        items.append(mrd.StreamItem.NdArrayComplexFloat(
            mrd.NdArray(data=g_res.getimg(imgtype.DPDYN).astype('complex64'), meta=dpmeta)))
    gcplx = getattr(g_res, 'gpdyn_complex', None)
    if gcplx is not None:
        # complex F*b per bin (single-channel): per-bin phase reference for the dissolved image.
        # Appended after the dissolved item so readers that take the first complex item are unaffected.
        items.append(mrd.StreamItem.NdArrayComplexFloat(
            mrd.NdArray(data=np.asarray(gcplx).astype('complex64'),
                        meta={'gas_phase_complex': [mrd.ArrayMetaValue.String('1')], 'deapod': deapod})))
    bmat = getattr(g_res, 'b', None)
    if bmat is not None and np.asarray(bmat).size:
        # calcb's phase reference b (nch, IS, IS, IS), cropped and max-normalised, exactly as multiplied
        # into every bin of every image type — for QC of what it imprints on the dissolved split.
        items.append(mrd.StreamItem.NdArrayComplexFloat(
            mrd.NdArray(data=np.asarray(bmat).astype('complex64'),
                        meta={'calcb_b': [mrd.ArrayMetaValue.String('1')]})))
    if nav is not None:
        for key, arr in nav.items():
            if arr is None or np.asarray(arr).size == 0:
                continue
            items.append(mrd.StreamItem.NdArrayFloat(
                mrd.NdArray(data=np.asarray(arr).astype('float32'),
                            meta={key: [mrd.ArrayMetaValue.String('1')]})))
    with mrd.BinaryMrdWriter(output) as writer:
        writer.write_header(header)
        writer.write_data(iter(items))


def reconstruct_from_mrd(input: BinaryIO, output: BinaryIO):
    # class objects init to store mrd streamitem
    g = gvar()
    g_traj = traj()
    g_raw = raw()
    g_res = results()

    # NdArray streamitems, discriminated by meta key
    ref_acq_arr = None
    dyn_acq_arr = None
    pneumo_arr = None
    pneumo_tmap = None
    gas_phase_traj_arr = None
    dissolved_phase_traj_arr = None
    with mrd.BinaryMrdReader(input) as reader:
        header = reader.read_header()
        for item in reader.read_data():
            if isinstance(item, (mrd.StreamItem.NdArrayDouble, mrd.StreamItem.NdArrayFloat)):
                if item.value.meta.get('gas_phase_trajectory'):
                    gas_phase_traj_arr = item.value.data
                elif item.value.meta.get('dissolved_phase_trajectory'):
                    dissolved_phase_traj_arr = item.value.data
                elif item.value.meta.get('pneumotach'):
                    pneumo_arr = item.value.data
                elif item.value.meta.get('pneumo_gas_true_s'):
                    pneumo_tmap = item.value.data
            elif isinstance(item, mrd.StreamItem.NdArrayComplexFloat):
                # raw acquisitions stored whole, shape (channels, samples, lines)
                if item.value.meta.get('reference_acquisition'):
                    ref_acq_arr = item.value.data
                elif item.value.meta.get('dynamic_acquisition'):
                    dyn_acq_arr = item.value.data

    # header user parameters → recon parameters / acquisition metadata
    user_long, user_double, user_string = {}, {}, {}
    if header.user_parameters is not None:
        user_long = {p.name: p.value for p in header.user_parameters.user_parameter_long}
        user_double = {p.name: p.value for p in header.user_parameters.user_parameter_double}
        user_string = {p.name: p.value for p in header.user_parameters.user_parameter_string}
    g.MS = int(user_long.get('MS', g.MS))
    g.IS = int(user_long.get('IS', g.IS))
    g.nbins = int(user_long.get('nbins', g.nbins))
    g.gplb = int(user_long.get('gplb', g.gplb))
    g.dplb = int(user_long.get('dplb', g.dplb))
    g.freqfilter = int(user_long.get('freqfilter', g.freqfilter))
    g.griddx = float(user_double.get('griddx', g.griddx))
    g.bindt = float(user_double.get('bindt', g.bindt))
    killpts = int(user_long.get('killpts', 2))
    g.binsigma = float(user_double.get('binsigma', g.binsigma))
    g.binpou = int(user_long.get('binpou', g.binpou))
    meta = {'TR': user_double.get('TR', 0.0), 'TE': user_double.get('TE', 0.0),
            'DPoff': user_double.get('DPoff', 0.0), 'dtdyn': user_double.get('dtdyn', 0.0),
            'dtspec': user_double.get('dtspec', 0.0), 'numspec': int(user_long.get('numspec', 0)),
            'sigminN': int(user_long.get('sigminN', 50))}
    if int(user_long.get('pneumoclock', 0)) and pneumo_tmap is not None:
        meta['pneumo_tmap'] = np.asarray(pneumo_tmap, dtype='float64')
        meta['pneumo_offset_s'] = float(user_double.get('pneumo_offset_s', 0.0))

    # load trajectories
    # dissolved trajectory is optional: gas-only sequences (e.g. v3_20230821) ship none
    if gas_phase_traj_arr is None:
        raise ValueError('MRD file missing gas_phase_trajectory NdArray')
    if dissolved_phase_traj_arr is None:
        print('no dissolved_phase_trajectory in MRD -- gas-phase only', file=stderr)
    g_traj.killpts = killpts
    g_traj.load_traj_from_array(gas_phase_traj_arr, dissolved_phase_traj_arr,
                                int(user_long.get('nusimg', 32)))

    # trim killpts on the samples axis of the (channels, samples, lines) arrays
    if dyn_acq_arr is None:
        raise ValueError('MRD file missing dynamic_acquisition NdArray')
    dyn_acq_arr = dyn_acq_arr[:, killpts:, :]
    if ref_acq_arr is not None:
        ref_acq_arr = ref_acq_arr[:, killpts:, :]
    print(f'read mrd: dyn={dyn_acq_arr.shape} ref={ref_acq_arr.shape if ref_acq_arr is not None else None}'
          f' pneumo={pneumo_arr is not None} MS={g.MS} IS={g.IS} nbins={g.nbins}', file=stderr)

    g_raw.load_from_arr(g_traj, ref_acq_arr, dyn_acq_arr, pneumo_arr, 'mrd_siemens', meta)
    print(f'SIGNAL: {len(g_raw.volmeasEEtime[bintype.SIGNAL])} EE minima (half-window {meta["sigminN"]})', file=stderr)

    # manual exclude ranges = Steve's GUI 'Exclude ranges' field (main.py stringtorangelist): pairs of integer
    # seconds on the gas-interleave clock (iilv * TR), added to load_from_arr's automatic low-signal ranges
    exstr = user_string.get('excluderanges', '')
    if exstr:
        a = list(map(int, re.findall(r'\d+', exstr)))
        manual = [] if len(a) % 2 else [range(a[j], a[j + 1]) for j in range(0, len(a), 2) if a[j + 1] > a[j]]
        g_raw.excluderanges += manual
        print(f'exclude ranges: manual {[(r.start, r.stop) for r in manual]} + auto, now '
              f'{[(r.start, r.stop) for r in g_raw.excluderanges]} s', file=stderr)

    g.usegpu = int(cuda.is_available())
    print(f'usegpu={g.usegpu}, calculating b-matrix...', file=stderr)
    g_res.calcb(g, g_raw, g_traj, g.usegpu)
    binning = user_string.get('binning', 'SIGNAL')

    # DIAPHRAGM needs a low-res navigator loop to populate ilvbin[DIAPHRAGM];
    # it temporarily drops g.MS, so recompute the full-res b-matrix afterward.
    nav = None
    if binning in ('DIAPHRAGM', 'MIX') and not len(g_raw.ilvbin[bintype.DIAPHRAGM]):
        nav = _diaphragm_navigator(g, g_raw, g_traj, g_res)
        print(f'recomputing full-res b (MS={g.MS})...', file=stderr)
        g_res.calcb(g, g_raw, g_traj, g.usegpu)

    binexp = {}
    if binning == 'MIX' and len(g_raw.ilvbin[bintype.DIAPHRAGM]):
        g_raw.ilvbin[bintype.MIX], binexp = _mix_binning(g, g_raw)
        bt = bintype.MIX
    elif binning == 'DIAPHRAGM' and len(g_raw.ilvbin[bintype.DIAPHRAGM]):
        bt = bintype.DIAPHRAGM
    elif binning == 'PNEUMOTACH' and len(g_raw.ilvbin[bintype.PNEUMOTACH]):
        bt = bintype.PNEUMOTACH
    else:
        bt = bintype.SIGNAL
    print(f'binned dynamic recon with {bt.name} binning...', file=stderr)
    g_res.dyn_recon(g, g_raw, g_traj, g_raw.ilvbin[bt], g.usegpu)

    # export the bins actually gridded (+ EE list, kernel weight per bin) so a reader can reproduce per-bin averages
    binexp.update(_bin_export(g, g_raw, bt))
    pc = getattr(g_raw, 'pneumo_clock', None)
    if pc:
        for k, v in pc.items():
            binexp[f'pneumo_clock_{k}'] = np.array([float(v)], dtype='float32')
        if len(g_raw.volmeastime[bintype.PNEUMOTACH]):
            binexp['pneumo_time_aligned'] = np.asarray(g_raw.volmeastime[bintype.PNEUMOTACH], dtype='float32')
            binexp['pneumo_vol_aligned'] = np.asarray(g_raw.volmeasvol[bintype.PNEUMOTACH], dtype='float32')
    if nav is None:
        nav = {}
    nav.update(binexp)

    print('writing reconstructed images to output mrd', file=stderr)
    _write_results_to_mrd(g_res, header, output, nav=nav if len(nav) else None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Path to input MRD binary file')
    parser.add_argument('--output', required=True, help='Path to output MRD binary file')
    args = parser.parse_args()

    with open(args.input, 'rb') as input_file:
        with open(args.output, 'wb') as output_file:
            reconstruct_from_mrd(input_file, output_file)
