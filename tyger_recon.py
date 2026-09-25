"""
Loads mrd file populated with siemens .dat file acquisitions and npy trajectory fields
in NdArray field. This script file is used as entrypoint of docker container to run reconstruction on tyger
"""
from typing import BinaryIO
from sys import stderr
import argparse

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

def _write_results_to_mrd(g_res, header, output, nav=None):
    """Write GPDYN and DPDYN reconstructed images as NdArray items to the output MRD stream.
    If `nav` (the DIAPHRAGM navigator dict) is given, also stream its arrays."""
    items = []
    if g_res.hasimg(imgtype.GPDYN):
        items.append(mrd.StreamItem.NdArrayFloat(
            mrd.NdArray(data=g_res.getimg(imgtype.GPDYN).astype('float32'),
                        meta={'gas_phase_image': [mrd.ArrayMetaValue.String('1')]})))
        gmag = getattr(g_res, 'gpdyn_magnitude', None)
        if gmag is not None:
            # |F*b| per bin, single-channel only: for videos/QC. real(F*b) above stays the
            # quantitative image (2steve/06: real() attenuates the moving rim outside b's mask).
            items.append(mrd.StreamItem.NdArrayFloat(
                mrd.NdArray(data=np.asarray(gmag).astype('float32'),
                            meta={'gas_phase_magnitude': [mrd.ArrayMetaValue.String('1')]})))
    if g_res.hasimg(imgtype.DPDYN):
        dpmeta = {'dissolved_phase_image': [mrd.ArrayMetaValue.String('1')],
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
    meta = {'TR': user_double.get('TR', 0.0), 'TE': user_double.get('TE', 0.0),
            'DPoff': user_double.get('DPoff', 0.0), 'dtdyn': user_double.get('dtdyn', 0.0),
            'dtspec': user_double.get('dtspec', 0.0), 'numspec': int(user_long.get('numspec', 0))}

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

    g.usegpu = int(cuda.is_available())
    print(f'usegpu={g.usegpu}, calculating b-matrix...', file=stderr)
    g_res.calcb(g, g_raw, g_traj, g.usegpu)
    binning = user_string.get('binning', 'SIGNAL')

    # DIAPHRAGM needs a low-res navigator loop to populate ilvbin[DIAPHRAGM];
    # it temporarily drops g.MS, so recompute the full-res b-matrix afterward.
    nav = None
    if binning == 'DIAPHRAGM' and not len(g_raw.ilvbin[bintype.DIAPHRAGM]):
        nav = _diaphragm_navigator(g, g_raw, g_traj, g_res)
        print(f'recomputing full-res b (MS={g.MS})...', file=stderr)
        g_res.calcb(g, g_raw, g_traj, g.usegpu)

    if binning == 'DIAPHRAGM' and len(g_raw.ilvbin[bintype.DIAPHRAGM]):
        bt = bintype.DIAPHRAGM
    elif binning == 'PNEUMOTACH' and len(g_raw.ilvbin[bintype.PNEUMOTACH]):
        bt = bintype.PNEUMOTACH
    else:
        bt = bintype.SIGNAL
    print(f'binned dynamic recon with {bt.name} binning...', file=stderr)
    g_res.dyn_recon(g, g_raw, g_traj, g_raw.ilvbin[bt], g.usegpu)

    print('writing reconstructed images to output mrd', file=stderr)
    _write_results_to_mrd(g_res, header, output, nav=nav)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Path to input MRD binary file')
    parser.add_argument('--output', required=True, help='Path to output MRD binary file')
    args = parser.parse_args()

    with open(args.input, 'rb') as input_file:
        with open(args.output, 'wb') as output_file:
            reconstruct_from_mrd(input_file, output_file)
