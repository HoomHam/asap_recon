import os
import numpy as np
import argparse

import mapvbvd
import mrd

from gtypes import gvar


def _classify_dat_files(dat_dir, dat_files=None, seqname=None):
    """
    Return (dynraw_path, refraw_path_or_None).
    Largest file = dynamic acquisition; second-largest = breath-hold reference.
    dat_files: optional pre-filtered list of .dat paths (e.g. from the GUI);
    seqname: optional substring filter on filenames.
    Raises FileNotFoundError if no .dat files are found.
    """
    if dat_files:
        dats = sorted(dat_files, key=os.path.getsize)
    else:
        dats = sorted(
            [os.path.join(dat_dir, f) for f in os.listdir(dat_dir) if f.endswith('.dat')],
            key=os.path.getsize
        )
    if seqname:
        dats = [f for f in dats if seqname in os.path.basename(f)]
    if not dats:
        raise FileNotFoundError(f'No matching .dat files found in {dat_dir}')
    dynraw  = dats[-1]
    refraw  = dats[-2] if len(dats) >= 2 else None
    print(f'dynamic:   {dynraw} ({os.path.getsize(dynraw)} bytes)')
    if refraw:
        print(f'reference: {refraw} ({os.path.getsize(refraw)} bytes)')
    return dynraw, refraw

def _read_twix(dat_file):
    """
    Read a single Siemens .dat file.

    Returns:
        raw   : np.ndarray, complex64, shape (npts_full, nch, nilv)
        meta  : dict with keys TR_us, TE_us, DPoff, dtdyn_ns, numspec_raw, dtspec_us
    """
    twix = mapvbvd.mapVBVD(dat_file)
    if isinstance(twix, list):
        twix = twix[-1]
    twix.image.flagRemoveOS = False
    twix.image.squeeze = True
    raw = twix.image.unsorted().astype('complex64')
    if raw.ndim == 2:
        raw = raw[:, np.newaxis, :]  # add singleton channel dim → (npts, 1, nilv)

    meta = dict(TR_us=0.0, TE_us=0.0, DPoff=0.0, dtdyn_ns=0.0,
                numspec_raw=0, dtspec_us=0.0)
    try:
        meta['TR_us'] = float(twix.hdr.MeasYaps[('alTR', '0')])
    except Exception:
        pass
    try:
        meta['TE_us'] = float(twix.hdr.MeasYaps[('alTE', '0')])
    except Exception:
        pass
    try:
        meta['DPoff'] = float(twix.hdr.MeasYaps[('sWipMemBlock', 'adFree', '2')])
    except Exception:
        pass
    try:
        meta['dtdyn_ns'] = float(twix.hdr.MeasYaps[('sRXSPEC', 'alDwellTime', '0')])
    except Exception:
        pass
    try:
        meta['dtdyn_ns'] = float(twix.hdr.MeasYaps[('sRXSPEC', 'alDwellTime', '1')])
    except Exception:
        pass
    try:
        meta['numspec_raw'] = int(
            twix.hdr.MeasYaps[('sWipMemBlock', 'alFree', '10')] *
            twix.hdr.MeasYaps[('sWipMemBlock', 'alFree', '11')] + 0.1
        )
    except Exception:
        pass
    try:
        meta['dtspec_us'] = float(twix.hdr.MeasYaps[('sWipMemBlock', 'alFree', '12')])
    except Exception:
        pass

    return raw, meta


def _parse_pneumotach(pneumo_file):
    """
    Parse the vendor pneumotach binary: packets marked by magic bytes 0xA6 0x20,
    each holding a millisecond timestamp and a pressure reading.

    Returns float32 array of shape (2, N): row 0 = time in seconds (zeroed to
    the first packet), row 1 = pressure. Smoothing/cropping/volume integration
    happen recon-side where ilvtime is known.
    """
    updatesendsize = 54
    with open(pneumo_file, mode='rb') as f:
        a = f.read()
    idx = [j for j in range(len(a) - updatesendsize) if a[j] == 0xA6 and a[j+1] == 0x20]
    print(f'found {len(idx)} pressure measurements in {pneumo_file}')
    t = np.zeros(len(idx), dtype='float64')
    P = np.zeros(len(idx), dtype='float64')
    for cnt, j in enumerate(idx):
        t[cnt] = a[j+5]*2**24 + a[j+4]*2**16 + a[j+3]*2**8 + a[j+2]
        P[cnt] = -20 + 90 * (a[j+33]*2**8 + a[j+32]) / 65535.0
    t -= t[0]
    t /= 1000
    return np.stack((t, P))


def convert_siemens_to_mrd(dat_dir, output_mrd_file,
                            gp_traj_file=None, dp_traj_file=None,
                            pneumotach_file=None, killpts=2,
                            params=None, dat_files=None, seqname=None):
    """
    Convert Siemens TWIX .dat file(s) in a directory to MRD binary format.

    The largest .dat file is used as the dynamic acquisition; the
    second-largest (if present) as the breath-hold reference. Header metadata
    is taken from the dynamic file. Each acquisition is stored whole as a
    single 3D NdArray of shape (channels, samples, lines), discriminated by
    meta key ('reference_acquisition' / 'dynamic_acquisition').

    NOTE: numspec is stored as the raw product alFree[10] * alFree[11].  The
    scaling kluge (÷20 × nsmpperusimg/npts) is applied recon-side in raw.py
    where the trajectory is available.
\

    Args:
        dat_dir:         Directory containing the Siemens .dat file(s).
        output_mrd_file: Path to output .mrd file (created or overwritten).
        gp_traj_file:    Optional path to gas-phase trajectory .npy file.
        dp_traj_file:    Optional path to dissolved-phase trajectory .npy file.
        pneumotach_file: Optional path to pneumotach binary file.
        killpts:         Initial samples to flag as discard_pre (default 2).
        params:          Optional dict of recon parameters from the GUI/CLI
                         (MS, IS, nbins, griddx, bindt, gplb, dplb, freqfilter,
                         binning) stored as MRD header user parameters.
        dat_files:       Optional pre-filtered list of .dat paths.
        seqname:         Optional substring filter on .dat filenames.
    """
    # --- 1. Identify and read each file independently ---
    dynraw_file, refraw_file = _classify_dat_files(dat_dir, dat_files, seqname)
    dynraw, meta = _read_twix(dynraw_file)
    npts_full, nch, nilv = dynraw.shape

    refraw = None
    if refraw_file is not None:
        refraw, _ = _read_twix(refraw_file)

    # --- 2. Build MRD Header (metadata from dynamic file) ---
    header = mrd.Header()

    seq = mrd.SequenceParametersType()
    seq.t_r = [meta['TR_us'] * 1e-3]   # µs → ms (informational; recon reads user params)
    seq.t_e = [meta['TE_us'] * 1e-3]
    header.sequence_parameters = seq

    sys_info = mrd.AcquisitionSystemInformationType()
    sys_info.receiver_channels = nch
    sys_info.system_vendor = 'Siemens'
    header.acquisition_system_information = sys_info

    if params is None:
        params = {}
    g_defaults = gvar()
    user = mrd.UserParametersType()
    user.user_parameter_double = [
        mrd.UserParameterDoubleType(name='TR',     value=meta['TR_us'] * 1e-6),     # s
        mrd.UserParameterDoubleType(name='TE',     value=meta['TE_us'] * 1e-6),     # s
        mrd.UserParameterDoubleType(name='DPoff',  value=meta['DPoff']),
        mrd.UserParameterDoubleType(name='dtdyn',  value=meta['dtdyn_ns'] * 1e-9),  # s
        mrd.UserParameterDoubleType(name='dtspec', value=meta['dtspec_us'] * 1e-6), # s
        mrd.UserParameterDoubleType(name='griddx', value=float(params.get('griddx', g_defaults.griddx))),
        mrd.UserParameterDoubleType(name='bindt',  value=float(params.get('bindt', g_defaults.bindt))),
    ]
    user.user_parameter_long = [
        mrd.UserParameterLongType(name='numspec',    value=meta['numspec_raw']),
        mrd.UserParameterLongType(name='nusimg',     value=32),
        mrd.UserParameterLongType(name='killpts',    value=killpts),
        mrd.UserParameterLongType(name='MS',         value=int(params.get('MS', g_defaults.MS))),
        mrd.UserParameterLongType(name='IS',         value=int(params.get('IS', g_defaults.IS))),
        mrd.UserParameterLongType(name='nbins',      value=int(params.get('nbins', g_defaults.nbins))),
        mrd.UserParameterLongType(name='gplb',       value=int(params.get('gplb', g_defaults.gplb))),
        mrd.UserParameterLongType(name='dplb',       value=int(params.get('dplb', g_defaults.dplb))),
        mrd.UserParameterLongType(name='freqfilter', value=int(params.get('freqfilter', g_defaults.freqfilter))),
    ]
    user.user_parameter_string = [
        mrd.UserParameterStringType(name='binning', value=params.get('binning', 'SIGNAL')),
    ]
    header.user_parameters = user

    # --- 3. Stream to MRD file ---
    def stream_items():
        # float64 throughout so the recon side is bit-identical to reading the
        # source .npy / pneumotach files directly
        if gp_traj_file is not None:
            gp = np.load(gp_traj_file).astype('float64')
            yield mrd.StreamItem.NdArrayDouble(
                mrd.NdArray(data=gp, meta={'gas_phase_trajectory': [mrd.ArrayMetaValue.String('1')]}))
        if dp_traj_file is not None:
            dp = np.load(dp_traj_file).astype('float64')
            yield mrd.StreamItem.NdArrayDouble(
                mrd.NdArray(data=dp, meta={'dissolved_phase_trajectory': [mrd.ArrayMetaValue.String('1')]}))
        if pneumotach_file is not None:
            pn = _parse_pneumotach(pneumotach_file)
            yield mrd.StreamItem.NdArrayDouble(
                mrd.NdArray(data=pn, meta={'pneumotach': [mrd.ArrayMetaValue.String('1')]}))
        # mapvbvd unsorted() is (samples, channels, lines); store channels-first
        if refraw is not None:
            yield mrd.StreamItem.NdArrayComplexFloat(
                mrd.NdArray(data=np.ascontiguousarray(refraw.transpose(1, 0, 2)),
                            meta={'reference_acquisition': [mrd.ArrayMetaValue.String('1')]}))
        yield mrd.StreamItem.NdArrayComplexFloat(
            mrd.NdArray(data=np.ascontiguousarray(dynraw.transpose(1, 0, 2)),
                        meta={'dynamic_acquisition': [mrd.ArrayMetaValue.String('1')]}))

    with mrd.BinaryMrdWriter(output_mrd_file) as writer:
        writer.write_header(header)
        writer.write_data(stream_items())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Siemens .dat file(s) in a directory to MRD binary format')
    parser.add_argument('-i', '--input',  required=True, metavar='DIR',
                        help='Directory containing Siemens .dat file(s)')
    parser.add_argument('-o', '--output', required=True, metavar='FILE',
                        help='Output .mrd file')
    parser.add_argument('--gp-traj',      default=None,  metavar='FILE',
                        help='Gas-phase trajectory .npy file')
    parser.add_argument('--dp-traj',      default=None,  metavar='FILE',
                        help='Dissolved-phase trajectory .npy file')
    parser.add_argument('--pneumotach',   default=None,  metavar='FILE',
                        help='Pneumotach binary file')
    parser.add_argument('--seqname',      default=None,  metavar='STR',
                        help='Substring filter for .dat filenames')
    parser.add_argument('--ms',         type=int,   default=gvar.MS,         help='Matrix size')
    parser.add_argument('--is',         type=int,   default=gvar.IS,         help='Image size', dest='IS')
    parser.add_argument('--nbins',      type=int,   default=gvar.nbins,      help='Number of respiratory bins')
    parser.add_argument('--griddx',     type=float, default=gvar.griddx,     help='Gridding kernel dx')
    parser.add_argument('--bindt',      type=float, default=gvar.bindt,      help='Binning kernel dt')
    parser.add_argument('--gplb',       type=int,   default=gvar.gplb,       help='Gas-phase line broadening')
    parser.add_argument('--dplb',       type=int,   default=gvar.dplb,       help='Dissolved-phase line broadening')
    parser.add_argument('--freqfilter', type=int,   default=gvar.freqfilter, help='Frequency filter')
    parser.add_argument('--binning',    default='SIGNAL', choices=['SIGNAL', 'PNEUMOTACH', 'DIAPHRAGM'],
                        help='Respiratory binning method')
    args = parser.parse_args()
    params = dict(MS=args.ms, IS=args.IS, nbins=args.nbins, griddx=args.griddx,
                  bindt=args.bindt, gplb=args.gplb, dplb=args.dplb,
                  freqfilter=args.freqfilter, binning=args.binning)
    convert_siemens_to_mrd(args.input, args.output,
                            args.gp_traj, args.dp_traj,
                            pneumotach_file=args.pneumotach, killpts=2,
                            params=params, seqname=args.seqname)
