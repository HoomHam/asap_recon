"""
Loads mrd file populated with siemens .dat file acquisitions and npy trajectory fields
in NdArray field. This script file is used as entrypoint of docker container to run reconstruction on tyger
"""
from typing import BinaryIO
from sys import stderr
import argparse

import matplotlib
matplotlib.use('Agg')  # headless subprocess: results.py calls plt.show()

from numba import cuda

import mrd

from gtypes import gvar, imgtype, bintype
from raw import traj, raw
from results import results

def _write_results_to_mrd(g_res, header, output):
    """Write GPDYN and DPDYN reconstructed images as NdArray items to the output MRD stream."""
    items = []
    if g_res.hasimg(imgtype.GPDYN):
        items.append(mrd.StreamItem.NdArrayFloat(
            mrd.NdArray(data=g_res.getimg(imgtype.GPDYN).astype('float32'),
                        meta={'gas_phase_image': [mrd.ArrayMetaValue.String('1')]})))
    if g_res.hasimg(imgtype.DPDYN):
        items.append(mrd.StreamItem.NdArrayComplexFloat(
            mrd.NdArray(data=g_res.getimg(imgtype.DPDYN).astype('complex64'),
                        meta={'dissolved_phase_image': [mrd.ArrayMetaValue.String('1')]})))
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
    if gas_phase_traj_arr is None or dissolved_phase_traj_arr is None:
        raise ValueError('MRD file missing gas_phase_trajectory or dissolved_phase_trajectory NdArray')
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
    bt = bintype.PNEUMOTACH if (binning == 'PNEUMOTACH' and len(g_raw.ilvbin[bintype.PNEUMOTACH])) \
            else bintype.SIGNAL
    print(f'binned dynamic recon with {bt.name} binning...', file=stderr)
    g_res.dyn_recon(g, g_raw, g_traj, g_raw.ilvbin[bt], g.usegpu)

    print('writing reconstructed images to output mrd', file=stderr)
    _write_results_to_mrd(g_res, header, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Path to input MRD binary file')
    parser.add_argument('--output', required=True, help='Path to output MRD binary file')
    args = parser.parse_args()

    with open(args.input, 'rb') as input_file:
        with open(args.output, 'wb') as output_file:
            reconstruct_from_mrd(input_file, output_file)
