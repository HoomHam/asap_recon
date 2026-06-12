import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mrd


def montage(vol, slice_idx, suptitle, out_path):
    nbins = vol.shape[0]
    grid = math.ceil(math.sqrt(nbins))
    fig, axes = plt.subplots(grid, grid, figsize=(2 * grid, 2 * grid))
    axes = np.atleast_1d(axes).ravel()
    for i in range(grid * grid):
        ax = axes[i]
        if i < nbins:
            ax.imshow(vol[i, slice_idx, :, :], cmap='gray')
            ax.set_title(f'bin {i}')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    fig.suptitle(suptitle)
    fig.savefig(out_path)
    print(f'saved {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', default='xe_dyn_recon.mrd')
    parser.add_argument('--slice', type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    gas = None
    dissolved = None
    with mrd.BinaryMrdReader(str(input_path)) as reader:
        reader.read_header()
        for item in reader.read_data():
            if isinstance(item, mrd.StreamItem.NdArrayFloat):
                if item.value.meta.get('gas_phase_image'):
                    gas = item.value.data
            elif isinstance(item, mrd.StreamItem.NdArrayComplexFloat):
                if item.value.meta.get('dissolved_phase_image'):
                    dissolved = item.value.data

    if args.slice is None:
        ref = gas if gas is not None else dissolved
        slice_idx = ref.shape[1] // 2
    else:
        slice_idx = args.slice

    if gas is not None:
        print(f'gas phase: shape={gas.shape} dtype={gas.dtype}')
        out = input_path.with_name(f'{input_path.stem}_gp.png')
        montage(gas, slice_idx, 'Gas phase', out)

    if dissolved is not None:
        print(f'dissolved phase: shape={dissolved.shape} dtype={dissolved.dtype}')
        out = input_path.with_name(f'{input_path.stem}_dp.png')
        montage(np.abs(dissolved), slice_idx, 'Dissolved phase (magnitude)', out)

    plt.show()


if __name__ == '__main__':
    main()
