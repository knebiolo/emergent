"""Convert directory of memmap .npy files produced by MemmapLogWriter to canonical HDF.

Usage:
    python tools/convert_memmap_to_hdf.py --memdir <dir> --out <hdf_path>

This script reads each `<var>.npy` file (created by numpy.lib.format.open_memmap)
and writes it into `/agent_data/<var>` dataset in the output HDF5 file.
"""
import os
import argparse
import h5py
import numpy as np


def convert(memdir, out_path):
    files = [f for f in os.listdir(memdir) if f.endswith('.npy')]
    if not files:
        print('No .npy files found in', memdir)
        return
    # open output HDF
    with h5py.File(out_path, 'w') as h:
        ad = h.create_group('agent_data')
        for fname in sorted(files):
            path = os.path.join(memdir, fname)
            var = os.path.splitext(fname)[0]
            # load memmap header and shape
            mm = np.lib.format.open_memmap(path, mode='r')
            data = np.asarray(mm, dtype='f4')
            # write dataset
            ad.create_dataset(var, data=data, dtype='f4', compression='gzip')
            print('Wrote', var, 'shape', data.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--memdir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    convert(args.memdir, args.out)
