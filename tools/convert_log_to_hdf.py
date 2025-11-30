"""Convert a directory of per-timestep `.npz` logs into a single HDF5 file.

Usage:
    python tools/convert_log_to_hdf.py --logdir path/to/logdir --out out.h5

This is intentionally conservative: it reads all `step_*.npz` files and creates
datasets under `/records/<var>` with shape (timesteps, ...) typed to float32.
"""
import argparse
import os
import numpy as np
import h5py


def convert(logdir, out_path):
    files = sorted([f for f in os.listdir(logdir) if f.startswith('step_') and f.endswith('.npz')])
    if not files:
        raise RuntimeError('No step_*.npz files found in ' + logdir)

    # inspect first file to detect per-agent arrays
    first = np.load(os.path.join(logdir, files[0]))
    nsteps = len(files)

    # Determine candidate agent arrays: 1D arrays whose length is consistent across files
    candidate_agent_vars = []
    other_vars = []
    for k in first.files:
        arr0 = first[k]
        if arr0.ndim == 1 and arr0.shape[0] > 1:
            # check consistency across files (same length)
            consistent = True
            for fn in files[1:]:
                arrn = np.load(os.path.join(logdir, fn))[k]
                if arrn.ndim != 1 or arrn.shape[0] != arr0.shape[0]:
                    consistent = False
                    break
            if consistent:
                candidate_agent_vars.append((k, arr0.shape[0]))
            else:
                other_vars.append(k)
        else:
            other_vars.append(k)

    # create HDF file and write datasets
    with h5py.File(out_path, 'w') as f:
        # write agent_data group for candidate agent vars
        if candidate_agent_vars:
            ag = f.create_group('agent_data')
            # determine num_agents from first candidate
            num_agents = candidate_agent_vars[0][1]
            for k, _ in candidate_agent_vars:
                ds = ag.create_dataset(k, (num_agents, nsteps), dtype='f4')
            # fill agent datasets per timestep
            for i, fn in enumerate(files):
                arrs = np.load(os.path.join(logdir, fn))
                for k, _ in candidate_agent_vars:
                    data = arrs[k].astype('f4')
                    ag[k][:, i] = data

        # write other variables under /records as before (timesteps, ...)
        if other_vars:
            rec = f.create_group('records')
            # create datasets for other vars based on first file shapes
            datasets = {}
            for k in other_vars:
                shape = first[k].shape
                ds_shape = (nsteps,) + shape
                datasets[k] = rec.create_dataset(k, ds_shape, dtype='f4')
            for i, fn in enumerate(files):
                arrs = np.load(os.path.join(logdir, fn))
                for k in other_vars:
                    data = arrs[k].astype('f4')
                    datasets[k][i] = data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    convert(args.logdir, args.out)


if __name__ == '__main__':
    main()
