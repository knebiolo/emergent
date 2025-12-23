#!/usr/bin/env python3
"""Create a small dummy HDF5 file with a positions dataset for viewer testing."""
import os
import numpy as np
import h5py

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'sim_db_test.h5')

def main(path=OUT):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    T = 200
    N = 50
    # create simple circular motion trajectories with small noise
    t = np.linspace(0, 2 * np.pi, T)
    positions = np.zeros((T, N, 2), dtype=np.float32)
    for i in range(N):
        r = 50.0 + 5.0 * (i / max(1, N-1))
        phase = (i / N) * 2 * np.pi
        positions[:, i, 0] = r * np.cos(t + phase) + 500.0  # x
        positions[:, i, 1] = r * np.sin(t + phase) + 1000.0  # y

    with h5py.File(path, 'w') as f:
        f.create_dataset('positions', data=positions, compression='gzip')
    print('Wrote', path, 'positions shape', positions.shape)


if __name__ == '__main__':
    main()
