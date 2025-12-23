"""Check HDF5 DB schema against expected datasets from sockeye.py

Run:
    python tools/check_db_schema.py /path/to/sim_db.h5

Prints presence and shapes of key datasets.
"""
import sys
import h5py
import os

EXPECTED = [
    # per-agent scalar datasets (1-D)
    'agent_data/sex',
    'agent_data/length',
    'agent_data/ucrit',
    'agent_data/weight',
    'agent_data/body_depth',
    'agent_data/too_shallow',
    'agent_data/opt_wat_depth',
    # environment rasters
    'environment/depth',
    'environment/vel_x',
    'environment/vel_y',
    'environment/vel_dir',
    'environment/vel_mag',
    # per-agent time-series (num_agents x num_timesteps)
    'agent_data/X',
    'agent_data/Y',
    'agent_data/Hz',
    'agent_data/thrust',
    'agent_data/drag',
    'agent_data/x_vel',
    'agent_data/y_vel',
    'agent_data/heading',
]


def check_db(path):
    if not os.path.exists(path):
        print('DB not found:', path)
        return 2
    with h5py.File(path, 'r') as f:
        print('Checking DB:', path)
        ok = True
        for name in EXPECTED:
            if name in f:
                ds = f[name]
                try:
                    shape = ds.shape
                except Exception:
                    # attributes or groups
                    shape = getattr(ds, 'shape', 'group/attr')
                print(f'  + {name}: present, shape={shape}')
            else:
                print(f'  - {name}: MISSING')
                ok = False
        return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/check_db_schema.py /path/to/sim_db.h5')
        sys.exit(2)
    sys.exit(check_db(sys.argv[1]))
