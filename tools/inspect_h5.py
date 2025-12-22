"""Simple HDF5 inspection utility for simulation outputs.

Usage: python tools/inspect_h5.py <path_to_h5>
"""
import sys
import h5py
import numpy as np


def summarize_dataset(ds):
    try:
        arr = ds[()]
        return {'shape': arr.shape, 'dtype': str(arr.dtype), 'min': float(np.nanmin(arr)), 'max': float(np.nanmax(arr))}
    except Exception:
        return {'shape': getattr(ds, 'shape', None), 'dtype': getattr(ds, 'dtype', None)}


def main():
    if len(sys.argv) < 2:
        print('Usage: python tools/inspect_h5.py <path_to_h5>')
        sys.exit(1)
    path = sys.argv[1]
    with h5py.File(path, 'r') as f:
        print('Top-level groups/datasets:')
        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                s = summarize_dataset(obj)
                print(f"- {name}: {s}")
            else:
                print(f"- {name}: group")
        f.visititems(print_item)


if __name__ == '__main__':
    main()
