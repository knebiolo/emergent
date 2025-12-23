"""Inspect diagnostics group contents for a specific sim DB.

Usage:
    python tools/deep_inspect_db.py outputs/sim_db_22lmm1bc.h5
"""
import sys
import h5py

if len(sys.argv) < 2:
    print('Usage: python tools/deep_inspect_db.py <db_path>')
    sys.exit(1)

p = sys.argv[1]
print('Inspecting', p)
with h5py.File(p, 'r') as f:
    print('Full HDF5 tree (groups/datasets):')
    def visitor(name, obj):
        t = 'Group' if isinstance(obj, h5py.Group) else 'Dataset'
        print('-', name, '(', t, ')')
        # print attributes
        try:
            for ak, av in obj.attrs.items():
                print('   @', ak, '=', av)
        except Exception:
            pass
        if isinstance(obj, h5py.Dataset):
            try:
                print('   dtype:', obj.dtype, 'shape:', obj.shape)
            except Exception:
                pass

    f.visititems(visitor)
