"""Inspect HECRAS HDF and print dataset paths, shapes and dtypes that match hydraulic keywords."""
import h5py
from pathlib import Path
plan = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf")
print('Plan:', plan)
keywords = ('water','wsel','velocity','vel','elev','depth','minimum elevation','cell velocity','water surface','cell points')
with h5py.File(plan,'r') as h:
    matches = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            p = name.lower()
            if any(k in p for k in keywords):
                try:
                    s = obj.shape
                except Exception:
                    s = None
                matches.append((name, s, getattr(obj, 'dtype', None)))
    h.visititems(visitor)

for name, shape, dtype in sorted(matches):
    print(name, shape, dtype)
print('\nTotal matches:', len(matches))
