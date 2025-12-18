import h5py
p = r"C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\Nuyakuk_Production_.p08.hdf"

def visit(name, obj):
    import numpy as np
    if isinstance(obj, h5py.Dataset):
        if name.lower().endswith('time') or 'time' in name.lower().split('/')[-1].lower():
            try:
                arr = obj[:]
                print('FOUND:', name, 'shape', getattr(arr,'shape',None), 'dtype', getattr(arr,'dtype',None))
                print(' sample first:', arr[:5])
                print(' sample last: ', arr[-5:])
            except Exception as e:
                print('FOUND but could not read:', name, e)

with h5py.File(p,'r') as h:
    print('Top level keys:', list(h.keys()))
    h.visititems(visit)
