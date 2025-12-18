import h5py
from pathlib import Path
p = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\Nuyakuk_Production_.p08.hdf")
with h5py.File(p,'r') as h:
    base = 'Results/Unsteady/Output/Output Blocks/Computation Block/Global'
    if base not in h:
        print('Global group not found at', base)
    else:
        g = h[base]
        def walk(name, obj):
            import h5py
            if isinstance(obj, h5py.Group):
                print('GROUP:', name)
            else:
                print(' DATASET:', name, 'shape', getattr(obj,'shape',None), 'dtype', getattr(obj,'dtype',None))
        g.visititems(walk)
print('done')
