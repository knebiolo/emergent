import h5py
from pathlib import Path
p = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\Nuyakuk_Production_.p08.hdf")
print('File:', p)
with h5py.File(p,'r') as h:
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and 'Cell Hydraulic Depth' in name:
            try:
                arr = obj
                shape = getattr(arr, 'shape', None)
                print('\nFOUND dataset:', name)
                print(' shape:', shape, 'dtype:', arr.dtype)
                if arr.size > 0 and arr.ndim >= 2:
                    # print first time index sample and last
                    print(' sample first row shape:', arr[0].shape if arr.shape[0] > 0 else None)
                    try:
                        print(' first values:', arr[0][:5])
                        print(' last values:', arr[-1][:5])
                    except Exception as e:
                        print(' could not sample rows:', e)
                elif arr.size > 0:
                    try:
                        print(' sample:', arr[:5])
                    except Exception as e:
                        print(' could not sample:', e)
            except Exception as ex:
                print('Error reading dataset', name, ex)
    h.visititems(visitor)
print('\nDone')
