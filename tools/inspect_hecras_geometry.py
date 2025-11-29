import h5py
import os
plan = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf"
print('Opening', plan)
with h5py.File(plan, 'r') as h:
    if 'Geometry' not in h:
        print('No Geometry group found')
    else:
        geom = h['Geometry']
        print('Geometry keys:')
        for k in geom.keys():
            obj = geom[k]
            print('-', k, 'type', type(obj))
            try:
                # if dataset
                if isinstance(obj, h5py.Dataset):
                    print('   shape', obj.shape, 'dtype', obj.dtype)
                    print('   attrs:', dict(obj.attrs))
                    try:
                        sample = obj[0:5]
                        print('   sample:', sample)
                    except Exception as e:
                        pass
                else:
                    print('   group keys:', list(obj.keys()))
            except Exception as e:
                print('   error reading', e)
