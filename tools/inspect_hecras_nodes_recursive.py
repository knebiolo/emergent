import h5py
plan = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf"
with h5py.File(plan, 'r') as h:
    base = 'Geometry/GeomPreprocess/Node Info'
    if base not in h:
        print(base, 'not found')
    else:
        grp = h[base]
        print('Keys under', base, ':')
        for k in grp.keys():
            obj = grp[k]
            print('-', k, type(obj))
            if isinstance(obj, h5py.Dataset):
                print('   shape', obj.shape, 'dtype', obj.dtype)
                try:
                    print('   sample:', obj[:5])
                except Exception as e:
                    print('   sample error', e)
            else:
                print('   subgroup keys:', list(obj.keys()))
                for k2 in obj.keys():
                    sub = obj[k2]
                    print('    -', k2, type(sub))
                    if isinstance(sub, h5py.Dataset):
                        print('      shape', getattr(sub,'shape',None), 'dtype', getattr(sub,'dtype',None))
                        try:
                            print('      sample', sub[:5])
                        except Exception:
                            pass
