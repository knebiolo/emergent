import h5py
plan = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf"
with h5py.File(plan, 'r') as h:
    path = 'Geometry/GeomPreprocess/Node Info'
    if path in h:
        ds = h[path]
        print('Dataset', path, 'shape', ds.shape, 'dtype', ds.dtype)
        print('Attributes:', dict(ds.attrs))
        # print first 10 rows
        try:
            sample = ds[:10]
            print('Sample rows (first 10):')
            for r in sample:
                print(r)
        except Exception as e:
            print('Could not read sample rows:', e)
    else:
        print(path, 'not found, available keys:', list(h['Geometry/GeomPreprocess'].keys()))
