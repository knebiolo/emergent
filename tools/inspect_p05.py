import h5py
p = r'data/salmon_abm/20240506/Nuyakuk_Production_.p05.hdf'
with h5py.File(p,'r') as h:
    print('root keys:', list(h.keys()))
    if 'environment' in h:
        print('environment keys:', list(h['environment'].keys()))
    else:
        # print first-level groups and some contents
        for k in list(h.keys()):
            print(k, '->', type(h[k]), 'children:', list(h[k].keys())[:10])
