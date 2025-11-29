import h5py
plan = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf"
coords_found = []
with h5py.File(plan, 'r') as h:
    def walk(g, path='/'):
        for k in g.keys():
            obj = g[k]
            p = path + k
            if isinstance(obj, h5py.Dataset):
                dt = obj.dtype
                # identify numeric float datasets
                if hasattr(dt, 'kind') and dt.kind == 'f':
                    coords_found.append((p, obj.shape, obj.dtype))
                # structured arrays that contain float fields
                if dt.names:
                    float_fields = [n for n in dt.names if dt[n].kind == 'f']
                    if float_fields:
                        coords_found.append((p, obj.shape, obj.dtype, float_fields))
            else:
                walk(obj, p + '/')
    walk(h['Geometry'], '/Geometry/')

print('Candidate coordinate datasets:')
for c in coords_found[:50]:
    print(c)
