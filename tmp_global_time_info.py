import h5py
p = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\Nuyakuk_Production_.p08.hdf"
with h5py.File(p,'r') as h:
    gp = 'Results/Unsteady/Output/Output Blocks/Computation Block/Global/Time'
    if gp in h:
        t = h[gp][:]
        print('global time shape', t.shape)
        print('first 5', t[:5])
        print('last 5', t[-5:])
    else:
        print('global time not found at', gp)
