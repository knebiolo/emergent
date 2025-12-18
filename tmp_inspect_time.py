import h5py
p = r"C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\Nuyakuk_Production_.p08.hdf"
with h5py.File(p,'r') as h:
    tpath = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Time'
    if tpath in h:
        t = h[tpath][:]
        print('Time shape:', getattr(t,'shape',None))
        print('First 5:', t[:5])
        print('Last 5:', t[-5:])
    else:
        print('Time dataset not found at expected path')
        # list children
        base = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area'
        print('children:', list(h[base].keys()))
