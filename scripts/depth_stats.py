import h5py
import numpy as np
from pathlib import Path
p=Path('data/Nuyakuk_Production_.p08.hdf')
with h5py.File(p,'r') as h:
    area_depth = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'
    area_time = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Time'
    if area_depth not in h:
        print('area depth dataset missing')
        raise SystemExit(1)
    depths = h[area_depth][:]  # shape (nt, ncell)
    # find a matching Time dataset under the same area path
    if area_time in h:
        times = h[area_time][:]
    else:
        # fallback: search for a Time dataset in the file
        times = None
        for name in h:
            pass
        # try the known global path
        if 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time' in h:
            times = h['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time'][:]
        else:
            print('Could not locate area Time dataset; aborting')
            raise SystemExit(1)
    print('depths shape', depths.shape)
    # use same mapping behavior as script
    global_time = h['Results/Unsteady/Output/Output Blocks/Computation Block/Global/Time'][:]
    gidx = 162000
    gval = float(global_time[gidx])
    # find nearest area index
    aidx = int(np.argmin(np.abs(times - gval)))
    arr = depths[aidx]
    print('using area index', aidx, 'area time', float(times[aidx]), 'global time', gval)
    arr_nonan = arr[~np.isnan(arr)]
    print('count cells:', arr.shape[0])
    print('NaNs:', np.isnan(arr).sum())
    print('min/median/mean/90p/99p/max:', float(np.nanmin(arr_nonan)), float(np.nanpercentile(arr_nonan,50)), float(np.nanmean(arr_nonan)), float(np.nanpercentile(arr_nonan,90)), float(np.nanpercentile(arr_nonan,99)), float(np.nanmax(arr_nonan)))
    thresh=0.05
    print('count <', thresh, ':', int((arr_nonan < thresh).sum()))
    print('percent <', thresh, ':', float((arr_nonan < thresh).sum())/arr_nonan.size)
