import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from emergent.salmon_abm.sockeye_SoA import simulation, compute_alongstream_raster
import numpy as np

# build the minimal sim using the profiling builder
from tools.profile_timestep_cprofile import build_sim
sim = build_sim(100)
print('HDF present?', hasattr(sim,'hdf5'))
# Ensure the attached HDF file is opened with write intent so we can create datasets
hf = getattr(sim, 'hdf5', None)
if hf is not None:
    fname = getattr(hf, 'filename', None) or getattr(hf, 'name', None)
    if fname:
        try:
            try:
                hf.close()
            except Exception:
                pass
            import h5py
            sim.hdf5 = h5py.File(fname, 'r+')
        except Exception:
            # if reopen fails, continue and the raster function will attempt a fallback
            pass

arr = compute_alongstream_raster(sim)
print('alongstream raster stats: min', np.nanmin(arr), 'max', np.nanmax(arr), 'mean', np.nanmean(arr))
arr2 = None
try:
    arr2 = None
    from emergent.salmon_abm.sockeye_SoA import compute_coarsened_alongstream_raster
    arr2 = compute_coarsened_alongstream_raster(sim, factor=4)
    print('coarsened alongstream raster stats: min', np.nanmin(arr2), 'max', np.nanmax(arr2), 'mean', np.nanmean(arr2))
except Exception as e:
    print('coarsened compute failed:', e)
