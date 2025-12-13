from emergent.salmon_abm.hecras_helpers import infer_wetted_perimeter_from_hecras
from pathlib import Path
import h5py, numpy as np

p = Path('tmp/debug_synthetic.h5')
def make_synthetic_hecras_hdf(path, n_cells=100, times=3):
	coords = np.column_stack((np.linspace(0, 9, int(np.sqrt(n_cells))).repeat(int(np.sqrt(n_cells))),
							  np.tile(np.linspace(0, 9, int(np.sqrt(n_cells))), int(np.sqrt(n_cells)))))
	with h5py.File(path, 'w') as hdf:
		grp_geom = hdf.create_group('Geometry/2D Flow Areas/2D area')
		grp_geom.create_dataset('Cells Center Coordinate', data=coords)
		grp_res = hdf.create_group('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area')
		depths = np.zeros((times, len(coords)), dtype=np.float32)
		for t in range(times):
			depths[t, : int(len(coords) * (0.2 + 0.2 * t))] = 0.2 + 0.1 * t
		grp_res.create_dataset('Cell Hydraulic Depth', data=depths)

make_synthetic_hecras_hdf(str(p), n_cells=100, times=4)
res = infer_wetted_perimeter_from_hecras(str(p), depth_threshold=0.1, raster_fallback_resolution=2.0, verbose=True, timestep=0)
print('Result type:', type(res))
print('Result:', res)
