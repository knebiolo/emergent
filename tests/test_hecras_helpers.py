import os
import tempfile
import h5py
import numpy as np

from emergent.salmon_abm.hecras_helpers import infer_wetted_perimeter_from_hecras
from emergent.salmon_abm.salmon_viewer import SalmonViewer
from emergent.salmon_abm.tin_helpers import triangulate_and_clip


def make_synthetic_hecras_hdf(path, n_cells=100, times=3):
    # simple grid coords and a depth timeseries with varying wetted cells
    coords = np.column_stack((np.linspace(0, 9, int(np.sqrt(n_cells))).repeat(int(np.sqrt(n_cells))),
                              np.tile(np.linspace(0, 9, int(np.sqrt(n_cells))), int(np.sqrt(n_cells)))))
    with h5py.File(path, 'w') as hdf:
        grp_geom = hdf.create_group('Geometry/2D Flow Areas/2D area')
        grp_geom.create_dataset('Cells Center Coordinate', data=coords)
        # Minimal facepoints/perimeter to exercise vector method may be absent; tests should accept raster fallback
        # Write a Results depth timeseries dataset: shape (times, n_cells)
        grp_res = hdf.create_group('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area')
        depths = np.zeros((times, len(coords)), dtype=np.float32)
        for t in range(times):
            # progressively flood more cells
            depths[t, : int(len(coords) * (0.2 + 0.2 * t))] = 0.2 + 0.1 * t
        grp_res.create_dataset('Cell Hydraulic Depth', data=depths)


def test_infer_wetted_perimeter_basic(tmp_path):
    p = tmp_path / 'synthetic_hecras.h5'
    make_synthetic_hecras_hdf(str(p), n_cells=100, times=4)

    # timestep 0
    perims0 = infer_wetted_perimeter_from_hecras(str(p), depth_threshold=0.1, raster_fallback_resolution=2.0, verbose=False, timestep=0)
    assert isinstance(perims0, list)
    assert len(perims0) >= 1

    # later timestep (more flooded)
    perims2 = infer_wetted_perimeter_from_hecras(str(p), depth_threshold=0.1, raster_fallback_resolution=2.0, verbose=False, timestep=2)
    assert isinstance(perims2, list)
    assert len(perims2) >= 1


def test_tin_loader_headless():
    # create a simple square grid of points with values
    xs = np.linspace(0, 10, 6)
    ys = np.linspace(0, 10, 6)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.column_stack([xv.flatten(), yv.flatten()])
    vals = np.sin(pts[:,0]*0.1) + np.cos(pts[:,1]*0.1)

    verts, faces = triangulate_and_clip(pts, vals, poly=None, max_nodes=200)
    # create payload dict like the mesh builder would
    zvals = verts[:, 2] if verts.shape[1] > 2 else np.zeros(len(verts))
    vmin, vmax = np.nanmin(zvals), np.nanmax(zvals) if len(zvals)>0 else (0,1)
    span = vmax - vmin if vmax > vmin else 1.0
    norm = (zvals - vmin) / span if len(zvals)>0 else np.zeros(len(verts))
    colors = np.zeros((len(verts), 4), dtype=float)
    if len(verts)>0:
        colors[:,0] = 0.2 + 0.8 * norm
        colors[:,1] = 0.4 * (1.0 - norm)
        colors[:,2] = 0.6 * (1.0 - norm)
        colors[:,3] = 1.0

    payload = {'verts': verts.astype(float), 'faces': faces.astype(int), 'colors': colors.astype(float)}

    # instantiate a minimal SalmonViewer (no GL required for loader)
    class DummySim:
        use_hecras = False

    sv = SalmonViewer(DummySim(), dt=0.1, T=1)
    verts_out, faces_out, colors_out = sv.load_tin_payload(payload)
    assert verts_out.shape[1] == 3
    assert faces_out.ndim == 2
    assert colors_out.shape[0] == verts_out.shape[0]
