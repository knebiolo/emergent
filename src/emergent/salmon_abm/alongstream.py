import numpy as np
import h5py
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from rasterio.transform import Affine as _Affine
from rasterio.transform import Affine
from .numba_wrappers import _project_points_onto_line_numba
from .numba_wrappers import _swim_speeds_numba


def pixel_to_geo(transform, rows, cols):
    xs = transform.c + transform.a * (cols + 0.5)
    ys = transform.f + transform.e * (rows + 0.5)
    return xs, ys


def compute_alongstream_raster(simulation, outlet_xy=None, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist'):
    hdf = getattr(simulation, 'hdf5', None)
    if hdf is None:
        raise RuntimeError('simulation.hdf5 is required')

    env = hdf.get('environment')
    if env is None:
        raise RuntimeError('environment group missing in HDF')

    # read rasters
    if depth_name in env:
        depth = np.asarray(env[depth_name][:], dtype=np.float32)
        mask = np.isfinite(depth) & (depth > 0.0)
    elif wetted_name in env:
        wett = np.asarray(env[wetted_name][:])
        mask = (wett != 0)
    else:
        raise RuntimeError('Neither depth nor wetted raster found')

    try:
        t = getattr(simulation, 'depth_rast_transform', None)
        if t is None:
            t = getattr(simulation, 'vel_mag_rast_transform', None)
    except Exception:
        t = None
    if t is None:
        px = py = 1.0
    else:
        px = abs(t.a)
        py = abs(t.e)

    h, w = mask.shape

    idx = -np.ones(mask.shape, dtype=np.int32)
    mask_flat = mask.ravel()
    node_ids = np.nonzero(mask_flat)[0]
    if node_ids.size == 0:
        arr = np.full(mask.shape, np.nan, dtype=np.float32)
        env.create_dataset(out_name, data=arr, dtype='f4')
        try:
            if hasattr(hdf, 'flush'):
                hdf.flush()
        except Exception:
            pass
        return arr

    idx_flat = -np.ones(h * w, dtype=np.int32)
    idx_flat[node_ids] = np.arange(node_ids.size, dtype=np.int32)
    idx = idx_flat.reshape(h, w)

    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    rows = []
    cols = []
    data = []
    for r in range(h):
        for c in range(w):
            nid = idx[r, c]
            if nid < 0:
                continue
            for dr, dc in nbrs:
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                nid2 = idx[rr, cc]
                if nid2 < 0:
                    continue
                dist = np.hypot(dr * py, dc * px)
                rows.append(nid)
                cols.append(nid2)
                data.append(dist)

    n_nodes = node_ids.size
    graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    if outlet_xy is not None:
        ox, oy = outlet_xy
        try:
            orow, ocol = pixel_to_geo(simulation.depth_rast_transform, [oy], [ox])
            orow = int(orow[0]); ocol = int(ocol[0])
        except Exception:
            orow = None
        if orow is None or orow < 0 or orow >= h or ocol < 0 or ocol >= w or idx[orow, ocol] < 0:
            flat_xy = np.column_stack((env['x_coords'][:].ravel(), env['y_coords'][:].ravel()))
            dists = np.hypot(flat_xy[:, 0] - ox, flat_xy[:, 1] - oy)
            cand = np.argmin(dists)
            if mask_flat[cand]:
                outlet_nodes = [idx_flat[cand]]
            else:
                wett_inds = np.nonzero(mask_flat)[0]
                nearest = wett_inds[np.argmin(dists[wett_inds])]
                outlet_nodes = [idx_flat[nearest]]
        else:
            outlet_nodes = [int(idx[orow, ocol])]
    else:
        xcoords = env['x_coords'][:]
        ycoords = env['y_coords'][:]
        flat_y = ycoords.ravel()
        wett_inds = np.nonzero(mask_flat)[0]
        if wett_inds.size == 0:
            outlet_nodes = [0]
        else:
            out_ind = wett_inds[np.argmin(flat_y[wett_inds])]
            outlet_nodes = [int(idx_flat[out_ind])]

    dist_matrix = dijkstra(csgraph=graph, directed=False, indices=outlet_nodes)
    if dist_matrix.ndim == 2:
        dist = dist_matrix.min(axis=0)
    else:
        dist = dist_matrix

    out_arr = np.full(h * w, np.nan, dtype=np.float32)
    out_arr[node_ids] = dist.astype(np.float32)
    out_arr = out_arr.reshape(h, w)

    wrote = False
    try:
        if out_name in env:
            env[out_name][:] = out_arr
        else:
            env.create_dataset(out_name, data=out_arr, dtype='f4')
        try:
            if hasattr(hdf, 'flush'):
                hdf.flush()
        except Exception:
            pass
        wrote = True
    except (TypeError, ValueError, RuntimeError):
        fname = getattr(hdf, 'filename', None) or getattr(hdf, 'name', None)
        if fname:
            try:
                with h5py.File(fname, 'r+') as hw:
                    envw = hw.require_group('environment')
                    if out_name in envw:
                        envw[out_name][:] = out_arr
                    else:
                        envw.create_dataset(out_name, data=out_arr, dtype='f4')
                    try:
                        hw.flush()
                    except Exception:
                        pass
                    wrote = True
            except Exception:
                wrote = False
        if not wrote:
            pass
    return out_arr


def compute_coarsened_alongstream_raster(simulation, factor=4, outlet_xy=None, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist'):
    hdf = getattr(simulation, 'hdf5', None)
    if hdf is None:
        raise RuntimeError('simulation.hdf5 is required')
    env = hdf.get('environment')
    if env is None:
        raise RuntimeError('environment group missing in HDF')

    if depth_name in env:
        depth = np.asarray(env[depth_name][:], dtype=np.float32)
        mask = np.isfinite(depth) & (depth > 0.0)
    elif wetted_name in env:
        wett = np.asarray(env[wetted_name][:])
        mask = (wett != 0)
    else:
        raise RuntimeError('Neither depth nor wetted raster found')

    h, w = mask.shape
    ch = max(1, h // factor)
    cw = max(1, w // factor)

    depth_coarse = np.full((ch, cw), np.nan, dtype=np.float32)
    mask_coarse = np.zeros((ch, cw), dtype=bool)
    for i in range(ch):
        for j in range(cw):
            r0 = i * factor
            c0 = j * factor
            block = mask[r0:r0 + factor, c0:c0 + factor]
            if np.any(block):
                mask_coarse[i, j] = True
                if depth_name in env:
                    db = depth[r0:r0 + factor, c0:c0 + factor]
                    vals = db[np.isfinite(db) & (db > 0.0)]
                    if vals.size:
                        depth_coarse[i, j] = float(np.mean(vals))
                    else:
                        depth_coarse[i, j] = np.nan
                else:
                    depth_coarse[i, j] = 1.0

    class _MiniSim:
        pass
    minisim = _MiniSim()
    try:
        t = getattr(simulation, 'depth_rast_transform', None)
        if t is None:
            t = getattr(simulation, 'vel_mag_rast_transform', None)
    except Exception:
        t = None
    if t is None:
        tcoarse = _Affine.scale(1, -1)
    else:
        tcoarse = _Affine(t.a * factor, t.b, t.c, t.d, t.e * factor, t.f)

    minisim.depth_rast_transform = tcoarse
    fname = getattr(hdf, 'filename', None) or getattr(hdf, 'name', None)
    if not fname:
        return compute_alongstream_raster(simulation, outlet_xy=outlet_xy, depth_name=depth_name, wetted_name=wetted_name, out_name=out_name)

    tmp_name = 'tmp_coarse'
    with h5py.File(fname, 'r+') as hw:
        if tmp_name in hw:
            del hw[tmp_name]
        g = hw.create_group(tmp_name)
        envc = g.create_group('environment')
        envc.create_dataset('depth', data=depth_coarse.astype('f4'))
        envc.create_dataset('wetted', data=mask_coarse.astype('i1'))
        cols = np.arange(cw, dtype=np.float32)
        rows = np.arange(ch, dtype=np.float32)
        colg, rowg = np.meshgrid(cols, rows)
        xs, ys = pixel_to_geo(tcoarse, rowg, colg)
        envc.create_dataset('x_coords', data=xs.astype('f4'))
        envc.create_dataset('y_coords', data=ys.astype('f4'))
        minisim.hdf5 = hw[tmp_name]
        minisim.depth_rast_transform = tcoarse
        coarse_out = compute_alongstream_raster(minisim, outlet_xy=outlet_xy, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist')
        upsampled = np.repeat(np.repeat(coarse_out, factor, axis=0), factor, axis=1)
        upsampled = upsampled[:h, :w]
        try:
            env = hw.require_group('environment')
            if out_name in env:
                try:
                    env[out_name][:] = upsampled.astype('f4')
                except Exception:
                    del env[out_name]
                    env.create_dataset(out_name, data=upsampled.astype('f4'), dtype='f4')
            else:
                env.create_dataset(out_name, data=upsampled.astype('f4'), dtype='f4')
            hw.flush()
        except Exception:
            pass

    try:
        with h5py.File(fname, 'r+') as hw2:
            if tmp_name in hw2:
                del hw2[tmp_name]
    except Exception:
        pass

    return upsampled.astype(np.float32)
