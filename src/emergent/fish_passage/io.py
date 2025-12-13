"""Minimal HECRAS IO helpers for fish_passage.

Compact, test-focused implementations used during migration. Keep
behaviour deterministic and small so unit tests can validate core IO
logic without depending on legacy complexity.
"""

from typing import List, Dict, Any, Optional, Tuple, Sequence
import h5py
import numpy as np
import logging

from emergent.fish_passage.utils import safe_build_kdtree as _safe_build_kdtree, safe_log_exception as _safe_log_exception

logger = logging.getLogger(__name__)
"""HECRAS IO helpers for fish_passage.

This module provides a compact, test-focused set of helpers for reading
HECRAS HDF5 plan files, building a KDTree of cell centers, and mapping
query points to HECRAS fields using inverse-distance weighting (IDW).

The implementations aim for parity with legacy helpers in
`emergent.salmon_abm.sockeye` but remain small and deterministic so
unit tests can validate behavior during migration.
"""

from typing import List, Dict, Any, Optional, Tuple, Sequence
import h5py
import numpy as np
import logging

from emergent.fish_passage.utils import safe_build_kdtree as _safe_build_kdtree, safe_log_exception as _safe_log_exception
from emergent.fish_passage.geometry import compute_affine_from_hecras, pixel_to_geo
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class HECRASMap:
    """Legacy-parity HECRASMap ported from `salmon_abm.sockeye`.

    Behavior matches the original implementation:
    - dataset discovery via substring match, preferring 'Results' paths
    - timestep handling when datasets are time-series
    - normalization to align field arrays with coords length
    - primary-field masking (first requested field) to remove invalid cells
    - KDTree built via `_safe_build_kdtree`
    - `map_idw` returns dict[field_name] -> (N,) ndarray
    """

    def __init__(self, plan_path: str, field_names: Optional[Sequence[str]] = None, timestep: int = 0):
        self.plan_path = str(plan_path)
        self.timestep = int(timestep) if timestep is not None else 0
        # Preserve whether the caller passed a single-string for legacy behavior
        was_string = isinstance(field_names, str)
        if field_names is None:
            field_names = ['Cells Minimum Elevation']
        elif was_string:
            field_names = [field_names]
        self.field_names = list(field_names)
        self._return_single = was_string
        self._load_plan()

    def _find_dataset_by_name(self, hdf: h5py.File, name_pattern: str) -> Optional[str]:
        name_pattern = name_pattern.lower()
        candidates = []

        def visitor(path, obj):
            if isinstance(obj, h5py.Dataset):
                p = path.lower()
                if name_pattern in p or name_pattern in obj.name.lower():
                    try:
                        shape = obj.shape
                    except Exception:
                        shape = None
                    candidates.append((path, shape))

        hdf.visititems(visitor)
        if not candidates:
            return None

        results_cands = [c for c in candidates if 'results/' in c[0].lower()]
        if results_cands:
            return results_cands[0][0]
        return candidates[0][0]

    def _load_plan(self) -> None:
        with h5py.File(self.plan_path, 'r') as h:
            coords = h['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]

            n_coords = coords.shape[0]

            fields = {}
            for fname in self.field_names:
                geom_path = f'/Geometry/2D Flow Areas/2D area/{fname}'
                if geom_path in h:
                    node = h[geom_path]
                    # If the path points to a dataset, take it. If it's a group, try common child names.
                    if isinstance(node, h5py.Dataset):
                        arr = node[:]
                    else:
                        # prefer 'Values' child dataset
                        if 'Values' in node:
                            arr = node['Values'][:]
                        else:
                            # fall back to first dataset inside the group
                            found = None
                            for name, obj in node.items():
                                if isinstance(obj, h5py.Dataset):
                                    found = obj
                                    break
                            if found is not None:
                                arr = found[:]
                            else:
                                arr = np.array([])
                else:
                    ds_path = self._find_dataset_by_name(h, fname)
                    if ds_path is not None:
                        ds = h[ds_path]
                        # read full dataset and let heuristics pick the right slice
                        data = np.asarray(ds[:])
                        if data.ndim == 2:
                            # Prefer interpretation where one axis matches n_coords
                            if data.shape[1] == n_coords and data.shape[0] > 1:
                                # likely (timesteps, n_cells)
                                t = min(self.timestep, data.shape[0] - 1)
                                arr = data[t]
                            elif data.shape[0] == n_coords and data.shape[1] >= 1:
                                # likely (n_cells, features)
                                arr = data[:, 0]
                            else:
                                # fallback: attempt to flatten conservatively
                                arr = data.reshape(-1)
                        else:
                            arr = data
                    else:
                        raise KeyError(f"Field '{fname}' not found in HECRAS HDF: {self.plan_path}")
                fields[fname] = np.asarray(arr)

        # normalize field arrays to align with coords length

        def normalize_field_array(arr):
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.shape[0] == n_coords:
                return arr
            if arr.size == n_coords:
                return arr.reshape(n_coords,)
            for axis, dim in enumerate(arr.shape):
                if dim == n_coords:
                    idx = []
                    for i in range(arr.ndim):
                        if i == axis:
                            idx.append(slice(None))
                        else:
                            idx.append(-1)
                    sliced = arr[tuple(idx)]
                    return np.asarray(sliced).reshape(n_coords,)
            return np.full((n_coords,), np.nan)

        primary = self.field_names[0]
        normed = {k: normalize_field_array(v) for k, v in fields.items()}
        mask = np.isfinite(normed[primary])

        self.coords = coords[mask].astype(np.float64)
        self.fields = {k: np.asarray(v[mask], dtype=np.float64) for k, v in normed.items()}
        self.tree = _safe_build_kdtree(self.coords, name='hecras_plan_tree')
        if self.tree is None:
            try:
                logger.warning('HECRAS plan: KDTree build failed; certain queries will be disabled')
            except Exception as e:
                _safe_log_exception('Failed while logging KDTree build warning', e, file='io.py')

    def map_idw(self, query_pts, k=8, eps=1e-8):
        query = np.asarray(query_pts, dtype=np.float64)
        if query.ndim == 1:
            query = query.reshape(1, 2)
        if getattr(self, 'tree', None) is None:
            raise RuntimeError('IDW mapping requested but KDTree is unavailable (HECRAS plan tree build failed)')
        dists, inds = self.tree.query(query, k=k)
        if k == 1:
            dists = dists[:, None]
            inds = inds[:, None]
        inv = 1.0 / (dists + eps)
        w = inv / np.sum(inv, axis=1)[:, None]
        out = {}
        for fname, arr in self.fields.items():
            vals = arr[inds]
            mapped = np.sum(vals * w, axis=1)
            out[fname] = mapped
        # Legacy behavior: if caller passed a single-string, return ndarray
        if getattr(self, '_return_single', False) and len(self.field_names) == 1:
            return out[self.field_names[0]]
        if len(self.field_names) == 1:
            return out
        return out


def map_hecras_for_agents(simulation_or_plan, pts: np.ndarray, plan_path: Optional[str] = None, field_names: List[str] = None, k: int = 8):
    """Wrapper supporting two calling patterns:

    - Legacy: map_hecras_for_agents(simulation, agent_xy, plan_path, field_names=..., k=...)
    - Simple: map_hecras_for_agents(plan_path, agent_xy, field_names=..., k=...)
    """
    # If first arg is a string, treat as (plan_path, pts, ...)
    if isinstance(simulation_or_plan, str):
        plan = simulation_or_plan
        points = pts
        m = HECRASMap(plan, field_names=field_names)
        return m.map_idw(points, k=k)

    # Otherwise expect a simulation-like object with _hecras_maps registered
    sim = simulation_or_plan
    # plan_path may be provided as third positional arg
    plan = str(plan_path) if plan_path is not None else getattr(sim, 'hecras_plan_path', '')
    key = (plan, tuple(field_names) if field_names is not None else None)
    maps = getattr(sim, '_hecras_maps', None)
    if not maps or key not in maps:
        raise KeyError(f"No adapter registered for plan {plan} and fields {field_names}")
    adapter = maps[key]
    return adapter.map_idw(pts, k=k)


def initialize_hecras_geometry(sim: Any, plan_path: str, depth_threshold: float = 0.05, create_rasters: bool = False) -> Dict[str, Any]:
    """Minimal initializer used by tests. Reads basic datasets from an
    HDF5-like plan (duck-typed) and registers coordinates on `sim.hdf5`.

    Parameters
    - sim: simulation object with `hdf5` attribute (h5py.File-like)
    - plan_path: path to an on-disk HECRAS plan HDF5 file
    - depth_threshold: threshold used to decide wetted cells (unused in skeleton)
    - create_rasters: whether to create rasters in `sim.hdf5` (basic behavior)

    Returns a dict containing `coords` and simple metadata.
    """
    with h5py.File(plan_path, 'r') as f:
        key = 'Geometry/2D Flow Areas/2D area/Cells Center Coordinate'
        if key in f:
            coords = np.asarray(f[key])
        else:
            raise RuntimeError(f"Missing expected geometry dataset in {plan_path}")

    xs = coords[:, 0]
    ys = coords[:, 1]
    # create or replace lightweight x/y coord datasets on sim.hdf5
    try:
        if 'x_coords' in sim.hdf5:
            del sim.hdf5['x_coords']
    except Exception:
        pass
    try:
        if 'y_coords' in sim.hdf5:
            del sim.hdf5['y_coords']
    except Exception:
        pass
    sim.hdf5.create_dataset('x_coords', data=xs)
    sim.hdf5.create_dataset('y_coords', data=ys)
    return {'coords': coords, 'n_cells': coords.shape[0]}


def ensure_hdf_coords_from_hecras(sim: Any, plan_path: str, target_shape: Optional[Tuple[int, int]] = None) -> None:
    """Populate `sim.hdf5` with x/y coordinate datasets derived from HECRAS plan.

    This implementation follows legacy behavior: if the plan provides cell
    centers, use them; otherwise create a simple regular grid of requested
    `target_shape` (or 10x10 default). It does not overwrite existing
    datasets.
    """
    # If already present, nothing to do
    if 'x_coords' in sim.hdf5 and 'y_coords' in sim.hdf5:
        return

    coords = None
    try:
        with h5py.File(str(plan_path), 'r') as ph:
            coords = np.asarray(ph['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
    except KeyError:
        coords = None
    except Exception as e:
        _safe_log_exception('Failed opening HECRAS plan in ensure_hdf_coords_from_hecras', e, file='io.py')
        coords = None

    if coords is None:
        # fallback: build a simple grid
        if target_shape is None:
            nx, ny = 10, 10
        else:
            ny, nx = target_shape
        xs = np.linspace(0.0, 1.0, nx * ny).reshape((ny, nx))
        ys = np.linspace(0.0, 1.0, nx * ny).reshape((ny, nx))
    else:
        # If a target_shape is provided, rasterize coords into a regular grid
        if target_shape is not None:
            height, width = target_shape
            aff = compute_affine_from_hecras(coords)
            cols = np.arange(width, dtype=np.float64)
            rows = np.arange(height, dtype=np.float64)
            col_grid, row_grid = np.meshgrid(cols, rows)
            xs_grid, ys_grid = pixel_to_geo(aff, row_grid, col_grid)
            xs = np.asarray(xs_grid)
            ys = np.asarray(ys_grid)
        else:
            # If coords are a perfect square number, reshape to (side, side)
            n = coords.shape[0]
            side = int(np.round(np.sqrt(n)))
            if side * side == n:
                try:
                    xs = coords[:, 0].reshape((side, side))
                    ys = coords[:, 1].reshape((side, side))
                except Exception:
                    xs = coords[:, 0]
                    ys = coords[:, 1]
            else:
                xs = coords[:, 0]
                ys = coords[:, 1]

    # create datasets; if already exist, leave them
    if 'x_coords' not in sim.hdf5:
        sim.hdf5.create_dataset('x_coords', data=xs)
    if 'y_coords' not in sim.hdf5:
        sim.hdf5.create_dataset('y_coords', data=ys)


def map_hecras_to_env_rasters(sim: Any, plan_path: str, field_names: Sequence[str], k: int = 1) -> bool:
    """Map HECRAS fields onto the simulation raster grid and write into `simulation.hdf5['environment']`.

    Expects a mapping adapter registered on `sim._hecras_maps[(plan_path, tuple(field_names))]`
    which implements `.map_idw(pts, k=...)`.
    """
    maps = getattr(sim, '_hecras_maps', None)
    plan_key = str(plan_path)
    key_candidates = [(plan_key, tuple(field_names)), ('', tuple(field_names))]
    adapter = None
    if maps is not None:
        for k in key_candidates:
            if k in maps:
                adapter = maps[k]
                break
    if adapter is None:
        raise KeyError(f"No adapter registered for plan {plan_path} and fields {field_names}")

    # Ensure simulation raster grid xy is prepared
    if 'x_coords' in sim.hdf5 and 'y_coords' in sim.hdf5:
        xarr = np.asarray(sim.hdf5['x_coords'])
        yarr = np.asarray(sim.hdf5['y_coords'])
        h, w = xarr.shape
        XX = xarr.flatten()
        YY = yarr.flatten()
        grid_xy = np.column_stack((XX, YY))
        sim._hecras_grid_shape = (h, w)
        sim._hecras_grid_xy = grid_xy
    else:
        raise RuntimeError('x_coords/y_coords not present in simulation.hdf5; call ensure_hdf_coords_from_hecras first')

    env = sim.hdf5.require_group('environment')
    h, w = sim._hecras_grid_shape
    for fname in (field_names or []):
        try:
            # prefer using the adapter directly
            mapped = adapter.map_idw(sim._hecras_grid_xy, k=k)
            # adapter may return an ndarray for single-field adapters or a dict
            if isinstance(mapped, dict):
                arr = np.asarray(mapped.get(fname))
            else:
                arr = np.asarray(mapped)
            if arr is None:
                raise RuntimeError('Adapter returned no data')
            if arr.size != h * w:
                raise RuntimeError('Mapped array size mismatch')
        except Exception:
            arr = np.full((h * w,), np.nan, dtype=float)

        if fname in env:
            del env[fname]
        env.create_dataset(fname, (h, w), dtype='f4')
        env[fname][:, :] = arr.reshape(h, w)

    return True


def infer_wetted_perimeter_from_hecras(hdf_path_or_file, depth_threshold=0.05, max_nodes=5000, raster_fallback_resolution=5.0, verbose=False, timestep=0):
    # Vector-only implementation: read facepoints/perimeter/face-info and
    # extract ordered rings. No try/except blocks — errors will propagate.
    close_file = False
    if isinstance(hdf_path_or_file, str):
        hdf = h5py.File(hdf_path_or_file, 'r')
        close_file = True
    else:
        hdf = hdf_path_or_file

    # Read depth dataset (expecting the standard long path)
    ds_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'
    ds = hdf[ds_path]
    data = np.asarray(ds[:])
    if data.ndim > 1 and data.shape[0] > 1:
        t = int(min(timestep, data.shape[0] - 1))
        depth = np.asarray(data[t])
    else:
        depth = np.asarray(data).reshape(-1)

    # read geometry datasets required by vector method
    facepoints = np.asarray(hdf['Geometry/2D Flow Areas/2D area/FacePoints Coordinate'])
    is_perim = np.asarray(hdf['Geometry/2D Flow Areas/2D area/FacePoints Is Perimeter'])
    face_info = np.asarray(hdf['Geometry/2D Flow Areas/2D area/Cells Face and Orientation Info']).astype(int)

    wetted_mask = depth > float(depth_threshold)
    wetted_idx = np.nonzero(wetted_mask)[0]

    perim_mask = (is_perim == -1)
    perim_touch = np.zeros(len(facepoints), dtype=bool)

    for i in wetted_idx:
        start, count = face_info[i]
        idxs = np.arange(start, start + count)
        idxs = idxs[idxs < len(perim_mask)]
        perim_touch[idxs] |= perim_mask[idxs]

    if not perim_touch.any():
        raise RuntimeError('No perimeter facepoints touched by wetted cells — vector method cannot proceed')

    perim_coords = np.asarray(hdf['Geometry/2D Flow Areas/2D area/Perimeter'])

    # Map perimeter coords to facepoints by nearest neighbor
    tree = _safe_build_kdtree(facepoints, name='facepoints_tree')
    if tree is not None:
        dists, idxs = tree.query(perim_coords, k=1)
    else:
        dif = perim_coords[:, None, :] - facepoints[None, :, :]
        dists = np.sqrt(np.sum(dif * dif, axis=2))
        idxs = np.argmin(dists, axis=1)

    touched = perim_touch[idxs]
    if not np.any(touched):
        raise RuntimeError('Perimeter points mapping found no touched points — vector method cannot proceed')

    # Extract contiguous runs of True in touched to form rings
    rings = []
    cur = []
    for flag, coord in zip(touched, perim_coords):
        if flag:
            cur.append(tuple(coord))
        else:
            if cur:
                rings.append(cur)
                cur = []
    if cur:
        rings.append(cur)

    # Convert rings to polygons and union them
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polys = [Polygon(r) for r in rings if len(r) >= 3]
    if not polys:
        raise RuntimeError('No valid perimeter polygons from vector method')
    merged = unary_union(polys)

    out_rings = []
    if merged.geom_type == 'Polygon':
        out_rings = [np.asarray(merged.exterior.coords)]
    else:
        out_rings = [np.asarray(g.exterior.coords) for g in merged.geoms]

    if close_file:
        hdf.close()
    return out_rings


def compute_alongstream_raster(simulation, outlet_xy=None, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist'):
    """Compute along-stream distance raster (Dijkstra on 8-neighbor graph).

    Simplified, deterministic port of legacy function. Writes result to
    `simulation.hdf5['environment'][out_name]` and returns the 2D array.
    """
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

    # pixel spacing
    t = getattr(simulation, 'depth_rast_transform', None) or getattr(simulation, 'vel_mag_rast_transform', None)
    if t is None:
        px = py = 1.0
    else:
        px = abs(t.a)
        py = abs(t.e)

    h, w = mask.shape
    idx_flat = -np.ones(h * w, dtype=np.int32)
    mask_flat = mask.ravel()
    node_ids = np.nonzero(mask_flat)[0]
    if node_ids.size == 0:
        arr = np.full(mask.shape, np.nan, dtype=np.float32)
        env.create_dataset(out_name, data=arr, dtype='f4')
        return arr

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

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    n_nodes = node_ids.size
    graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # determine outlet node(s)
    if outlet_xy is not None:
        ox, oy = outlet_xy
        try:
            orow, ocol = geo_to_pixel(simulation.depth_rast_transform, [oy], [ox])
            orow = int(orow[0]); ocol = int(ocol[0])
        except Exception:
            orow = None
        if orow is None or orow < 0 or orow >= h or ocol < 0 or ocol >= w or idx[orow, ocol] < 0:
            flat_xy = np.column_stack((env['x_coords'][:].ravel(), env['y_coords'][:].ravel()))
            dists = np.hypot(flat_xy[:,0] - ox, flat_xy[:,1] - oy)
            cand = np.argmin(dists)
            if mask_flat[cand]:
                outlet_nodes = [int(idx_flat[cand])]
            else:
                wett_inds = np.nonzero(mask_flat)[0]
                nearest = wett_inds[np.argmin(dists[wett_inds])]
                outlet_nodes = [int(idx_flat[nearest])]
        else:
            outlet_nodes = [int(idx[orow, ocol])]
    else:
        flat_y = env['y_coords'][:].ravel()
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

    if out_name in env:
        env[out_name][:] = out_arr
    else:
        env.create_dataset(out_name, data=out_arr, dtype='f4')

    return out_arr


def compute_coarsened_alongstream_raster(simulation, factor=2, depth_name='depth', wetted_name='wetted', out_name='along_stream_dist_coarse'):
    """Compute along-stream raster on a coarsened grid and upsample back to original resolution.

    This helper creates temporary coarsened `environment` datasets, calls
    `compute_alongstream_raster` on the smaller grid (using the same simulation
    object but with adjusted `x_coords`/`y_coords` datasets), then bilinearly
    resamples the result back to the original shape.
    """
    hdf = getattr(simulation, 'hdf5', None)
    if hdf is None:
        raise RuntimeError('simulation.hdf5 is required')
    env = hdf.get('environment')
    if env is None:
        raise RuntimeError('environment group missing in HDF')

    if 'x_coords' not in hdf or 'y_coords' not in hdf:
        raise RuntimeError('x_coords/y_coords required in simulation.hdf5')

    xarr = np.asarray(hdf['x_coords'])
    yarr = np.asarray(hdf['y_coords'])
    h, w = xarr.shape
    # coarsened size
    ch = max(1, h // factor)
    cw = max(1, w // factor)

    # compute coarse pixel centers by sampling underlying coords
    cols = np.linspace(0, w - 1, cw, dtype=int)
    rows = np.linspace(0, h - 1, ch, dtype=int)
    col_grid, row_grid = np.meshgrid(cols, rows)
    xs_coarse = xarr[row_grid, col_grid]
    ys_coarse = yarr[row_grid, col_grid]

    # backup original coords
    orig_x = hdf['x_coords'][:]
    orig_y = hdf['y_coords'][:]

    # replace with coarse coords for computation
    del hdf['x_coords']
    del hdf['y_coords']
    hdf.create_dataset('x_coords', data=xs_coarse)
    hdf.create_dataset('y_coords', data=ys_coarse)

    try:
        coarse = compute_alongstream_raster(simulation, outlet_xy=None, depth_name=depth_name, wetted_name=wetted_name, out_name=out_name)
    finally:
        # restore original coords
        del hdf['x_coords']
        del hdf['y_coords']
        hdf.create_dataset('x_coords', data=orig_x)
        hdf.create_dataset('y_coords', data=orig_y)

    # bilinear upsample: simple nearest-neighbor upscale from coarse to original
    from scipy.ndimage import zoom
    zoom_h = h / coarse.shape[0]
    zoom_w = w / coarse.shape[1]
    up = zoom(coarse, (zoom_h, zoom_w), order=1)
    # ensure shape matches exactly
    up = up[:h, :w]

    # write into environment
    if out_name in env:
        env[out_name][:] = up
    else:
        env.create_dataset(out_name, data=up, dtype='f4')
    return up
