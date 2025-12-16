"""HECRAS mapping utilities extracted from io.py for clarity and reuse.

Provides `HECRASMap` which discovers HECRAS HDF5 datasets and maps query
points to nodal fields using inverse-distance weighting (IDW).
"""
from typing import Optional, Sequence
import h5py
import numpy as np
import logging

from emergent.fish_passage.utils import safe_build_kdtree as _safe_build_kdtree, safe_log_exception as _safe_log_exception

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

        # prefer candidates that contain the coords length as one axis
        results_cands = [c for c in candidates if 'results/' in c[0].lower()]
        if results_cands:
            return results_cands[0][0]
        return candidates[0][0]

    def _load_plan(self):
        with h5py.File(self.plan_path, 'r') as h:
            coords = h['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]

            # load each requested field, attempting Geometry first then Results
            fields = {}
            for fname in self.field_names:
                geom_path = f'/Geometry/2D Flow Areas/2D area/{fname}'
                if geom_path in h:
                    arr = h[geom_path][:]
                else:
                    ds_path = self._find_dataset_by_name(h, fname)
                    if ds_path is not None:
                        ds = h[ds_path]
                        # If dataset is multi-dimensional, it may be either:
                        # - a per-node array shaped (n_coords, M) (M may be 1), or
                        # - a time-series shaped (T, ...). Prefer the per-node
                        # interpretation when the leading axis matches coords.
                        if ds.ndim > 1:
                            if ds.shape[0] == coords.shape[0]:
                                arr = ds[:]
                            else:
                                t = min(self.timestep, ds.shape[0] - 1)
                                arr = ds[t]
                        else:
                            arr = ds[:]
                    else:
                        raise KeyError(f"Field '{fname}' not found in HECRAS HDF: {self.plan_path}")
                fields[fname] = np.asarray(arr)

        # normalize field arrays to align with coords length
        n_coords = coords.shape[0]

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
                _safe_log_exception('Failed while logging KDTree build warning', e, file='hecras.py')

    def map_idw(self, query_pts, k=8, eps=1e-8):
        query = np.asarray(query_pts, dtype=np.float64)
        if query.ndim == 1:
            query = query.reshape(1, 2)
        if getattr(self, 'tree', None) is None:
            raise RuntimeError('IDW mapping requested but KDTree is unavailable (HECRAS plan tree build failed)')
        try:
            n_coords = self.coords.shape[0]
        except Exception:
            n_coords = None
        if n_coords is not None and k is not None:
            k = max(1, min(int(k), int(n_coords)))
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
        if getattr(self, '_return_single', False) and len(self.field_names) == 1:
            return out[self.field_names[0]]
        if len(self.field_names) == 1:
            return out
        return out
__all__ = ['HECRASMap']


def ensure_hdf_coords_from_hecras(sim: object, plan_path: str, target_shape: Optional[tuple] = None) -> None:
    """Populate `sim.hdf5` with x/y coordinate datasets derived from HECRAS plan.

    This implementation follows legacy behavior but keeps dependencies on
    `emergent.fish_passage.geometry` and `emergent.fish_passage.utils` only.
    """
    try:
        import h5py
        from emergent.fish_passage.geometry import compute_affine_from_hecras, pixel_to_geo
        from emergent.fish_passage.utils import _safe_log_exception
    except Exception:
        return

    # If already present, nothing to do
    try:
        if 'x_coords' in sim.hdf5 and 'y_coords' in sim.hdf5:
            return
    except Exception:
        pass

    coords = None
    try:
        with h5py.File(str(plan_path), 'r') as ph:
            coords = np.asarray(ph['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
    except Exception:
        coords = None

    if coords is None:
        # fallback: create a simple grid if requested
        if target_shape is None:
            nx, ny = 10, 10
        else:
            ny, nx = target_shape
        xs = np.linspace(0.0, 1.0, nx * ny).reshape((ny, nx))
        ys = np.linspace(0.0, 1.0, nx * ny).reshape((ny, nx))
    else:
        if target_shape is not None:
            height, width = target_shape
            aff = compute_affine_from_hecras(coords)
            cols = np.arange(width, dtype=np.float64)
            rows = np.arange(height, dtype=np.float64)
            col_grid, row_grid = np.meshgrid(cols, rows)
            xs, ys = pixel_to_geo(aff, row_grid, col_grid)
            xs = np.asarray(xs)
            ys = np.asarray(ys)
        else:
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

    # create datasets if missing
    try:
        if 'x_coords' not in sim.hdf5:
            sim.hdf5.create_dataset('x_coords', data=xs.astype('float32'))
        if 'y_coords' not in sim.hdf5:
            sim.hdf5.create_dataset('y_coords', data=ys.astype('float32'))
    except Exception as e:
        try:
            _safe_log_exception('Failed creating x/y coord datasets', e, file='hecras.py')
        except Exception:
            pass


def map_hecras_to_env_rasters(sim: object, plan_path: str, field_names: Sequence[str], k: int = 1) -> bool:
    """Map HECRAS fields onto the simulation raster grid and write into `simulation.hdf5['environment']`.

    Delegates IDW mapping to `HECRASMap` implemented in this module.
    """
    # Build adapter
    try:
        env = sim.hdf5.require_group('environment')
    except Exception:
        return False

    # ensure x/y coords are present
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
        # try to build grid from HECRAS plan
        try:
            m = HECRASMap(plan_path, field_names=[field_names[0]] if field_names else None)
            coords = m.coords
            from emergent.fish_passage.geometry import compute_affine_from_hecras, pixel_to_geo
            aff = compute_affine_from_hecras(coords)
            cell = abs(aff.a)
            minx = float(coords[:, 0].min())
            maxx = float(coords[:, 0].max())
            miny = float(coords[:, 1].min())
            maxy = float(coords[:, 1].max())
            w = max(1, int(np.ceil((maxx - minx) / cell)))
            h = max(1, int(np.ceil((maxy - miny) / cell)))
            cols = np.arange(w)
            rows = np.arange(h)
            col_grid, row_grid = np.meshgrid(cols, rows)
            xs, ys = pixel_to_geo(aff, row_grid, col_grid)
            sim._hecras_grid_shape = (h, w)
            sim._hecras_grid_xy = np.column_stack((xs.flatten(), ys.flatten()))
        except Exception:
            return False

    grid_xy = sim._hecras_grid_xy
    try:
        m = HECRASMap(plan_path, field_names=field_names)
        mapped = m.map_idw(grid_xy, k=k)
    except Exception:
        return False

    if isinstance(mapped, dict):
        for name, arr in mapped.items():
            h, w = sim._hecras_grid_shape
            if name in env:
                del env[name]
            env.create_dataset(name, (h, w), dtype='f4')
            env[name][:, :] = np.asarray(arr).reshape(h, w)
    else:
        arr = np.asarray(mapped)
        h, w = sim._hecras_grid_shape
        ds_name = field_names[0] if field_names else 'field'
        if ds_name in env:
            del env[ds_name]
        env.create_dataset(ds_name, (h, w), dtype='f4')
        env[ds_name][:, :] = arr.reshape(h, w)
    return True


def initialize_hecras_geometry(sim: object, plan_path: str, depth_threshold: float = 0.05, create_rasters: bool = False):
    """Read basic HECRAS datasets and register coordinates on `sim.hdf5`.

    Returns a dict with coords and n_cells similar to prior implementations.
    """
    try:
        import h5py
    except Exception:
        raise RuntimeError('h5py required')

    with h5py.File(plan_path, 'r') as f:
        key = 'Geometry/2D Flow Areas/2D area/Cells Center Coordinate'
        if key in f:
            coords = np.asarray(f[key])
        else:
            raise RuntimeError(f"Missing expected geometry dataset in {plan_path}")

    xs = coords[:, 0]
    ys = coords[:, 1]
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
    sim.hdf5.create_dataset('x_coords', data=xs.astype('float32'))
    sim.hdf5.create_dataset('y_coords', data=ys.astype('float32'))
    return {'coords': coords, 'n_cells': coords.shape[0]}


def infer_wetted_perimeter_from_hecras(plan_path: str, depth_threshold: float = 0.05, max_nodes: int = 5000, raster_fallback_resolution: float = 5.0, verbose: bool = False, timestep: int = 0):
    """Delegate to `centerline.infer_wetted_perimeter_from_hecras` where available."""
    try:
        from emergent.fish_passage.centerline import infer_wetted_perimeter_from_hecras as _inf
        return _inf(plan_path, depth_threshold=depth_threshold, max_nodes=max_nodes, raster_fallback_resolution=raster_fallback_resolution, verbose=verbose, timestep=timestep)
    except Exception:
        # Fallback to attempting to use HECRASMap-based rasterization via io-style helper
        raise

__all__ = ['HECRASMap', 'ensure_hdf_coords_from_hecras', 'map_hecras_to_env_rasters', 'initialize_hecras_geometry', 'infer_wetted_perimeter_from_hecras']

