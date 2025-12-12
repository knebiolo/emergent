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
    def __init__(self, plan_path: str, field_names: List[str] = None, timestep: int = 0):
        self.plan_path = plan_path
        self.timestep = int(timestep) if timestep is not None else 0
        if field_names is None:
            field_names = ['Cells Minimum Elevation']
        self.field_names = list(field_names)
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

            fields = {}
            for fname in self.field_names:
                geom_path = f'/Geometry/2D Flow Areas/2D area/{fname}'
                if geom_path in h:
                    arr = h[geom_path][:]
                else:
                    ds_path = self._find_dataset_by_name(h, fname)
                    if ds_path is not None:
                        ds = h[ds_path]
                        if ds.ndim > 1:
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
        return out


def map_hecras_for_agents(plan_path: str, points: np.ndarray, field_names: List[str] = None, k: int = 8) -> Dict[str, np.ndarray]:
    m = HECRASMap(plan_path, field_names=field_names)
    return m.map_idw(points, k=k)


def initialize_hecras_geometry(sim: Any, plan_path: str, depth_threshold: float = 0.05, create_rasters: bool = False) -> Dict[str, Any]:
    with h5py.File(plan_path, 'r') as f:
        key = 'Geometry/2D Flow Areas/2D area/Cells Center Coordinate'
        if key in f:
            coords = np.asarray(f[key])
        else:
            raise RuntimeError(f"Missing expected geometry dataset in {plan_path}")

    xs = coords[:, 0]
    ys = coords[:, 1]
    sim.hdf5.create_dataset('x_coords', data=xs)
    sim.hdf5.create_dataset('y_coords', data=ys)
    return {'coords': coords, 'n_cells': coords.shape[0]}


def ensure_hdf_coords_from_hecras(sim: Any, plan_path: str, target_shape: Optional[Tuple[int, int]] = None) -> None:
    if 'x_coords' in sim.hdf5 and 'y_coords' in sim.hdf5:
        return
    try:
        with h5py.File(str(plan_path), 'r') as ph:
            coords = np.asarray(ph['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
    except Exception:
        coords = None

    if coords is None:
        if target_shape is None:
            nx, ny = 10, 10
        else:
            ny, nx = target_shape
        xs = np.linspace(0.0, 1.0, nx * ny)
        ys = np.linspace(0.0, 1.0, nx * ny)
    else:
        xs = coords[:, 0]
        ys = coords[:, 1]

    sim.hdf5.create_dataset('x_coords', data=xs)
    sim.hdf5.create_dataset('y_coords', data=ys)


def map_hecras_to_env_rasters(sim: Any, plan_path: str, field_names: Sequence[str], k: int = 1) -> bool:
    key_candidates = [(plan_path, tuple(field_names)), ('', tuple(field_names))]
    adapter = None
    for key in key_candidates:
        adapter = sim._hecras_maps.get(key)
        if adapter is not None:
            break
    if adapter is None:
        raise KeyError(f"No adapter registered for plan {plan_path} and fields {field_names}")

    pts = np.zeros((1, 2))
    vals = adapter.map_idw(pts, k=k)
    arr = np.asarray(vals)
    grid_shape = getattr(adapter, 'grid_shape', None)
    if grid_shape is None:
        n = arr.size
        side = int(np.sqrt(n))
        grid_shape = (side, side) if side * side == n else (n,)

    env = sim.hdf5.require_group('environment')
    for idx, fname in enumerate(field_names):
        data = arr
        if data.size == 0:
            data = np.zeros(int(np.prod(grid_shape)))
        env.create_dataset(fname, data=np.asarray(data).reshape(grid_shape))

    return True
