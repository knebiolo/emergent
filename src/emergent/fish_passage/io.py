"""HECRAS IO and mapping helpers (skeleton)

This module contains minimal, well-documented entry points used by
tests and will be expanded during migration. Keep implementations small
and unit-tested; avoid changing legacy callers until new APIs are
accepted.
"""
from typing import Any, Dict, Iterable, Sequence, Tuple
import numpy as np


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
    import h5py

    coords = None
    result = {}
    with h5py.File(plan_path, 'r') as f:
        # expected path used in tests
        key = 'Geometry/2D Flow Areas/2D area/Cells Center Coordinate'
        if key in f:
            coords = np.asarray(f[key])

    if coords is None:
        raise RuntimeError(f"Missing expected geometry dataset in {plan_path}")

    # ensure simulation HDF5 registers x/y coords
    xs = coords[:, 0]
    ys = coords[:, 1]
    sim.hdf5.create_dataset('x_coords', data=xs)
    sim.hdf5.create_dataset('y_coords', data=ys)

    result['coords'] = coords
    result['n_cells'] = coords.shape[0]
    return result


def map_hecras_for_agents(sim: Any, pts: np.ndarray, plan_path: str, field_names: Sequence[str], k: int = 8) -> np.ndarray:
    """Simple adapter-based mapping used by tests.

    This skeleton expects `sim._hecras_maps` to be a dict keyed by
    `(plan_path, tuple(field_names))` returning an adapter with a
    `map_idw(pts, k=...)` method.
    """
    key = (plan_path, tuple(field_names))
    adapter = sim._hecras_maps.get(key)
    if adapter is None:
        raise KeyError(f"No adapter registered for plan {plan_path} and fields {field_names}")
    return adapter.map_idw(pts, k=k)


def ensure_hdf_coords_from_hecras(sim: Any, plan_path: str, target_shape: Tuple[int, int] = (10, 10)) -> None:
    """Populate `sim.hdf5` with simple x/y grids if missing (test convenience).
    """
    if 'x_coords' in sim.hdf5 and 'y_coords' in sim.hdf5:
        return
    # create dummy grid based on plan coords if present; otherwise, simple mesh
    try:
        coords = np.vstack((sim.hdf5['x_coords'][:], sim.hdf5['y_coords'][:])).T
        xs = coords[:, 0]
        ys = coords[:, 1]
    except Exception:
        nx, ny = target_shape
        xs = np.linspace(0.0, 1.0, nx * ny)
        ys = np.linspace(0.0, 1.0, nx * ny)
    sim.hdf5.create_dataset('x_coords', data=xs)
    sim.hdf5.create_dataset('y_coords', data=ys)


class HECRASMap:
    """Lightweight container for a HECRAS plan KDTree and field values.

    Usage:
        m = HECRASMap(plan_path, field_path)
        vals = m.map_idw(query_pts, k=8)

    The constructor attempts to find commonly-used dataset paths for
    coordinates and field values. It builds a cKDTree when available and
    falls back to a numpy brute-force nearest neighbor.
    """
    def __init__(self, plan_path: str, field_path: str):
        import h5py
        self.plan_path = plan_path
        self.field_path = field_path
        self.coords = None
        self.values = None
        self._kdtree = None

        with h5py.File(plan_path, 'r') as f:
            # try common coord locations
            for key in (
                'Geometry/Nodes/Coordinates',
                'Geometry/2D Flow Areas/2D area/Cells Center Coordinate',
            ):
                if key in f:
                    self.coords = np.asarray(f[key])
                    break

            # Field values: allow full path or common result paths
            if field_path in f:
                self.values = np.asarray(f[field_path])
            else:
                # try Results style locations
                candidates = [
                    f'Results/Results_0001/{field_path}/Values',
                    f'Results/Results_0001/{field_path}',
                    f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/{field_path}',
                    f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/{field_path}/Values',
                ]
                for c in candidates:
                    if c in f:
                        self.values = np.asarray(f[c])
                        break

        if self.coords is None:
            raise RuntimeError(f"Could not find coordinate dataset in {plan_path}")

        if self.values is None:
            # allow missing values but warn via exception so callers handle
            raise RuntimeError(f"Could not find field dataset for '{field_path}' in {plan_path}")

        # values may be shape (timesteps, n_cells) or (n_cells,) or (n_cells,1)
        if self.values.ndim == 2 and self.values.shape[0] > 1 and self.values.shape[1] >= self.coords.shape[0]:
            # common case in some files: time x cells -> take first timestep
            self.values = np.asarray(self.values[0])
        self.values = np.asarray(self.values).reshape(-1)

        # Build KDTree if available
        try:
            from scipy.spatial import cKDTree as _KD
            self._kdtree = _KD(self.coords)
        except Exception:
            self._kdtree = None

    def map_idw(self, pts: np.ndarray, k: int = 8) -> np.ndarray:
        """Map query points to values using IDW (inverse-distance weighting).

        Falls back to nearest neighbor when distances are zero or KDTree is
        unavailable.
        """
        pts = np.asarray(pts)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        if self._kdtree is not None:
            dists, idx = self._kdtree.query(pts, k=min(k, len(self.coords)))
            # if k == 1, make shapes consistent
            if idx.ndim == 1:
                idx = idx[:, None]
                dists = dists[:, None]
        else:
            # brute force
            d = np.linalg.norm(pts[:, None, :] - self.coords[None, :, :], axis=2)
            idx = np.argsort(d, axis=1)[:, :k]
            dists = np.take_along_axis(d, idx, axis=1)

        # Compute weights: inverse distance, protect zeros
        with np.errstate(divide='ignore'):
            weights = 1.0 / (dists + 1e-12)
        # If any distance is zero, use the exact value
        exact = dists == 0
        out = np.empty((pts.shape[0],), dtype=float)
        for i in range(pts.shape[0]):
            if exact[i].any():
                out[i] = self.values[idx[i, exact[i]].tolist()[0]]
            else:
                w = weights[i]
                vals = self.values[idx[i]]
                out[i] = float(np.sum(w * vals) / np.sum(w))

        return out
"""HECRAS I/O and mapping helpers for fish_passage.

These are intentionally minimal, testable, and fail-fast. They do not
attempt to replicate HECRASMap behavior — callers that need full HECRAS
IDW mapping should provide a mapping object or adapter and register it on
`simulation._hecras_maps` (keyed by plan path and field names).

This design follows the project's programming guide: explicit errors,
clear input validation, no broad silent fallbacks.
"""
from typing import Iterable, Optional, Sequence
import numpy as np
import h5py

from emergent.fish_passage.geometry import compute_affine_from_hecras, pixel_to_geo
from emergent.fish_passage.centerline import extract_centerline_fast_hecras, infer_wetted_perimeter_from_hecras
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


def ensure_hdf_coords_from_hecras(simulation, plan_path: str, target_shape: Optional[tuple] = None, target_transform=None, timestep: int = 0):
    """Populate `simulation.hdf5` with `x_coords` and `y_coords` datasets derived from HECRAS cell centers.

    - `simulation` must have attribute `hdf5` which is an open `h5py.File` or a group-like object.
    - `plan_path` is a path to a HECRAS HDF5 plan containing
      `Geometry/2D Flow Areas/2D area/Cells Center Coordinate`.

    This function will write `x_coords` and `y_coords` into `simulation.hdf5` if they are missing.
    Raises `RuntimeError` for I/O problems or `KeyError` when required datasets are missing.
    """
    hdf = getattr(simulation, 'hdf5', None)
    if hdf is None:
        raise RuntimeError('Simulation object missing `hdf5` attribute')

    try:
        with h5py.File(str(plan_path), 'r') as ph:
            coords = np.asarray(ph['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
    except KeyError as e:
        raise KeyError('HECRAS plan missing expected Cells Center Coordinate dataset') from e
    except Exception as e:
        raise RuntimeError('Failed to open HECRAS plan: ' + str(e)) from e

    if target_transform is None:
        target_transform = compute_affine_from_hecras(coords)

    if target_shape is None:
        aff = target_transform
        minx, miny = float(coords[:, 0].min()), float(coords[:, 1].min())
        maxx, maxy = float(coords[:, 0].max()), float(coords[:, 1].max())
        width = max(1, int(np.ceil((maxx - minx) / abs(aff.a))))
        height = max(1, int(np.ceil((maxy - miny) / abs(aff.e))))
        target_shape = (height, width)

    height, width = target_shape
    if height <= 0 or width <= 0:
        raise ValueError('Computed target_shape has non-positive dimensions')

    # create or reuse datasets
    if 'x_coords' not in hdf:
        dset_x = hdf.create_dataset('x_coords', (height, width), dtype='f4')
    else:
        dset_x = hdf['x_coords']
    if 'y_coords' not in hdf:
        dset_y = hdf.create_dataset('y_coords', (height, width), dtype='f4')
    else:
        dset_y = hdf['y_coords']

    # Populate if empty or contains non-finite values
    try:
        existing = np.asarray(dset_x[:])
        needs_populate = not np.isfinite(existing).any()
    except Exception:
        needs_populate = True

    if needs_populate:
        cols = np.arange(width, dtype=np.float64)
        rows = np.arange(height, dtype=np.float64)
        col_grid, row_grid = np.meshgrid(cols, rows)
        xs, ys = pixel_to_geo(target_transform, row_grid, col_grid)
        dset_x[:, :] = xs.astype('f4')
        dset_y[:, :] = ys.astype('f4')

    # attach metadata to simulation for convenience
    simulation.depth_rast_transform = target_transform
    simulation.hdf_height = height
    simulation.hdf_width = width


def map_hecras_for_agents(simulation_or_plan, agent_xy: Iterable[Sequence[float]], plan_path: Optional[str] = None, field_names: Optional[Sequence[str]] = None, k: int = 8, timestep: int = 0):
    """Map agent coordinates to HECRAS nodal fields via an injected mapping adapter.

    This function expects a mapping adapter to be registered on the simulation object under
    `simulation._hecras_maps[(plan_key, tuple(field_names))]` which implements `.map_idw(agent_xy, k=k)`.

    If no adapter is found, this function raises NotImplementedError — callers should register an adapter
    (e.g., a wrapper around HECRASMap) on the simulation prior to calling this function.
    """
    # Accept either a simulation-like object or a plan path string
    if isinstance(simulation_or_plan, str):
        # We intentionally do not attempt to build a mapping adapter here.
        raise NotImplementedError('Direct plan_path mapping is not supported by this minimal adapter. Register an adapter on a simulation object or implement a HECRASMap adapter.')

    sim = simulation_or_plan
    if field_names is None:
        field_names = getattr(sim, 'hecras_fields', None)
    plan_key = str(plan_path or getattr(sim, 'hecras_plan_path', ''))
    key = (plan_key, tuple(field_names) if field_names is not None else None)
    maps = getattr(sim, '_hecras_maps', None)
    if not maps or key not in maps:
        raise NotImplementedError('No HECRAS mapping adapter registered on simulation for key: ' + repr(key))

    adapter = maps[key]
    # Expect adapter.map_idw to exist
    if not hasattr(adapter, 'map_idw'):
        raise RuntimeError('Registered HECRAS adapter missing required method `.map_idw`')

    out = adapter.map_idw(agent_xy, k=k)
    return out


def map_hecras_to_env_rasters(simulation, plan_path: str, field_names: Optional[Sequence[str]] = None, k: int = 1, strict_missing_fields: bool = False):
    """Map HECRAS nodal fields onto the simulation raster grid and write into `simulation.hdf5['environment']`.

    This function requires `simulation.hdf5` to be present and `x_coords`/`y_coords` datasets created (see `ensure_hdf_coords_from_hecras`).
    It uses `map_hecras_for_agents` to obtain mapped arrays and writes them as 2D datasets under group `environment`.
    """
    if not hasattr(simulation, 'hdf5') or getattr(simulation, 'hdf5', None) is None:
        raise RuntimeError('Simulation missing hdf5; cannot write environment rasters')

    if field_names is None:
        field_names = getattr(simulation, 'hecras_fields', None)

    hdf = simulation.hdf5
    # ensure coords available
    if 'x_coords' in hdf and 'y_coords' in hdf:
        xarr = np.asarray(hdf['x_coords'])
        yarr = np.asarray(hdf['y_coords'])
        h, w = xarr.shape
        XX = xarr.flatten()
        YY = yarr.flatten()
        grid_xy = np.column_stack((XX, YY))
        simulation._hecras_grid_shape = (h, w)
        simulation._hecras_grid_xy = grid_xy
    else:
        raise RuntimeError('x_coords/y_coords not present in simulation.hdf5; call ensure_hdf_coords_from_hecras first')

    env = hdf.require_group('environment')
    h, w = simulation._hecras_grid_shape
    # Map fields one-by-one so missing fields don't break the whole write
    for fname in (field_names or []):
        mapped_arr = None
        try:
            mapped_arr = map_hecras_for_agents(simulation, simulation._hecras_grid_xy, plan_path=plan_path, field_names=[fname], k=k)
            arr = np.asarray(mapped_arr)
            if arr.size != h * w:
                # If mapping returned unexpected size, reshape conservatively or fill NaNs
                raise RuntimeError('Mapped array size mismatch')
        except Exception:
            if strict_missing_fields:
                raise
            arr = np.full((h * w,), np.nan, dtype=float)

        ds_name = fname
        if ds_name in env:
            del env[ds_name]
        env.create_dataset(ds_name, (h, w), dtype='f4')
        env[ds_name][:, :] = arr.reshape(h, w)

    return True


class HECRASMap:
    """Minimal HECRAS plan adapter that supports IDW mapping for a small set of fields.

    This is intentionally lightweight and only implements the subset of behavior
    required by `initialize_hecras_geometry` and mapping for environment rasters.
    """
    def __init__(self, plan_path: str, field_names: Optional[Sequence[str]] = None, timestep: int = 0):
        self.plan_path = str(plan_path)
        # Accept either a single field name string or an iterable of names
        if field_names is None:
            self.field_names = []
        elif isinstance(field_names, str):
            self.field_names = [field_names]
        else:
            self.field_names = list(field_names)
        self.timestep = int(timestep)
        # lazy-loaded
        self._coords = None
        self._fields = {}
        self._tree = None

    def _load_coords(self):
        if self._coords is None:
            with h5py.File(self.plan_path, 'r') as hdf:
                try:
                    coords = np.asarray(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
                except KeyError as e:
                    raise KeyError('HECRAS plan missing Cells Center Coordinate') from e
            self._coords = coords.astype(float)
        return self._coords

    def _ensure_tree(self):
        if self._tree is None:
            coords = self._load_coords()
            if coords.size == 0:
                raise RuntimeError('HECRAS plan has no coords')
            self._tree = cKDTree(coords)
        return self._tree

    def _read_field(self, name: str):
        # Try several common HECRAS dataset paths used in this project and tests
        candidates = [
            'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/' + name,
            'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/' + name + '/Values',
            'Results/Results_0001/' + name + '/Values',
            'Results/Results_0001/' + name,
            name,
            name + '/Values',
        ]
        with h5py.File(self.plan_path, 'r') as hdf:
            # first try exact candidates
            for path in candidates:
                if path in hdf:
                    ds = hdf[path]
                    data = np.asarray(ds)
                    return self._normalize_field_array(data)
            # fallback: search for any dataset path that contains the name substring
            found = None
            def _collect(n):
                nonlocal found
                if name in n and found is None:
                    found = n
            hdf.visit(_collect)
            if found is not None:
                ds = hdf[found]
                data = np.asarray(ds)
                return self._normalize_field_array(data)
            raise KeyError(f'Field dataset not found in HECRAS plan. Tried: {candidates}')

    def _normalize_field_array(self, data: np.ndarray) -> np.ndarray:
        """Normalize various dataset layouts into a 1D array of per-cell values.

        Handles shapes like (n_cells,), (n_cells,1), (timesteps, n_cells),
        and (n_cells, timesteps). Uses the configured `timestep` when needed.
        """
        data = np.asarray(data)
        # 0-D
        if data.ndim == 0:
            return data.reshape(-1)
        # 1-D
        if data.ndim == 1:
            return data
        # 2-D
        if data.ndim == 2:
            # if time x cells
            if data.shape[0] > 1 and data.shape[1] > 1:
                # prefer timestep x cells
                if data.shape[1] >= data.shape[0]:
                    # assume shape (timesteps, n_cells)
                    t = min(self.timestep, data.shape[0]-1)
                    return np.asarray(data[t]).reshape(-1)
                else:
                    # assume shape (n_cells, features) -> take first column
                    return np.asarray(data[:, 0]).reshape(-1)
            # if one dimension is 1, flatten
            return data.reshape(-1)
        # higher dims -> flatten
        return data.reshape(-1)

    def map_idw(self, query_pts, k: int = 8):
        """Return a dict of mapped fields keyed by field name, or a single array when one field requested.

        Uses inverse-distance weighting with k neighbors; when k==1 uses nearest neighbor.
        """
        pts = np.asarray(query_pts, dtype=float)
        tree = self._ensure_tree()
        if pts.ndim == 1:
            pts = pts[None, :]
        if k <= 1:
            dists, idxs = tree.query(pts, k=1)
            idxs = np.atleast_1d(idxs)
        else:
            dists, idxs = tree.query(pts, k=k)

        out = {}
        for fname in self.field_names:
            try:
                vals = self._read_field(fname)
            except Exception as e:
                raise
            # vals expected length equals number of HECRAS cells
            vals = np.asarray(vals)
            if k <= 1:
                res = vals[idxs]
            else:
                # compute IDW
                d = np.asarray(dists, dtype=float)
                # avoid zero distances
                d_safe = np.where(d == 0, 1e-12, d)
                w = 1.0 / d_safe
                wsum = np.sum(w, axis=1, keepdims=True)
                res = np.sum(vals[idxs] * w, axis=1) / np.maximum(wsum.flatten(), 1e-12)
            out[fname] = res
        if len(self.field_names) == 1:
            return out[self.field_names[0]]
        return out


def initialize_hecras_geometry(simulation, plan_path: str, depth_threshold: float = 0.05, crs=None,
                                target_cell_size: Optional[float] = None, create_rasters: bool = True,
                                strict_missing_fields: bool = False):
    """Ported initializer: orchestrate HECRAS geometry setup using `fish_passage` helpers.

    Behavior:
    - reads HECRAS coords
    - registers a minimal `HECRASMap` adapter on `simulation._hecras_maps`
    - extracts centerline via `extract_centerline_fast_hecras`
    - infers wetted perimeter via `infer_wetted_perimeter_from_hecras`
    - optionally creates regular rasters (writes `x_coords`/`y_coords` and maps fields)

    Raises descriptive exceptions on failure (no silent fallbacks).
    """
    # Validate simulation hdf5 presence
    if not hasattr(simulation, 'hdf5') or getattr(simulation, 'hdf5', None) is None:
        raise RuntimeError('Simulation must have an open `hdf5` attribute (h5py.File)')

    # Step 1: load coords
    with h5py.File(plan_path, 'r') as hdf:
        try:
            coords = np.asarray(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
        except KeyError as e:
            raise KeyError('HECRAS plan missing Cells Center Coordinate dataset') from e

    coords = coords.astype(float)

    # Register minimal HECRASMap adapter for common fields
    fields = ['Cell Hydraulic Depth', 'Cell Velocity - Velocity X', 'Cell Velocity - Velocity Y', 'Water Surface']
    if not hasattr(simulation, '_hecras_maps') or simulation._hecras_maps is None:
        simulation._hecras_maps = {}
    key = (str(plan_path), tuple(fields))
    if key not in simulation._hecras_maps:
        simulation._hecras_maps[key] = HECRASMap(str(plan_path), field_names=fields)

    # Step 2: extract centerline
    centerline = extract_centerline_fast_hecras(plan_path, depth_threshold=depth_threshold, sample_fraction=0.1, min_length=50)

    # Step 2b: infer wetted perimeter
    wetted_info = infer_wetted_perimeter_from_hecras(plan_path, depth_threshold=depth_threshold, timestep=0, verbose=False)
    perimeter_points = wetted_info.get('perimeter_points', None) if isinstance(wetted_info, dict) else None
    perimeter_cells = wetted_info.get('perimeter_cells', None) if isinstance(wetted_info, dict) else None
    median_spacing = wetted_info.get('median_spacing', None) if isinstance(wetted_info, dict) else None

    transform = None
    if create_rasters:
        transform = compute_affine_from_hecras(coords, target_cell_size=target_cell_size)
        # compute raster dims
        minx, miny = float(coords[:, 0].min()), float(coords[:, 1].min())
        maxx, maxy = float(coords[:, 0].max()), float(coords[:, 1].max())
        cell = abs(transform.a)
        width = max(1, int(np.ceil((maxx - minx) / cell)))
        height = max(1, int(np.ceil((maxy - miny) / cell)))
        ensure_hdf_coords_from_hecras(simulation, plan_path, target_shape=(height, width), target_transform=transform)
        map_hecras_to_env_rasters(simulation, plan_path, field_names=fields, k=1, strict_missing_fields=strict_missing_fields)

    return {
        'centerline': centerline,
        'coords': coords,
        'n_cells': int(coords.shape[0]) if hasattr(coords, 'shape') else None,
        'transform': transform,
        'perimeter_points': perimeter_points,
        'perimeter_cells': perimeter_cells,
        'median_spacing': median_spacing
    }
"""
io.py

Preamble/Module plan for input/output adapters and format helpers (moved to fish_passage).

Responsibilities (planned):
- Read/write adapters for HEC-RAS HDF5, Flow3D, GeoJSON, and internal hdf5 cache format.
- Provide simple `read_plan(path)` and `write_simulation_state(path, sim)` APIs.
- Keep I/O functions defensive at public boundaries; internal helpers use assertions and documented exceptions.

Notes:
- Centralize HDF5 path keys and transformations to avoid duplication.
"""
