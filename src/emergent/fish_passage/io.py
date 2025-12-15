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
        # clamp k so we never request more neighbors than exist
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


def initialize_hdf5(sim: Any, num_agents: int, num_timesteps: int, model_name: str = 'model') -> None:
    """Create the `agent_data` group and expected datasets on `sim.hdf5`.

    This is a compact, test-first implementation matching the legacy
    dataset names and shapes. It will not overwrite existing datasets
    unless they are missing.
    """
    if getattr(sim, 'hdf5', None) is None:
        raise RuntimeError('Simulation object must have a writable `hdf5` attribute (h5py.File)')

    h5 = sim.hdf5
    if 'agent_data' not in h5:
        agent_data = h5.create_group('agent_data')
    else:
        agent_data = h5['agent_data']

    # helper to create dataset if missing
    def _ensure(name, shape, dtype='f4'):
        if name not in agent_data:
            agent_data.create_dataset(name, shape, dtype=dtype)

    _ensure('sex', (num_agents,))
    _ensure('length', (num_agents,))
    _ensure('ucrit', (num_agents,))
    _ensure('weight', (num_agents,))
    _ensure('body_depth', (num_agents,))
    _ensure('too_shallow', (num_agents,))
    _ensure('opt_wat_depth', (num_agents,))

    tshape = (num_agents, num_timesteps)
    _ensure('X', tshape)
    _ensure('Y', tshape)
    _ensure('Z', tshape)
    _ensure('prev_X', tshape)
    _ensure('prev_Y', tshape)
    _ensure('heading', tshape)
    _ensure('sog', tshape)
    _ensure('ideal_sog', tshape)
    _ensure('swim_speed', tshape)
    _ensure('battery', tshape)
    _ensure('swim_behav', tshape)
    _ensure('swim_mode', tshape)
    _ensure('recover_stopwatch', tshape)
    _ensure('ttfr', tshape)
    _ensure('time_out_of_water', tshape)
    _ensure('drag', tshape)
    _ensure('thrust', tshape)
    _ensure('Hz', tshape)
    _ensure('bout_no', tshape)
    _ensure('dist_per_bout', tshape)
    _ensure('bout_dur', tshape)
    _ensure('time_of_jump', tshape)
    _ensure('kcal', tshape)

    # metadata attrs
    try:
        h5.attrs['simulation_name'] = f"{model_name} Fish Passage Simulation"
        h5.attrs['num_agents'] = num_agents
        h5.attrs['num_timesteps'] = num_timesteps
    except Exception:
        pass

    try:
        h5.flush()
    except Exception:
        pass


def sample_environment(sim: Any, transform, raster_name: str):
    """Sample raster values at agent X/Y positions.

    Parity-focused port of legacy `sample_environment`.
    Accepts `sim` (object with `hdf5`, `X`, `Y`, `num_agents`) and an
    affine `transform` and raster dataset name. Returns a flattened array
    of length `sim.num_agents` of sampled values.
    """
    from emergent.fish_passage.geometry import geo_to_pixel

    rows, cols = geo_to_pixel(transform, sim.X, sim.Y)

    # prefer cache if present
    cache = getattr(sim, '_env_cache', None)
    if cache is not None and raster_name in cache and cache[raster_name] is not None:
        data = cache[raster_name]
        rows = np.clip(np.round(rows).astype(int), 0, data.shape[0] - 1)
        cols = np.clip(np.round(cols).astype(int), 0, data.shape[1] - 1)

        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()
        if (rmax - rmin + 1) * (cmax - cmin + 1) <= max(4 * sim.num_agents, 256):
            block = data[rmin:rmax+1, cmin:cmax+1]
            vals = block[rows - rmin, cols - cmin]
            return np.asarray(vals).flatten()

        vals = data[rows, cols]
        return np.asarray(vals).flatten()

    # fallback to hdf5 reads
    env = sim.hdf5.get('environment', None)
    if env is None:
        return np.full(getattr(sim, 'num_agents', rows.size), np.nan, dtype=float)

    # allow dataset to be specified as 'depth' or full path
    raster_dataset = env[raster_name] if raster_name in env else sim.hdf5.get(f'environment/{raster_name}')

    rows = np.clip(np.round(rows).astype(int), 0, raster_dataset.shape[0] - 1)
    cols = np.clip(np.round(cols).astype(int), 0, raster_dataset.shape[1] - 1)

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    if (rmax - rmin + 1) * (cmax - cmin + 1) <= max(4 * getattr(sim, 'num_agents', rows.size), 256):
        block = raster_dataset[rmin:rmax+1, cmin:cmax+1]
        vals = block[rows - rmin, cols - cmin]
        return np.asarray(vals).flatten()

    # grouped-row efficient indexing fallback: read by rows
    try:
        h5idx = getattr(sim, '_h5_advanced_index', None)
        if h5idx is not None:
            return np.asarray(h5idx(raster_dataset, rows, cols)).flatten()
    except Exception:
        pass

    # final fallback: iterate
    out = np.empty(rows.shape[0], dtype=float)
    for i in range(rows.shape[0]):
        out[i] = raster_dataset[int(rows[i]), int(cols[i])]
    return out


def initialize_mental_map(sim: Any, avoid_cell_size: float = 5.0) -> None:
    """Create per-agent memory maps under `memory/` in sim.hdf5.

    The legacy implementation created one dataset per agent under 'memory',
    sized according to the simulation `height`/`width` and `avoid_cell_size`.
    This compact version computes a grid size, creates datasets, and zeroes them.
    """
    h5 = getattr(sim, 'hdf5', None)
    if h5 is None:
        raise RuntimeError('Simulation object must have an open `hdf5` file')

    # determine map sizes
    height = getattr(sim, 'height', None)
    width = getattr(sim, 'width', None)
    if height is None or width is None:
        raise RuntimeError('Simulation must have `height` and `width` attributes for mental map initialization')

    avoid_h = int(np.round(height / avoid_cell_size)) + 1
    avoid_w = int(np.round(width / avoid_cell_size)) + 1

    if 'memory' not in h5:
        mem = h5.create_group('memory')
    else:
        mem = h5['memory']

    num_agents = getattr(sim, 'num_agents', None)
    if num_agents is None:
        raise RuntimeError('Simulation must have `num_agents` attribute')

    for i in range(int(num_agents)):
        name = f"{i}"
        if name not in mem:
            mem.create_dataset(name, (avoid_h, avoid_w), dtype='f4')
            mem[name][:, :] = np.zeros((avoid_h, avoid_w), dtype='f4')

    # store transform metadata similar to legacy
    try:
        from affine import Affine
        depth_aff = getattr(sim, 'depth_rast_transform', None)
        if depth_aff is not None:
            # store a simple representation
            h5.attrs['mental_map_transform'] = str(depth_aff)
    except Exception:
        pass

    try:
        h5.flush()
    except Exception:
        pass


def initialize_refugia_map(sim: Any, refugia_cell_size: float = 5.0) -> None:
    """Create per-agent refugia rasters under `refugia/` in sim.hdf5.

    Mirrors the legacy behavior: datasets per agent sized from simulation
    dimensions and `refugia_cell_size`, initialized to zeros.
    """
    h5 = getattr(sim, 'hdf5', None)
    if h5 is None:
        raise RuntimeError('Simulation object must have an open `hdf5` file')

    height = getattr(sim, 'height', None)
    width = getattr(sim, 'width', None)
    if height is None or width is None:
        raise RuntimeError('Simulation must have `height` and `width` attributes for refugia map initialization')

    refug_h = int(np.round(height / refugia_cell_size)) + 1
    refug_w = int(np.round(width / refugia_cell_size)) + 1

    if 'refugia' not in h5:
        mem = h5.create_group('refugia')
    else:
        mem = h5['refugia']

    num_agents = getattr(sim, 'num_agents', None)
    if num_agents is None:
        raise RuntimeError('Simulation must have `num_agents` attribute')

    for i in range(int(num_agents)):
        name = f"{i}"
        if name not in mem:
            mem.create_dataset(name, (refug_h, refug_w), dtype='f4')
            mem[name][:, :] = np.zeros((refug_h, refug_w), dtype='f4')

    try:
        h5.attrs['refugia_map_transform'] = str(getattr(sim, 'depth_rast_transform', None))
    except Exception:
        pass

    try:
        h5.flush()
    except Exception:
        pass


def timestep_flush(sim: Any, timestep: int, flush_interval: int = 100) -> None:
    """Write the current timestep slice of agent arrays from simulation into HDF5.

    Expects that `sim` has attributes corresponding to agent arrays (e.g.
    `X`, `Y`, `battery`, etc.) and that `sim.hdf5['agent_data']` exists with
    matching datasets. Only datasets present are written. Periodically flushes
    according to `flush_interval`.
    """
    h5 = getattr(sim, 'hdf5', None)
    if h5 is None:
        raise RuntimeError('Simulation object must have an open `hdf5` file')

    if 'agent_data' not in h5:
        raise RuntimeError('HDF5 file missing `agent_data` group; call initialize_hdf5 first')

    ag = h5['agent_data']

    # map of attribute name -> dataset name (here they match)
    for name in list(ag.keys()):
        try:
            ds = ag[name]
        except Exception:
            continue
        # skip static datasets (1D)
        if ds.ndim != 2:
            continue
        # get array attribute
        arr = getattr(sim, name, None)
        if arr is None:
            # allow some derived writes (e.g., drag computed as norm); skip if absent
            continue
        a = np.asarray(arr)
        # If sim attribute is 2D (num_agents, num_timesteps), write the column
        if a.ndim == 2 and a.shape[0] == ds.shape[0]:
            if a.shape[1] > timestep:
                try:
                    ds[:, timestep] = a[:, timestep].astype('float32')
                    continue
                except Exception:
                    pass
        # If sim attribute is 1D (num_agents,), write whole column
        if a.ndim == 1 and a.shape[0] == ds.shape[0]:
            try:
                ds[:, timestep] = a.astype('float32')
                continue
            except Exception:
                pass
        # fallback: try to flatten and broadcast
        try:
            ds[:, timestep] = a.reshape(ds.shape[0], -1)[:, 0].astype('float32')
        except Exception:
            continue

    if flush_interval and (timestep % flush_interval == 0):
        try:
            h5.flush()
        except Exception:
            pass


def enviro_import(sim: Any, data, surface_type: str, transform: Optional[Any] = None, no_data_value: Optional[float] = None) -> None:
    """Import an environmental raster into `sim.hdf5['environment']`.

    This compact implementation accepts either a NumPy array `data` (height, width)
    or a path to a raster file (will attempt to use rasterio). It will create
    `x_coords`/`y_coords` if missing when provided a transform or when `data` is a
    grid and `transform` is provided.
    """
    h5 = getattr(sim, 'hdf5', None)
    if h5 is None:
        raise RuntimeError('Simulation object must have an open `hdf5` file')

    # Ensure environment group
    if 'environment' not in h5:
        env = h5.create_group('environment')
    else:
        env = h5['environment']

    arr = None
    height = width = None
    # If caller provided a numpy array
    if isinstance(data, (list, tuple)) or hasattr(data, 'ndim'):
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError('Numeric environment data must be 2D (height, width)')
        height, width = arr.shape
    else:
        # data is likely a path; attempt to use rasterio if available
        try:
            import rasterio
        except Exception:
            raise RuntimeError('rasterio required to read raster file paths')
        with rasterio.open(str(data)) as src:
            arr = src.read(1)
            height, width = arr.shape
            if transform is None:
                transform = src.transform
            if no_data_value is None:
                no_data_value = src.nodatavals[0] if src.nodatavals else None

    # create x/y coords if missing and a transform is available
    if 'x_coords' not in h5 or 'y_coords' not in h5:
        if transform is not None and height is not None and width is not None:
            # build simple x/y grids using affine if provided
            try:
                from emergent.fish_passage.geometry import pixel_to_geo
                cols = np.arange(width, dtype=np.float64)
                rows = np.arange(height, dtype=np.float64)
                col_grid, row_grid = np.meshgrid(cols, rows)
                xs, ys = pixel_to_geo(transform, row_grid, col_grid)
                h5.create_dataset('x_coords', data=xs.astype('float32'))
                h5.create_dataset('y_coords', data=ys.astype('float32'))
            except Exception:
                # fallback: simple index grid
                xs = np.tile(np.arange(width, dtype=np.float32), (height, 1))
                ys = np.tile(np.arange(height, dtype=np.float32).reshape((height, 1)), (1, width))
                h5.create_dataset('x_coords', data=xs)
                h5.create_dataset('y_coords', data=ys)

    # write the raster into environment group
    name = surface_type
    if name in env:
        del env[name]
    env.create_dataset(name, (height, width), dtype='f4')
    env[name][:, :] = np.asarray(arr).astype('float32')

    # record transforms if provided
    if transform is not None:
        try:
            h5.attrs[f'{surface_type}_transform'] = str(transform)
        except Exception:
            pass

    if no_data_value is not None:
        try:
            h5.attrs['no_data_value'] = float(no_data_value)
        except Exception:
            pass

    try:
        h5.flush()
    except Exception:
        pass


def infer_wetted_perimeter_from_hecras(hdf_path_or_file, depth_threshold=0.05, max_nodes=5000, raster_fallback_resolution=5.0, verbose=False, timestep=0):
    """Vector-first wetted perimeter extraction with raster fallback.

    Returns a list of rings (each ring is an Nx2 numpy array of coordinates).
    """
    close_file = False
    if isinstance(hdf_path_or_file, str):
        hdf = h5py.File(hdf_path_or_file, 'r')
        close_file = True
    else:
        hdf = hdf_path_or_file

    try:
        # Read depth dataset (long-form path used in tests/fixtures)
        ds_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'
        ds = hdf[ds_path]
        data = np.asarray(ds[:])
        if data.ndim > 1 and data.shape[0] > 1:
            t = int(min(timestep, data.shape[0] - 1))
            depth = np.asarray(data[t])
        else:
            depth = np.asarray(data).reshape(-1)

        wetted_mask = depth > float(depth_threshold)

        # Attempt vector workflow
        vector_failed = False
        try:
            facepoints = np.asarray(hdf['Geometry/2D Flow Areas/2D area/FacePoints Coordinate'])
            is_perim = np.asarray(hdf['Geometry/2D Flow Areas/2D area/FacePoints Is Perimeter'])
            face_info = np.asarray(hdf['Geometry/2D Flow Areas/2D area/Cells Face and Orientation Info']).astype(int)
            perim_coords = np.asarray(hdf['Geometry/2D Flow Areas/2D area/Perimeter'])

            perim_mask = (is_perim == -1)
            perim_touch = np.zeros(len(facepoints), dtype=bool)
            wetted_idx = np.nonzero(wetted_mask)[0]
            for i in wetted_idx:
                start, count = face_info[i]
                idxs = np.arange(start, start + count)
                idxs = idxs[idxs < len(perim_mask)]
                perim_touch[idxs] |= perim_mask[idxs]

            if not perim_touch.any():
                raise RuntimeError('No perimeter facepoints touched by wetted cells')

            tree = _safe_build_kdtree(facepoints, name='facepoints_tree')
            if tree is not None:
                _, idxs = tree.query(perim_coords, k=1)
            else:
                dif = perim_coords[:, None, :] - facepoints[None, :, :]
                dists = np.sqrt(np.sum(dif * dif, axis=2))
                idxs = np.argmin(dists, axis=1)

            touched = perim_touch[idxs]
            if not np.any(touched):
                raise RuntimeError('Perimeter points mapping found no touched points')

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

            return out_rings

        except Exception:
            vector_failed = True

        # Raster fallback
        if vector_failed:
            coords = np.asarray(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
            x = coords[:, 0]
            y = coords[:, 1]
            xmin, xmax = float(x.min()), float(x.max())
            ymin, ymax = float(y.min()), float(y.max())
            res = float(raster_fallback_resolution)
            # avoid zero division
            span_x = max(1e-6, xmax - xmin)
            span_y = max(1e-6, ymax - ymin)
            nx = max(3, int(np.ceil(span_x / res)))
            ny = max(3, int(np.ceil(span_y / res)))

            gx = np.clip(((x - xmin) / span_x * (nx - 1)).astype(int), 0, nx - 1)
            gy = np.clip(((y - ymin) / span_y * (ny - 1)).astype(int), 0, ny - 1)
            grid = np.zeros((ny, nx), dtype=np.uint8)
            # assign by iterating to avoid shape/broadcast surprises
            for i_val, (gy_i, gx_i) in enumerate(zip(gy, gx)):
                if wetted_mask[i_val]:
                    grid[gy_i, gx_i] = 1

            dx = span_x / (nx - 1) if nx > 1 else span_x
            dy = span_y / (ny - 1) if ny > 1 else span_y
            from shapely.geometry import box
            from shapely.ops import unary_union

            polys = []
            ys_idx, xs_idx = np.where(grid == 1)
            for gy_i, gx_i in zip(ys_idx, xs_idx):
                x0 = xmin + gx_i * dx
                x1 = xmin + (gx_i + 1) * dx
                y0 = ymin + gy_i * dy
                y1 = ymin + (gy_i + 1) * dy
                polys.append(box(x0, y0, x1, y1))

            if not polys:
                raise RuntimeError('Raster fallback produced no polygons')

            merged = unary_union(polys)
            if merged.geom_type == 'Polygon':
                rings = [np.asarray(merged.exterior.coords)]
            else:
                rings = [np.asarray(g.exterior.coords) for g in merged.geoms]

            return rings

    finally:
        if close_file:
            hdf.close()


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

    # locate x_coords/y_coords either under environment group or at root
    target_group = None
    if 'x_coords' in env and 'y_coords' in env:
        target_group = env
    elif 'x_coords' in hdf and 'y_coords' in hdf:
        target_group = hdf
    else:
        raise RuntimeError('x_coords/y_coords required in simulation.hdf5')

    xarr = np.asarray(target_group['x_coords'])
    yarr = np.asarray(target_group['y_coords'])
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


    # backup original coords from the same group and replace with coarse coords
    orig_x = target_group['x_coords'][:]  # read copy
    orig_y = target_group['y_coords'][:]
    del target_group['x_coords']
    del target_group['y_coords']
    target_group.create_dataset('x_coords', data=xs_coarse)
    target_group.create_dataset('y_coords', data=ys_coarse)

    # If original coords lived at root but compute_alongstream_raster expects
    # coords under `environment`, create temporary copies under env so the
    # callee can find them. Track whether we created them so we can remove later.
    created_env_coords = False
    env_had = ('x_coords' in env and 'y_coords' in env)
    if target_group is hdf and not env_had:
        env.create_dataset('x_coords', data=target_group['x_coords'][:])
        env.create_dataset('y_coords', data=target_group['y_coords'][:])
        created_env_coords = True

    # create coarse depth/wetted rasters in env so compute_alongstream_raster sees matching shapes
    created_coarse_depth = False
    orig_depth = None
    orig_wetted = None
    if depth_name in env:
        orig_depth = env[depth_name][:]
        # simple block-mean downsample
        depth_arr = orig_depth
        block_h = max(1, int(np.ceil(h / ch)))
        block_w = max(1, int(np.ceil(w / cw)))
        # sample indices grid already computed (row_grid, col_grid)
        coarse_depth = depth_arr[row_grid, col_grid]
        if depth_name in env:
            del env[depth_name]
        env.create_dataset(depth_name, data=coarse_depth)
        created_coarse_depth = True
    elif wetted_name in env:
        orig_wetted = env[wetted_name][:]
        wett_arr = orig_wetted
        coarse_wett = wett_arr[row_grid, col_grid]
        if wetted_name in env:
            del env[wetted_name]
        env.create_dataset(wetted_name, data=coarse_wett)

    try:
        coarse = compute_alongstream_raster(simulation, outlet_xy=None, depth_name=depth_name, wetted_name=wetted_name, out_name=out_name)
    finally:
        # restore original coords in the same group
        del target_group['x_coords']
        del target_group['y_coords']
        target_group.create_dataset('x_coords', data=orig_x)
        target_group.create_dataset('y_coords', data=orig_y)
        # remove any temporary env copies we created
        if created_env_coords:
            del env['x_coords']
            del env['y_coords']
        # restore any coarse depth/wetted datasets
        if created_coarse_depth:
            del env[depth_name]
            env.create_dataset(depth_name, data=orig_depth)
        if orig_wetted is not None:
            del env[wetted_name]
            env.create_dataset(wetted_name, data=orig_wetted)

    # bilinear upsample: simple nearest-neighbor upscale from coarse to original
    from scipy.ndimage import zoom
    zoom_h = h / coarse.shape[0]
    zoom_w = w / coarse.shape[1]
    up = zoom(coarse, (zoom_h, zoom_w), order=1)
    # ensure shape matches exactly
    up = up[:h, :w]

    # write into environment (ensure dataset shape matches upsampled array)
    if out_name in env:
        del env[out_name]
    env.create_dataset(out_name, data=up, dtype='f4')
    return up
