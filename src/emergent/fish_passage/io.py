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
from emergent.fish_passage.hecras import HECRASMap, ensure_hdf_coords_from_hecras as _ensure_hdf_coords_from_hecras, map_hecras_to_env_rasters as _map_hecras_to_env_rasters, initialize_hecras_geometry as _initialize_hecras_geometry, infer_wetted_perimeter_from_hecras as _infer_wetted_perimeter_from_hecras


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
    return _initialize_hecras_geometry(sim, plan_path, depth_threshold=depth_threshold, create_rasters=create_rasters)


def ensure_hdf_coords_from_hecras(sim: Any, plan_path: str, target_shape: Optional[Tuple[int, int]] = None) -> None:
    """Populate `sim.hdf5` with x/y coordinate datasets derived from HECRAS plan.

    This implementation follows legacy behavior: if the plan provides cell
    centers, use them; otherwise create a simple regular grid of requested
    `target_shape` (or 10x10 default). It does not overwrite existing
    datasets.
    """
    return _ensure_hdf_coords_from_hecras(sim, plan_path, target_shape=target_shape)


def map_hecras_to_env_rasters(sim: Any, plan_path: str, field_names: Sequence[str], k: int = 1) -> bool:
    """Map HECRAS fields onto the simulation raster grid and write into `simulation.hdf5['environment']`.

    Expects a mapping adapter registered on `sim._hecras_maps[(plan_path, tuple(field_names))]`
    which implements `.map_idw(pts, k=...)`.
    """
    return _map_hecras_to_env_rasters(sim, plan_path, field_names=field_names, k=k)


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


def get_agent_flow_components(simulation: Any, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-agent flow components (vel_x, vel_y) preferring HECRAS mapping.

    Strategy:
    - If `simulation.use_hecras` and `hecras_plan_path` available, attempt to map
      nodal `Velocity X`/`Velocity Y` fields to agent XY positions.
    - Else, sample `environment/vel_x` and `environment/vel_y` rasters if present.
    - Else, if only `vel_dir` raster exists, return unit vectors from direction.
    - If none available, return NaNs arrays so callers can detect missing flow.
    """
    n = int(getattr(simulation, 'num_agents', 0))
    # try HECRAS mapping first
    if getattr(simulation, 'use_hecras', False) and getattr(simulation, 'hecras_plan_path', None):
        try:
            pts = np.column_stack((simulation.X, simulation.Y))
            k_ = getattr(simulation, 'hecras_k', 8) if k is None else k
            vx = map_hecras_for_agents(simulation, pts, simulation.hecras_plan_path, field_names=['Velocity X'], k=k_)
            vy = map_hecras_for_agents(simulation, pts, simulation.hecras_plan_path, field_names=['Velocity Y'], k=k_)
            if isinstance(vx, dict):
                vx = np.asarray(vx.get('Velocity X'))
            else:
                vx = np.asarray(vx)
            if isinstance(vy, dict):
                vy = np.asarray(vy.get('Velocity Y'))
            else:
                vy = np.asarray(vy)
            if vx is not None and vy is not None and vx.size == n and vy.size == n:
                return np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)
        except Exception:
            # fall through to raster sampling
            pass

    # raster fallback: try vel_x/vel_y datasets
    env = getattr(simulation, 'hdf5', None)
    if env is not None and 'environment' in env:
        env_grp = env['environment']
        if 'vel_x' in env_grp and 'vel_y' in env_grp:
            try:
                # use sample_environment helper with transforms if present
                t_x = getattr(simulation, 'vel_x_rast_transform', getattr(simulation, 'depth_rast_transform', None))
                t_y = getattr(simulation, 'vel_y_rast_transform', getattr(simulation, 'depth_rast_transform', None))
                vx = sample_environment(simulation, t_x, 'vel_x')
                vy = sample_environment(simulation, t_y, 'vel_y')
                return np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)
            except Exception:
                pass

        # if only vel_dir available, return unit vectors
        if 'vel_dir' in env_grp:
            try:
                t = getattr(simulation, 'vel_dir_rast_transform', getattr(simulation, 'depth_rast_transform', None))
                dirs = sample_environment(simulation, t, 'vel_dir')
                dirs = np.asarray(dirs, dtype=float)
                vx = np.cos(dirs)
                vy = np.sin(dirs)
                return vx, vy
            except Exception:
                pass

    # nothing available — return NaNs so callers can detect missing flow
    return np.full(n, np.nan, dtype=float), np.full(n, np.nan, dtype=float)


def sample_environment(sim: Any, transform, raster_name: str):
    """Sample raster values at agent X/Y positions.

    Parity-focused port of legacy `sample_environment`.
    Accepts `sim` (object with `hdf5`, `X`, `Y`, `num_agents`) and an
    affine `transform` and raster dataset name. Returns a flattened array
    of length `sim.num_agents` of sampled values.
    """
    from emergent.fish_passage.geometry import geo_to_pixel_from_inv
    from emergent.fish_passage.utils import get_inv_transform

    inv = get_inv_transform(getattr(transform, '__self__', None) or getattr(sim, 'sim', None), transform)
    rows, cols = geo_to_pixel_from_inv(inv, sim.X, sim.Y)

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


def boundary_surface(sim: Any, wetted_name: str = 'wetted', distance_name: str = 'distance_to') -> None:
    """Compute distance-to-boundary raster from a binary `wetted` raster and store in environment.

    This is a compact parity implementation: where `wetted` is True, compute
    distance to the nearest non-wetted cell along raster connectivity using
    Euclidean distance on the raster grid; non-wetted cells get NaN.
    """
    h5 = getattr(sim, 'hdf5', None)
    if h5 is None:
        raise RuntimeError('Simulation object must have an open `hdf5` file')

    env = h5.get('environment', None)
    if env is None or wetted_name not in env:
        raise RuntimeError(f"Environment raster '{wetted_name}' missing")

    wetted = np.asarray(env[wetted_name])
    # create boolean mask of wetted cells
    mask = np.asarray(wetted, dtype=bool)

    # compute distance transform on inverted mask (distance to nearest False)
    try:
        from scipy.ndimage import distance_transform_edt
        # distance in pixels from wetted cells to nearest non-wetted cell
        dist = distance_transform_edt(mask).astype('f4')
    except Exception:
        # fallback: naive per-pixel loop (O(n^3) worst-case) for tiny rasters
        h, w = mask.shape
        dist = np.full((h, w), np.inf, dtype='f4')
        false_idxs = np.argwhere(~mask)
        true_idxs = np.argwhere(mask)
        if false_idxs.size == 0:
            # no boundary -> distances are inf; represent as NaN
            dist = np.full((h, w), np.nan, dtype='f4')
        else:
            for r, c in true_idxs:
                dif = false_idxs - np.array([r, c])
                d2 = np.sum(dif * dif, axis=1)
                d = np.sqrt(d2).min()
                dist[r, c] = float(d)

    # Make non-wetted cells NaN to mirror legacy semantics
    out = np.full_like(dist, np.nan, dtype='f4')
    out[mask] = dist[mask]

    if 'environment' not in h5:
        env = h5.create_group('environment')
    else:
        env = h5['environment']

    if distance_name in env:
        del env[distance_name]
    env.create_dataset(distance_name, data=out, dtype='f4')
    try:
        h5.flush()
    except Exception:
        pass


def update_mental_map(sim: Any, current_timestep: int, raster_name: str = 'depth') -> None:
    """Update per-agent mental maps stored under `memory/` in sim.hdf5.

    Behavior mirrors legacy semantics at a compact level:
    - Ensure `memory/<agent>` datasets exist (created by `initialize_mental_map`).
    - Use `sim.mental_map_transform` or `sim.depth_rast_transform` to map agent
      geographic positions into mental-map cell indices.
    - For each agent, sample the requested raster (using `sample_environment`) and
      write that scalar into the agent's memory cell.
    """
    h5 = getattr(sim, 'hdf5', None)
    if h5 is None:
        raise RuntimeError('Simulation object must have an open `hdf5` file')

    # prepare memory group
    if 'memory' not in h5:
        raise RuntimeError('Memory maps not initialized; call initialize_mental_map first')
    mem = h5['memory']

    num_agents = getattr(sim, 'num_agents', None)
    if num_agents is None:
        raise RuntimeError('Simulation must have `num_agents` attribute')

    # choose transform for mental map indexing
    transform = getattr(sim, 'mental_map_transform', getattr(sim, 'depth_rast_transform', None))
    if transform is None:
        raise RuntimeError('No affine transform available for mental map sampling')

    # sample requested raster values at agent positions using existing helper
    try:
        values = sample_environment(sim, transform, raster_name)
    except Exception:
        # If sampling fails, set NaNs
        values = np.full(int(num_agents), np.nan, dtype=float)

    # compute mental map indices using geo_to_pixel
    from emergent.fish_passage.geometry import geo_to_pixel_from_inv
    from emergent.fish_passage.utils import get_inv_transform
    inv = get_inv_transform(getattr(transform, '__self__', None) or getattr(sim, 'sim', None), transform)
    rows, cols = geo_to_pixel_from_inv(inv, sim.X, sim.Y)

    # per-agent writes
    for i in range(int(num_agents)):
        name = f"{i}"
        if name not in mem:
            # create a small default memory if missing
            h, w = max(3, int(np.round(getattr(sim, 'height', 10) / 5))), max(3, int(np.round(getattr(sim, 'width', 10) / 5)))
            mem.create_dataset(name, (h, w), dtype='f4')
            mem[name][:, :] = np.zeros((h, w), dtype='f4')

        ds = mem[name]
        # map row/col into ds bounds
        r = int(np.clip(int(round(rows[i])), 0, ds.shape[0] - 1))
        c = int(np.clip(int(round(cols[i])), 0, ds.shape[1] - 1))
        try:
            ds[r, c] = float(values[i])
        except Exception:
            # skip write if value not scalar
            continue

    try:
        h5.flush()
    except Exception:
        pass


def update_refugia_map(sim: Any, current_velocity: float = None, raster_name: str = 'depth') -> None:
    """Update per-agent refugia maps stored under `refugia/` in sim.hdf5.

    Compact parity implementation:
    - Ensures `refugia/<agent>` datasets exist (created by `initialize_refugia_map`).
    - Uses `sim.refugia_map_transform` or `sim.depth_rast_transform` to map agent
      positions into refugia-map cell indices.
    - Samples `raster_name` via `sample_environment` and writes scalar into
      refugia per-agent dataset at the computed cell.
    """
    h5 = getattr(sim, 'hdf5', None)
    if h5 is None:
        raise RuntimeError('Simulation object must have an open `hdf5` file')

    if 'refugia' not in h5:
        raise RuntimeError('Refugia maps not initialized; call initialize_refugia_map first')
    ref = h5['refugia']

    num_agents = getattr(sim, 'num_agents', None)
    if num_agents is None:
        raise RuntimeError('Simulation must have `num_agents` attribute')

    transform = getattr(sim, 'refugia_map_transform', getattr(sim, 'depth_rast_transform', None))
    if transform is None:
        raise RuntimeError('No affine transform available for refugia map sampling')

    try:
        values = sample_environment(sim, transform, raster_name)
    except Exception:
        values = np.full(int(num_agents), np.nan, dtype=float)

    from emergent.fish_passage.geometry import geo_to_pixel_from_inv
    from emergent.fish_passage.utils import get_inv_transform
    inv = get_inv_transform(getattr(transform, '__self__', None) or getattr(sim, 'sim', None), transform)
    rows, cols = geo_to_pixel_from_inv(inv, sim.X, sim.Y)

    for i in range(int(num_agents)):
        name = f"{i}"
        if name not in ref:
            h, w = max(3, int(np.round(getattr(sim, 'height', 10) / 5))), max(3, int(np.round(getattr(sim, 'width', 10) / 5)))
            ref.create_dataset(name, (h, w), dtype='f4')
            ref[name][:, :] = np.zeros((h, w), dtype='f4')
        ds = ref[name]
        r = int(np.clip(int(round(rows[i])), 0, ds.shape[0] - 1))
        c = int(np.clip(int(round(cols[i])), 0, ds.shape[1] - 1))
        try:
            ds[r, c] = float(values[i])
        except Exception:
            continue

    try:
        h5.flush()
    except Exception:
        pass


def initial_heading(sim: Any, default_heading: Optional[float] = None) -> np.ndarray:
    """Initialize agent headings.

    Strategy:
    - If `environment/vel_dir` raster exists, sample it with `sample_environment`.
    - Else if HECRAS is enabled and a hecras adapter is registered, map `vel_x` and `vel_y` to agents and compute `atan2`.
    - Else, use `default_heading` if provided, otherwise NaN.

    Writes into `agent_data/heading` first column if present and returns heading array.
    """
    num_agents = int(getattr(sim, 'num_agents', 0))
    headings = None

    # If HECRAS mapping or raster sampling produced flow components, compute headings
    try:
        vx, vy = get_agent_flow_components(sim)
        if vx is not None and vy is not None:
            headings = np.mod(np.arctan2(vy, vx), 2 * np.pi)
    except Exception:
        headings = None

    # last resort: default or NaN
    if headings is None:
        if default_heading is not None:
            headings = np.full(num_agents, float(default_heading), dtype=float)
        else:
            headings = np.full(num_agents, np.nan, dtype=float)

    # Write to agent_data if present
    try:
        h5 = sim.hdf5
        if 'agent_data' in h5 and 'heading' in h5['agent_data']:
            ds = h5['agent_data/heading']
            # write first column if ds has timesteps
            if ds.ndim == 2:
                ds[:, 0] = np.asarray(headings, dtype='f4')
            else:
                ds[:] = np.asarray(headings, dtype='f4')
    except Exception:
        pass

    try:
        if getattr(sim, 'hdf5', None) is not None:
            sim.hdf5.flush()
    except Exception:
        pass

    return headings




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
    """Delegating wrapper to the canonical HECRAS helper in hecras.py.

    This keeps the public `io` surface stable while the canonical
    implementation lives in `emergent.fish_passage.hecras`.
    """
    from emergent.fish_passage.hecras import infer_wetted_perimeter_from_hecras as _central
    return _central(hdf_path_or_file, depth_threshold=depth_threshold, max_nodes=max_nodes, raster_fallback_resolution=raster_fallback_resolution, verbose=verbose, timestep=timestep)


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
            from emergent.fish_passage.geometry import geo_to_pixel_from_inv
            from emergent.fish_passage.utils import get_inv_transform

            inv = get_inv_transform(getattr(simulation, '__self__', None) or getattr(simulation, 'sim', None), simulation.depth_rast_transform)
            orow, ocol = geo_to_pixel_from_inv(inv, [oy], [ox])
            orow = int(orow[0]); ocol = int(ocol[0])
        except Exception:
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
