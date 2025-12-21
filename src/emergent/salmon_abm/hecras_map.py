import h5py
import numpy as np
from scipy.spatial import cKDTree


class HECRASMap:
    """Lightweight container for a HECRAS plan KDTree and field values.

    Usage:
      m = HECRASMap(plan_path, field_name='Cells Minimum Elevation')
      vals = m.map_idw(query_pts, k=8)
    """
    def __init__(self, plan_path, field_names=None):
        self.plan_path = plan_path
        if field_names is None:
            field_names = ['Cells Minimum Elevation']
        # store requested field names (list)
        self.field_names = list(field_names)
        self._load_plan()

    def _find_dataset_by_name(self, hdf, name_pattern):
        """Search HDF recursively for a dataset whose name contains name_pattern (case-insensitive).

        Returns the dataset path if found, otherwise None.
        """
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
        # n_coords is unknown here; we'll return best available: prefer Results paths
        # First, look for any candidate with 'results' in path
        results_cands = [c for c in candidates if 'results/' in c[0].lower()]
        if results_cands:
            return results_cands[0][0]
        # otherwise return the first candidate
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
                    # try to find a dataset anywhere that contains fname
                    ds_path = self._find_dataset_by_name(h, fname)
                    if ds_path is not None:
                        ds = h[ds_path]
                        # if time series (first dim > 1), take last timestep
                        if ds.ndim > 1:
                            arr = ds[-1]
                        else:
                            arr = ds[:]
                    else:
                        raise KeyError(f"Field '{fname}' not found in HECRAS HDF: {self.plan_path}")
                fields[fname] = np.asarray(arr)

        # normalize field arrays to align with coords length
        n_coords = coords.shape[0]
        def normalize_field_array(arr):
            arr = np.asarray(arr)
            # if already 1D and matches
            if arr.ndim == 1 and arr.shape[0] == n_coords:
                return arr
            # if total size matches, reshape
            if arr.size == n_coords:
                return arr.reshape(n_coords,)
            # try to find axis with length == n_coords
            for axis, dim in enumerate(arr.shape):
                if dim == n_coords:
                    # build index tuple: select last on other axes
                    idx = []
                    for i in range(arr.ndim):
                        if i == axis:
                            idx.append(slice(None))
                        else:
                            idx.append(-1)
                    sliced = arr[tuple(idx)]
                    return np.asarray(sliced).reshape(n_coords,)
            # As a last resort, return nan-filled array
            return np.full((n_coords,), np.nan)

        # choose primary field (for valid-cell masking) as first field
        primary = self.field_names[0]
        # normalize all fields first
        normed = {k: normalize_field_array(v) for k, v in fields.items()}
        mask = np.isfinite(normed[primary])

        self.coords = coords[mask].astype(np.float64)
        # store each field masked and casted
        self.fields = {k: np.asarray(v[mask], dtype=np.float64) for k, v in normed.items()}
        # build KDTree once
        self.tree = cKDTree(self.coords)

    def map_idw(self, query_pts, k=8, eps=1e-8):
        """Map `query_pts` (N x 2) to a dict of field_name -> mapped values via IDW.

        Returns: dict where each key is a field name and value is a (N,) array.
        """
        query = np.asarray(query_pts, dtype=np.float64)
        if query.ndim == 1:
            query = query.reshape(1, 2)
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


def load_hecras_plan_cached(simulation, plan_path, field_names=None):
    """Load a HECRAS plan and cache HECRASMap on the simulation object.

    Stores in `simulation._hecras_maps` dict keyed by (plan_path, tuple(field_names)).
    """
    if not hasattr(simulation, '_hecras_maps'):
        simulation._hecras_maps = {}
    if field_names is None:
        field_names = ['Cells Minimum Elevation']
    key = (str(plan_path), tuple(field_names))
    if key not in simulation._hecras_maps:
        simulation._hecras_maps[key] = HECRASMap(str(plan_path), field_names=field_names)
    return simulation._hecras_maps[key]


def map_hecras_for_agents(simulation, agent_xy, plan_path, field_names=None, k=8):
    """Map `agent_xy` (N x 2) to one or more HECRAS fields.

    Returns:
      - If one field requested: (N,) array
      - If multiple fields: dict field_name -> (N,) array
    """
    m = load_hecras_plan_cached(simulation, plan_path, field_names=field_names)
    out = m.map_idw(agent_xy, k=k)
    # if single field requested, return array directly
    if len(out) == 1:
        return next(iter(out.values()))
    return out
