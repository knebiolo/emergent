import numpy as np
import h5py
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from scipy.ndimage import label
from scipy.spatial import cKDTree
from shapely.prepared import prep
from shapely.geometry import Point, LineString
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def map_hecras_for_agents(simulation_or_plan, agent_xy, plan_path=None, field_names=None, k=8, timestep=0):
    """Map agent coordinates to HECRAS nodal fields using IDW via HECRASMap.

    This function accepts either a `simulation` object that caches HECRASMap
    instances (preferred) or the raw `plan_path` and will return arrays
    for the requested field names.
    """
    # Lazy import of HECRASMap if available in caller environment
    try:
        # If a simulation-like object provided
        sim = simulation_or_plan
        if not hasattr(sim, '_hecras_maps'):
            sim._hecras_maps = {}
        if field_names is None:
            field_names = ['Cells Minimum Elevation']
        key = (str(plan_path or getattr(sim, 'hecras_plan_path', '')), tuple(field_names))
        if key not in sim._hecras_maps:
            sim._hecras_maps[key] = load_hecras_plan_cached(plan_path or getattr(sim, 'hecras_plan_path', None), field_names=field_names, timestep=timestep)
        m = sim._hecras_maps[key]
        out = m.map_idw(agent_xy, k=k)
        if len(out) == 1:
            return next(iter(out.values()))
        return out
    except Exception:
        # Fallback: if a plan path string was passed as first arg
        if isinstance(simulation_or_plan, str):
            plan = simulation_or_plan
            # Build a temporary HECRASMap and map
            m = HECRASMap(str(plan), field_names=field_names, timestep=timestep)
            out = m.map_idw(agent_xy, k=k)
            if len(out) == 1:
                return next(iter(out.values()))
            return out
        raise


def ensure_hdf_coords_from_hecras(simulation, plan_path, target_shape=None, target_transform=None, timestep=0):
    """Create `x_coords` and `y_coords` datasets in `simulation.hdf5` when missing."""
    hdf = getattr(simulation, 'hdf5', None)
    if hdf is None:
        return

    try:
        with h5py.File(str(plan_path), 'r') as ph:
            hecras_coords = ph['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]
    except Exception:
        hecras_coords = None

    if hecras_coords is None:
        return

    if target_transform is None:
        target_transform = compute_affine_from_hecras(hecras_coords)
    if target_shape is None:
        aff = target_transform
        minx, miny = float(hecras_coords[:, 0].min()), float(hecras_coords[:, 1].min())
        maxx, maxy = float(hecras_coords[:, 0].max()), float(hecras_coords[:, 1].max())
        width = max(1, int(np.ceil((maxx - minx) / abs(aff.a))))
        height = max(1, int(np.ceil((maxy - miny) / abs(aff.e))))
        target_shape = (height, width)

    height, width = target_shape
    if height == 0 or width == 0:
        return

    if 'x_coords' not in hdf:
        dset_x = hdf.create_dataset('x_coords', (height, width), dtype='float32')
    else:
        dset_x = hdf['x_coords']
    if 'y_coords' not in hdf:
        dset_y = hdf.create_dataset('y_coords', (height, width), dtype='float32')
    else:
        dset_y = hdf['y_coords']

    try:
        existing = np.asarray(dset_x[:])
        needs_populate = not np.isfinite(existing).any() or np.allclose(existing, 0.0)
    except Exception:
        needs_populate = True

    if needs_populate:
        cols = np.arange(width, dtype=np.float64)
        rows = np.arange(height, dtype=np.float64)
        col_grid, row_grid = np.meshgrid(cols, rows)
        xs, ys = pixel_to_geo(target_transform, row_grid, col_grid)
        dset_x[:, :] = xs.astype('float32')
        dset_y[:, :] = ys.astype('float32')

    simulation.depth_rast_transform = target_transform
    simulation.hdf_height = height
    simulation.hdf_width = width


def map_hecras_to_env_rasters(simulation, plan_path, field_names=None, k=1):
    # accept optional timestep forwarded from callers, but keep signature for compatibility
    timestep = 0
    if isinstance(field_names, dict) and 'timestep' in field_names:
        # legacy callers might pass a dict; support minimal unpack
        timestep = int(field_names.pop('timestep'))
    if hasattr(simulation, '_hecras_timestep'):
        timestep = getattr(simulation, '_hecras_timestep')
    """Map HECRAS nodal fields onto the full environment raster grid and write into `simulation.hdf5['environment']`."""
    if not hasattr(simulation, 'hdf5') or getattr(simulation, 'hdf5', None) is None:
        return False
    if field_names is None:
        field_names = getattr(simulation, 'hecras_fields', None)

    try:
        ensure_hdf_coords_from_hecras(simulation, plan_path, target_transform=getattr(simulation, 'depth_rast_transform', None), timestep=timestep)
    except Exception:
        pass

    env = simulation.hdf5.require_group('environment')
    if not hasattr(simulation, '_hecras_grid_xy') or simulation._hecras_grid_xy is None:
        if 'x_coords' in simulation.hdf5 and 'y_coords' in simulation.hdf5:
            xarr = np.asarray(simulation.hdf5['x_coords'])
            yarr = np.asarray(simulation.hdf5['y_coords'])
            h, w = xarr.shape
            XX = xarr.flatten()
            YY = yarr.flatten()
            simulation._hecras_grid_shape = (h, w)
            simulation._hecras_grid_xy = np.column_stack((XX, YY))
        else:
            m = load_hecras_plan_cached(simulation, plan_path, field_names=[field_names[0]] if field_names else None, timestep=timestep)
            coords = m.coords
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
            simulation._hecras_grid_shape = (h, w)
            simulation._hecras_grid_xy = np.column_stack((xs.flatten(), ys.flatten()))

    # flatten grid coords and map
    grid_xy = simulation._hecras_grid_xy
    mapped = map_hecras_for_agents(simulation, grid_xy, plan_path, field_names=field_names, k=k, timestep=timestep)
    # Write mapped rasters into env group as 2D arrays
    if isinstance(mapped, dict):
        for name, arr in mapped.items():
            h, w = simulation._hecras_grid_shape
            ds_name = name
            if ds_name in env:
                del env[ds_name]
            env.create_dataset(ds_name, (h, w), dtype='f4')
            env[ds_name][:, :] = np.asarray(arr).reshape(h, w)
    else:
        # single field
        arr = np.asarray(mapped)
        h, w = simulation._hecras_grid_shape
        ds_name = field_names[0] if field_names else 'field'
        if ds_name in env:
            del env[ds_name]
        env.create_dataset(ds_name, (h, w), dtype='f4')
        env[ds_name][:, :] = arr.reshape(h, w)
    return True


def compute_distance_to_bank_hecras(wetted_info, coords, median_spacing=None):
    """Compute distance-to-bank for HECRAS irregular mesh cells using graph Dijkstra."""
    wetted_mask = wetted_info['wetted_mask']
    perimeter_indices = wetted_info['perimeter_cells']
    if median_spacing is None:
        sample_size = min(1000, len(coords))
        sample_idx = np.random.choice(len(coords), size=sample_size, replace=False)
        sample_coords = coords[sample_idx]
        sample_tree = cKDTree(sample_coords)
        dists, _ = sample_tree.query(sample_coords, k=2)
        median_spacing = np.median(dists[:, 1])

    wetted_coords = coords[wetted_mask]
    wetted_indices = np.where(wetted_mask)[0]
    n_wetted = len(wetted_coords)
    wetted_tree = cKDTree(wetted_coords)
    connectivity_radius = median_spacing * 1.5
    pairs = wetted_tree.query_pairs(r=connectivity_radius, output_type='ndarray')
    if len(pairs) > 0:
        row = pairs[:, 0]
        col = pairs[:, 1]
        edge_coords_i = wetted_coords[row]
        edge_coords_j = wetted_coords[col]
        edge_dists = np.sqrt(np.sum((edge_coords_i - edge_coords_j)**2, axis=1))
        row_sym = np.concatenate([row, col])
        col_sym = np.concatenate([col, row])
        data_sym = np.concatenate([edge_dists, edge_dists])
        graph = csr_matrix((data_sym, (row_sym, col_sym)), shape=(n_wetted, n_wetted))
        perimeter_wetted_indices = []
        for perim_idx in perimeter_indices:
            pos = np.where(wetted_indices == perim_idx)[0]
            if len(pos) > 0:
                perimeter_wetted_indices.append(pos[0])
        perimeter_wetted_indices = np.array(perimeter_wetted_indices, dtype=np.int32)
        if len(perimeter_wetted_indices) > 0:
            dist_matrix = dijkstra(csgraph=graph, directed=False, indices=perimeter_wetted_indices)
            if dist_matrix.ndim == 2:
                distances_wetted = np.min(dist_matrix, axis=0)
            else:
                distances_wetted = dist_matrix
        else:
            distances_wetted = np.full(n_wetted, np.inf)
    else:
        distances_wetted = np.full(n_wetted, np.inf)

    distances_all = np.full(len(coords), np.nan, dtype=np.float32)
    distances_all[wetted_indices] = distances_wetted.astype(np.float32)
    distances_all[perimeter_indices] = 0.0
    return distances_all


def derive_centerline_from_hecras_distance(coords, distances, wetted_mask, crs=None, min_distance_threshold=None, min_length=50):
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter1d
    valid_mask = wetted_mask & np.isfinite(distances)
    valid_coords = coords[valid_mask]
    valid_distances = distances[valid_mask]
    if len(valid_coords) == 0:
        return None
    if min_distance_threshold is None:
        min_distance_threshold = np.percentile(valid_distances, 75)
    ridge_mask = valid_distances >= min_distance_threshold
    ridge_coords = valid_coords[ridge_mask]
    ridge_distances = valid_distances[ridge_mask]
    if len(ridge_coords) < 10:
        return None
    start_idx = np.argmax(ridge_distances)
    ordered_indices = [start_idx]
    remaining = set(range(len(ridge_coords))) - {start_idx}
    current_idx = start_idx
    ridge_tree = cKDTree(ridge_coords)
    while remaining:
        current_pt = ridge_coords[current_idx]
        dists, indices = ridge_tree.query(current_pt, k=len(ridge_coords))
        next_idx = None
        for idx in indices:
            if idx in remaining:
                next_idx = idx
                break
        if next_idx is None:
            break
        ordered_indices.append(next_idx)
        remaining.remove(next_idx)
        current_idx = next_idx
    ordered_coords = ridge_coords[ordered_indices]
    if len(ordered_coords) > 5:
        sigma = max(1, len(ordered_coords) // 20)
        smoothed_x = gaussian_filter1d(ordered_coords[:, 0], sigma=sigma)
        smoothed_y = gaussian_filter1d(ordered_coords[:, 1], sigma=sigma)
        ordered_coords = np.column_stack((smoothed_x, smoothed_y))
    centerline = LineString(ordered_coords)
    if centerline.length < min_length:
        return None
    return centerline


def extract_centerline_fast_hecras(plan_path, depth_threshold=0.05, sample_fraction=0.1, min_length=50):
    """Fast centerline extraction from HECRAS by sampling wetted cells."""
    with h5py.File(plan_path, 'r') as hdf:
        coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
        depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0])
    wetted_mask = depth > depth_threshold
    coords_wet = coords[wetted_mask]
    if len(coords_wet) < 10:
        return None
    # sample
    n_sample = max(50, int(len(coords_wet) * sample_fraction))
    idx = np.random.choice(len(coords_wet), size=n_sample, replace=False)
    sample = coords_wet[idx]
    # PCA along-stream
    mean = sample.mean(axis=0)
    cov = np.cov((sample - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    proj = (sample - mean).dot(principal)
    order = np.argsort(proj)
    ordered = sample[order]
    # smooth
    from scipy.ndimage import gaussian_filter1d
    if len(ordered) > 5:
        sigma = max(1, len(ordered) // 50)
        sx = gaussian_filter1d(ordered[:, 0], sigma=sigma)
        sy = gaussian_filter1d(ordered[:, 1], sigma=sigma)
        ordered = np.column_stack((sx, sy))
    centerline = LineString(ordered)
    if centerline.length < min_length:
        return None
    return centerline

def infer_wetted_perimeter_from_hecras(hdf_path_or_file, depth_threshold=0.05, max_nodes=5000, raster_fallback_resolution=5.0, verbose=False, timestep=0):
    """Infer wetted perimeter using HECRAS vector geometry if possible.

    Returns a list of (x,y) points representing the wetted perimeter polygon(s) in order.

    Parameters
    ----------
    hdf_path_or_file : str or h5py.File
        Path to HEC-RAS HDF5 plan or an open h5py.File object.
    depth_threshold : float
        Threshold for hydraulic depth to consider a cell wetted.
    max_nodes : int
        If number of wetted cells >> max_nodes, this function still computes perimeter but
        caller may wish to thin before building Delaunay.
    raster_fallback_resolution : float
        Cell size (in model units) for coarse raster fallback when vector method is too costly.

    Notes
    -----
    This function favors a vector method that inspects facepoints and cell-face mappings to
    extract the exact wetted boundary. If that fails or is too expensive for the input size,
    it falls back to a coarse rasterization + polygonize approach.
    """
    close_file = False
    if isinstance(hdf_path_or_file, str):
        hdf = h5py.File(hdf_path_or_file, 'r')
        close_file = True
    else:
        hdf = hdf_path_or_file

    try:
        ds = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth']
        # pick requested timestep if dataset is time-series
        if getattr(ds, 'ndim', 0) > 0 and ds.shape[0] > 1:
            t = int(min(timestep, ds.shape[0]-1))
            depth = np.array(ds[t])
        else:
            depth = np.array(ds[0])
        wetted_mask = depth > depth_threshold

        # Try vector method using facepoints and perimeter flags
        try:
            facepoints = np.array(hdf['Geometry/2D Flow Areas/2D area/FacePoints Coordinate'])
            is_perim = np.array(hdf['Geometry/2D Flow Areas/2D area/FacePoints Is Perimeter'])
            face_info = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Face and Orientation Info']).astype(int)

            perim_mask = (is_perim == -1)
            perim_touch = np.zeros(len(facepoints), dtype=bool)

            wetted_idx = np.nonzero(wetted_mask)[0]
            # vectorized-ish loop over wetted cells (still Python loop but only over wetted cells)
            for i in wetted_idx:
                start, count = face_info[i]
                if count <= 0:
                    continue
                idxs = np.arange(start, start+count)
                # guard indices
                idxs = idxs[idxs < len(perim_mask)]
                perim_touch[idxs] |= perim_mask[idxs]

            # If no perimeter points touched, fall back to raster
            if not perim_touch.any():
                if verbose:
                    print('No perimeter facepoints touched by wetted cells — falling back to raster')
                raise RuntimeError('No perimeter facepoints touched by wetted cells — falling back to raster')

            # Now extract ordered perimeter coordinates from 'Perimeter' dataset
            perim_coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Perimeter'])
            # Map perimeter coordinates to the facepoints index by nearest neighbor (facepoints may duplicate)
            # Build KD-tree for facepoints for mapping if sizes are large — use simple nearest lookup here for clarity
            from scipy.spatial import cKDTree
            tree = cKDTree(facepoints)
            dists, idxs = tree.query(perim_coords, k=1)
            # keep only the perimeter coordinates that map to a touched facepoint
            touched = perim_touch[idxs]
            if not np.any(touched):
                if verbose:
                    print('Perimeter points mapping found no touched points — fallback')
                raise RuntimeError('Perimeter points mapping found no touched points — fallback')
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

            # Convert rings to polygons and union small gaps
            polys = [Polygon(r) for r in rings if len(r) >= 3]
            if not polys:
                if verbose:
                    print('No valid perimeter polygons from vector method')
                raise RuntimeError('No valid perimeter polygons from vector method')
            merged = unary_union(polys)
            # Return exterior coords for merged polygon(s)
            if merged.geom_type == 'Polygon':
                return [list(merged.exterior.coords)]
            else:
                return [list(g.exterior.coords) for g in merged.geoms]

        except Exception as e:
            # Vector method failed — log and fall back to raster
            if verbose:
                print('vector perimeter method failed:', e)
            pass

        # Raster fallback: rasterize wetted cells to coarse grid and polygonize
        try:
            # Get cell centers and build a coarse raster bounds
            coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
            x = coords[:,0]
            y = coords[:,1]
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            # build coarse grid
            res = raster_fallback_resolution
            nx = max(3, int(np.ceil((xmax - xmin) / res)))
            ny = max(3, int(np.ceil((ymax - ymin) / res)))
            # map each cell center to grid index
            gx = np.clip(((x - xmin) / (xmax - xmin) * (nx-1)).astype(int), 0, nx-1)
            gy = np.clip(((y - ymin) / (ymax - ymin) * (ny-1)).astype(int), 0, ny-1)
            grid = np.zeros((ny, nx), dtype=np.uint8)
            grid[gy, gx] = wetted_mask.astype(np.uint8)

            # Fill small holes via label and area threshold
            lbl, n = label(grid==1)
            # keep labels with area > 1 (very coarse)
            keep = set(np.where([np.sum(lbl==i) for i in range(1, n+1)]) [0] + 1)
            mask = np.isin(lbl, list(keep))

            # polygonize mask: extract boundaries from mask contours
            # Simple approach: convert mask to polygons using shapely (via marching squares not available here)
            from shapely.geometry import shape, mapping
            from shapely.geometry import Polygon as ShPolygon
            polys = []
            # brute-force: find connected components and make bounding polygons (coarse)
            for i in range(1, n+1):
                if np.sum(lbl==i) == 0:
                    continue
                inds = np.column_stack(np.where(lbl==i))
                # convert grid indices back to coords for polygon approx
                pts = []
                for gy_i, gx_i in inds:
                    px = xmin + (gx_i + 0.5) * (xmax - xmin) / (nx-1)
                    py = ymin + (gy_i + 0.5) * (ymax - ymin) / (ny-1)
                    pts.append((px, py))
                if len(pts) >= 3:
                    polys.append(ShPolygon(pts).convex_hull)
            if not polys:
                if verbose:
                    print('Raster fallback produced no polygons')
                raise RuntimeError('Raster fallback produced no polygons')
            merged = unary_union(polys)
            if merged.geom_type == 'Polygon':
                return [list(merged.exterior.coords)]
            else:
                return [list(g.exterior.coords) for g in merged.geoms]
        except Exception as e:
            # both methods failed
            raise RuntimeError('Both vector and raster wetted-perimeter inference failed: ' + str(e))

    finally:
        if close_file:
            hdf.close()
