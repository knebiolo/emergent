import numpy as np
import h5py
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from scipy.ndimage import label

def infer_wetted_perimeter_from_hecras(hdf_path_or_file, depth_threshold=0.05, max_nodes=5000, raster_fallback_resolution=5.0):
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
        depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0])
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
                raise RuntimeError('No valid perimeter polygons from vector method')
            merged = unary_union(polys)
            # Return exterior coords for merged polygon(s)
            if merged.geom_type == 'Polygon':
                return [list(merged.exterior.coords)]
            else:
                return [list(g.exterior.coords) for g in merged.geoms]

        except Exception as e:
            # Vector method failed — log and fall back to raster
            # (avoid printing loudly; raise a controlled signal)
            # print('vector perimeter method failed:', e)
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
