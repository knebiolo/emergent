"""HECRAS adapter utilities for viewer_v3.

Provide a minimal `extract_depth_points` function that reads Cell Hydraulic
Depth and Cell Center Coordinates and returns (pts, vals) suitable for
`mesh_builder.build_mesh`.
"""
from typing import Tuple
import numpy as np
import h5py

from emergent.salmon_abm.tin_helpers import sample_evenly


def extract_depth_points(hdf_path_or_file, timestep: int = 0, depth_thresh: float | None = None, max_nodes: int | None = None, use_wetted_perimeter: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Extract point coordinates and depth values from a HECRAS HDF5 plan.

    Args:
        hdf_path_or_file: Path to an HDF5 file or an open `h5py.File`.
        timestep: Time index to read (default 0).
        depth_thresh: If provided, keep only cells with depth > depth_thresh.
        max_nodes: If provided and number of points > max_nodes, downsample via `sample_evenly`.

    Returns:
        pts: (N,2) array of X,Y coordinates.
        vals: (N,) array of depth values.

    Raises:
        ValueError: on invalid inputs or missing datasets.
    """
    close = False
    if isinstance(hdf_path_or_file, str):
        hdf = h5py.File(hdf_path_or_file, 'r')
        close = True
    else:
        hdf = hdf_path_or_file

    try:
        try:
            coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
        except Exception as e:
            raise ValueError('HECRAS HDF5 missing Cells Center Coordinate') from e

        try:
            ds = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth']
        except Exception as e:
            raise ValueError('HECRAS HDF5 missing Cell Hydraulic Depth dataset') from e

        # pick timestep safely
        if getattr(ds, 'ndim', 0) > 0 and ds.shape[0] > 1:
            t = int(min(timestep, ds.shape[0] - 1))
            depth = np.array(ds[t])
        else:
            try:
                depth = np.array(ds[0])
            except Exception:
                depth = np.array(ds[:])

        if depth.shape[0] != coords.shape[0]:
            # try flattening or reshaping if necessary
            depth = depth.flatten()
            if depth.shape[0] != coords.shape[0]:
                raise ValueError('Depth array length does not match coordinate count')

        # optionally mask by depth threshold
        if depth_thresh is not None:
            mask = np.isfinite(depth) & (depth > float(depth_thresh))
        else:
            mask = np.isfinite(depth)

        # optionally refine mask via connectivity (keep largest connected wetted region)
        if use_wetted_perimeter and np.any(mask):
            try:
                from scipy.spatial import cKDTree
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import connected_components
                # coords of candidate wetted cells
                cand_idx = np.nonzero(mask)[0]
                cand_coords = coords[cand_idx][:, :2]
                if len(cand_coords) > 1:
                    # connectivity radius: median neighbor spacing * 1.5 (heuristic)
                    tree = cKDTree(cand_coords)
                    dists, inds = tree.query(cand_coords, k=2)
                    median_spacing = float(np.median(dists[:, 1])) if dists.shape[1] > 1 else float(np.median(dists))
                    radius = max(median_spacing * 1.5, median_spacing + 1e-6)
                    pairs = tree.query_pairs(r=radius, output_type='ndarray')
                    if pairs.size > 0:
                        row = pairs[:, 0]
                        col = pairs[:, 1]
                        data = np.ones(len(row), dtype=np.int8)
                        # undirected graph
                        row_sym = np.concatenate([row, col])
                        col_sym = np.concatenate([col, row])
                        data_sym = np.concatenate([data, data])
                        graph = csr_matrix((data_sym, (row_sym, col_sym)), shape=(len(cand_coords), len(cand_coords)))
                        ncomp, labels = connected_components(csgraph=graph, directed=False)
                        # find largest component
                        counts = np.bincount(labels)
                        largest = int(np.argmax(counts))
                        keep_local = (labels == largest)
                        # build global mask
                        new_mask = np.zeros_like(mask, dtype=bool)
                        new_mask[cand_idx[keep_local]] = True
                        mask = new_mask
            except Exception:
                # on any error fallback to depth-only mask
                pass

        if not np.any(mask):
            # return empty arrays rather than raise
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

        pts = coords[mask][:, :2].astype(float)
        vals = depth[mask].astype(float)

        # downsample if requested
        if max_nodes is not None and len(pts) > int(max_nodes):
            try:
                pts, vals = sample_evenly(pts, vals, max_nodes=int(max_nodes), grid_dim=120)
            except Exception:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(pts), size=int(max_nodes), replace=False)
                pts = pts[idx]
                vals = vals[idx]

        return pts, vals
    finally:
        if close:
            hdf.close()


def build_mesh_from_hecras(hdf_path_or_file, timestep: int = 0, depth_thresh: float | None = 0.05, max_nodes: int | None = 5000, vert_exag: float = 1.0, use_wetted_perimeter: bool = False, alpha: float | None = None):
    """Convenience helper: extract depth points and build a TIN mesh.

    Returns (verts, faces, colors) suitable for ModernglViewerWidget.set_mesh.
    If no valid points are found, returns empty arrays with the expected shapes.
    """
    from emergent.salmon_abm.viewer_v3 import mesh_builder

    pts, vals = extract_depth_points(hdf_path_or_file, timestep=timestep, depth_thresh=depth_thresh, max_nodes=max_nodes, use_wetted_perimeter=use_wetted_perimeter)
    if pts is None or pts.size == 0:
        import numpy as _np
        return _np.zeros((0, 3), dtype='f4'), _np.zeros((0, 3), dtype='i4'), _np.zeros((0, 4), dtype='f4')

    verts, faces, colors = mesh_builder.build_mesh(pts, vals, vert_exag=float(vert_exag), alpha=alpha)
    # ensure types expected by renderer
    verts = verts.astype('f4')
    faces = faces.astype('i4')
    colors = colors.astype('f4')
    return verts, faces, colors
