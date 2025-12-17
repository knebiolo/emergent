"""HECRAS adapter utilities for viewer_v3.

Provide a minimal `extract_depth_points` function that reads Cell Hydraulic
Depth and Cell Center Coordinates and returns (pts, vals) suitable for
`mesh_builder.build_mesh`.
"""
from typing import Tuple
import numpy as np
import h5py

from emergent.salmon_abm.tin_helpers import sample_evenly


def extract_depth_points(hdf_path_or_file, timestep: int = 0, depth_thresh: float | None = None, max_nodes: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
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
