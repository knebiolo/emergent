"""HECRAS HDF5 direct reader + mapping helpers for salmon_abm.

This module supports direct, time-varying sampling from a HECRAS plan
without rasterizing each timestep. It provides:
- cached KDTree over HECRAS cell centers
- time-indexed field reads
- IDW mapping to arbitrary points (agents, grids)
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import h5py

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

SECONDS_PER_DAY = 86400.0

FIELD_ALIASES = {
    "depth": "Cell Hydraulic Depth",
    "vel_x": "Cell Velocity - Velocity X",
    "vel_y": "Cell Velocity - Velocity Y",
    "wsel": "Water Surface",
}


def _geom_path(area_name: str) -> str:
    return f"Geometry/2D Flow Areas/{area_name}"


def _results_base(area_name: str) -> str:
    return (
        "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
        f"2D Flow Areas/{area_name}"
    )


def _time_path() -> str:
    return "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time"


def _apply_idw(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply IDW weights, ignoring non-finite values per-row."""
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(vals)
    if np.all(valid):
        return np.sum(vals * w, axis=1)
    w = np.where(valid, w, 0.0)
    wsum = np.sum(w, axis=1)
    out = np.full(vals.shape[0], np.nan, dtype=float)
    ok = wsum > 0
    if np.any(ok):
        out[ok] = np.sum(vals * w, axis=1)[ok] / wsum[ok]
    return out


def compute_grid_from_coords(
    coords: np.ndarray, target_cell_size: float | None = None
) -> Tuple[Tuple[float, float, float, float, float, float], Tuple[int, int]]:
    """Compute a simple north-up grid transform and shape from HECRAS coords."""
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"Invalid HECRAS coords shape: {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("HECRAS coords empty; cannot build grid")

    if target_cell_size is not None:
        cell = float(target_cell_size)
        if not np.isfinite(cell) or cell <= 0.0:
            raise ValueError(f"Invalid target_cell_size: {target_cell_size}")
    else:
        if cKDTree is None:
            raise RuntimeError("cKDTree not available; cannot infer HECRAS spacing")
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        spacing = np.median(dists[:, 1])
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = 1.0
        cell = float(spacing)

    minx = float(np.min(pts[:, 0]))
    miny = float(np.min(pts[:, 1]))
    maxx = float(np.max(pts[:, 0]))
    maxy = float(np.max(pts[:, 1]))

    ncols = max(1, int(np.ceil((maxx - minx) / cell)) + 1)
    nrows = max(1, int(np.ceil((maxy - miny) / cell)) + 1)

    # north-up grid: x = minx + cell * col, y = maxy - cell * row
    transform = (cell, 0.0, minx, 0.0, -cell, maxy)
    return transform, (nrows, ncols)


def iter_grid_points(
    transform: Tuple[float, float, float, float, float, float],
    shape: Tuple[int, int],
    max_points: int = 250000,
):
    """Yield chunked grid points for a given transform/shape."""
    nrows, ncols = int(shape[0]), int(shape[1])
    if nrows <= 0 or ncols <= 0:
        return
    max_points = int(max_points) if max_points is not None else 0
    if max_points <= 0:
        max_points = ncols
    rows_per_chunk = max(1, int(max_points // max(1, ncols)))

    a, b, c, d, e, f = transform
    cols = np.arange(ncols, dtype=float)
    for row_start in range(0, nrows, rows_per_chunk):
        row_end = min(nrows, row_start + rows_per_chunk)
        rows = np.arange(row_start, row_end, dtype=float)
        col_idx, row_idx = np.meshgrid(cols, rows)
        x_coords = a * col_idx + b * row_idx + c
        y_coords = d * col_idx + e * row_idx + f
        pts = np.column_stack((x_coords.ravel(), y_coords.ravel()))
        yield row_start, row_end, pts


class HecrasPlan:
    """HECRAS plan reader with cached KDTree over cell centers."""

    def __init__(self, plan_path: str, area_name: str = "2D area"):
        if cKDTree is None:
            raise RuntimeError("cKDTree not available; cannot build HECRAS KDTree")
        self.plan_path = str(plan_path)
        self.area_name = str(area_name)
        self._hdf = h5py.File(self.plan_path, "r")
        self._coord_mask = None
        self.coords = self._load_coords()
        self.tree = cKDTree(self.coords)
        self.time_days = self._read_time()
        self.time_s = (
            np.asarray(self.time_days, dtype=float) * SECONDS_PER_DAY
            if self.time_days is not None
            else None
        )

    def close(self) -> None:
        try:
            self._hdf.close()
        except Exception:
            pass

    def _load_coords(self) -> np.ndarray:
        path = f"/{_geom_path(self.area_name)}/Cells Center Coordinate"
        if path not in self._hdf:
            raise KeyError(f"HECRAS coords not found: {path}")
        coords = np.asarray(self._hdf[path][:], dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"Invalid HECRAS coords shape: {coords.shape}")
        mask = np.isfinite(coords).all(axis=1)
        if not np.any(mask):
            raise ValueError("HECRAS coords are all non-finite")
        self._coord_mask = mask
        return coords[mask]

    def _read_time(self) -> np.ndarray | None:
        path = _time_path()
        if path not in self._hdf:
            return None
        arr = np.asarray(self._hdf[path][:], dtype=float)
        if arr.ndim != 1:
            return None
        return arr

    def read_field(self, field_name: str, timestep_idx: int) -> np.ndarray:
        base = _results_base(self.area_name)
        path = f"{base}/{field_name}"
        if path not in self._hdf:
            raise KeyError(f"HECRAS field not found: {path}")
        ds = self._hdf[path]
        if ds.ndim <= 1:
            arr = np.asarray(ds[:], dtype=float)
        else:
            arr = np.asarray(ds[int(timestep_idx)], dtype=float)

        if self._coord_mask is not None and arr.shape[0] == self._coord_mask.shape[0]:
            arr = arr[self._coord_mask]
        return arr

    def read_field_alias(self, alias: str, timestep_idx: int) -> np.ndarray:
        if alias not in FIELD_ALIASES:
            raise KeyError(f"Unknown HECRAS field alias: {alias}")
        return self.read_field(FIELD_ALIASES[alias], timestep_idx)

    def read_fields(
        self, timestep_idx: int, field_aliases: Iterable[str]
    ) -> Dict[str, np.ndarray]:
        out = {}
        for alias in field_aliases:
            out[str(alias)] = self.read_field_alias(alias, timestep_idx)
        return out

    def _query_weights(
        self, points: np.ndarray, k: int = 8, eps: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape((1, 2))
        n_nodes = int(self.coords.shape[0])
        if n_nodes <= 0:
            raise ValueError("HECRAS coords empty; cannot query")
        k_use = min(max(1, int(k)), n_nodes)
        dists, inds = self.tree.query(pts, k=k_use)
        if k_use == 1:
            dists = dists[:, None]
            inds = inds[:, None]
        inv = 1.0 / (dists + eps)
        wsum = np.sum(inv, axis=1, keepdims=True)
        if not np.all(np.isfinite(wsum)) or np.any(wsum <= 0):
            raise ValueError("Invalid HECRAS IDW weights (zero or non-finite)")
        weights = inv / wsum
        return inds, weights

    def map_values_to_points(
        self,
        field_values: Dict[str, np.ndarray],
        points: np.ndarray,
        k: int = 8,
        eps: float = 1e-8,
    ) -> Dict[str, np.ndarray]:
        inds, weights = self._query_weights(points, k=k, eps=eps)
        out: Dict[str, np.ndarray] = {}
        for alias, values in field_values.items():
            vals = np.asarray(values, dtype=float)
            if vals.shape[0] != self.coords.shape[0]:
                raise ValueError(
                    f"HECRAS field length mismatch for {alias}: "
                    f"{vals.shape[0]} vs {self.coords.shape[0]}"
                )
            mapped = _apply_idw(vals[inds], weights)
            out[alias] = mapped
        return out

    def map_fields_to_points(
        self,
        points: np.ndarray,
        timestep_idx: int,
        field_aliases: Iterable[str],
        k: int = 8,
        eps: float = 1e-8,
    ) -> Dict[str, np.ndarray]:
        values = self.read_fields(timestep_idx, field_aliases)
        return self.map_values_to_points(values, points, k=k, eps=eps)

    def map_values_to_grid(
        self,
        field_values: Dict[str, np.ndarray],
        shape: Tuple[int, int],
        transform: Tuple[float, float, float, float, float, float],
        k: int = 8,
        eps: float = 1e-8,
        max_points: int = 250000,
    ) -> Dict[str, np.ndarray]:
        nrows, ncols = int(shape[0]), int(shape[1])
        if nrows <= 0 or ncols <= 0:
            raise ValueError(f"Invalid grid shape: {shape}")
        out = {alias: np.empty((nrows, ncols), dtype=np.float32) for alias in field_values}
        for row_start, row_end, pts in iter_grid_points(transform, shape, max_points=max_points):
            inds, weights = self._query_weights(pts, k=k, eps=eps)
            nrows_chunk = row_end - row_start
            for alias, values in field_values.items():
                vals = np.asarray(values, dtype=float)
                if vals.shape[0] != self.coords.shape[0]:
                    raise ValueError(
                        f"HECRAS field length mismatch for {alias}: "
                        f"{vals.shape[0]} vs {self.coords.shape[0]}"
                    )
                mapped = _apply_idw(vals[inds], weights)
                out[alias][row_start:row_end, :] = mapped.reshape((nrows_chunk, ncols))
        return out

    def map_fields_to_grid(
        self,
        timestep_idx: int,
        field_aliases: Iterable[str],
        shape: Tuple[int, int],
        transform: Tuple[float, float, float, float, float, float],
        k: int = 8,
        eps: float = 1e-8,
        max_points: int = 250000,
    ) -> Dict[str, np.ndarray]:
        values = self.read_fields(timestep_idx, field_aliases)
        return self.map_values_to_grid(
            values, shape, transform, k=k, eps=eps, max_points=max_points
        )


__all__ = [
    "HecrasPlan",
    "FIELD_ALIASES",
    "compute_grid_from_coords",
    "iter_grid_points",
    "SECONDS_PER_DAY",
]

