"""Simple headless runner for quick smoke tests without PyQt/GIS deps.

Creates a simulation, assigns simple wind/current samplers, prepopulates
waypoints, calls spawn(), and advances the sim for a few steps.
"""
import time
import os
from datetime import datetime
import numpy as np
import sys

# Ensure src is on sys.path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Lightweight shim for geopandas so tests can run without installing the full stack.
# simulation_core imports geopandas at module import time for bbox reprojection; provide
# a minimal GeoDataFrame with to_crs() -> object with total_bounds that matches expectations.
try:
    import geopandas as _gpd
except Exception:
    import types as _types
    def _make_gdf(geometry, crs=None):
        class _GDF:
            def __init__(self, geometry, crs=None):
                self.geometry = geometry
                self.crs = crs
            def to_crs(self, target_crs):
                # geometry is expected to be a shapely geometry or list thereof
                geom = self.geometry[0] if isinstance(self.geometry, (list, tuple)) else self.geometry
                bounds = geom.bounds
                return _types.SimpleNamespace(total_bounds=bounds)
        return _GDF(geometry, crs)
    class _GeoSeries:
        def __init__(self, geoms, crs=None):
            self.geometry = list(geoms)
            self.crs = crs
            self.values = self.geometry
        def simplify(self, tol, preserve_topology=True):
            # trivial no-op
            return self
        def plot(self, ax=None, **kwargs):
            # no-op for headless tests
            return []

    def _gdf_factory(geometry, crs=None):
        return _types.SimpleNamespace(geometry=geometry, crs=crs, to_crs=lambda target: _types.SimpleNamespace(total_bounds=geometry[0].bounds))

    _gpd = _types.SimpleNamespace(GeoDataFrame=_gdf_factory, GeoSeries=_GeoSeries)
    sys.modules['geopandas'] = _gpd

# Minimal shapely.geometry shim used by simulation_core during startup
try:
    import shapely.geometry as _sh_geom
except Exception:
    class _Box:
        def __init__(self, minx, miny, maxx, maxy):
            self._bounds = (minx, miny, maxx, maxy)
        @property
        def bounds(self):
            return self._bounds

    def _box(minx, miny, maxx, maxy):
        return _Box(minx, miny, maxx, maxy)

    import types as _types2
    _sh_mod = _types2.ModuleType('shapely')
    _sh_geom_mod = _types2.ModuleType('shapely.geometry')
    _sh_geom_mod.box = _box
    _sh_geom_mod.Point = lambda *a, **k: None
    _sh_geom_mod.LineString = lambda *a, **k: None
    class _Polygon:
        def __init__(self, coords):
            # coords could be a shapely-like sequence; compute simple bounds
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            self._bounds = (min(xs), min(ys), max(xs), max(ys))
        @property
        def bounds(self):
            return self._bounds

    _sh_geom_mod.MultiPolygon = lambda *a, **k: None
    _sh_geom_mod.Polygon = _Polygon
    _sh_mod.geometry = _sh_geom_mod
    sys.modules['shapely'] = _sh_mod
    sys.modules['shapely.geometry'] = _sh_geom_mod
    # Provide shapely.strtree.STRtree shim
    _strtree_mod = _types2.ModuleType('shapely.strtree')
    class STRtree:
        def __init__(self, objs):
            self._objs = objs
        def query(self, geom):
            return []
    _strtree_mod.STRtree = STRtree
    sys.modules['shapely.strtree'] = _strtree_mod
    # provide shapely.ops.unary_union used by simulation_core
    _sh_ops = _types2.ModuleType('shapely.ops')
    def _unary_union(objs):
        # naive union: return first object's bounds wrapper
        return objs[0] if objs else None
    _sh_ops.unary_union = _unary_union
    sys.modules['shapely.ops'] = _sh_ops
    # lightweight fiona shim
    _fiona = _types2.ModuleType('fiona')
    _fiona.open = lambda *a, **k: []
    sys.modules['fiona'] = _fiona

    # minimal pyproj shim with Transformer.from_crs(...).transform(x,y) -> returns (x,y)
    _pyproj = _types2.ModuleType('pyproj')
    class _Transformer:
        @staticmethod
        def from_crs(a, b, always_xy=True):
            class T:
                def transform(self, xs, ys):
                    # assume xs, ys are arrays or scalars; return them unchanged
                    return xs, ys
            return T()
    _pyproj.Transformer = _Transformer
    sys.modules['pyproj'] = _pyproj
    # minimal pyqtgraph shim (we only need QtCore later)
    _pg = _types2.ModuleType('pyqtgraph')
    _pg.Qt = _types2.SimpleNamespace(QtCore=_types2.SimpleNamespace())
    sys.modules['pyqtgraph'] = _pg
    # ensure pyqtgraph.Qt is importable
    _pg_qt = _types2.ModuleType('pyqtgraph.Qt')
    _pg_qt.QtCore = _types2.SimpleNamespace()
    sys.modules['pyqtgraph.Qt'] = _pg_qt

from emergent.ship_abm.simulation_core import simulation


def main():
    print('HEADLESS: creating simulation (no ENC)')
    sim = simulation(port_name='Seattle', dt=0.1, T=5.0, n_agents=1, load_enc=False)

    # simple zero environment
    sim._env_loaded = True
    sim.wind_fn = lambda lon, lat, when: np.zeros((len(np.atleast_1d(lon)), 2))
    sim.current_fn = lambda lon, lat, when: np.zeros((len(np.atleast_1d(lon)), 2))

    # set trivial waypoints and spawn
    cx = 0.5 * (sim.minx + sim.maxx)
    cy = 0.5 * (sim.miny + sim.maxy)
    sim.waypoints = [[(cx, cy), (cx + 100.0, cy)] for _ in range(sim.n)]
    print('HEADLESS: calling spawn()')
    sim.spawn()

    print('HEADLESS: running sim.run()')
    sim.run()
    print('HEADLESS: done')


if __name__ == '__main__':
    main()
