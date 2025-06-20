# -*- coding: utf-8 -*-
import numpy as np
from itertools import product
import geopandas as gpd
from shapely.strtree import STRtree
from collections import OrderedDict


TILE_METRES = 2000          #   2 km × 2 km tiles
LRU_CAP     = 32             #   max live tiles (tune to GPU RAM)

def _bbox_to_tiles(xmin, ymin, xmax, ymax, size=TILE_METRES):
    """Return all (ix, iy) touching this bbox."""
    eps = 1e-6 * size         # tiny buffer to avoid numerical jitter
    ix0 = int(np.floor((xmin - eps) / size))
    ix1 = int(np.floor((xmax + eps) / size))
    iy0 = int(np.floor((ymin - eps) / size))
    iy1 = int(np.floor((ymax + eps) / size))
    return list(product(range(ix0, ix1 + 1), range(iy0, iy1 + 1)))

def pre_tile(enc_series: gpd.GeoSeries, tol: float = 10.0):

    """
    Pre-chunk ENC geometries onto tile grid.
    Returns dict {tile_id: GeoSeries}.
    """
    enc = enc_series.simplify(tol, preserve_topology=True)
    tile_map = {}
    for geom in enc.geometry:
        if geom.is_empty:
            continue
        for tid in _bbox_to_tiles(*geom.bounds):
            tile_map.setdefault(tid, []).append(geom)
    # collapse lists → GeoSeries + STRtree for hit-testing
    for tid, geoms in tile_map.items():
        gs   = gpd.GeoSeries(geoms, crs=enc_series.crs)
        tree = STRtree(gs.values)            # ← build explicit spatial index
        tile_map[tid] = dict(gs=gs, tree=tree)
    return tile_map


# ─────────────────────────────────────────────────────────────────────────────
class TileCache:
    """LRU cache that stores already-drawn PathCollections per tile."""
    def __init__(self, ax, tile_dict):
        self.ax   = ax
        self.tmap = tile_dict          # output of pre_tile
        self.art  = OrderedDict()      # {tile_id: PathCollection}

    def _draw_tile(self, tid):
        gs = self.tmap[tid]["gs"]
        # add the tile’s PatchCollection and capture the handle
        before = set(self.ax.collections)

        gs.plot(ax=self.ax, facecolor='#f2e9dc', edgecolor='none',
                linewidth=0.0, zorder=0, antialiased=False)
        # the new collection(s) are whatever appeared since “before”
        new_artists = list(set(self.ax.collections) - before)
        if not new_artists:
            return                      # nothing drawn (empty geo)
        pc = new_artists[0]            # first (only) PatchCollection
        self.art[tid] = pc
        self.art.move_to_end(tid)
        # evict LRU
        while len(self.art) > LRU_CAP:
            old_tid, old_pc = self.art.popitem(last=False)
            if old_pc in self.ax.collections:   # only remove if still live
                try:
                    old_pc.remove()
                except ValueError:
                    pass               # already gone → ignore
                    
    def ensure_visible(self, view_bbox):
        xmin, xmax = view_bbox.xmin, view_bbox.xmax
        ymin, ymax = view_bbox.ymin, view_bbox.ymax
        needed = set(_bbox_to_tiles(xmin, ymin, xmax, ymax))
        changed = False
        for tid in needed.difference(self.art.keys()):
            if tid in self.tmap:
                self._draw_tile(tid)
                changed = True
        return changed 
        # optionally remove off-screen early (LRU handles it lazily)