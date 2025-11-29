import h5py
import numpy as np
from scipy.spatial import cKDTree
import fiona
from shapely.geometry import shape, Point

plan = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf"
start_shp = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\starting_location\starting_location.shp"

with h5py.File(plan, 'r') as h:
    coords = h['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]
    print('Cell centers shape:', coords.shape)
    # build KDTree
    tree = cKDTree(coords)
    # pick a few sample points from starting shapefile
    pts = []
    with fiona.open(start_shp, 'r') as src:
        for feat in src:
            geom = shape(feat['geometry'])
            if geom.geom_type == 'Point':
                pts.append((geom.x, geom.y))
            else:
                pts.append(geom.representative_point().coords[0])
    pts = np.array(pts)
    print('Starting points:', pts.shape)
    # query KDTree
    dists, inds = tree.query(pts, k=3)
    print('Distances:', dists)
    print('Indices:', inds)
    # sample a field (Cells Minimum Elevation)
    elev = h['/Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'][:]
    print('elev shape', elev.shape)
    mapped = np.sum(elev[inds] * (1.0 / (dists + 1e-6)) / np.sum(1.0 / (dists + 1e-6), axis=1)[:,None], axis=1)
    print('Mapped elevations for starting points:', mapped)
