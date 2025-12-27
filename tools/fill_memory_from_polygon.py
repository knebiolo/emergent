import sys
import os
import h5py
import numpy as np
from affine import Affine
from emergent.salmon_abm.utils import geo_to_pixel

def fill_memory(preseed_h5, polygon_shp, avoid_cell_size=50.0, start_ts=100, end_ts=1):
    preseed_h5 = os.path.abspath(preseed_h5)
    polygon_shp = os.path.abspath(polygon_shp) if polygon_shp else None
    if not os.path.exists(preseed_h5):
        print('Preseed file not found:', preseed_h5)
        return 2

    # open file and attempt to compute mental-map dimensions from environment/depth
    with h5py.File(preseed_h5, 'a') as f:
        # infer environment bounds
        if 'environment/depth' in f:
            depth = np.array(f['environment/depth'])
            nrows, ncols = depth.shape
            # get original depth transform (a,b,c,d,e,f) if present on sim attrs
            # approximate transform: a = pixel x size, e = -pixel y size, c/f = offsets
            # attempt to read stored transform values saved by io.write_raster_to_hdf5
            a = f.attrs.get('depth_transform_a', None)
            b = f.attrs.get('depth_transform_b', None)
            c = f.attrs.get('depth_transform_c', None)
            d = f.attrs.get('depth_transform_d', None)
            e = f.attrs.get('depth_transform_e', None)
            ff = f.attrs.get('depth_transform_f', None)
            # fallback transform if not saved
            if a is None:
                # assume a=1, e=-1, origin at 0,0
                a, b, c, d, e, ff = 1.0, 0.0, 0.0, 0.0, -1.0, 0.0
        else:
            # nothing to sample; create a reasonable grid
            nrows, ncols = 5000, 5000
            a, b, c, d, e, ff = 1.0, 0.0, 0.0, 0.0, -1.0, 0.0

        # compute mental map dims
        avoid_h = int(np.round(nrows / avoid_cell_size, 0)) + 1
        avoid_w = int(np.round(ncols / avoid_cell_size, 0)) + 1

        # Attempt to align mental-map grid origin with environment world coordinates.
        # Prefer explicit x_coords/y_coords datasets if present; otherwise try saved depth transform attrs.
        origin_x = 0.0
        origin_y = 0.0
        try:
            if 'environment/x_coords' in f and 'environment/y_coords' in f:
                xcoords = np.array(f['environment/x_coords'])
                ycoords = np.array(f['environment/y_coords'])
                # world coordinate of top-left pixel (row=0,col=0)
                origin_x = float(xcoords[0, 0])
                origin_y = float(ycoords[0, 0])
            else:
                # fallback: try transform attrs created by raster import in this file
                tr = f.attrs.get('depth_rast_transform', None) or f.attrs.get('depth_transform', None) or f.attrs.get('depth_transform_a', None)
                if tr is not None:
                    try:
                        tr = np.asarray(tr)
                        if tr.size >= 6:
                            a, b, c, d, e, ff = tr.ravel()[:6]
                            origin_x = float(c)
                            origin_y = float(ff)
                    except Exception:
                        pass
                else:
                    # try to discover a headless HDF5 in the same outputs/diagnostics folder
                    try:
                        d = os.path.dirname(preseed_h5)
                        # look for *_headless.h5 files
                        cand = [os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith('_headless.h5')]
                        if cand:
                            candf = cand[0]
                            with h5py.File(candf, 'r') as ch:
                                if 'environment/x_coords' in ch and 'environment/y_coords' in ch:
                                    xc = np.array(ch['environment/x_coords'])
                                    yc = np.array(ch['environment/y_coords'])
                                    origin_x = float(xc[0, 0])
                                    origin_y = float(yc[0, 0])
                                else:
                                    # maybe transform stored as attr
                                    tr2 = ch.attrs.get('depth_rast_transform', None) or ch.attrs.get('depth_transform', None)
                                    if tr2 is not None:
                                        tr2 = np.asarray(tr2)
                                        if tr2.size >= 6:
                                            a, b, c, d, e, ff = tr2.ravel()[:6]
                                            origin_x = float(c)
                                            origin_y = float(ff)
                    except Exception:
                        pass
        except Exception:
            pass

        # mental map transform with aligned origin (pixel size = avoid_cell_size)
        # Note: we use the origin at the top-left of the environment so mental grid coords map to same world CRS
        mental_aff = Affine(avoid_cell_size, 0.0, origin_x, 0.0, -avoid_cell_size, origin_y)

        # build a mask of the polygon area in mental-map pixel space
        poly_mask = np.zeros((avoid_h, avoid_w), dtype=np.float32)

        # if polygon provided, rasterize it into mask using naive bounding-box approach
        if polygon_shp and os.path.exists(polygon_shp):
            try:
                import geopandas as gpd
                from shapely.geometry import mapping
                gdf = gpd.read_file(polygon_shp)
                geom = gdf.unary_union if len(gdf) > 1 else gdf.geometry.iloc[0]
                # generate coords of mental-map pixel centers in world space
                cols = np.arange(avoid_w)
                rows = np.arange(avoid_h)
                col_inds, row_inds = np.meshgrid(cols, rows)
                xs = mental_aff * (col_inds + 0.5, row_inds + 0.5)
                xs = np.array(xs)
                # xs is (2, H, W)
                xs0 = xs[0]
                xs1 = xs[1]
                # check containment
                pts = np.column_stack((xs0.ravel(), xs1.ravel()))
                contains = np.array([geom.contains(type('P', (), {'x':float(p[0]), 'y':float(p[1])})()) for p in pts])
                poly_mask[:, :] = contains.reshape((avoid_h, avoid_w)).astype(np.float32)
            except Exception:
                # fallback: fill entire mask
                poly_mask[:, :] = 1.0
        else:
            # no polygon: fill entire area
            poly_mask[:, :] = 1.0

        # create or replace memory group
        if 'memory' in f:
            try:
                del f['memory']
            except Exception:
                pass
        mem = f.create_group('memory')

        # create a gradient of timestamps across the polygon mask
        # map mask True cells -> values from start_ts down to end_ts across one axis
        ys = np.linspace(start_ts, end_ts, avoid_h).astype(np.float32)
        grad = np.tile(ys[:, None], (1, avoid_w))
        grad_masked = grad * poly_mask

        # write datasets for requested number of agents equal to 1000 if present, else default
        # number of agents is not required for memory preseed; if more agents exist later the sim will ignore extras
        nagents = 1000
        for i in range(nagents):
            ds = mem.create_dataset(str(i), data=grad_masked.astype('f4'), dtype='f4')

        # set mental_map_transform attributes so sim can reconstruct
        # Use the full affine (including c,f origin) so geo->pixel mapping aligns
        mental_params = np.array([mental_aff.a, mental_aff.b, mental_aff.c, mental_aff.d, mental_aff.e, mental_aff.f], dtype=np.float32)
        f.attrs['mental_map_transform'] = mental_params

        # Debug check: map a few sample environment coordinates into mental pixels
        try:
            if 'environment/x_coords' in f and 'environment/y_coords' in f:
                xc = np.array(f['environment/x_coords'])
                yc = np.array(f['environment/y_coords'])
                # pick a few sample indices (center and corners)
                rr = [0, xc.shape[0]//2, xc.shape[0]-1]
                cc = [0, xc.shape[1]//2, xc.shape[1]-1]
                samples = []
                for r in rr:
                    for c in cc:
                        samples.append((float(xc[r, c]), float(yc[r, c])))
                print('Debug: mapping sample environment coords to mental-map pixels using mental_map_transform:')
                for sx, sy in samples:
                    try:
                        prow, pcol = geo_to_pixel(sx, sy, mental_params)
                        print(f'  world=({sx:.3f},{sy:.3f}) -> pixel=(row={prow},col={pcol})')
                    except Exception as ex:
                        print('  mapping failed for', (sx, sy), '->', ex)
        except Exception:
            pass
        try:
            f.flush()
        except Exception:
            pass

    print(f'Filled memory in {preseed_h5} for polygon={polygon_shp or "<all>"} shape=({avoid_h},{avoid_w})')
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python fill_memory_from_polygon.py <preseed.h5> [polygon_shp] [--cell-size=N] [--start-ts=S] [--end-ts=E]')
        sys.exit(2)
    pre = sys.argv[1]
    shp = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    cell = 50.0
    st = 100
    et = 1
    for a in sys.argv[2:]:
        if a.startswith('--cell-size='):
            cell = float(a.split('=')[1])
        if a.startswith('--start-ts='):
            st = int(a.split('=')[1])
        if a.startswith('--end-ts='):
            et = int(a.split('=')[1])
    sys.exit(fill_memory(pre, shp, avoid_cell_size=cell, start_ts=st, end_ts=et))
