import h5py
import numpy as np
import sys
from emergent.salmon_abm.utils import geo_to_pixel

pre = sys.argv[1]
headless = sys.argv[2]
trace = sys.argv[3] if len(sys.argv)>3 else None

with h5py.File(pre,'r') as f:
    print('Preseed keys:', list(f.keys()))
    mental = f.attrs.get('mental_map_transform', None)
    print('Preseed mental_map_transform:', mental)
    if 'memory' in f:
        keys = list(f['memory'].keys())
        print('Memory keys count:', len(keys))
        if keys:
            print('memory[0] shape:', f['memory'][keys[0]].shape)

with h5py.File(headless,'r') as h:
    print('Headless keys:', list(h.keys()))
    if 'x_coords' in h and 'y_coords' in h:
        xc = np.array(h['x_coords'])
        yc = np.array(h['y_coords'])
        print('x_coords shape', xc.shape)
        pts = [ (float(xc[0,0]), float(yc[0,0])), (float(xc[xc.shape[0]//2, xc.shape[1]//2]), float(yc[yc.shape[0]//2, yc.shape[1]//2])), (float(xc[-1,-1]), float(yc[-1,-1])) ]
        print('Sample world points (UL, center, LR):')
        for p in pts:
            print('  ', p)
        print('Mapping using preseed mental_params:')
        if mental is not None:
            for p in pts:
                try:
                    r,c = geo_to_pixel(p[0], p[1], mental)
                    print(f'  world={p} -> pixel=(row={r},col={c})')
                except Exception as ex:
                    print('  mapping failed',ex)

# If trace provided, map first few agent positions
if trace:
    import csv
    with open(trace,'r') as fh:
        rdr = csv.DictReader(fh)
        rows = list(rdr)
        if rows:
            print('First trace row keys:', rows[0].keys())
            samples = []
            for idx in [0,1,2,10,50,100]:
                if idx < len(rows):
                    row = rows[idx]
                    x = float(row.get('X', row.get('x', row.get('agent_x',0))))
                    y = float(row.get('Y', row.get('y', row.get('agent_y',0))))
                    samples.append((idx,x,y))
            print('Mapping sample agent positions:')
            for i,x,y in samples:
                try:
                    r,c = geo_to_pixel(x,y,mental)
                    print(f'  agent {i} world=({x},{y}) -> pixel=(row={r},col={c})')
                except Exception as ex:
                    print('  mapping failed for agent',i,ex)
