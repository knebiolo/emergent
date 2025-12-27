import argparse
import csv
import h5py
import numpy as np
from affine import Affine

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preseed', required=True, help='Preseed HDF5 path containing memory/')
    p.add_argument('--trace', required=True, help='Headless trace CSV (timestep,agent,x,y)')
    p.add_argument('--agents', default='0,1,2', help='Comma-separated agent indices to inspect')
    return p.parse_args()

def main():
    args = parse_args()
    pre = args.preseed
    trace = args.trace
    agents = [int(x) for x in args.agents.split(',') if x.strip()]

    # read trace and extract timestep 0 rows for agents of interest
    samples = {}
    with open(trace, 'r', newline='') as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            if int(row['timestep']) == 0:
                a = int(row['agent'])
                if a in agents:
                    samples[a] = (float(row['x']), float(row['y']))

    print('Loaded trace samples for agents:', sorted(samples.keys()))

    with h5py.File(pre, 'r') as fh:
        if 'memory' not in fh:
            print('No memory group in', pre)
            return 1
        # read mental_map_transform attr if present
        tr = fh.attrs.get('mental_map_transform', None)
        if tr is None:
            print('No mental_map_transform attribute found; assuming cellsize=10 with origin 0,0')
            a,b,c,d,e,fv = 10.0,0.0,0.0,0.0,-10.0,0.0
        else:
            tr = np.asarray(tr, dtype=float)
            a,b,c,d,e,fv = tr.tolist()
        m_aff = Affine(a,b,c,d,e,fv)
        inv = ~m_aff

        for agent in sorted(samples.keys()):
            x,y = samples[agent]
            colf, rowf = inv * (x,y)
            col = int(round(colf))
            row = int(round(rowf))
            dsname = str(agent)
            info = {'agent':agent, 'x':x, 'y':y, 'row':row, 'col':col}
            print('Agent', agent, 'pos=({:.3f},{:.3f}) -> pixel (row,col)=({}, {})'.format(x,y,row,col))
            if dsname in fh['memory']:
                ds = fh['memory'][dsname]
                h,w = ds.shape
                if 0 <= row < h and 0 <= col < w:
                    val = float(ds[row, col])
                    print('  memory[{}][{},{}] = {}'.format(dsname, row, col, val))
                else:
                    print('  pixel out of bounds for memory[{}] shape={} -> skipping'.format(dsname, ds.shape))
            else:
                print('  no dataset memory/{} in preseed file'.format(dsname))

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
