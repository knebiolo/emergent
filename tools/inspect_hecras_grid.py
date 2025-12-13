import h5py, numpy as np
from pathlib import Path
p = Path('tmp/debug_synthetic.h5')
with h5py.File(p,'r') as h:
    ds = h['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth']
    print('ds.shape', getattr(ds,'shape',None), 'ndim', getattr(ds,'ndim',None))
    depth = np.array(ds[0])
    print('depth len', len(depth), 'min/max', depth.min(), depth.max(), 'sum>', (depth>0.1).sum())
    coords = np.array(h['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
    x = coords[:,0]; y = coords[:,1]
    xmin,xmax = x.min(), x.max(); ymin,ymax = y.min(), y.max()
    res=2.0
    nx = max(3, int(np.ceil((xmax - xmin) / res)))
    ny = max(3, int(np.ceil((ymax - ymin) / res)))
    print('nx,ny', nx, ny)
    gx = np.clip(((x - xmin) / (xmax - xmin) * (nx-1)).astype(int), 0, nx-1)
    gy = np.clip(((y - ymin) / (ymax - ymin) * (ny-1)).astype(int), 0, ny-1)
    print('gx unique', np.unique(gx))
    print('gy unique', np.unique(gy))
    grid = np.zeros((ny,nx), dtype=int)
    wetted_mask = depth>0.1
    print('wetted_mask sum', wetted_mask.sum())
    # assign elementwise to ensure correct mapping
    for i,(gy_i,gx_i) in enumerate(zip(gy,gx)):
        val = int(wetted_mask[i])
        if i < 30:
            print('assign', i, 'gy,gx', gy_i, gx_i, 'val', val)
        grid[gy_i,gx_i] = val
    print('grid sum', grid.sum())
    print('grid array:\n', grid)
    from scipy.ndimage import label
    lbl,n = label(grid==1)
    print('labels n', n)
    for i in range(1,n+1):
        print('label',i,'count', (lbl==i).sum())
