import h5py, numpy as np, json, sys
from pathlib import Path

h5path = Path('outputs/diagnostics/mini_rheo_test_headless.h5')
weights_file = Path('outputs/diagnostics/test_weights_rheo_only_small.json')
if not h5path.exists():
    print('Missing h5:', h5path)
    sys.exit(1)

weights = {}
if weights_file.exists():
    with open(weights_file,'r') as fh:
        weights = json.load(fh)

with h5py.File(h5path,'r') as f:
    print('HDF5 keys:', list(f.keys()))
    # try agent positions from agent_data/X and Y time 0
    agent_X = None
    agent_Y = None
    if 'agent_data' in f and 'X' in f['agent_data']:
        Xarr = f['agent_data']['X'][:]
        Yarr = f['agent_data']['Y'][:]
        # time index 0
        agent_X = Xarr[:,0] if Xarr.ndim==2 else Xarr
        agent_Y = Yarr[:,0] if Yarr.ndim==2 else Yarr
    elif 'X' in f and 'Y' in f:
        agent_X = f['X'][:]
        agent_Y = f['Y'][:]
    else:
        print('No agent X/Y found')
        sys.exit(1)
    n = agent_X.size
    print('num agents:', n)
    # load vel_x/vel_y or vel_mag+dir
    env = f.get('environment', None)
    if env is None:
        print('No environment group found')
        sys.exit(1)
    velx = env.get('vel_x', None)
    vely = env.get('vel_y', None)
    x_coords = env.get('x_coords', None)
    y_coords = env.get('y_coords', None)
    if velx is None or vely is None:
        print('vel_x or vel_y missing in HDF5')
        sys.exit(1)
    velx_arr = np.array(velx)
    vely_arr = np.array(vely)
    x_coords_arr = np.array(x_coords) if x_coords is not None else None
    y_coords_arr = np.array(y_coords) if y_coords is not None else None
    # for each agent, pick nearest pixel by comparing to x_coords/y_coords
    sampled_vx = np.full(n, np.nan)
    sampled_vy = np.full(n, np.nan)
    if x_coords_arr is not None and y_coords_arr is not None:
        # assume x_coords arr shape (rows, cols), x_coords[0,:] varies with cols, y_coords[:,0] varies with rows
        cols_vals = x_coords_arr[0,:]
        rows_vals = y_coords_arr[:,0]
        for i in range(n):
            # find closest col
            col = int(np.argmin(np.abs(cols_vals - agent_X[i])))
            row = int(np.argmin(np.abs(rows_vals - agent_Y[i])))
            # clip
            row = max(0, min(row, velx_arr.shape[0]-1))
            col = max(0, min(col, velx_arr.shape[1]-1))
            sampled_vx[i] = velx_arr[row, col]
            sampled_vy[i] = vely_arr[row, col]
    else:
        # fallback: assume raster coords are pixel indices and agent_X/Y are indices
        rows = np.clip(agent_Y.astype(int), 0, velx_arr.shape[0]-1)
        cols = np.clip(agent_X.astype(int), 0, velx_arr.shape[1]-1)
        sampled_vx = velx_arr[rows, cols]
        sampled_vy = vely_arr[rows, cols]

    # compute upstream unit vectors
    v = np.column_stack((sampled_vx, sampled_vy))
    # upstream = -v
    upstream = -v
    norms = np.linalg.norm(upstream, axis=1)
    unit_up = np.zeros_like(upstream)
    valid = norms>0
    unit_up[valid] = (upstream[valid].T / norms[valid]).T

    # expected rheotaxis vector (weight * unit_up)
    wr = float(weights.get('rheotaxis', 25000.0))
    expected_rheo = unit_up * wr

    # try to read final headings from agent_data heading or top-level heading
    heading_arr = None
    if 'agent_data' in f and 'heading' in f['agent_data']:
        harr = f['agent_data']['heading'][:]
        heading_arr = harr[:,0] if harr.ndim==2 else harr
    elif 'heading' in f:
        heading_arr = np.array(f['heading'][:])
    else:
        print('No heading data found')

    head_dir = None
    if heading_arr is not None:
        head_dir = np.column_stack((np.cos(heading_arr), np.sin(heading_arr)))

    # also try to read head_vec stored in behavior NPZ not in HDF5; skip

    # compute dot products
    dots_rheo_up = []
    dots_head_up = []
    for i in range(n):
        eu = unit_up[i]
        # rheotaxis vector from NPZ might differ; but expected rheo is eu*wr
        # compute normalized rheo unit (eu)
        if not np.any(np.isfinite(eu)):
            dots_rheo_up.append(np.nan)
            dots_head_up.append(np.nan)
            continue
        # dot between expected rheo direction and unit_up is 1 trivially
        dots_rheo_up.append(np.dot(eu, eu))
        if head_dir is not None:
            dots_head_up.append(np.dot(head_dir[i], eu))
        else:
            dots_head_up.append(np.nan)

    # print per-agent summary (first 10) and aggregate stats
    print('\nSampled vel_x (first 10):', sampled_vx[:10].tolist())
    print('Sampled vel_y (first 10):', sampled_vy[:10].tolist())
    print('\nPer-agent dot(head_dir, upstream_unit) (first 10):')
    for i,d in enumerate(dots_head_up[:10]):
        print(f'{i}: {d:.4f}')
    valid_head = [d for d in dots_head_up if np.isfinite(d)]
    if valid_head:
        print('\nHead vs upstream: mean dot =', float(np.mean(valid_head)), 'median=', float(np.median(valid_head)))
    else:
        print('\nNo heading data to compare')
    print('\nDone')
