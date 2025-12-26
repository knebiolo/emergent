import numpy as np
import glob, os, h5py, math

# helper
def angle_between(u, v):
    if u is None or v is None:
        return np.nan
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0 or not np.isfinite(nu) or not np.isfinite(nv):
        return np.nan
    cosang = np.dot(u, v) / (nu * nv)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

# Try to find latest NPZ behavior dump
files_all = sorted(glob.glob('outputs/diagnostics/behavior_debug_step_*.npz'), key=os.path.getmtime)
use_npz = False
rheo = None
head = None
if files_all:
    files = [f for f in files_all if not f.endswith('_alignment.npz')]
    if not files:
        files = files_all
    npz_path = files[-1]
    print('Found behavior NPZ:', npz_path)
    try:
        npz = np.load(npz_path)
        if 'rheotaxis_vec' in npz and 'head_vec' in npz:
            rheo = npz['rheotaxis_vec']
            head = npz['head_vec']
            use_npz = True
            print('Using rheotaxis/head vectors from NPZ')
        else:
            print('NPZ missing expected keys; falling back to HDF5-only analysis')
    except Exception as e:
        print('Failed to load NPZ:', e)

# locate latest headless HDF5
h5_files = sorted(glob.glob('outputs/diagnostics/*headless.h5'), key=os.path.getmtime)
if h5_files:
    h5_path = h5_files[-1]
    print('Using HDF5:', h5_path)
else:
    print('No headless HDF5 found in outputs/diagnostics; aborting')
    raise SystemExit(1)

h5 = h5py.File(h5_path, 'r')

# If no NPZ provided, build rheo/head from HDF5
if not use_npz:
    # read agent positions
    if 'agent_data/X' in h5:
        agent_X = h5['agent_data/X'][:, 0]
        agent_Y = h5['agent_data/Y'][:, 0]
    else:
        agent_X = h5['X'][:]
        agent_Y = h5['Y'][:]
    # read vel rasters and coords
    vx = h5['environment/vel_x'][:]
    vy = h5['environment/vel_y'][:]
    x_coords = h5['environment/x_coords'][:]
    y_coords = h5['environment/y_coords'][:]
    n_agents = agent_X.shape[0]
    sampled_vx = np.full(n_agents, np.nan)
    sampled_vy = np.full(n_agents, np.nan)
    cols_vals = x_coords[0, :]
    rows_vals = y_coords[:, 0]
    for i in range(n_agents):
        col = int(np.argmin(np.abs(cols_vals - agent_X[i])))
        row = int(np.argmin(np.abs(rows_vals - agent_Y[i])))
        col = max(0, min(col, vx.shape[1]-1))
        row = max(0, min(row, vx.shape[0]-1))
        sampled_vx[i] = vx[row, col]
        sampled_vy[i] = vy[row, col]
    # rheotaxis points upstream i.e. -vel
    rheo = np.column_stack((-sampled_vx, -sampled_vy))
    # headings from persisted headings if available
    if 'agent_data/heading' in h5:
        head_angles = h5['agent_data/heading'][:, 0]
        head = np.column_stack((np.cos(head_angles), np.sin(head_angles)))
    elif 'heading' in h5:
        head_angles = h5['heading'][:]
        head = np.column_stack((np.cos(head_angles), np.sin(head_angles)))
    else:
        print('No headings in HDF5; cannot compute head vs upstream. Aborting')
        h5.close()
        raise SystemExit(1)
    print(f'Built rheo/head from HDF5; n_agents={rheo.shape[0]}')

# Now compute angles
n = min(rheo.shape[0], head.shape[0])
ang_head_rheo = np.zeros(n)
ang_rheo_up = np.zeros(n)
ang_head_up = np.zeros(n)
for i in range(n):
    rv = rheo[i]
    hv = head[i]
    ang_head_rheo[i] = angle_between(hv, rv)
    # upstream vector is -vel; derive from rheo since rheo = -vel
    ang_rheo_up[i] = angle_between(rv, -rv)
    ang_head_up[i] = angle_between(hv, -rv)

# summary function
import numpy as _np
def summarize(arr, name):
    valid = _np.isfinite(arr)
    if not valid.any():
        print(f'{name}: no valid samples')
        return
    a = arr[valid]
    print(f'{name}: mean_abs={_np.mean(_np.abs(a)):.2f} deg, median_abs={_np.median(_np.abs(a)):.2f} deg, n={a.size}')
    print(f'  pct within 30deg: {(_np.sum(_np.abs(a)<=30)/a.size*100):.1f}%')
    print(f'  pct within 90deg: {(_np.sum(_np.abs(a)<=90)/a.size*100):.1f}%')

print('\nSample (first 10 agents) angles:')
for i in range(min(10,n)):
    print(f'agent {i}: ang(head, rheo)={ang_head_rheo[i]:.1f} deg, ang(rheo,up)={ang_rheo_up[i]:.1f} deg, ang(head,up)={ang_head_up[i]:.1f} deg')

print('\nAggregates:')
summarize(ang_head_rheo, 'head vs rheotaxis')
summarize(ang_rheo_up, 'rheotaxis vs upstream')
summarize(ang_head_up, 'head vs upstream')

# close h5
if h5 is not None:
    h5.close()

print('\nDone')
