import numpy as np
import glob, os, h5py, math

# find latest behavior NPZ (non-alignment preferred)
files_all = sorted(glob.glob('outputs/diagnostics/behavior_debug_step_*.npz'), key=os.path.getmtime)
if not files_all:
    print('No behavior NPZs found')
    raise SystemExit(1)
files = [f for f in files_all if not f.endswith('_alignment.npz')]
if not files:
    files = files_all
npz_path = files[-1]
print('Using NPZ:', npz_path)
npz = np.load(npz_path)
if 'rheotaxis_vec' not in npz or 'head_vec' not in npz:
    print('NPZ missing required keys')
    print('keys:', list(npz.keys()))
    raise SystemExit(1)
rheo = npz['rheotaxis_vec']
head = npz['head_vec']
# try to find a headless h5 in outputs/diagnostics
h5_files = sorted(glob.glob('outputs/diagnostics/*headless.h5'), key=os.path.getmtime)
if h5_files:
    h5_path = h5_files[-1]
else:
    # fallback known name
    h5_path = 'outputs/diagnostics/mini_rheo_test_headless.h5'
print('Using HDF5:', h5_path)
if not os.path.exists(h5_path):
    print('HDF5 not found, will skip upstream comparisons')
    h5 = None
else:
    h5 = h5py.File(h5_path,'r')

# sample flow from HDF5 if possible: try environment vel_x vel_y and sample at agent X/Y from HDF5
vel = None
agent_X = None
agent_Y = None
if h5 is not None:
    try:
        vx = h5['environment/vel_x'][:]
        vy = h5['environment/vel_y'][:]
        # try X/Y top-level or agent_data
        if 'agent_data/X' in h5:
            agent_X = h5['agent_data/X'][:,0]
            agent_Y = h5['agent_data/Y'][:,0]
        elif 'X' in h5:
            agent_X = h5['X'][:]
            agent_Y = h5['Y'][:]
        # try x_coords/y_coords
        if 'environment/x_coords' in h5 and 'environment/y_coords' in h5 and agent_X is not None:
            x_coords = h5['environment/x_coords'][:]
            y_coords = h5['environment/y_coords'][:]
            # pick nearest pixel
            n = len(agent_X)
            sampled_vx = np.full(n, np.nan)
            sampled_vy = np.full(n, np.nan)
            cols_vals = x_coords[0,:]
            rows_vals = y_coords[:,0]
            for i in range(n):
                col = int(np.argmin(np.abs(cols_vals - agent_X[i])))
                row = int(np.argmin(np.abs(rows_vals - agent_Y[i])))
                col = max(0, min(col, vx.shape[1]-1))
                row = max(0, min(row, vx.shape[0]-1))
                sampled_vx[i] = vx[row, col]
                sampled_vy[i] = vy[row, col]
            vel = np.column_stack((sampled_vx, sampled_vy))
    except Exception as e:
        print('Failed to read vel rasters from HDF5:', e)

# helper
def angle_between(u, v):
    # returns angle in degrees between vectors u and v, shape (2,)
    if u is None or v is None:
        return np.nan
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return np.nan
    cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.degrees(math.acos(cosang))
    return ang

# compute per-agent angles
n_rheo = rheo.shape[0]
n_head = head.shape[0]
n = min(n_rheo, n_head)
print(f'NPZ counts: rheo={n_rheo}, head={n_head}')
ang_head_rheo = np.zeros(n)
ang_rheo_up = np.zeros(n)
ang_head_up = np.zeros(n)
for i in range(n):
    rv = rheo[i]
    hv = head[i]
    ang_head_rheo[i] = angle_between(hv, rv)
    if vel is not None:
        # ensure we have matching HDF5 vel rows; if not, use min length
        up = -vel[i] if i < vel.shape[0] else None
        ang_rheo_up[i] = angle_between(rv, up)
        ang_head_up[i] = angle_between(hv, up)
    else:
        ang_rheo_up[i] = np.nan
        ang_head_up[i] = np.nan

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
