import numpy as np
import h5py, os, glob

h5path = sorted(glob.glob('outputs/diagnostics/*headless.h5'))[-1]
print('Using HDF5:', h5path)
with h5py.File(h5path, 'r') as h5:
    # prefer agent_data/heading if present
    if 'agent_data/heading' in h5:
        headings = np.array(h5['agent_data/heading'])
        print('agent_data/heading shape:', headings.shape)
    elif 'heading' in h5:
        headings = np.array(h5['heading'])[None, :]
        print('heading shape:', headings.shape)
    else:
        raise SystemExit('No heading dataset found in h5')

npz_files = sorted(glob.glob('outputs/diagnostics/forced_rawvecs/*.npz'))
if not npz_files:
    raise SystemExit('No authoritative NPZs found')

rows = []
for f in npz_files:
    d = np.load(f)
    step = int(os.path.basename(f).split('_')[3])
    head_vec = d['head_vec'] if 'head_vec' in d.files else None
    # choose HDF5 timestep column: step (0-based)
    try:
        h_t = headings[:, step]
    except Exception:
        h_t = headings[0]
    # compute per-agent heading angle from head_vec if present
    if head_vec is not None:
        # head_vec shape should be (n_agents,2)
        hv = np.asarray(head_vec)
        agent_angles = np.arctan2(hv[:,1], hv[:,0])
        # diff to actual heading
        diffs = np.abs((agent_angles - h_t + np.pi) % (2*np.pi) - np.pi)
        mean_deg = np.degrees(np.nanmean(diffs))
        print(f'{os.path.basename(f)} mean_abs_angle_deg={mean_deg:.3f}')
    else:
        print(os.path.basename(f), 'no head_vec key')
