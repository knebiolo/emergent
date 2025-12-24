import glob, os, numpy as np
files = glob.glob('outputs/diagnostics/behavior_debug_step_*.npz')
if not files:
    print('No NPZ files found')
    raise SystemExit(1)
f = max(files, key=os.path.getmtime)
print('Inspecting', f)
d = np.load(f, allow_pickle=True)
print('keys:', list(d.keys()))
for k in ['head_vec','alignment_vec','cohesion_vec','neighbor_counts','neighbors_concat','neighbor_headings','neighbor_rel_heading','alignment_used_velocity','neighbors_owner','closest_agent','nearest_neighbor_distance']:
    if k in d:
        v = d[k]
        try:
            print(k, 'shape=', getattr(v, 'shape', None), 'sample=', v if getattr(v, 'shape', None) is None else (v[:10] if v.size>10 else v))
        except Exception as e:
            print(k, 'could not display:', e)

# print head_vec and alignment magnitudes
if 'head_vec' in d:
    hv = d['head_vec']
    print('head_vec norms sample=', np.linalg.norm(hv, axis=1))

print('--- simulation-level check ---')
# try to load sim db heading if available in outputs
try:
    # some runs write agent_data/heading into sim db; attempt to open outputs/sim_db_*.h5
    import h5py
    h5s = glob.glob('outputs/sim_db_*.h5')
    if h5s:
        h = max(h5s, key=os.path.getmtime)
        print('Opening DB', h)
        f_h5 = h5py.File(h, 'r')
        if 'agent_data/heading' in f_h5:
            heading_ts = f_h5['agent_data/heading'][:]
            print('agent_data/heading shape', heading_ts.shape)
            print('agent_data/heading col0 sample', heading_ts[:,0])
        f_h5.close()
except Exception as e:
    print('Could not open sim DB:', e)
