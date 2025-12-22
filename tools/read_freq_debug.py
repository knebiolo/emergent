import os, glob, json
import h5py
out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
patterns = glob.glob(os.path.join(out_dir, 'sim_db_*.h5'))
if not patterns:
    print('No sim DB files found')
    raise SystemExit(1)
latest = max(patterns, key=os.path.getmtime)
print('DB:', latest)
with h5py.File(latest,'r') as f:
    if 'diagnostics' in f:
        if 'freq_debug_history_json' in f['diagnostics']:
            arr = f['diagnostics']['freq_debug_history_json'][:]
            try:
                j = json.loads(arr[0].decode() if isinstance(arr[0], bytes) else arr[0])
                print('freq_debug_history snapshots:', len(j))
                if len(j) > 0:
                    print('\nFirst snapshot keys:', list(j[0].keys()))
                    # print first snapshot for small fields
                    s = j[0]
                    for k in ['drags_J_s','denom_si','Hz_raw','Hz','A_m','B_m','V_m_s','U_m_s','safe_si']:
                        if k in s:
                            print(k, ':', s[k])
            except Exception as e:
                print('Failed to parse JSON freq_debug_history:', e)
        else:
            print('diagnostics/freq_debug_history_json missing')
    else:
        print('diagnostics group missing')
