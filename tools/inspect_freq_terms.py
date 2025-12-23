import json
import h5py
import glob
import os

# find most recent sim_db file in outputs/
files = glob.glob(os.path.join('outputs', 'sim_db_*.h5'))
if not files:
    print('No sim_db files found in outputs/')
    raise SystemExit(1)

latest = max(files, key=os.path.getmtime)
print('Using DB:', latest)

with h5py.File(latest, 'r') as f:
    key = 'diagnostics/freq_terms_history_json'
    if key not in f:
        print('Key not found:', key)
        raise SystemExit(1)
    raw = f[key][0].astype('U')
    data = json.loads(raw)

# data is a list of per-step diagnostics lists; each element corresponds to a step
print('Steps recorded:', len(data))

# aggregate counts
total_steps = len(data)

neg_denom_counts = []
zero_num_counts = []
safe_ratio_counts = []

for step_idx, step in enumerate(data):
    # step is a list of per-agent snapshots (compact up to Nsnap)
    # each item is a dict; aggregate from counts field if present
    if isinstance(step, dict):
        counts = step.get('counts', {})
        neg = counts.get('N_invalid_ratio', None)
        safe = counts.get('N_safe_ratio', None)
        neg_denom_counts.append(neg)
        safe_ratio_counts.append(safe)
    else:
        # previously the serialization wrapped lists differently; attempt best-effort
        try:
            first = step[0]
            counts = first.get('counts', {})
            neg_denom_counts.append(counts.get('N_invalid_ratio', None))
            safe_ratio_counts.append(counts.get('N_safe_ratio', None))
        except Exception:
            neg_denom_counts.append(None)
            safe_ratio_counts.append(None)

print('Per-step invalid-ratio counts (first 10):', neg_denom_counts[:10])
print('Per-step safe-ratio counts (first 10):', safe_ratio_counts[:10])

# summarize
valid_steps = [v for v in neg_denom_counts if v is not None]
if valid_steps:
    print('Mean invalid ratio per-step:', sum(valid_steps)/len(valid_steps))

# inspect a representative step (last)
rep = data[-1]
if isinstance(rep, dict):
    print('\nRepresentative step keys:', list(rep.keys()))
    print('Sample term1_si[:10]:', rep.get('term1_si')[:10])
    print('Sample term2_si[:10]:', rep.get('term2_si')[:10])
    print('Sample term3_si[:10]:', rep.get('term3_si')[:10])
    print('Sample denom_si[:10]:', rep.get('denom_si')[:10])
    print('Sample num_si[:10]:', rep.get('num_si')[:10])
    print('safe_ratio[:10]:', rep.get('safe_ratio')[:10])
else:
    print('Representative step is not dict; raw:', rep)
