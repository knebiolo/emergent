import pandas as pd
import os

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
inst_dir = os.path.join(base, 'sweep_results', 'instrumentation')

f_on = os.path.join(inst_dir, 'pid_trace_osc_keepKp_lowerKd_dtau2.0_full_events.csv')
f_off = os.path.join(inst_dir, 'pid_trace_osc_keepKp_lowerKd_dtau2.0_full_nodead_events.csv')

out = os.path.join(inst_dir, 'compare_full_vs_nodead_events.txt')

if not os.path.exists(f_on) or not os.path.exists(f_off):
    raise FileNotFoundError('One or both event files missing')

df_on = pd.read_csv(f_on)
df_off = pd.read_csv(f_off)

summary = []
summary.append(f'on rows: {len(df_on)}')
summary.append(f'off rows: {len(df_off)}')
summary.append(f'on first/last: {df_on.time.min()} -> {df_on.time.max()}')
summary.append(f'off first/last: {df_off.time.min()} -> {df_off.time.max()}')

# exact equality check
exact = df_on.equals(df_off)
summary.append(f'exact equality: {exact}')

# check row-wise differences if not exact
if not exact:
    merged = df_on.merge(df_off, on=['time','event','info'], how='outer', indicator=True)
    only_on = merged[merged['_merge']=='left_only']
    only_off = merged[merged['_merge']=='right_only']
    summary.append(f'only_on count: {len(only_on)}')
    summary.append(f'only_off count: {len(only_off)}')
    # capture a few examples
    if len(only_on)>0:
        summary.append('examples only_on:')
        summary.extend([str(x) for x in only_on.head(10).to_dict(orient='records')])
    if len(only_off)>0:
        summary.append('examples only_off:')
        summary.extend([str(x) for x in only_off.head(10).to_dict(orient='records')])
else:
    summary.append('no differences found')

# timestamp equality check (floats) within tolerance
import numpy as np
times_on = np.array(df_on['time'])
times_off = np.array(df_off['time'])
if len(times_on)==len(times_off):
    dt = np.abs(times_on - times_off)
    maxdt = dt.max()
    summary.append(f'max timestamp delta (paired rows): {maxdt}')
else:
    summary.append('row counts differ, skip paired timestamp delta')

with open(out, 'w') as fh:
    fh.write('\n'.join(summary))

print('\n'.join(summary))
print('wrote', out)
