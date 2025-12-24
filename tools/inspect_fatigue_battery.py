import numpy as np
import glob, json, os

out_dir = 'outputs/diagnostics'
npzs = glob.glob(os.path.join(out_dir, 'behavior_debug_step_*.npz'))
if not npzs:
    print('No behavior NPZs found in', out_dir); raise SystemExit(1)
# pick the most recently modified NPZ (safer than lexicographic filename sort)
last = max(npzs, key=os.path.getmtime)
print('Inspecting', last)
npz = np.load(last)
keys = list(npz.keys())
print('keys:', keys)
report = {'file': last, 'keys': keys}
for k in ['battery','swim_behav','swim_mode','ideal_sog','sog','ttf','bout_dur','dist_per_bout']:
    if k in npz:
        a = npz[k]
        try:
            amin = float(np.nanmin(a))
            amax = float(np.nanmax(a))
            amean = float(np.nanmean(a))
            report[k] = {'shape': a.shape, 'min': amin, 'max': amax, 'mean': amean}
        except Exception as e:
            report[k] = {'shape': a.shape, 'error': str(e)}
    else:
        report[k] = None

out_path = os.path.join(out_dir, 'fatigue_battery_stats.json')
with open(out_path, 'w') as f:
    json.dump(report, f, indent=2)
print('Wrote', out_path)
print('Summary:')
for k in ['battery','swim_behav','swim_mode','ideal_sog','sog']:
    print(k, '->', report.get(k))
