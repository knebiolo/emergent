import glob, os, json, csv
import numpy as np

outdir = 'outputs/diagnostics'
files = sorted(glob.glob(os.path.join(outdir, 'move_debug_step_*.npz')))
if not files:
    print('No move_debug NPZs found in', outdir); raise SystemExit(1)

rows = []
for f in files:
    try:
        data = np.load(f)
    except Exception:
        continue
    # attempt to parse step index from filename
    basename = os.path.basename(f)
    parts = basename.split('_')
    step = -1
    if len(parts) >= 4:
        try:
            step = int(parts[3])
        except Exception:
            step = -1
    hz = data['Hz'] if 'Hz' in data else np.array([])
    thrust = data['thrust'] if 'thrust' in data else np.array([])
    drag = data['drag'] if 'drag' in data else np.array([])
    ideal = data['ideal_drag'] if 'ideal_drag' in data else np.array([])

    def mean_norm(arr):
        if arr is None or getattr(arr, 'size', 0) == 0:
            return None
        try:
            if arr.ndim == 1:
                return float(np.nanmean(arr))
            else:
                norms = np.linalg.norm(arr, axis=1)
                return float(np.nanmean(norms))
        except Exception:
            return None

    rows.append({
        'file': f,
        'step': step,
        'mean_Hz': mean_norm(hz),
        'mean_thrust': mean_norm(thrust),
        'mean_drag': mean_norm(drag),
        'mean_ideal_drag': mean_norm(ideal)
    })

csv_path = os.path.join(outdir, 'movement_timeseries.csv')
with open(csv_path, 'w', newline='') as cf:
    writer = csv.DictWriter(cf, fieldnames=['file','step','mean_Hz','mean_thrust','mean_drag','mean_ideal_drag'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

json_path = os.path.join(outdir, 'movement_timeseries_summary.json')
with open(json_path, 'w') as jf:
    json.dump(rows, jf, indent=2)

print('Wrote', csv_path, 'and', json_path)
