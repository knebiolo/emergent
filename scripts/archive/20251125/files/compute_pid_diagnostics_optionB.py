import pandas as pd
import numpy as np

fn = 'scripts/pid_trace_repro_t478.5.csv'
print('Reading', fn)
df = pd.read_csv(fn)
# ensure time column present
if 't' not in df.columns:
    raise SystemExit('t column missing')

times = df['t'].to_numpy()
dt = np.median(np.diff(times)) if len(times) > 1 else 0.5

err = df['err_deg'].to_numpy()
I = df['I_deg'].to_numpy()
raw = df['raw_deg'].to_numpy()
rud = df['rud_deg'].to_numpy()

peak_err = np.max(np.abs(err))
max_I = np.max(np.abs(I))
max_raw = np.max(np.abs(raw))
max_rud = np.max(np.abs(rud))

# sat_time: count rows where abs(rud) is within 1e-6 of max_rud
sat_mask = np.isclose(np.abs(rud), max_rud, atol=1e-6) | (np.abs(rud) > (max_rud - 1e-6))
sat_time_s = sat_mask.sum() * dt
first_sat_t = float(times[sat_mask.argmax()]) if sat_mask.any() else None

print('\nDiagnostics (Option B run - reduced Ndelta)')
print('peak_err =', round(float(peak_err), 3), 'deg')
print('max_I =', round(float(max_I), 3), 'deg')
print('max_raw =', round(float(max_raw), 3), 'deg')
print('max_rud =', round(float(max_rud), 3), 'deg')
print('sat_time_s =', round(float(sat_time_s), 3), 's')
print('first_sat_t =', first_sat_t, 's')
