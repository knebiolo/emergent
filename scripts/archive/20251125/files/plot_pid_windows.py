import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import fabs
p = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\scripts\pid_trace_gui_rosario.csv"
rows = []
with open(p, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            t = float(row['t'])
            err = float(row['err_deg'])
            raw = float(row['raw_deg'])
            rud = float(row['rud_deg'])
        except Exception:
            continue
        rows.append((t, err, raw, rud))

# select windows
windows = [(0.0,2.0),(30.0,45.0),(480.0,490.0)]
fig, axes = plt.subplots(len(windows), 1, figsize=(10, 3*len(windows)), sharex=False)
for ax, (t0,t1) in zip(axes, windows):
    sel = [(t,err,raw,rud) for (t,err,raw,rud) in rows if t0 <= t <= t1]
    if not sel:
        ax.text(0.5,0.5,f'No data in {t0}-{t1}s', ha='center')
        continue
    t = [s[0] for s in sel]
    err = [s[1] for s in sel]
    raw = [s[2] for s in sel]
    rud = [s[3] for s in sel]
    ax.plot(t, err, label='err_deg')
    ax.plot(t, raw, label='raw_deg')
    ax.plot(t, rud, label='rud_deg')
    ax.set_xlim(t0, t1)
    ax.set_ylabel('deg')
    ax.legend()
    ax.grid(True)
axes[-1].set_xlabel('time (s)')
plt.tight_layout()
out = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\scripts\pid_plot_windows.png"
plt.savefig(out)
print('Saved', out)
