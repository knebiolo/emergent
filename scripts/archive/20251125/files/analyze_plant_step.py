"""Analyze plant step-response CSV and print concise sign/gain summary.

Usage: python scripts/analyze_plant_step.py [path]
"""
import sys
import numpy as np
import csv
path = sys.argv[1] if len(sys.argv)>1 else 'scripts/plant_step_response.csv'
rows=[]
with open(path,'r',encoding='utf-8') as fh:
    r=csv.reader(fh)
    for i,row in enumerate(r):
        if i==0: continue
        if not row: continue
        vals = [float(x) if x!='' else np.nan for x in row]
        rows.append(vals)
if not rows:
    print('No data in', path)
    sys.exit(2)
arr = np.array(rows)
t=arr[:,0]; psi=arr[:,1]; rmeas=arr[:,2]; rud=arr[:,3]
dt = t[1]-t[0] if len(t)>1 else 0.5
last_mask = t >= (t.max()-10)
steady_rmeas = np.nanmedian(rmeas[last_mask])
steady_rud = np.nanmedian(rud[last_mask])
print('Plant step-response summary for', path)
print(' samples=', len(t))
print(' steady_rmeas_deg_s (median last10s)=', round(float(steady_rmeas),4))
print(' steady_rud_deg (median last10s)=', round(float(steady_rud),4))
if not np.isnan(steady_rud) and abs(steady_rud)>1e-6:
    gain = steady_rmeas/steady_rud
    print(' approx yaw-rate per deg rudder (deg/s per deg rudder)=', round(float(gain),4))
else:
    print(' cannot compute gain (steady rudder near zero)')
