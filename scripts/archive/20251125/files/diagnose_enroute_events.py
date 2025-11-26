#!/usr/bin/env python3
"""Annotate en-route correlated PID events with simple diagnostics.

Usage: python scripts/diagnose_enroute_events.py <in_correlated_enroute.csv> [out.csv]
Writes a CSV with additional columns:
 - is_applied_saturated: abs(rud_deg) >= 0.95*max_rudder
 - is_raw_saturated: abs(raw_deg) >= 0.95*max_rudder
 - rmeas_vs_raw_mismatch: r_meas exists and sign(r_meas)*sign(raw_deg) < 0 and abs(raw_deg) > 5_deg
 - psi_minus_hd_cmd_deg: wrapped difference in degrees

Also prints a short summary.
"""
import sys, csv, math
from math import copysign
from emergent.ship_abm import config

inpath = sys.argv[1] if len(sys.argv)>1 else 'scripts/pid_trace_waypoint_crosswind_correlated_15deg_enroute.csv'
outpath = sys.argv[2] if len(sys.argv)>2 else inpath.replace('.csv','_diagnostics.csv')

max_rud_rad = config.SHIP_PHYSICS.get('max_rudder', None)
if max_rud_rad is None:
    max_rud_deg = 20.0
else:
    max_rud_deg = abs(max_rud_rad) * 180.0 / 3.141592653589793
sat_thresh = 0.95 * max_rud_deg

rows = []
with open(inpath,'r',encoding='utf-8') as fh:
    rdr = csv.DictReader(fh)
    for r in rdr:
        if not r: continue
        try:
            t = float(r.get('t','nan'))
        except Exception:
            continue
        def tof(k):
            v = r.get(k,'')
            try:
                return float(v) if v!='' else float('nan')
            except Exception:
                return float('nan')
        raw = tof('raw_deg')
        rud = tof('rud_deg')
        rmeas = tof('r_meas_deg_s')
        psi = tof('psi_deg')
        hd = tof('hd_cmd_deg')
        err = tof('err_deg')
        # saturation flags
        is_applied_saturated = False
        is_raw_saturated = False
        if not math.isnan(rud):
            is_applied_saturated = abs(rud) >= sat_thresh
        if not math.isnan(raw):
            is_raw_saturated = abs(raw) >= sat_thresh
        # rmeas_vs_raw_mismatch: rmeas sign opposite raw sign and raw magnitude > 5 deg
        mismatch = False
        if (not math.isnan(rmeas)) and (not math.isnan(raw)) and abs(raw) > 5.0:
            # sign compare
            if (rmeas > 0 and raw < 0) or (rmeas < 0 and raw > 0):
                mismatch = True
        # psi-hd wrapped diff
        psi_minus_hd = float('nan')
        if not math.isnan(psi) and not math.isnan(hd):
            diff = psi - hd
            # wrap to [-180,180]
            diff_wrapped = (diff + 180.0) % 360.0 - 180.0
            psi_minus_hd = diff_wrapped
        out = {
            't': t,
            'agent': r.get('agent',''),
            'err_deg': err,
            'raw_deg': raw,
            'rud_deg': rud,
            'psi_deg': psi,
            'hd_cmd_deg': hd,
            'r_meas_deg_s': rmeas,
            'is_applied_saturated': int(is_applied_saturated),
            'is_raw_saturated': int(is_raw_saturated),
            'rmeas_vs_raw_mismatch': int(mismatch),
            'psi_minus_hd_cmd_deg': psi_minus_hd,
        }
        rows.append(out)

# write
fieldnames = ['t','agent','err_deg','raw_deg','rud_deg','psi_deg','hd_cmd_deg','r_meas_deg_s','is_applied_saturated','is_raw_saturated','rmeas_vs_raw_mismatch','psi_minus_hd_cmd_deg']
with open(outpath,'w',encoding='utf-8',newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# summary
n = len(rows)
nsat = sum(1 for r in rows if int(r['is_applied_saturated']))
nmismatch = sum(1 for r in rows if int(r['rmeas_vs_raw_mismatch']))
med_diff = float('nan')
try:
    diffs = [abs(r['psi_minus_hd_cmd_deg']) for r in rows if not (r['psi_minus_hd_cmd_deg']!=r['psi_minus_hd_cmd_deg'])]
    if diffs:
        med_diff = sorted(diffs)[len(diffs)//2]
except Exception:
    med_diff = float('nan')

print(f'Diagnostics written to {outpath}  rows={n}  applied_saturated={nsat}  rmeas_raw_mismatch={nmismatch}  median|psi-hd|={med_diff}')
