#!/usr/bin/env python3
"""Print PID rows near t=454.3s for quick inspection."""
import csv
IN='logs/pid_trace_window.csv'
start=453.8
end=454.8
rows=[]
with open(IN,'r',newline='') as f:
    r=csv.DictReader(f)
    for row in r:
        try:
            t=float(row.get('t','nan'))
        except:
            continue
        if start <= t <= end:
            rows.append(row)

if not rows:
    print('No rows found in',IN,'for',start,end)
else:
    print(f'Found {len(rows)} rows in {start}..{end}')
    # print sorted by time, agent
    rows=sorted(rows,key=lambda r:(float(r['t']),int(r['agent'])))
    for r in rows:
        # safely cast expected numeric fields to float for formatted printing
        def safef(key, default=float('nan')):
            try:
                return float(r.get(key, default))
            except Exception:
                return default

        def safestr(key):
            return r.get(key, '')

        t_val = safef('t')
        agent = safestr('agent')
        role = safestr('role')
        crossing_lock = safestr('crossing_lock')
        flagged_give_way = safestr('flagged_give_way')
        hd_cmd_deg = safef('hd_cmd_deg')
        psi_deg = safef('psi_deg')
        err_deg = safef('err_deg')
        raw_preinv_deg = safef('raw_preinv_deg')
        raw_deg = safef('raw_deg')
        rud_preinv_deg = safef('rud_preinv_deg')
        rud_deg = safef('rud_deg')
        x_m = safef('x_m')
        y_m = safef('y_m')
        event = safestr('event')

        print(f"t={t_val:.2f} ag={agent} role={role} lock={crossing_lock} flag={flagged_give_way} hd_cmd={hd_cmd_deg:.2f} psi={psi_deg:.2f} err={err_deg:.2f} raw_preinv={raw_preinv_deg:.2f} raw={raw_deg:.2f} rud_preinv={rud_preinv_deg:.2f} rud={rud_deg:.2f} x={x_m:.2f} y={y_m:.2f} ev={event}")
