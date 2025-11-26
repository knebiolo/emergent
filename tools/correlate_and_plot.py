#!/usr/bin/env python3
"""Correlate pid_trace CSV with run log messages and plot per-agent timelines.

Usage: python correlate_and_plot.py csv_file run_log start end

Produces:
- logs/correlated_events.csv
- logs/agent_0_300_460.png, logs/agent_1_300_460.png
"""
import sys, os, csv, re
from collections import defaultdict
import math

csv_file = sys.argv[1] if len(sys.argv)>1 else 'logs/pid_trace_300_500.csv'
run_log = sys.argv[2] if len(sys.argv)>2 else 'logs/run_ship_Seattle_gui.out'
start = float(sys.argv[3]) if len(sys.argv)>3 else 300.0
end = float(sys.argv[4]) if len(sys.argv)>4 else 460.0

out_dir = 'logs'
os.makedirs(out_dir, exist_ok=True)

# read csv rows into memory
rows = []
with open(csv_file,'r',newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            t = float(row.get('t','nan'))
        except:
            continue
        if t < start - 1e-6 or t > end + 1e-6:
            continue
        try:
            agent = int(row.get('agent',0))
        except:
            agent = 0
        def g(k):
            try:
                return float(row.get(k,''))
            except:
                return math.nan
        rows.append({'t':t,'agent':agent,'rud':g('rud_deg'),'hd_cmd':g('hd_cmd_deg'),'psi':g('psi_deg'),'I_raw':g('I_raw_deg'),'flag':row.get('flagged_give_way','')})

rows = sorted(rows, key=lambda x: (x['agent'], x['t']))

# parse run log for COLREGS/CTRL/Collision lines
events = []
pat_colregs = re.compile(r"\[COLREGS\].*?t[ =]*([0-9]+\.?[0-9]*)", re.IGNORECASE)
# many [CTRL] lines don't include 'agent='; capture t, hd_cmd, rud and optional agent
pat_ctrl = re.compile(r"\[CTRL\].*?t=\s*([0-9]+\.?[0-9]*)s?.*?hd_cmd=\s*([\-0-9\.eE]+).*?rud=\s*([\-0-9\.eE]+)(?:.*?agent[= ]*(\d+))?", re.IGNORECASE)
pat_collision = re.compile(r"Collision: Ships.*?at t=([0-9]+\.?[0-9]*)s?", re.IGNORECASE)

def read_text_with_bom(path):
    b = open(path,'rb').read()
    # BOM detection
    if b.startswith(b'\xff\xfe') or b.startswith(b'\xfe\xff'):
        try:
            return b.decode('utf-16')
        except:
            return b.decode('utf-16', errors='ignore')
    if b.startswith(b'\xef\xbb\xbf'):
        return b.decode('utf-8-sig', errors='ignore')
    # fallback
    try:
        return b.decode('utf-8')
    except:
        return b.decode('utf-8', errors='ignore')

file_text = read_text_with_bom(run_log)

# quick scans for counts
ncol = len(pat_collision.findall(file_text))
nctrl = len(pat_ctrl.findall(file_text))
ncolregs = file_text.count('[COLREGS]')
print('run_log counts: COLLISION={}, CTRL(matches)={}, COLREGS_lines={}'.format(ncol,nctrl,ncolregs))
# iterate by lines from decoded text
for line in file_text.splitlines():
    if 'COLREGS' in line or '[CTRL]' in line or 'Collision' in line:
        m = pat_ctrl.search(line)
        if m:
            t = float(m.group(1))
            hd_cmd = float(m.group(2))
            rud = float(m.group(3))
            agent = int(m.group(4)) if m.group(4) else None
            if t < start-1 or t> end+1:
                continue
            events.append({'t':t,'type':'CTRL','agent':agent,'hd_cmd':hd_cmd,'rud':rud,'raw':line.strip()})
            continue
        if '[COLREGS]' in line:
            events.append({'t':None,'type':'COLREGS','line':line.strip()})
            continue
        m = pat_collision.search(line)
        if m:
            t = float(m.group(1))
            if t < start-1 or t> end+1:
                continue
            events.append({'t':t,'type':'COLLISION','line':line.strip()})

events = sorted(events, key=lambda x: (float('inf') if x.get('t') is None else x.get('t')))

# align events to nearest CSV timestamp per agent when applicable
def find_nearest(agent, t):
    # binary search in rows for given agent
    arr = [r for r in rows if r['agent']==agent]
    if not arr:
        return None
    lo=0; hi=len(arr)-1
    while lo<hi:
        mid=(lo+hi)//2
        if arr[mid]['t']<t: lo=mid+1
        else: hi=mid
    # check lo and lo-1
    cand = arr[lo]
    best = cand
    best_dt = abs(cand['t']-t)
    if lo>0:
        c2=arr[lo-1]
        dt = abs(c2['t']-t)
        if dt<best_dt:
            best=c2; best_dt=dt
    return best

correlated = []
for e in events:
    if e['type']=='CTRL':
        nearest = find_nearest(e['agent'], e['t'])
        correlated.append({'event_t':e['t'],'etype':'CTRL','agent':e['agent'],'evt_line':e['raw'],'csv_t': nearest['t'] if nearest else '', 'csv_rud': nearest['rud'] if nearest else '', 'csv_hd_cmd': nearest['hd_cmd'] if nearest else ''})
    else:
        correlated.append({'event_t':e['t'],'etype':e['type'],'agent':'','evt_line':e.get('line',''),'csv_t':'','csv_rud':'','csv_hd_cmd':''})

# write correlated CSV
out_csv = os.path.join(out_dir,'correlated_events.csv')
with open(out_csv,'w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['event_t','etype','agent','evt_line','csv_t','csv_rud','csv_hd_cmd'])
    w.writeheader()
    for c in correlated:
        w.writerow(c)

print('Wrote',out_csv,'events:',len(correlated))

# Now create plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print('matplotlib not available:',e)
    sys.exit(0)

agents = sorted(set([r['agent'] for r in rows]))
for agent in agents:
    ag_rows = [r for r in rows if r['agent']==agent]
    t = [r['t'] for r in ag_rows]
    rud = [r['rud'] for r in ag_rows]
    hd = [r['hd_cmd'] for r in ag_rows]
    psi = [r['psi'] for r in ag_rows]

    plt.figure(figsize=(12,6))
    plt.plot(t, rud, label='rud_deg')
    plt.plot(t, hd, label='hd_cmd_deg', alpha=0.8)
    # overlay events
    for ev in events:
        if ev.get('agent','')==agent and ev['type']=='CTRL':
            plt.axvline(ev['t'], color='orange', alpha=0.3)
    plt.xlabel('t (s)')
    plt.ylabel('deg')
    plt.title(f'Agent {agent} rudder & hd_cmd {start}-{end}s')
    plt.legend()
    out_png = os.path.join(out_dir,f'agent_{agent}_{int(start)}_{int(end)}.png')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print('Wrote',out_png)

# mark todo done in manage file by writing a short summary to stdout
print('Done. Outputs:')
print(' -',out_csv)
for agent in agents:
    print(' -', os.path.join(out_dir,f'agent_{agent}_{int(start)}_{int(end)}.png'))
