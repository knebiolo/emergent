"""
Inspect rosario PID trace for suspicious rudder/heading behavior.
Prints summary stats and small context windows around events where:
 - |rud_deg| > 15 deg (near saturation at 20deg)
 - |delta_psi| between timesteps > 30 deg (fast spin events)
 - abs(err_deg) > 30 deg

Usage: python scripts/inspect_rosario_trace.py
"""
import csv
import math
from collections import defaultdict

TRACE = 'traces/rosario_2_agent_pid_trace.csv'


def wrap_deg(x):
    return ((x + 180) % 360) - 180


def heading_diff_deg(a, b):
    # minimal signed difference a - b in degrees
    return wrap_deg(a - b)


def read_trace(path):
    agents = defaultdict(list)
    with open(path, 'r', newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = {k: i for i, k in enumerate(header)}
        for row in reader:
            try:
                t = float(row[idx['t']])
                agent = int(row[idx['agent']])
                rud_deg = float(row[idx['rud_deg']])
                psi_deg = float(row[idx['psi_deg']])
                err_deg = float(row[idx['err_deg']])
                r_meas_deg = float(row[idx['r_meas_deg']]) if 'r_meas_deg' in idx else 0.0
                agents[agent].append({'t': t, 'rud_deg': rud_deg, 'psi_deg': psi_deg, 'err_deg': err_deg, 'r_meas_deg': r_meas_deg, 'row': row})
            except Exception as e:
                # skip malformed
                continue
    return agents


def analyze_agent(rows):
    out = {}
    max_rud = 0.0
    sat_count = 0
    large_rud_events = []
    large_delta_events = []
    large_err_events = []
    prev = None
    for i, r in enumerate(rows):
        rud = abs(r['rud_deg'])
        if rud > max_rud:
            max_rud = rud
        if rud >= 19.0:
            sat_count += 1
        if rud >= 15.0:
            large_rud_events.append(i)
        if prev is not None:
            dt = r['t'] - prev['t']
            delta_psi = heading_diff_deg(r['psi_deg'], prev['psi_deg'])
            # deg per second
            rate = delta_psi / (dt if dt > 0 else 1)
            if abs(delta_psi) > 30.0:
                large_delta_events.append(i)
        if abs(r['err_deg']) > 30.0:
            large_err_events.append(i)
        prev = r
    out['max_rud'] = max_rud
    out['sat_fraction'] = sat_count / max(1, len(rows))
    out['large_rud_events'] = large_rud_events
    out['large_delta_events'] = large_delta_events
    out['large_err_events'] = large_err_events
    return out


def dump_window(rows, idx, window=6):
    start = max(0, idx - window)
    end = min(len(rows), idx + window + 1)
    lines = []
    header = 't,agent,err_deg,r_des_deg,derr_deg,P_deg,I_deg,D_deg,raw_deg,rud_deg,psi_deg,hd_cmd_deg,r_meas_deg,x_m,y_m'
    lines.append(header)
    for i in range(start, end):
        lines.append(','.join(rows[i]['row']))
    return '\n'.join(lines)


def main():
    agents = read_trace(TRACE)
    if not agents:
        print('No agents or trace not found at', TRACE)
        return
    for agent, rows in agents.items():
        print('\nAgent', agent, 'rows', len(rows))
        stats = analyze_agent(rows)
        print(' max |rud_deg| =', stats['max_rud'])
        print(' sat fraction (>=19deg) =', stats['sat_fraction'])
        print(' large rudder events (>15deg):', len(stats['large_rud_events']))
        print(' large psi delta events (>30deg):', len(stats['large_delta_events']))
        print(' large err events (>30deg):', len(stats['large_err_events']))

        # show a few windows for each symptom type
        shown = 0
        for idx in stats['large_delta_events'][:3]:
            print('\n=== Large delta window index', idx, 't=', rows[idx]['t'], '===')
            print(dump_window(rows, idx))
            shown += 1
        for idx in stats['large_rud_events'][:3]:
            print('\n=== Large rud window index', idx, 't=', rows[idx]['t'], '===')
            print(dump_window(rows, idx))
            shown += 1
        for idx in stats['large_err_events'][:3]:
            print('\n=== Large err window index', idx, 't=', rows[idx]['t'], '===')
            print(dump_window(rows, idx))
            shown += 1
        if shown == 0:
            print(' No suspicious windows to show for this agent.')

if __name__ == '__main__':
    main()
