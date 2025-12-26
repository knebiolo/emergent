import csv
from pathlib import Path
import statistics

OUT = Path('outputs/diagnostics')


def analyze_rheo_peragent():
    rows = []
    for p in OUT.glob('*_rheo_peragent.csv'):
        with open(p, 'r', newline='') as fh:
            reader = csv.DictReader(fh)
            deltas = []
            for r in reader:
                # delta_deg column is signed; we want absolute angular difference
                d = float(r.get('delta_deg', '0'))
                # normalize to [-180,180]
                while d <= -180:
                    d += 360
                while d > 180:
                    d -= 360
                deltas.append(abs(d))
            if deltas:
                rows.append({'file': p.name,
                             'count': len(deltas),
                             'mean_abs_deg': statistics.mean(deltas),
                             'median_abs_deg': statistics.median(deltas),
                             'pct_within_30': sum(1 for x in deltas if x<=30)/len(deltas),
                             'pct_within_90': sum(1 for x in deltas if x<=90)/len(deltas)})
    return rows


def aggregate_cue_summaries():
    cue_map = {}
    for p in OUT.glob('*_cue_summary.csv'):
        with open(p, 'r', newline='') as fh:
            reader = list(csv.DictReader(fh))
            if not reader:
                continue
            last = reader[-1]
            # infer cue
            name = p.name
            cue = 'unknown'
            if 'cue_test_' in name:
                cue = name.split('cue_test_')[1].split('_')[0]
            else:
                for token in ['rheotaxis','alignment','cohesion','collision','low_speed']:
                    if token in name:
                        cue = token
                        break
            # parse metrics
            n_agents = int(last.get('n_agents',0)) if last.get('n_agents') else None
            mean_nn = float(last.get('mean_nn_distance', 'nan')) if last.get('mean_nn_distance') else None
            circ = float(last.get('heading_circ_var', 'nan')) if last.get('heading_circ_var') else None
            cue_map.setdefault(cue, []).append({'file': p.name, 'n_agents': n_agents, 'mean_nn_distance': mean_nn, 'heading_circ_var': circ})
    # compute aggregated stats
    out = []
    for cue, items in cue_map.items():
        mean_nn = statistics.mean([it['mean_nn_distance'] for it in items if it['mean_nn_distance'] is not None]) if items else None
        mean_circ = statistics.mean([it['heading_circ_var'] for it in items if it['heading_circ_var'] is not None]) if items else None
        out.append({'cue': cue, 'n_runs': len(items), 'mean_last_nn_distance': mean_nn, 'mean_last_heading_circ_var': mean_circ})
    return out


def main():
    rheo = analyze_rheo_peragent()
    cues = aggregate_cue_summaries()
    # print concise report
    print('Rheotaxis per-agent aggregations:')
    for r in rheo:
        print(f"- {r['file']}: count={r['count']}, mean_abs_deg={r['mean_abs_deg']:.2f}, median_abs_deg={r['median_abs_deg']:.2f}, pct<=30={r['pct_within_30']:.2f}, pct<=90={r['pct_within_90']:.2f}")
    print('\nCue summary aggregates:')
    for c in cues:
        print(f"- {c['cue']}: runs={c['n_runs']}, mean_nn={c['mean_last_nn_distance']:.3f}, mean_heading_circ_var={c['mean_last_heading_circ_var']:.3f}")

if __name__ == '__main__':
    main()
