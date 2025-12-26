import csv
from pathlib import Path

OUT_DIR = Path('outputs/diagnostics')

def find_cue_summaries():
    for p in OUT_DIR.glob('*_cue_summary.csv'):
        yield p

def read_last_row(csv_path):
    with open(csv_path, 'r', newline='') as fh:
        reader = list(csv.reader(fh))
        if not reader:
            return None
        header = reader[0]
        last = reader[-1]
        return dict(zip(header, last))

def read_rheo_deep():
    p = OUT_DIR / 'rheo_deep_summary.csv'
    if not p.exists():
        return {}
    res = {}
    with open(p, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # use filename field (npz name) as key
            key = row['file']
            res[key] = row
    return res

def infer_cue_from_filename(p: Path):
    name = p.name
    # Prefer pattern: cue_test_<cue>_...
    if 'cue_test_' in name:
        after = name.split('cue_test_')[1]
        # first token after cue_test_ is the cue name in most runs
        cue = after.split('_')[0]
        return cue
    # fallback: if filename ends with _cue_summary.csv, take the token before that
    if name.endswith('_cue_summary.csv'):
        base = name.rsplit('_cue_summary.csv', 1)[0]
        # last token of base is likely the cue or an identifier; try to extract a known cue
        tokens = base.split('_')
        # prefer common cues if present
        common = ['rheotaxis','alignment','cohesion','collision','low_speed','avoid','refugia','border','shallow','wave_drag']
        for t in tokens[::-1]:
            if t in common:
                return t
        # otherwise return last token
        return tokens[-1]
    # generic fallback
    return name.split('_cue_summary')[0]

def main():
    rheo = read_rheo_deep()
    out_path = OUT_DIR / 'per_cue_summary.csv'
    fields = ['run_file','cue','step','n_agents','mean_nn_distance','heading_circ_var',
              'rheo_mean_abs_deg','rheo_pct_within_30','rheo_pct_within_90']
    with open(out_path, 'w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for p in find_cue_summaries():
            last = read_last_row(p)
            if last is None:
                continue
            cue = infer_cue_from_filename(p)
            # try map to rheo entry by matching behavior_debug_step_<step> in filenames
            # some cue summaries are per-step; for now attach rheo stats when names match prefix
            row = {
                'run_file': p.name,
                'cue': cue,
                'step': last.get('step',''),
                'n_agents': last.get('n_agents',''),
                'mean_nn_distance': last.get('mean_nn_distance',''),
                'heading_circ_var': last.get('heading_circ_var',''),
                'rheo_mean_abs_deg': '',
                'rheo_pct_within_30': '',
                'rheo_pct_within_90': ''
            }
            # try find matching rheo by run_file prefix
            # e.g., behavior_debug_step_0_1766681085.npz -> behavior_debug_step_0_1766681085.npz
            for k,v in rheo.items():
                if k.startswith('behavior_debug_step') and p.name.find(k.split('.npz')[0])!=-1:
                    row['rheo_mean_abs_deg'] = v.get('mean_abs_deg','')
                    row['rheo_pct_within_30'] = v.get('pct_within_30','')
                    row['rheo_pct_within_90'] = v.get('pct_within_90','')
                    break
            writer.writerow(row)

if __name__ == '__main__':
    main()
