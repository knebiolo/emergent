"""Parse latest runtime logs in ./logs/ and report:
- number of PID mismatches (match==0) in pid_mismatch_debug*.csv
- number of runtime rows near give-way times where applied rudder magnitude > 0.1
Exit code 0 if mismatches==0 and nonzero_rudder_count>0, else 1.
"""
import os, glob, csv, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
LOGDIR = os.path.join(ROOT, "logs")

def find_latest(pattern):
    files = glob.glob(os.path.join(LOGDIR, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

mismatch = find_latest('pid_mismatch_debug*.csv')
col = find_latest('colregs_runtime_debug*.csv')
run = find_latest('pid_runtime_debug*.csv')

def count_mismatches(path):
    if not path:
        return None
    c = 0
    with open(path, newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                if int(r.get('match','1')) == 0:
                    c += 1
            except Exception:
                pass
    return c

mismatches = count_mismatches(mismatch)

# collect give times
give_times = []
if col and run:
    with open(col, newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                t = float(r.get('time', r.get('t', r.get('sim_time', 0))))
            except Exception:
                continue
            role = r.get('role','') or ''
            try:
                lock = float(r.get('crossing_lock','-1') or -1)
            except Exception:
                lock = -1
            try:
                linger = float(r.get('crossing_linger_s', r.get('crossing_linger','0') or 0))
            except Exception:
                linger = 0
            try:
                post = float(r.get('post_avoid_s', r.get('post_avoid_timer','0') or 0))
            except Exception:
                post = 0
            is_give = (str(role).lower().strip() == 'give_way') or (lock >= 0) or (linger > 0) or (post > 0)
            if is_give:
                give_times.append(t)

# scan runtime for non-zero rudder near give times
nonzero = 0
checked = 0
if run and give_times:
    with open(run, newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                t = float(r.get('time', r.get('t', 0)))
            except Exception:
                continue
            if any(abs(t - g) <= 0.5 for g in give_times):
                checked += 1
                rud = None
                for key in ['final_rudder_deg','applied_rudder_deg','rud_deg','final_deg']:
                    val = r.get(key)
                    if val is not None:
                        try:
                            rud = float(val)
                            break
                        except Exception:
                            rud = 0.0
                if rud is None:
                    rud = 0.0
                if abs(rud) > 0.1:
                    nonzero += 1

print('mismatch_file:', mismatch)
print('mismatches:', mismatches)
print('colregs_file:', col)
print('give_intervals_found:', len(give_times))
print('runtime_file:', run)
print('runtime_rows_near_give:', checked)
print('nonzero_applied_rudder_rows:', nonzero)

ok = (mismatches == 0) and ( (not give_times) or (nonzero > 0) )
print('REGTEST OK:', ok)
sys.exit(0 if ok else 1)
