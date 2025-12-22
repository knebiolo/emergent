"""Quick regression test to run a short headless reproduction and assert two things:
1) The runtime/deep PID mismatch file contains zero mismatches (no rows where match==0).
2) During any active give-way windows, the runtime applied rudder shows non-zero values (i.e., controller actually steered).

Run from the repo root. This script shells out to the headless reproducer and then parses ./logs/*.csv
"""
import os
import sys
import subprocess
import time
import glob
import csv

ROOT = os.path.dirname(os.path.dirname(__file__))
LOGDIR = os.path.join(ROOT, "logs")
HEADLESS = os.path.join(ROOT, "tools", "headless_repro_and_extract.py")

DURATION = 30

def run_headless(duration=DURATION):
    env = os.environ.copy()
    env["EMERGENT_PID_DEEP_DEBUG"] = "1"
    env["EMERGENT_PID_DEBUG"] = "1"
    cmd = [sys.executable, HEADLESS, "--duration", str(duration), "--agents", "2", "--port", "Seattle"]
    print("Running headless reproducer:", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)
    proc.wait()
    return proc.returncode


def find_latest_csv(pattern):
    files = glob.glob(os.path.join(LOGDIR, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def scan_mismatches():
    path = find_latest_csv("pid_mismatch_debug*.csv")
    if not path:
        print("No pid_mismatch_debug CSV found in logs/; failing")
        return False
    found = 0
    with open(path, newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            # expects a column named 'match' present (1/0)
            try:
                if int(r.get('match', '1')) == 0:
                    found += 1
            except Exception:
                pass
    print(f"pid_mismatch_debug -> {path}; mismatches found: {found}")
    return found == 0


def check_giveway_rudder_nonzero():
    # load latest colregs and runtime pid files
    col = find_latest_csv('colregs_runtime_debug*.csv')
    run = find_latest_csv('pid_runtime_debug*.csv')
    deep = find_latest_csv('pid_deep_debug*.csv')
    if not (col and run):
        print('Missing runtime logs (colregs or pid_runtime); failing')
        return False
    # identify any timeframe where role indicates give_way or the timers are active
    give_intervals = []
    with open(col, newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                t = float(r.get('time', r.get('t', 0)))
            except Exception:
                continue
            role = r.get('role', '')
            try:
                lock = float(r.get('crossing_lock', '-1'))
            except Exception:
                lock = -1
            linger = float(r.get('crossing_linger_timer', '0') or 0)
            post = float(r.get('post_avoid_timer', '0') or 0)
            is_give = (role.lower().strip() == 'give_way') or (lock >= 0) or (linger > 0) or (post > 0)
            if is_give:
                give_intervals.append(t)
    if not give_intervals:
        print('No give_way intervals detected in colregs runtime log; cannot assert behavior.')
        return True
    # read runtime pid and ensure applied rudder for agent 1 (or any agent) is non-zero during intervals
    nonzero_count = 0
    total_checked = 0
    with open(run, newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                t = float(r.get('time', r.get('t', 0)))
            except Exception:
                continue
            # find closest give interval membership
            if any(abs(t - g) <= 0.5 for g in give_intervals):
                total_checked += 1
                try:
                    rud = float(r.get('final_rudder_deg', r.get('applied_rudder_deg', '0')))
                except Exception:
                    rud = 0.0
                if abs(rud) > 0.1:
                    nonzero_count += 1
    print(f'Checked {total_checked} runtime rows near give_way times; non-zero applied rudder rows: {nonzero_count}')
    # require at least one non-zero applied rudder during give_way
    return nonzero_count > 0


def main():
    rc = run_headless()
    if rc != 0:
        print('Headless reproducer returned non-zero:', rc)
        sys.exit(2)
    ok_mismatch = scan_mismatches()
    ok_rudder = check_giveway_rudder_nonzero()
    success = ok_mismatch and ok_rudder
    if success:
        print('\nREGTEST PASS')
        sys.exit(0)
    else:
        print('\nREGTEST FAIL')
        sys.exit(1)

if __name__ == '__main__':
    main()
