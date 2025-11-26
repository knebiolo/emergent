"""
Extract last N [CTRL] ticks from a run log and include nearby COLREGS flagged_give_way state.
Produces CSV to logs/ctrl_window_extract.csv

Usage:
  python scripts\extract_ctrl_window.py --lines 400 --out logs/ctrl_window_extract.csv
"""
import re
import sys
import argparse
from collections import defaultdict

LOG = r"logs/run_ship_Seattle_chicken.out"

CTRL_RE = re.compile(r"\[CTRL\].*t=(?P<t>[0-9]+\.?[0-9]*)s.*hd_cmd=(?P<hd_cmd>-?[0-9]+\.?[0-9]*).*hd_cur=(?P<hd_cur>-?[0-9]+\.?[0-9]*).*err=(?P<err>-?[0-9]+\.?[0-9]*).*rud=(?P<rud>-?[0-9]+\.?[0-9]*)")
GIVE_RE = re.compile(r"\[COLREGS\] flagged_give_way persists agent=(?P<agent>\d+) post_avoid=(?P<post_avoid>[0-9]+\.?[0-9]*)s linger=(?P<linger>[0-9]+\.?[0-9]*)s lock=(?P<lock>\d+)")
TICK_RE = re.compile(r"^\u25b6 tick t=(?P<t>[0-9]+\.?[0-9]*)")


def parse_log(path):
    ctrls = []
    gives = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = CTRL_RE.search(ln)
            if m:
                ctrls.append({
                    "t": float(m.group("t")),
                    "hd_cmd": float(m.group("hd_cmd")),
                    "hd_cur": float(m.group("hd_cur")),
                    "err": float(m.group("err")),
                    "rud": float(m.group("rud")),
                    "raw": ln.strip()
                })
                continue
            mg = GIVE_RE.search(ln)
            if mg:
                gives.append({
                    "t": None,
                    "agent": int(mg.group("agent")),
                    "post_avoid": float(mg.group("post_avoid")),
                    "linger": float(mg.group("linger")),
                    "lock": int(mg.group("lock")),
                    "raw": ln.strip()
                })
                # try to extract a tick time from the most recent tick line in same vicinity
                # (we'll match times later by proximity)
                continue
            # try to capture tick lines to associate gives with times
            mt = TICK_RE.search(ln)
            if mt and gives:
                # assign time to the last unmatched give entry if it has no time
                if gives and gives[-1]["t"] is None:
                    gives[-1]["t"] = float(mt.group("t"))
    return ctrls, gives


def stitch(ctrls, gives):
    # for each ctrl sample, find the nearest give (within 1s) and attach fields
    gives_by_time = sorted([g for g in gives if g["t"] is not None], key=lambda x: x["t"]) if gives else []
    out = []
    import bisect
    times = [g["t"] for g in gives_by_time]
    for c in ctrls:
        entry = dict(c)
        entry.update({"give_agent":"", "post_avoid":"", "linger":"", "lock":""})
        if gives_by_time:
            i = bisect.bisect_left(times, c["t"]) 
            # check nearest left and right
            candidates = []
            if i>0: candidates.append(gives_by_time[i-1])
            if i<len(gives_by_time): candidates.append(gives_by_time[i])
            best = None
            best_dt = 1e9
            for g in candidates:
                dt = abs(g["t"] - c["t"]) if g["t"] is not None else 1e9
                if dt < best_dt:
                    best = g; best_dt = dt
            if best and best_dt <= 2.0: # within 2s
                entry["give_agent"] = best["agent"]
                entry["post_avoid"] = best["post_avoid"]
                entry["linger"] = best["linger"]
                entry["lock"] = best["lock"]
        out.append(entry)
    return out


def write_csv(rows, out_path):
    import csv
    keys = ["t","hd_cmd","hd_cur","err","rud","give_agent","post_avoid","linger","lock"]
    with open(out_path, "w", newline='', encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in keys}
            w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lines", type=int, default=400, help="how many last [CTRL] entries to include")
    p.add_argument("--out", default="logs/ctrl_window_extract.csv")
    p.add_argument("--log", default=LOG)
    args = p.parse_args()

    ctrls, gives = parse_log(args.log)
    if not ctrls:
        print("No [CTRL] entries found in", args.log)
        return 2
    print(f"Found {len(ctrls)} [CTRL] samples and {len(gives)} flagged_give_way entries")
    rows = stitch(ctrls, gives)
    rows = rows[-args.lines:]
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} samples to {args.out}. First t={rows[0]['t'] if rows else 'n/a'} last t={rows[-1]['t'] if rows else 'n/a'}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
