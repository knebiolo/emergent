import os, runpy, shutil, multiprocessing, time, csv, subprocess
from pathlib import Path
import psutil

OUT_DIR = Path('outputs/profiling')
OUT_DIR.mkdir(parents=True, exist_ok=True)
BS = [64,128,256,512,1024]
AGENTS = [10000,20000]

rows = []
proc = psutil.Process()
for n in AGENTS:
    for b in BS:
        print(f'Running sweep: n={n} b={b}')
        os.environ['BEHAVIOR_BATCH_SIZE'] = str(b)
        os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count())

        # 1) Warmup run to force JIT compilation and stabilize caches
        warm_tag = f'n{n}_b{b}_warm'
        cmd_warm = ['python', 'tools/profile_simulation.py', '--mode', 'hotspot', '--nagents', str(n), '--iters', '10', '--tag', warm_tag]
        if os.environ.get('SWEEP_DEBUG') == '1':
            cmd_warm.append('--debug-behavior')
        subprocess.check_call(cmd_warm)

        # 2) Two measured runs (meas1 + meas2) to reduce single-run variance
        measured_tags = []
        for meas_i in (1, 2):
            measured_tag = f'n{n}_b{b}_meas{meas_i}'
            t0 = time.time()
            cmd_meas = ['python', 'tools/profile_simulation.py', '--mode', 'hotspot', '--nagents', str(n), '--iters', '100', '--tag', measured_tag]
            if os.environ.get('SWEEP_DEBUG') == '1':
                cmd_meas.append('--debug-behavior')
            subprocess.check_call(cmd_meas)
            t1 = time.time()

            # copy the measured hotspot profile to a named file (profile name already contains tag)
            pfile = OUT_DIR / f'hotspot_profile_{measured_tag}.txt'
            dst = OUT_DIR / f'hotspot_profile_{n}_b{b}_{meas_i}.txt'
            try:
                shutil.copy(pfile, dst)
            except Exception:
                pass

            rss = proc.memory_info().rss
            rows.append({'n_agents': n, 'batch': b, 'wallclock': round(t1-t0, 6), 'rss': rss, 'tag': measured_tag})
            measured_tags.append(measured_tag)
            print('done', n, b, 'meas', meas_i, t1-t0, rss)

out = OUT_DIR / 'sweep_monitor.csv'
with out.open('w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=['n_agents','batch','wallclock','rss','tag'])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print('Wrote', out)
