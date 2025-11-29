"""Run a series of HECRAS-only benchmarks varying k and number of agents.

This script calls the existing `run_headless_hecras_sim.py` script as a subprocess
for the desired combinations and stores CSV outputs under `outputs/benchmarks/`.
"""
import subprocess
from pathlib import Path

ks = [3, 8, 16]
ns = [50, 200, 1000]
timesteps = 200
outdir = Path('outputs/benchmarks')
outdir.mkdir(parents=True, exist_ok=True)

for k in ks:
    for n in ns:
        out = outdir / f'bench_k{k}_n{n}.csv'
        print(f'Running k={k} n={n} -> {out}')
        cmd = ['python', 'tools/run_headless_hecras_sim.py', '--timesteps', str(timesteps), '--num-agents', str(n), '--out', str(out), '--hecras_k', str(k)]
        subprocess.check_call(cmd)

print('Sweep complete')
