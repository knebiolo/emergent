#!/usr/bin/env python3
"""Run `launch_viewer.py --diag` multiple times with alternate QSurfaceFormat options.

Variants tried:
  0: default (what launch_viewer.py already requests: 3.3 core)
  1: disable alpha channel
  2: request compatibility/profile (NoCore)

This script spawns separate processes and captures each run to outputs/launch_diag_variant{n}.log
"""
import subprocess, os, sys
variants = [
    {'name':'core_default','env':{}},
    {'name':'no_alpha','env':{'EMERGENT_QSURFACE_NO_ALPHA':'1'}},
    {'name':'compat_profile','env':{'EMERGENT_QSURFACE_COMPAT':'1'}},
]
this_dir = os.getcwd()
launcher = os.path.join(this_dir, 'scripts', 'launch_viewer.py')
for i,v in enumerate(variants):
    outlog = os.path.join(this_dir, 'outputs', f'launch_diag_variant{i}_{v["name"]}.log')
    env = os.environ.copy()
    env.update(v['env'])
    cmd = [sys.executable, launcher, '--diag', f'--diag-variant={i}']
    print('Running variant', i, v['name'], '->', outlog)
    with open(outlog, 'wb') as f:
        p = subprocess.Popen(cmd, env=env, cwd=this_dir, stdout=f, stderr=subprocess.STDOUT)
        p.wait()
    print('Variant', i, 'finished with exit', p.returncode)
print('All variants completed')
