#!/usr/bin/env python3
"""Run the viewer with different QT_OPENGL environment settings to test ANGLE vs desktop vs software.
Saves logs to outputs/launch_qt_opengl_{mode}.log
"""
import os, subprocess, sys
modes = ['desktop', 'angle', 'software']
this_dir = os.getcwd()
launcher = os.path.join(this_dir, 'scripts', 'launch_viewer.py')
for mode in modes:
    outlog = os.path.join(this_dir, 'outputs', f'launch_qt_opengl_{mode}.log')
    env = os.environ.copy()
    env['QT_OPENGL'] = mode
    cmd = [sys.executable, launcher, '--diag']
    print('Running QT_OPENGL=', mode)
    with open(outlog, 'wb') as f:
        p = subprocess.Popen(cmd, env=env, cwd=this_dir, stdout=f, stderr=subprocess.STDOUT)
        p.wait()
    print('Mode', mode, 'exit', p.returncode)
print('Completed all QT_OPENGL modes')
