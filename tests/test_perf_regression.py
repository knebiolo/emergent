import os
import subprocess
import sys

# Lightweight perf regression: run the profiler for a tiny scenario and ensure it completes

def test_small_profiler_run():
    cmd = [sys.executable, 'tools/profile_timestep_cprofile.py', '--agents', '50', '--timesteps', '2', '--out', 'tools/test_perf_reg.pstats']
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert res.returncode == 0, f'Profiler failed: {res.stderr.decode()}'
    assert os.path.exists('tools/test_perf_reg.pstats')
