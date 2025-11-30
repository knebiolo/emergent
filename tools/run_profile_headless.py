import runpy
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

out_text = os.path.join(repo_root, 'tools', 'profile_text_after_numba_expand.txt')
# set CLI args for the profiling script
sys_argv_backup = sys.argv
sys.argv = ['profile_timestep_cprofile.py', '--agents', '200', '--timesteps', '50', '--out', os.path.join('tools','profile_after_numba_expand.pstats')]

# redirect stdout
with open(out_text, 'w', encoding='utf-8') as f:
    sys_stdout_backup = sys.stdout
    try:
        sys.stdout = f
        runpy.run_path(os.path.join(repo_root, 'tools', 'profile_timestep_cprofile.py'), run_name='__main__')
    finally:
        sys.stdout = sys_stdout_backup
        sys.argv = sys_argv_backup

print('Wrote profile text to', out_text)
