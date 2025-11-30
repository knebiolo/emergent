import runpy
import os
import sys

# Ensure repo root is on sys.path for local imports
repo_root = os.path.abspath(os.getcwd())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

script = os.path.join(repo_root, 'tools', 'profile_timestep_cprofile.py')
print('Running', script)
runpy.run_path(script, run_name='__main__')
print('Done')
