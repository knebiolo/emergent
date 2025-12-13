import subprocess, sys, os
cmd=[sys.executable,'-X','faulthandler','-m','pytest','-q','--basetemp','tmp/pytest_full','--junit-xml','tmp/junit.xml']
print('Running:', ' '.join(cmd))
env = os.environ.copy()
env['PYTHONFAULTHANDLER'] = '1'
try:
    proc=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=300)
except subprocess.TimeoutExpired:
    print('pytest timed out after 300s')
    with open('tmp/subproc_partial_out.txt','wb') as f:
        f.write(b'')
    sys.exit(2)
with open('tmp/subproc_out.txt','wb') as f:
    f.write(proc.stdout)
with open('tmp/subproc_err.txt','wb') as f:
    f.write(proc.stderr)
print('Return code:', proc.returncode)
