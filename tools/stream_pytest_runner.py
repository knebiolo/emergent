"""
Stream pytest stdout/stderr to files and record exit code.
Usage: python tools/stream_pytest_runner.py
"""
import subprocess, sys, os
out_path = 'tmp/full_tests_out.txt'
err_path = 'tmp/full_tests_err.txt'
code_path = 'tmp/full_tests_exitcode.txt'
# ensure tmp exists
os.makedirs('tmp', exist_ok=True)
cmd = [sys.executable, '-X', 'faulthandler', '-m', 'pytest', '-q', '--basetemp', 'tmp/pytest_full', '--junit-xml', 'tmp/junit.xml']
print('Running:', ' '.join(cmd))
with open(out_path, 'wb') as out_f, open(err_path, 'wb') as err_f:
    # limit BLAS/OMP thread counts to avoid native crashes on some Windows/BLAS builds
    env = dict(os.environ)
    env.update({
        'PYTHONFAULTHANDLER': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        # Qt/OpenGL headless settings to reduce driver/GPU interaction during collection
        'QT_QPA_PLATFORM': 'offscreen',
        'AA_UseDesktopOpenGL': '0',
        'QT_XCB_GL_INTEGRATION': 'none',
    })
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    # stream
    try:
        while True:
            o = proc.stdout.read(1024)
            if o:
                out_f.write(o)
                out_f.flush()
            e = proc.stderr.read(1024)
            if e:
                err_f.write(e)
                err_f.flush()
            if o == b'' and e == b'' and proc.poll() is not None:
                break
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
    rc = proc.poll()
    with open(code_path, 'w') as f:
        f.write(str(rc))
print('Completed with rc', rc)
