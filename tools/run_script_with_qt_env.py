import os
import sys
import subprocess
from pathlib import Path


def run_script(script_path, qt_plugin_path=None, qt_platform='minimal'):
    env = os.environ.copy()
    if qt_plugin_path:
        env['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt_plugin_path)
    env['QT_QPA_PLATFORM'] = qt_platform
    env['OPENBLAS_NUM_THREADS'] = env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'

    proc = subprocess.run([sys.executable, script_path], env=env, capture_output=True)

    out_dir = Path('tmp')
    out_dir.mkdir(exist_ok=True)
    name = Path(script_path).stem
    out_file = out_dir / f"{name}_script_out.txt"
    rc_file = out_dir / f"{name}_script_rc.txt"

    with open(out_file, 'wb') as f:
        f.write(proc.stdout)
        f.write(proc.stderr)

    with open(rc_file, 'w') as f:
        f.write(str(proc.returncode))

    print(f"Wrote {out_file} and {rc_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: run_script_with_qt_env.py <script_path> [qt_plugin_path] [qt_platform]')
        sys.exit(2)
    script = sys.argv[1]
    plugin = sys.argv[2] if len(sys.argv) > 2 else None
    platform = sys.argv[3] if len(sys.argv) > 3 else 'minimal'
    run_script(script, plugin, platform)
