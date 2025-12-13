import os
import sys
import subprocess
from pathlib import Path


TOOLS = [
    'tools/test_opengl_and_qt.py',
    'tools/test_gl_viewwidget.py',
    'tools/test_tin_gl.py',
]

PLATFORMS = ['minimal', 'offscreen', 'windows']


def run_one(script, platform, plugin_path=None):
    env = os.environ.copy()
    if plugin_path:
        env['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugin_path)
    env['QT_QPA_PLATFORM'] = platform
    env['OPENBLAS_NUM_THREADS'] = env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'

    proc = subprocess.run([sys.executable, script], env=env, capture_output=True)
    return proc.returncode, proc.stdout.decode(errors='replace'), proc.stderr.decode(errors='replace')


def main():
    root = Path.cwd()
    tmp = root / 'tmp'
    tmp.mkdir(exist_ok=True)
    plugin_default = Path(r"C:\Users\Kevin.Nebiolo\.conda\envs\emergent\Library\plugins\platforms")

    report_lines = []
    for script in TOOLS:
        report_lines.append(f'--- {script} ---')
        for platform in PLATFORMS:
            rc, out, err = run_one(script, platform, plugin_default)
            out_file = tmp / f"{Path(script).stem}_{platform}_out.txt"
            rc_file = tmp / f"{Path(script).stem}_{platform}_rc.txt"
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(out)
                f.write('\n')
                f.write(err)
            with open(rc_file, 'w', encoding='utf-8') as f:
                f.write(str(rc))
            summary = f'{script} [{platform}] -> rc={rc} (logs: {out_file.name})'
            report_lines.append(summary)
            print(summary)

    report = tmp / 'tools_tests_report.txt'
    report.write_text('\n'.join(report_lines), encoding='utf-8')
    print('\nWrote report to', report)


if __name__ == '__main__':
    main()
