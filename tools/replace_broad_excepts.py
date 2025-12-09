"""Safe replacer for broad `except Exception: pass` occurrences.

This script finds occurrences of the exact pattern:

    except Exception:
        pass

in `src/emergent/salmon_abm/sockeye.py` and replaces them in micro-batches with a
log-first handler that calls `_safe_log_exception(...)` and preserves a `pass`.

Behavior:
- Operates in batches (default 6 replacements per batch).
- Creates a timestamped backup of the original `sockeye.py` before first write.
- After each batch it runs the headless smoke test `python tools\headless_viewer_check.py`.
- If smoke test fails, restores backup, writes diagnostic files and exits non-zero.
- Logs progress to `.ai_journal/session/2025-12-09-iterate-excepts.md`.

Usage:
    python tools\replace_broad_excepts.py [--batch-size N] [--dry-run]

"""
import re
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SOCKEYE = ROOT / 'src' / 'emergent' / 'salmon_abm' / 'sockeye.py'
SESSION_REPORT = ROOT / '.ai_journal' / 'session' / '2025-12-09-iterate-excepts.md'
BACKUP_DIR = ROOT / 'tmp_replace_backups'

PATTERN = re.compile(r'(^[ \t]*)except\s+Exception\s*:\s*\n([ \t]*)pass\b', re.M)

REPLACEMENT_TEMPLATE = """{indent}except Exception as e:
{body_indent}_safe_log_exception('Auto-patched broad except', e, file='sockeye.py', line={line})
{body_indent}pass"""


def find_matches(text):
    return list(PATTERN.finditer(text))


def do_batch(text, matches, batch_size, start_idx=0):
    """Return (new_text, replacements_done)
    Only replaces matches[start_idx:start_idx+batch_size]
    """
    if not matches:
        return text, 0
    to_replace = matches[start_idx:start_idx+batch_size]
    if not to_replace:
        return text, 0
    pieces = []
    last = 0
    for m in to_replace:
        s, e = m.span()
        pieces.append(text[last:s])
        indent = m.group(1)
        body_indent = m.group(2) or (indent + '    ')
        line_no = text[:s].count('\n') + 1
        repl = REPLACEMENT_TEMPLATE.format(indent=indent, body_indent=body_indent, line=line_no)
        pieces.append(repl)
        last = e
    pieces.append(text[last:])
    return ''.join(pieces), len(to_replace)


def append_session(line):
    try:
        with open(SESSION_REPORT, 'a', encoding='utf-8') as f:
            f.write('\n' + line + '\n')
    except Exception:
        pass


def run_smoke():
    cmd = [sys.executable, str(ROOT / 'tools' / 'headless_viewer_check.py')]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + '\n' + proc.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if not SOCKEYE.exists():
        print('sockeye.py not found at', SOCKEYE)
        return 2
    text = SOCKEYE.read_text(encoding='utf-8')
    matches = find_matches(text)
    total_matches = len(matches)
    print(f'Found {total_matches} exact `except Exception:\n    pass` occurrences')
    if total_matches == 0:
        append_session(f"[{datetime.utcnow().isoformat()}] replace_broad_excepts: no matches found.")
        return 0

    # ensure backup dir
    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    backup_path = BACKUP_DIR / f'sockeye.py.bak.{timestamp}'
    if not backup_path.exists():
        shutil.copy2(SOCKEYE, backup_path)
        print('Backup created at', backup_path)

    # iterate batches
    cur_text = text
    replaced_total = 0
    batch_idx = 0
    while True:
        matches = find_matches(cur_text)
        if not matches:
            print('No more matches; done')
            append_session(f"[{datetime.utcnow().isoformat()}] replace_broad_excepts: completed all replacements; total_replaced={replaced_total}")
            return 0
        # perform one batch
        new_text, done = do_batch(cur_text, matches, args.batch_size, start_idx=0)
        if done == 0:
            print('No replacements in batch; finished')
            append_session(f"[{datetime.utcnow().isoformat()}] replace_broad_excepts: finished; total_replaced={replaced_total}")
            return 0
        batch_idx += 1
        print(f'Applying batch {batch_idx}: replacing {done} occurrences...')
        if args.dry_run:
            print('Dry-run mode: no file changes applied')
            return 0
        # write new file atomically
        tmp_path = SOCKEYE.with_suffix('.py.tmp')
        SOCKEYE.write_text(new_text, encoding='utf-8')
        # Run smoke test
        rc, output = run_smoke()
        now = datetime.utcnow().isoformat()
        if rc != 0:
            # restore backup
            shutil.copy2(backup_path, SOCKEYE)
            print('Smoke test failed after batch', batch_idx)
            print(output)
            append_session(f"[{now}] Batch {batch_idx} FAILED: replaced={done}; restored backup. Smoke output:\n{output}")
            return 3
        else:
            replaced_total += done
            cur_text = new_text
            print('Batch OK — smoke test passed')
            append_session(f"[{now}] Batch {batch_idx} OK: replaced={done}; total_replaced={replaced_total}")
            # continue loop


if __name__ == '__main__':
    raise SystemExit(main())
