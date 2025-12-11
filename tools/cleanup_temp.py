"""Cleanup project temp directory

Removes files and directories older than a configured age (default 48 hours)
from `tmp/pytest_tmp`. Designed to be safe: only operates within the configured
project temp dir and never ascends above it.

Usage:
    python tools/cleanup_temp.py --path tmp/pytest_tmp --max-age-hours 48

This script exits with code 0 on success and prints a summary of removed items.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path


def remove_old(path: Path, max_age_seconds: int) -> int:
    now = time.time()
    removed = 0
    path = path.resolve()
    if not path.exists():
        print(f'Path does not exist: {path}')
        return 0
    if not str(path).startswith(str(Path.cwd())):
        raise RuntimeError('Refusing to operate outside the repository root')

    for root, dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                mtime = fp.stat().st_mtime
                if (now - mtime) > max_age_seconds:
                    fp.unlink()
                    removed += 1
                    print(f'Removed file: {fp}')
            except Exception as e:
                print(f'Failed to remove file {fp}: {e}')
        for name in dirs:
            dp = Path(root) / name
            try:
                mtime = dp.stat().st_mtime
                # only remove empty directories older than threshold
                if (now - mtime) > max_age_seconds and not any(dp.iterdir()):
                    dp.rmdir()
                    removed += 1
                    print(f'Removed empty dir: {dp}')
            except Exception as e:
                print(f'Failed to remove dir {dp}: {e}')
    return removed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', default='tmp/pytest_tmp', help='Temp dir to clean')
    p.add_argument('--max-age-hours', type=float, default=48.0, help='Max age in hours')
    args = p.parse_args()

    path = Path(args.path)
    max_age_seconds = int(args.max_age_hours * 3600)

    removed = remove_old(path, max_age_seconds)
    print(f'Done. Removed {removed} items from {path}')


if __name__ == '__main__':
    main()
