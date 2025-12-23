#!/usr/bin/env python3
"""Start the viewer and stream demo frames to it in one command (Windows-friendly)."""
import subprocess
import time
import sys
import os

VIEWER_CMD = [sys.executable, '-m', 'emergent.salmon_abm.realtime_viewer', '--live', '--host', '127.0.0.1', '--port', '50007']
STREAMER_CMD = [sys.executable, 'tools/stream_dummy_to_viewer.py', '--host', '127.0.0.1', '--port', '50007', '--frames', '300', '--agents', '60', '--delay', '0.03']

def main():
    # start viewer as a subprocess
    print('Starting streamer (server)...')
    with open('streamer_stdout.txt', 'wb') as sf_out, open('streamer_stderr.txt', 'wb') as sf_err:
        streamer_proc = subprocess.Popen(STREAMER_CMD, stdout=sf_out, stderr=sf_err)
        time.sleep(0.5)
        try:
            print('Starting viewer (client)...')
            with open('viewer_stdout.txt', 'wb') as vf_out, open('viewer_stderr.txt', 'wb') as vf_err:
                viewer_proc = subprocess.Popen(VIEWER_CMD, stdout=vf_out, stderr=vf_err)
                viewer_proc.wait()
        finally:
            try:
                streamer_proc.terminate()
            except Exception:
                pass
    # print logs for diagnosis
    try:
        print('\n=== viewer stderr ===')
        with open('viewer_stderr.txt', 'r', encoding='utf-8', errors='ignore') as fh:
            print(fh.read())
    except Exception:
        pass
    try:
        print('\n=== streamer stderr ===')
        with open('streamer_stderr.txt', 'r', encoding='utf-8', errors='ignore') as fh:
            print(fh.read())
    except Exception:
        pass

if __name__ == '__main__':
    main()
