#!/usr/bin/env python3
"""Stream a high-agent-count test to the viewer using the raw protocol.

Usage:
  python tools/stream_load_test.py --host 127.0.0.1 --port 50007 --agents 1000 --fps 20 --duration 30
"""
import argparse
import socket
import struct
import time
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=50007)
    parser.add_argument('--agents', type=int, default=1000)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--duration', type=float, default=30.0)
    args = parser.parse_args(argv)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.listen(1)
    print(f'Listening for viewer client on {args.host}:{args.port}...')
    conn, addr = s.accept()
    print('Viewer connected from', addr)

    N = args.agents
    period = 1.0 / max(1.0, float(args.fps))
    steps = int(max(1, args.duration * args.fps))
    t0 = time.time()
    try:
        for step in range(steps):
            t = time.time() - t0
            # create a synthetic swirl pattern scaled with agent index
            arr = np.empty((N, 2), dtype=np.float32)
            theta = 2 * np.pi * (t / max(1.0, args.duration))
            radii = 20.0 + np.arange(N, dtype=np.float32) * 0.5
            phases = (np.arange(N, dtype=np.float32) / max(1.0, N)) * 2 * np.pi
            arr[:, 0] = radii * np.cos(theta + phases) + 500.0
            arr[:, 1] = radii * np.sin(theta + phases) + 1000.0
            data = arr.tobytes()
            # send with raw protocol: 'R' + len + payload
            try:
                conn.sendall(b'R' + struct.pack('!I', len(data)) + data)
            except Exception as e:
                print('send error', e)
                break
            time.sleep(period)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            s.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
