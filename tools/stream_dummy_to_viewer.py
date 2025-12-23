#!/usr/bin/env python3
"""Stream synthetic frames to the viewer live TCP server for testing.

Usage:
  python tools/stream_dummy_to_viewer.py --host 127.0.0.1 --port 50007
"""
import argparse
import socket
import struct
import time
import io
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=50007)
    parser.add_argument('--frames', type=int, default=500)
    parser.add_argument('--agents', type=int, default=50)
    parser.add_argument('--delay', type=float, default=0.05)
    parser.add_argument('--raw', action='store_true', help='Send raw float32 payloads (faster)')
    args = parser.parse_args(argv)

    # Act as server: bind and listen, accept one client
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.listen(1)
    print(f'Listening for viewer client on {args.host}:{args.port}...')
    conn, addr = s.accept()
    print('Viewer connected from', addr)

    T = args.frames
    N = args.agents
    for t in range(T):
        theta = 2 * np.pi * (t / max(1, T))
        arr = np.zeros((N, 2), dtype=float)
        for i in range(N):
            r = 50.0 + 2.0 * i
            phase = (i / max(1, N)) * 2 * np.pi
            arr[i, 0] = r * np.cos(theta + phase) + 500.0
            arr[i, 1] = r * np.sin(theta + phase) + 1000.0
        if args.raw:
            # prepare raw float32 payload: N x 2 float32 array, row-major
            data = arr.astype(np.float32).tobytes()
            # protocol: 1 byte 'R' then 4-byte length then payload
            try:
                conn.sendall(b'R' + struct.pack('!I', len(data)) + data)
            except Exception:
                break
        else:
            bio = io.BytesIO()
            np.save(bio, arr)
            data = bio.getvalue()
            try:
                conn.sendall(struct.pack('!I', len(data)))
                conn.sendall(data)
            except Exception:
                break
        time.sleep(args.delay)

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
