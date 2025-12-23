#!/usr/bin/env python3
"""Simple dev server: listens for one client, accepts and streams raw frames with verbose logs.
Usage:
  python tools/dev_stream_server.py --host 127.0.0.1 --port 50007 --agents 1000 --fps 10
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
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--duration', type=float, default=0.0, help='0 = run forever')
    args = parser.parse_args(argv)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.listen(1)
    print(f'Listening for viewer client on {args.host}:{args.port}...')
    try:
        while True:
            try:
                conn, addr = s.accept()
            except KeyboardInterrupt:
                print('Server interrupted, exiting')
                break
            except Exception as e:
                print('Accept failed:', e)
                time.sleep(0.2)
                continue
            print('Viewer connected from', addr)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            N = args.agents
            period = 1.0 / max(1.0, float(args.fps))
            steps = None if args.duration <= 0 else int(max(1, args.duration * args.fps))
            step = 0
            try:
                while steps is None or step < steps:
                    t = time.time()
                    # synthetic swirl
                    arr = np.empty((N, 2), dtype=np.float32)
                    theta = 2 * np.pi * ((t % 60.0) / 60.0)
                    radii = 20.0 + np.arange(N, dtype=np.float32) * 0.5
                    phases = (np.arange(N, dtype=np.float32) / max(1.0, N)) * 2 * np.pi
                    arr[:, 0] = radii * np.cos(theta + phases)
                    arr[:, 1] = radii * np.sin(theta + phases)
                    data = arr.tobytes()
                    packet = b'R' + struct.pack('!I', len(data)) + data
                    try:
                        conn.sendall(packet)
                    except Exception as e:
                        print('send error, closing connection:', e)
                        break
                    if step % max(1, int(args.fps)) == 0:
                        print(f'sent frame {step} to {addr} (nbytes={len(data)})')
                    step += 1
                    time.sleep(period)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                print('Connection closed, waiting for next client...')
    finally:
        try:
            s.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
