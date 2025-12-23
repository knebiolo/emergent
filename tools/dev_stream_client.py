#!/usr/bin/env python3
"""Simple dev client that connects to a server, receives one raw frame, prints metadata, saves payload to disk, and exits.
Usage:
  python tools/dev_stream_client.py --host 127.0.0.1 --port 50007
"""
import argparse
import socket
import struct
import numpy as np

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=50007)
    args = parser.parse_args(argv)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.connect((args.host, args.port))
    except Exception as e:
        print('connect failed:', e)
        return
    print('connected to server', args.host, args.port)
    try:
        # read first byte
        first = s.recv(1)
        if not first:
            print('no data received')
            return
        if first == b'R':
            lb = s.recv(4)
            if not lb or len(lb) < 4:
                print('incomplete length header', lb)
                return
            (nbytes,) = struct.unpack('!I', lb)
            print('raw frame nbytes=', nbytes)
            buf = bytearray()
            while len(buf) < nbytes:
                chunk = s.recv(nbytes - len(buf))
                if not chunk:
                    break
                buf.extend(chunk)
            if len(buf) < nbytes:
                print('incomplete payload', len(buf))
                return
            arr = np.frombuffer(bytes(buf), dtype=np.float32).reshape((-1,2))
            print('received arr shape', arr.shape)
            np.save('dev_client_frame.npy', arr)
            print('saved dev_client_frame.npy')
        else:
            rest = s.recv(3)
            if not rest or len(rest) < 3:
                print('incomplete length for npy mode')
                return
            (nbytes,) = struct.unpack('!I', first+rest)
            print('npy length=', nbytes)
    finally:
        try:
            s.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
