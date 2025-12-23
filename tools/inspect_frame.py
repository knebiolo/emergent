#!/usr/bin/env python3
"""Inspect a saved live frame (.npy) and print simple stats.
Usage:
  python tools/inspect_frame.py latest_live_frame.npy
"""
import sys
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def inspect(path):
    p = Path(path)
    if not p.exists():
        print('File not found:', path)
        return 1
    arr = np.load(str(p))
    print('shape', arr.shape)
    try:
        xs = arr[:, :, 0]
        ys = arr[:, :, 1]
        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.any():
            print('xmin xmax ymin ymax', float(np.nanmin(xs[valid])), float(np.nanmax(xs[valid])), float(np.nanmin(ys[valid])), float(np.nanmax(ys[valid])))
        else:
            print('no finite coords')
    except Exception:
        # maybe it's (N,2)
        try:
            xs = arr[:, 0]
            ys = arr[:, 1]
            valid = np.isfinite(xs) & np.isfinite(ys)
            if valid.any():
                print('xmin xmax ymin ymax', float(np.nanmin(xs[valid])), float(np.nanmax(xs[valid])), float(np.nanmin(ys[valid])), float(np.nanmax(ys[valid])))
        except Exception:
            print('Could not interpret array shape for coords')
    print('sample points (first 10):')
    try:
        flat = arr.reshape(-1, 2)
        for i, (x, y) in enumerate(flat[:10]):
            print(i, x, y)
    except Exception:
        pass

    if PIL_AVAILABLE:
        try:
            w = 800; h = 600
            img = Image.new('RGB', (w, h), (255,255,255))
            draw = ImageDraw.Draw(img)
            # mapping
            try:
                if arr.ndim == 3:
                    pos = arr[0]
                else:
                    pos = arr
                xs = pos[:,0]; ys = pos[:,1]
                valid = np.isfinite(xs) & np.isfinite(ys)
                xmin = float(np.nanmin(xs[valid])); xmax = float(np.nanmax(xs[valid]))
                ymin = float(np.nanmin(ys[valid])); ymax = float(np.nanmax(ys[valid]))
                dx = xmax - xmin if xmax != xmin else 1.0
                dy = ymax - ymin if ymax != ymin else 1.0
                s = min(w/dx, h/dy) * 0.9
                tx = (w - s*dx)/2.0
                ty = (h - s*dy)/2.0
                for x,y in pos:
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    sxp = tx + (x - xmin)*s
                    syp = ty + (ymax - y)*s
                    r = max(1, int(min(w,h)*0.01))
                    draw.ellipse((sxp-r, syp-r, sxp+r, syp+r), fill=(200,30,30))
                out = p.with_suffix('.png')
                img.save(str(out))
                print('Wrote snapshot', out)
            except Exception as e:
                print('Failed to make snapshot:', e)
        except Exception:
            pass
    else:
        print('Pillow not available; no snapshot written')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: tools/inspect_frame.py <file.npy>')
        sys.exit(2)
    sys.exit(inspect(sys.argv[1]))
