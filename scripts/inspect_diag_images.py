#!/usr/bin/env python3
from PIL import Image
import numpy as np
import os

out_lines = []
for name in ['diag_snapshot.png','diag_snapshot_mgl.png','diag_snapshot_fbo.png']:
    p = os.path.join('outputs', name)
    if not os.path.exists(p):
        out_lines.append(f"{name}: MISSING")
        continue
    im = Image.open(p).convert('RGBA')
    arr = np.array(im)
    mins = arr.min(axis=(0,1)).tolist()
    maxs = arr.max(axis=(0,1)).tolist()
    shape = im.size
    # compute unique color count (may be large)
    flat = arr.reshape(-1,4)
    uniques = np.unique(flat, axis=0)
    ucount = len(uniques)
    out_lines.append(f"{name}: size={shape} dtype={arr.dtype} min={mins} max={maxs} unique_colors={ucount}")
    if ucount <= 64:
        out_lines.append(f"  colors={uniques.tolist()}")

print('\n'.join(out_lines))
