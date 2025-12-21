"""Inspect PNG diagnostic images and report sizes and average brightness.

Usage: python tools/inspect_images.py
"""
import os
from PIL import Image, ImageStat

paths = [
    'outputs/latest_preview.png',
    'outputs/cpu_raster_snapshot.png',
    'outputs/diag_snapshot.png',
    'outputs/diag_snapshot_fbo.png',
]

for p in paths:
    if not os.path.exists(p):
        print(p, 'MISSING')
        continue
    try:
        im = Image.open(p).convert('RGBA')
        stat = ImageStat.Stat(im)
        # average brightness per channel
        means = stat.mean
        size = os.path.getsize(p)
        print(p, 'size=', size, 'meanRGBA=', [round(m,2) for m in means])
    except Exception as e:
        print(p, 'ERR', e)
