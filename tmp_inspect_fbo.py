from PIL import Image
import os
p = os.path.join(os.getcwd(), 'outputs', 'diag_snapshot_fbo.png')
print('path exists:', os.path.exists(p), p)
if not os.path.exists(p):
    raise SystemExit(1)
im = Image.open(p).convert('RGBA')
import numpy as np
arr = np.array(im)
print('format, size, mode:', im.format, im.size, im.mode)
print('pixels:', arr.size//4)
print('mins:', arr.reshape(-1,4).min(axis=0).tolist())
print('maxs:', arr.reshape(-1,4).max(axis=0).tolist())
print('means:', arr.reshape(-1,4).mean(axis=0).tolist())
print('center pixel RGBA:', tuple(arr[im.size[1]//2, im.size[0]//2]))
print('top-left 5 pixels:', [tuple(arr[0,i]) for i in range(min(5, im.size[0]))])
