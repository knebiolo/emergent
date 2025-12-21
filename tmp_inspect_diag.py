from PIL import Image
import os
p = os.path.join(os.getcwd(), 'outputs', 'diag_snapshot.png')
print('path exists:', os.path.exists(p), p)
if not os.path.exists(p):
    raise SystemExit(1)
img = Image.open(p)
print('format, size, mode:', img.format, img.size, img.mode)
arr = img.convert('RGBA')
px = list(arr.getdata())
N = len(px)
print('pixels:', N)
channels = 4
sums = [0]*channels
mins = [255]*channels
maxs = [0]*channels
for c in px:
    for i,v in enumerate(c):
        sums[i]+=v
        mins[i]=min(mins[i], v)
        maxs[i]=max(maxs[i], v)
means = [s / N for s in sums]
print('mins:', mins)
print('maxs:', maxs)
print('means:', means)
# Print center pixel and a small crop sample
w,h = arr.size
print('center pixel RGBA:', arr.getpixel((w//2, h//2)))
print('top-left 5 pixels:', [arr.getpixel((x,0)) for x in range(min(5,w))])
