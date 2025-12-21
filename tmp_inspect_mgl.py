from PIL import Image
import os

def inspect(fn):
    p = os.path.join(os.getcwd(), 'outputs', fn)
    print('path exists:', os.path.exists(p), p)
    if not os.path.exists(p):
        return
    im = Image.open(p).convert('RGBA')
    arr = im
    w,h = im.size
    px = list(arr.getdata())
    N = len(px)
    print('format, size, mode:', im.format, im.size, im.mode)
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
    print('center pixel RGBA:', arr.getpixel((w//2, h//2)))
    print('top-left 5 pixels:', [arr.getpixel((x,0)) for x in range(min(5,w))])

if __name__ == '__main__':
    inspect('diag_snapshot.png')
    inspect('diag_snapshot_mgl.png')
