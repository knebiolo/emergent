from PIL import Image, ImageStat
import glob, os

candidates = glob.glob('outputs/diag/outputs/tin_preview*.png') + glob.glob('outputs/diag/outputs/tin_preview.png') + glob.glob('outputs/tin_preview*.png')
if not candidates:
    # fallback to known path
    candidates = ['outputs/diag/outputs/tin_preview.png', 'outputs/tin_preview.png']

for p in candidates:
    if not os.path.exists(p):
        continue
    try:
        im = Image.open(p).convert('RGBA')
        st = ImageStat.Stat(im)
        print('path:', p)
        print('size:', im.size)
        print('mode:', im.mode)
        print('bands:', im.getbands())
        print('min:', [e[0] for e in st.extrema])
        print('max:', [e[1] for e in st.extrema])
        print('mean:', st.mean)
        # quick center pixel
        cx = im.size[0]//2
        cy = im.size[1]//2
        print('center pixel RGBA:', im.getpixel((cx,cy)))
        # count non-black pixels
        pix = im.getdata()
        nonblack = sum(1 for px in pix if px[0] or px[1] or px[2])
        total = im.size[0]*im.size[1]
        print('non-black pixels:', nonblack, '/', total)
        break
    except Exception as e:
        print('error reading', p, e)
else:
    print('no preview images found')
