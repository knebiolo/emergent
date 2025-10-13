import sys
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path
import fiona


def check_cell(cell_id, url=None):
    if url is None:
        url = f"https://www.charts.noaa.gov/ENCs/{cell_id}.zip"
    tmpd = Path(tempfile.mkdtemp(prefix=f"enc_check_{cell_id}_"))
    try:
        print(f"Downloading {url} -> {tmpd}")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        zpath = tmpd / f"{cell_id}.zip"
        with open(zpath, 'wb') as fh:
            for chunk in r.iter_content(1024*32):
                fh.write(chunk)
        print(f"Extracting {zpath}")
        with zipfile.ZipFile(zpath, 'r') as z:
            z.extractall(tmpd)
        # find .000 file
        found = list(tmpd.rglob('*.000'))
        if not found:
            print("No .000 files found in archive")
            return 2
        f000 = found[0]
        print(f"Found .000: {f000}")
        try:
            layers = fiona.listlayers(str(f000))
        except Exception as e:
            print(f"fiona.listlayers failed: {e}")
            return 3
        print(f"Layers: {layers}")
        print(f"LNDARE present: {'LNDARE' in layers}")
        print(f"COALNE present: {'COALNE' in layers}")
        return 0
    finally:
        try:
            shutil.rmtree(tmpd)
        except Exception:
            pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python check_enc_cell.py CELL_ID')
        sys.exit(1)
    cell = sys.argv[1]
    sys.exit(check_cell(cell))
