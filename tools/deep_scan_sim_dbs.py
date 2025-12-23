"""Deep-scan sim DB files and report any dataset paths that contain 'freq' or 'diagnostics'."""
import os
import glob
import h5py

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
PAT = os.path.join(OUT_DIR, 'sim_db_*.h5')


def scan():
    files = glob.glob(PAT)
    if not files:
        print('No sim_db files found')
        return
    files.sort(key=os.path.getmtime, reverse=True)
    for p in files[:50]:
        try:
            with h5py.File(p, 'r') as f:
                found = []
                def visitor(name, obj):
                    if 'freq' in name or 'diagnostics' in name:
                        found.append(name)
                f.visititems(visitor)
                print(p, '->', found)
        except Exception as e:
            print(p, '-> open failed:', e)

if __name__ == '__main__':
    scan()
