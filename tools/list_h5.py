import glob, os, h5py
files = glob.glob(os.path.join('outputs','sim_db_*.h5'))
if not files:
    print('no sim db files')
    raise SystemExit(1)
latest = max(files, key=os.path.getmtime)
print('Latest DB:', latest)
with h5py.File(latest,'r') as f:
    def walk(name, obj):
        print(name)
    f.visititems(walk)
