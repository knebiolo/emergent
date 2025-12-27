import h5py
import sys
import numpy as np
import os

def seed_memory(preseed_path, nagents=1000, timestamp=50, avoid_cell_size=10.0):
    preseed_path = os.path.abspath(preseed_path)
    if not os.path.exists(preseed_path):
        print('Preseed file not found:', preseed_path)
        return 2
    with h5py.File(preseed_path, 'a') as f:
        # attempt to infer raster height/width from environment/depth
        if 'environment/depth' in f:
            depth = np.array(f['environment/depth'])
            height, width = depth.shape
        elif 'environment/x_coords' in f:
            arr = np.array(f['environment/x_coords'])
            height, width = arr.shape
        else:
            # fallback to small default
            height, width = 1000, 1000

        avoid_h = int(np.round(height / avoid_cell_size, 0)) + 1
        avoid_w = int(np.round(width / avoid_cell_size, 0)) + 1

        # replace existing memory group
        if 'memory' in f:
            try:
                del f['memory']
            except Exception:
                pass
        mem = f.create_group('memory')

        # create each agent dataset and mark a center cell as visited at `timestamp`
        center_r = avoid_h // 2
        center_c = avoid_w // 2
        for i in range(int(nagents)):
            dname = str(i)
            ds = mem.create_dataset(dname, (avoid_h, avoid_w), dtype='f4')
            ds[:, :] = 0.0
            # set a small neighborhood so geo->pixel rounding issues still pick it up
            r0 = max(0, center_r - 1)
            r1 = min(avoid_h, center_r + 2)
            c0 = max(0, center_c - 1)
            c1 = min(avoid_w, center_c + 2)
            ds[r0:r1, c0:c1] = float(timestamp)

        try:
            f.flush()
        except Exception:
            pass
    print(f'Seeded memory for {nagents} agents in {preseed_path} shape=({avoid_h},{avoid_w}) timestamp={timestamp}')
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python seed_memory_all_agents.py <preseed.h5> [--nagents N] [--timestamp T]')
        sys.exit(2)
    path = sys.argv[1]
    nagents = 1000
    timestamp = 50
    for arg in sys.argv[2:]:
        if arg.startswith('--nagents'):
            nagents = int(arg.split('=')[1]) if '=' in arg else int(sys.argv[sys.argv.index(arg)+1])
        if arg.startswith('--timestamp'):
            timestamp = int(arg.split('=')[1]) if '=' in arg else int(sys.argv[sys.argv.index(arg)+1])
    sys.exit(seed_memory(path, nagents=nagents, timestamp=timestamp))
