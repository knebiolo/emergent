"""Inspect rheotaxis: compare agent headings and velocities to local flow.

Usage:
    python tools/inspect_rheotaxis.py

Prints a short summary to stdout.
"""
import os, glob, math
import numpy as np
import h5py

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'outputs')
PAT = os.path.join(OUT, 'sim_db_*.h5')


def find_latest_db():
    files = glob.glob(PAT)
    if not files:
        raise FileNotFoundError('No sim_db_*.h5 found in outputs/')
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def safe_read(ds, default=None):
    try:
        return ds[()]
    except Exception:
        return default


def main():
    db = find_latest_db()
    print('Using DB:', db)
    with h5py.File(db, 'r') as f:
        # List top-level groups/datasets
        print('\nTop-level keys:')
        for k in sorted(f.keys()):
            print(' ', k)

        # Helper to find dataset by possible names
        def find(keys):
            for key in keys:
                if key in f:
                    return key
                # check nested agent_data
                if 'agent_data/' + key in f:
                    return 'agent_data/' + key
            return None

        # Try to find agent positions
        x_key = find(['X', 'x', 'positions/X', 'positions/x'])
        y_key = find(['Y', 'y', 'positions/Y', 'positions/y'])
        if x_key is None or y_key is None:
            # see if agent_data group exists with children
            if 'agent_data' in f:
                print('\nagent_data group keys:', list(f['agent_data'].keys()))
            else:
                print('\nNo agent position datasets found in DB')
        else:
            X = safe_read(f[x_key])
            Y = safe_read(f[y_key])
            print('\nFound positions:', x_key, y_key, 'shapes', X.shape, Y.shape)

        # Find heading/orientation dataset
        heading_key = None
        for k in f:
            lk = k.lower()
            if 'heading' in lk or 'orient' in lk or 'theta' in lk or 'angle' in lk:
                heading_key = k
                break
            if k == 'agent_data':
                for kk in f['agent_data'].keys():
                    lk = kk.lower()
                    if 'heading' in lk or 'orient' in lk or 'theta' in lk or 'angle' in lk:
                        heading_key = 'agent_data/' + kk
                        break
                if heading_key:
                    break
        if heading_key:
            H = safe_read(f[heading_key])
            print('Heading dataset:', heading_key, 'shape', getattr(H,'shape',None))
        else:
            H = None
            print('No heading dataset found')

        # Swim behavior
        swim_key = find(['swim_behav', 'swimbehav', 'behav'])
        SB = safe_read(f[swim_key]) if swim_key else None
        print('Swim behavior dataset:', swim_key, 'present' if SB is not None else 'missing')

        # Agent velocities (instantaneous)
        vx_key = find(['x_vel', 'vx', 'v_x'])
        vy_key = find(['y_vel', 'vy', 'v_y'])
        VX = safe_read(f[vx_key]) if vx_key else None
        VY = safe_read(f[vy_key]) if vy_key else None
        print('Agent velocity datasets:', vx_key, vy_key)

        # Environment flow fields
        velx = safe_read(f.get('environment/vel_x'))
        vely = safe_read(f.get('environment/vel_y'))
        x_coords = safe_read(f.get('environment/x_coords'))
        y_coords = safe_read(f.get('environment/y_coords'))
        print('Environment vel present:', velx is not None and vely is not None)

        # Compute sample comparisons for up to 10 agents
        if 'X' in locals() and velx is not None and vely is not None and x_coords is not None and y_coords is not None:
            if X.ndim > 1:
                last_idx = X.shape[1] - 1
                last_X = X[:, last_idx]
                last_Y = Y[:, last_idx]
            else:
                last_X = X
                last_Y = Y
            n = min(20, last_X.shape[0])
            print('\nSample agent -> local flow and heading (first', n, 'agents):')
            align_counts = {'upstream':0,'downstream':0,'cross':0,'unknown_heading':0}
            ang_diffs = []
            for i in range(n):
                xg = float(last_X[i]); yg = float(last_Y[i])
                flatx = x_coords.flatten(); flaty = y_coords.flatten()
                d2 = (flatx - xg)**2 + (flaty - yg)**2
                idx = int(np.argmin(d2))
                r = idx // velx.shape[1]; c = idx % velx.shape[1]
                fx = float(velx[r,c]); fy = float(vely[r,c])
                flow_ang = math.atan2(fy, fx)  # radians
                # agent heading
                hval = None
                if H is not None:
                    try:
                        if H.ndim>1:
                            hval = float(H[i, -1])
                        else:
                            hval = float(H[i])
                    except Exception:
                        hval = None
                # agent velocity direction
                vax = None; vay = None; agent_ang = None
                if VX is not None and VY is not None:
                    try:
                        if VX.ndim>1:
                            vax = float(VX[i, -1]); vay = float(VY[i, -1])
                        else:
                            vax = float(VX[i]); vay = float(VY[i])
                        agent_ang = math.atan2(vay, vax)
                    except Exception:
                        vax = vay = agent_ang = None
                # compute angular difference agent heading vs flow (smallest signed)
                angdiff = None
                if hval is not None:
                    # ensure heading in radians; assume stored in radians or degrees? try heuristic
                    if abs(hval) > 2*math.pi:
                        # probably degrees
                        h = math.radians(hval)
                    else:
                        h = hval
                    # compute smallest angular difference between heading and flow
                    d = (h - flow_ang + math.pi) % (2*math.pi) - math.pi
                    angdiff = d
                    ang_diffs.append(d)
                    # classify: upstream means heading approximately opposite to flow (d ~ pi or -pi)
                    if abs(abs(d) - math.pi) < math.radians(45):
                        align_counts['upstream'] += 1
                    elif abs(d) < math.radians(45):
                        align_counts['downstream'] += 1
                    else:
                        align_counts['cross'] += 1
                else:
                    align_counts['unknown_heading'] += 1

                print(f' agent{i}: pos=({xg:.1f},{yg:.1f}) vx_flow={fx:.3f}, vy_flow={fy:.3f}, flow_ang={flow_ang:.2f}', end='')
                if hval is not None:
                    print(f', heading={hval:.3f}, angdiff={angdiff:.2f}', end='')
                if agent_ang is not None:
                    print(f', agent_move_ang={agent_ang:.2f}', end='')
                print('')

            print('\nAlignment counts (first', n, '):', align_counts)
            if ang_diffs:
                angs = np.array(ang_diffs)
                print(' mean angular diff (rad)=', float(np.nanmean(angs)), 'median=', float(np.nanmedian(angs)))
        else:
            print('\nInsufficient datasets to compare flow vs heading/velocity. Available keys maybe:')
            print('  has_X=', 'X' in locals())
            print('  env vel present=', velx is not None and vely is not None)
            print('  x_coords present=', x_coords is not None, 'y_coords present=', y_coords is not None)


if __name__ == '__main__':
    main()
