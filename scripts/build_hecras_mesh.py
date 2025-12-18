"""Build a TIN mesh from a HEC-RAS HDF5 plan and save outputs.

Usage:
    python scripts/build_hecras_mesh.py <plan.hdf>

Saves:
    outputs/<plan_basename>_mesh.npz  (verts, faces, colors)
    outputs/<plan_basename>_preview.png  (offscreen preview if possible)
"""
import sys, os
from pathlib import Path

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            print('Usage: python scripts/build_hecras_mesh.py <plan.hdf>')
            sys.exit(2)
        plan = sys.argv[1]
        planp = Path(plan)
        if not planp.exists():
            print('File not found:', plan)
            sys.exit(1)

        # import local package
        sys.path.insert(0, '.')
        try:
            from emergent.salmon_abm.viewer_v3.hecras_adapter import extract_depth_points, build_mesh_from_hecras
        except Exception as e:
            print('Failed to import viewer_v3.hecras_adapter:', e)
            raise

        import h5py
        # open and determine time indexing options
        TIME_SOURCE = os.environ.get('TIME_SOURCE', os.environ.get('TIME', 'area'))  # 'area' or 'global'
        TIME_INDEX = os.environ.get('TIME_INDEX', None)
        DEPTH_THRESH = float(os.environ.get('DEPTH_THRESH', '0.05'))
        MAX_NODES = int(os.environ.get('MAX_NODES', '5000'))
        VERT_EXAG = float(os.environ.get('VERT_EXAG', '1.0'))
        USE_WETTED = bool(int(os.environ.get('USE_WETTED', '0')))
        ALPHA = os.environ.get('ALPHA', None)
        if ALPHA is not None:
            try:
                ALPHA = float(ALPHA)
            except Exception:
                ALPHA = None

        with h5py.File(plan, 'r') as hdf:
            area_path = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area'
            global_path = 'Results/Unsteady/Output/Output Blocks/Computation Block/Global'

            # pick dataset for cell hydraulic depth depending on time source
            # prefer area output depths by default
            area_depth_path = area_path + '/Cell Hydraulic Depth'
            area_time_path = area_path + '/Time'
            global_time_path = global_path + '/Time'

            if TIME_SOURCE.lower().startswith('global'):
                # try to find a global high-resolution Cell Hydraulic Depth; if missing, we'll map a global time index
                ds_candidate = global_path + '/Cell Hydraulic Depth'
                if ds_candidate in hdf:
                    ds_path = ds_candidate
                else:
                    print('Global Cell Hydraulic Depth not found; will map global time index to area output time')
                    ds_path = area_depth_path
            else:
                ds_path = area_depth_path

            if ds_path not in hdf:
                print('HECRAS depth dataset not found at expected path:', ds_path)
                sys.exit(1)

            ds = hdf[ds_path]
            nt = ds.shape[0] if getattr(ds, 'ndim', 0) > 0 else 1

            # determine numeric tindex to pass to mesh builder
            if TIME_SOURCE.lower().startswith('global') and (global_time_path in hdf) and (area_time_path in hdf):
                # map the requested global time index -> actual time value -> nearest area time index
                gtime = hdf[global_time_path][:]
                atime = hdf[area_time_path][:]
                if TIME_INDEX is None:
                    gidx = len(gtime)//2
                else:
                    gidx = int(TIME_INDEX)
                    gidx = max(0, min(len(gtime)-1, gidx))
                gval = float(gtime[gidx])
                # find nearest index in area time array
                import numpy as _np
                tindex = int(_np.argmin(_np.abs(atime - gval)))
                print(f'Mapped global index {gidx} (time={gval}) to area index {tindex} (area_time={float(atime[tindex])})')
            else:
                # fallback: use TIME_INDEX relative to the chosen depth dataset
                if TIME_INDEX is not None:
                    try:
                        tindex = int(TIME_INDEX)
                    except Exception:
                        tindex = max(0, min(nt-1, int(float(TIME_INDEX))))
                else:
                    tindex = max(0, int(nt // 2))

            print(f'Using time source="{TIME_SOURCE}", dataset="{ds_path}", nt={nt}, index={tindex}, depth_thresh={DEPTH_THRESH}, max_nodes={MAX_NODES}, vert_exag={VERT_EXAG}')

        # build mesh (pass numeric timestep index)
        verts, faces, colors = build_mesh_from_hecras(plan, timestep=tindex, depth_thresh=DEPTH_THRESH, max_nodes=MAX_NODES, vert_exag=VERT_EXAG, use_wetted_perimeter=USE_WETTED, alpha=ALPHA)
        print('Mesh shapes:', getattr(verts,'shape',None), getattr(faces,'shape',None), getattr(colors,'shape',None))

        outdir = Path('outputs')
        outdir.mkdir(parents=True, exist_ok=True)
        base = planp.stem
        npz_path = outdir / f'{base}_mesh.npz'
        # write compressed mesh file
        import numpy as np
        np.savez_compressed(str(npz_path), verts=verts, faces=faces, colors=colors)
        print('Saved mesh to', npz_path)

        # attempt quick preview: if Qt available, use OffscreenQtFBORenderer, else fallback to 2D scatter via matplotlib
        try:
            from emergent.salmon_abm.salmon_viewer import OffscreenQtFBORenderer
            rend = OffscreenQtFBORenderer(800,600)
            payload = {'verts': verts, 'faces': faces, 'colors': colors}
            img = rend.render(payload, size=(800,600))
            try:
                p = outdir / f'{base}_preview.png'
                img.save(str(p))
                print('Saved preview to', p)
            except Exception as e:
                print('Failed to save QImage preview:', e)
        except Exception:
            print('Qt Offscreen renderer unavailable; falling back to 2D scatter preview')
            try:
                import matplotlib.pyplot as plt
                if verts is not None and verts.shape[0] > 0 and faces is not None and faces.shape[0] > 0:
                    x = verts[:,0]
                    y = verts[:,1]
                    z = verts[:,2]
                    plt.figure(figsize=(10,8))
                    try:
                        # use tricontourf for a smooth viridis background from the mesh z values
                        plt.tricontourf(x, y, faces, z, levels=256, cmap='viridis')
                    except Exception:
                        # fallback to scatter colored by z
                        plt.scatter(x, y, c=z, cmap='viridis', s=1)
                    # overlay mesh edges subtly to show triangulation
                    try:
                        plt.triplot(x, y, faces, linewidth=0.2, color='k', alpha=0.2)
                    except Exception:
                        pass
                    plt.axis('equal')
                    plt.axis('off')
                    p = outdir / f'{base}_preview_matplotlib.png'
                    plt.savefig(p, dpi=200, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    print('Saved matplotlib preview to', p)
                else:
                    print('Not enough verts/faces for matplotlib preview')
            except Exception as e:
                print('Matplotlib preview failed:', e)

        from emergent.salmon_abm.viewer_v3.viewer_shim import launch_viewer, SalmonViewer
        # create or load simulation object `sim` then:
        viewer = SalmonViewer(sim)
        viewer.load_last_mesh()   # loads outputs/*_mesh.npz into GL widget
        viewer.run()

        print('Done')
    except Exception as exc:
        import traceback, sys as _sys
        print('ERROR: build_hecras_mesh failed:', exc, file=_sys.stderr)
        traceback.print_exc()
        raise
