"""Launch the Salmon/OpenGL viewer for sockeye simulations.

Usage examples:
  python scripts/run_sockeye_viewer.py                # load latest mesh from outputs/
  python scripts/run_sockeye_viewer.py --plan path/to/plan.hdf5
  python scripts/run_sockeye_viewer.py --plan plan.hdf5 --full
  python scripts/run_sockeye_viewer.py --diag --cpu

This script prefers the existing `SalmonViewer` compatibility shim which will
select an appropriate GL/CPU widget. When `--full` is provided we attempt to
instantiate `sockeye.simulation` and let the viewer read attributes from it.
"""
from __future__ import annotations
import sys
import os
import argparse
import glob
import numpy as np


def find_latest_mesh():
    outdir = os.path.join(os.getcwd(), 'outputs')
    if not os.path.isdir(outdir):
        return None
    files = glob.glob(os.path.join(outdir, '*_mesh.npz'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def build_qsurface_format():
    try:
        from PyQt5.QtGui import QSurfaceFormat
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        # prefer core profile; keep alpha unless env overrides
        try:
            fmt.setProfile(QSurfaceFormat.CoreProfile)
        except Exception:
            pass
        QSurfaceFormat.setDefaultFormat(fmt)
        print('Requested QSurfaceFormat 3.3 core')
    except Exception:
        print('Failed to request QSurfaceFormat; continuing with default')


def main(argv=None):
    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser(description='Launch SalmonViewer for sockeye/sim data')
    p.add_argument('--plan', '-p', help='Path to HECRAS HDF5 plan file or mesh .npz')
    p.add_argument('--full', action='store_true', help='Instantiate a full sockeye.simulation using the provided --plan (if any)')
    p.add_argument('--diag', action='store_true', help='Diagnostic mode (capture/inspect GL state)')
    p.add_argument('--cpu', action='store_true', help='Prefer CPU renderer / avoid ModernGL')
    args = p.parse_args(argv)

    # Try to request a reasonable QSurfaceFormat early
    build_qsurface_format()

    # Import viewer shim (this module provides a compatibility `SalmonViewer`)
    try:
        from PyQt5 import QtWidgets
    except Exception:
        QtWidgets = None

    try:
        # prefer the public viewer API which wraps v3 internals
        from emergent.salmon_abm.salmon_viewer import SalmonViewer
        launch_name = 'salmon_viewer.SalmonViewer'
    except Exception:
        # fallback to v3 shim directly
        try:
            from emergent.salmon_abm.viewer_v3.viewer_shim import SalmonViewer
            launch_name = 'viewer_v3.viewer_shim.SalmonViewer'
        except Exception as e:
            print('Failed to import SalmonViewer:', e)
            raise

    # Create a QApplication if Qt available
    if QtWidgets is not None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    else:
        app = None

    sim = None

    # If full simulation requested, attempt to construct sockeye.simulation
    if args.full:
        try:
            from emergent.salmon_abm import sockeye
            plan = args.plan
            model_name = os.path.splitext(os.path.basename(plan))[0] if plan else 'sockeye_preview'
            sim = sockeye.simulation(
                model_dir=os.getcwd(),
                model_name=model_name,
                crs=None,
                basin='preview',
                water_temp=10.0,
                start_polygon=None,
                centerline=None,
                env_files=None,
                num_timesteps=100,
                num_agents=100,
                use_hecras=bool(plan),
                hecras_plan_path=plan,
                hecras_fields=None,
                hecras_k=8,
                hecras_write_rasters=False,
                defer_hdf=False,
            )
            print('Instantiated sockeye.simulation (preview)')
        except Exception as e:
            print('Failed to instantiate full sockeye.simulation:', e)
            raise

    # If no sim created, provide a light-weight placeholder that the viewer expects
    if sim is None:
        class DummySim:
            def __init__(self):
                self.hdf5 = None
                self.perimeter_points = None
                self.vert_exag = 1.0

        sim = DummySim()

    viewer = SalmonViewer(sim)

    # Apply CPU-only preference if requested
    if args.cpu:
        try:
            viewer._force_cpu = True
        except Exception:
            pass

    if args.diag:
        try:
            viewer._diag_mode = True
        except Exception:
            pass

    # If a plan was provided and not running full sim, try loading it via viewer helper
    if args.plan and not args.full:
        plan = args.plan
        try:
            # If it's a prebuilt mesh npz, load verts/faces/colors directly
            if plan.lower().endswith('.npz'):
                d = np.load(plan)
                verts = d['verts']
                faces = d['faces']
                colors = d['colors']
                try:
                    viewer.load_tin_payload({'verts': verts, 'faces': faces, 'colors': colors})
                except Exception:
                    try:
                        if getattr(viewer, 'gl_widget', None) is not None:
                            viewer.gl_widget.set_mesh(verts, faces, colors, vert_exag=getattr(viewer.sim, 'vert_exag', 1.0))
                        else:
                            viewer.last_mesh_payload = {'verts': verts, 'faces': faces, 'colors': colors}
                    except Exception:
                        pass
                print('Loaded mesh file', plan)
            else:
                # assume HECRAS HDF: prefer viewer API load routine
                try:
                    viewer.load_hecras_mesh(plan, timestep=0)
                    print('Requested viewer to load HECRAS plan:', plan)
                except Exception:
                    # older API name
                    try:
                        viewer.load_hecras_plan(plan)
                        print('Requested viewer to load HECRAS plan (alt API):', plan)
                    except Exception as e:
                        print('Failed to instruct viewer to load plan:', e)
        except Exception as e:
            print('Failed while loading provided --plan:', e)

    # If no explicit mesh/plan, try to load the latest mesh from outputs/
    if not args.plan:
        mesh = find_latest_mesh()
        if mesh is not None:
            try:
                d = np.load(mesh)
                verts = d['verts']
                faces = d['faces']
                colors = d['colors']
                try:
                    viewer.load_tin_payload({'verts': verts, 'faces': faces, 'colors': colors})
                except Exception:
                    try:
                        if getattr(viewer, 'gl_widget', None) is not None:
                            viewer.gl_widget.set_mesh(verts, faces, colors, vert_exag=getattr(viewer.sim, 'vert_exag', 1.0))
                        else:
                            viewer.last_mesh_payload = {'verts': verts, 'faces': faces, 'colors': colors}
                    except Exception:
                        pass
                print('Loaded latest mesh', mesh)
            except Exception:
                print('Found mesh but failed to load it:', mesh)

    # If diagnostic mode, schedule a small inspection after event loop starts
    try:
        if args.diag and getattr(viewer, 'gl_widget', None) is not None:
            from PyQt5.QtCore import QTimer

            def inspect_gl():
                try:
                    from PyQt5.QtGui import QOpenGLContext
                    ctx = QOpenGLContext.currentContext()
                    if ctx is not None:
                        fmt = ctx.format()
                        print('Current QOpenGLContext format: version', fmt.majorVersion(), fmt.minorVersion(), 'profile', fmt.profile())
                    else:
                        print('No current QOpenGLContext')
                except Exception as e:
                    print('Error inspecting QOpenGLContext:', e)

                try:
                    gw = viewer.gl_widget
                    print('gl_widget inspection:', 'exists' if gw is not None else 'None')
                    if gw is not None:
                        cls = gw.__class__
                        print('  widget class:', cls.__name__, 'module:', cls.__module__)
                except Exception as e:
                    print('Error inspecting gl_widget attributes:', e)

            QTimer.singleShot(400, inspect_gl)
    except Exception:
        pass

    # Start the viewer (blocking call for GUI)
    try:
        viewer.run()
    except Exception as e:
        print('Viewer failed to run:', e)
        # fallback: try opening the latest preview image
        try:
            p = os.path.join(os.getcwd(), 'outputs', 'latest_preview.png')
            if os.path.exists(p):
                try:
                    os.startfile(p)
                except Exception:
                    import matplotlib.pyplot as plt
                    img = plt.imread(p)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
        except Exception:
            pass


if __name__ == '__main__':
    main()
