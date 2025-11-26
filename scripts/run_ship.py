"""
Launch a normal ENC-backed harbour simulation.

usage:
    python scripts/run_ship.py        # defaults: Baltimore, 1 agent
    python scripts/run_ship.py --port Oakland --agents 3
"""

import argparse, sys
from PyQt5 import QtWidgets, QtCore

# local imports
from emergent.ship_abm.ship_viewer import ship_viewer
from emergent.ship_abm.config      import xml_url      # ENC catalog URL


def main(argv=None):
    parser = argparse.ArgumentParser(description='Launch ENC-backed harbour simulation viewer')
    parser.add_argument('--port', default='Galveston', help='Port name (matches config entries)')
    parser.add_argument('--agents', type=int, default=1, help='Number of agents to spawn')
    parser.add_argument('--no-enc', dest='no_enc', action='store_true', help='Skip downloading/loading ENC data (fast startup)')
    parser.add_argument('--auto-start', dest='auto_start', action='store_true', help='Automatically start the simulation after the UI appears')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args(argv or sys.argv[1:])

    # ── Spin up Qt & viewer ────────────────────────────────────────────────
    app = QtWidgets.QApplication(sys.argv)

    sim_instance = None
    # Preload ENC synchronously before starting the Qt app so the UI appears after ENCs loaded
    # Default behavior: preload ENCs unless the user explicitly requests --no-enc
    if not args.no_enc:
        from emergent.ship_abm.simulation_core import simulation
        print("[Launcher] Preloading ENC synchronously before UI (default)...")
        sim_instance = simulation(
            port_name=args.port,
            dt=0.1,
            T=9000,
            n_agents=args.agents,
            coast_simplify_tol=2.0,
            light_bg=True,
            verbose=args.verbose,
            use_ais=False,
            load_enc=True
        )

    view = ship_viewer(
        port_name          = args.port,
        xml_url            = xml_url,
        dt                 = 0.1,
        T                  = 9000,
        n_agents           = args.agents,
        coast_simplify_tol = 2.0,
        light_bg           = True,
        verbose            = args.verbose,
        use_ais            = False,
        load_enc           = not args.no_enc,
        sim_instance       = sim_instance
    )
    view.show()

    # optionally auto-start the simulation after the event loop starts
    if getattr(args, 'auto_start', False):
        # start after a short delay so the UI finishes initial layout
        QtCore.QTimer.singleShot(200, lambda: view._start_simulation())

    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
  