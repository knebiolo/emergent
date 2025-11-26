"""
Quick launcher for the Ship ABM GUI that explicitly disables ENC loading
to allow fast startup for development and debugging.

Usage:
    python scripts/run_ship_quick_gui.py --port Galveston --agents 1 --auto-start

This mirrors `scripts/run_ship.py` but never preloads ENCs and always asks
the viewer to construct the simulation with `load_enc=False` (deferred).
"""
import argparse
import sys
from PyQt5 import QtWidgets, QtCore

from emergent.ship_abm.ship_viewer import ship_viewer
from emergent.ship_abm.config import xml_url


def main(argv=None):
    parser = argparse.ArgumentParser(description='Quick GUI launcher (no ENC preload)')
    parser.add_argument('--port', default='Galveston', help='Port name (matches config entries)')
    parser.add_argument('--agents', type=int, default=1, help='Number of agents to spawn')
    parser.add_argument('--auto-start', dest='auto_start', action='store_true', help='Automatically start the simulation after the UI appears')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args(argv or sys.argv[1:])

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)

    # Build the viewer but explicitly avoid preloading ENCs or constructing a heavy simulation.
    # ship_viewer will create a simulation with load_enc=False by default when sim_instance is None.
    view = ship_viewer(
        port_name=args.port,
        xml_url=xml_url,
        dt=0.1,
        T=9000,
        n_agents=args.agents,
        coast_simplify_tol=2.0,
        light_bg=True,
        verbose=args.verbose,
        use_ais=False,
        load_enc=False,
        sim_instance=None
    )
    view.show()

    if getattr(args, 'auto_start', False):
        QtCore.QTimer.singleShot(200, lambda: view._start_simulation())

    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
