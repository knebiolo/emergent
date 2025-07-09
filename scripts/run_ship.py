"""
Launch a normal ENC-backed harbour simulation.

usage:
    python scripts/run_ship.py        # defaults: Baltimore, 1 agent
    python scripts/run_ship.py --port Oakland --agents 3
"""

import argparse, sys
from PyQt5 import QtWidgets

# local imports
from emergent.ship_abm.ship_viewer import ship_viewer
from emergent.ship_abm.config      import xml_url      # ENC catalog URL

# ── Spin up Qt & viewer ────────────────────────────────────────────────
app = QtWidgets.QApplication(sys.argv)

view = ship_viewer(
    port_name          = "Galveston",
    xml_url            = xml_url,          # from config.py
    dt                 = 0.1,
    T                  = 9000,
    n_agents           = 2,
    coast_simplify_tol = 2.0,
    light_bg           = True,
    verbose            = True,
    use_ais            = False              # viewer knows what to do
)
view.show()

sys.exit(app.exec_())
  