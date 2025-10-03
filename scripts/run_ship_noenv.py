"""
Launch ship simulation WITHOUT environmental forcing (no currents/winds).
This bypasses the Qt threading issue for now.

usage:
    python scripts/run_ship_noenv.py
"""

import argparse, sys
from PyQt5 import QtWidgets

# local imports
from emergent.ship_abm.ship_viewer import ship_viewer
from emergent.ship_abm.config      import xml_url      # ENC catalog URL

# ── Spin up Qt & viewer ────────────────────────────────────────────────
app = QtWidgets.QApplication(sys.argv)

# Use test_mode to bypass environmental forcing
view = ship_viewer(
    port_name          = "Galveston",
    xml_url            = xml_url,
    dt                 = 0.1,
    T                  = 9000,
    n_agents           = 1,
    coast_simplify_tol = 2.0,
    light_bg           = True,
    verbose            = True,
    use_ais            = False,
    test_mode          = "calm"  # This should skip environmental forcing
)
view.show()

sys.exit(app.exec_())
