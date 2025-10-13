"""
Quick-launch the ship viewer without loading ENCs so we can test winds/currents quickly.
"""
import sys
from PyQt5 import QtWidgets
from emergent.ship_abm.ship_viewer import ship_viewer
from emergent.ship_abm.config import xml_url

app = QtWidgets.QApplication(sys.argv)

view = ship_viewer(
    port_name = "Galveston",
    xml_url = xml_url,
    dt = 0.1,
    T = 9000,
    n_agents = 1,
    coast_simplify_tol = 2.0,
    light_bg = True,
    verbose = True,
    use_ais = False,
    # NOTE: supply load_enc=False if the viewer accepts it; ship_viewer currently defers ENC loading
)
view.show()

sys.exit(app.exec_())
