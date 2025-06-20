from PyQt5 import QtWidgets
import sys
import numpy as np
from emergent.ship_abm.simulation_core import compute_zigzag_metrics
from emergent.ship_abm.ship_viewer     import ship_viewer
from emergent.ship_abm.config          import xml_url      # same URL you use now

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    view = ship_viewer(
        port_name   = "Baltimore",     # any port name is fine – ENC ignored
        xml_url     = xml_url,
        n_agents    = 1,
        dt          = 0.1,
        T           = 600,
        use_ais     = False,
        test_mode   = "zigzag",        # ← activates ±zig-zag
        zigzag_deg  = 10,
        zigzag_hold = 30
    )
    view.show()

    
    # run the GUI / simulation
    exit_code = app.exec_()

    # pull out the recorded histories (make sure you've populated these)
    t_data         = np.array(view.sim.t_history)        # [s]
    actual_heading = np.array(view.sim.psi_history)      # [rad]
    cmd_heading    = np.array(view.sim.hd_cmd_history)   # [rad]

    metrics = compute_zigzag_metrics(t_data, actual_heading, cmd_heading, tol = 5.)
    print("Zig-zag metrics:", metrics)

    sys.exit(exit_code)
