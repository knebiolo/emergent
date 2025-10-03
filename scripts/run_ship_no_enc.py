"""
Minimal ship viewer test - NO ENC charts, just basic GUI
"""
import sys
from emergent.ship_abm.ship_viewer import ship_viewer

if __name__ == "__main__":
    # Disable ENC loading by setting enc_catalog_url to None
    view = ship_viewer(
        port_name          = "Galveston",
        dt                 = 0.1,
        T                  = 9000,
        n_agents           = 1,
        light_bg           = True,
        verbose            = True,
        test_mode          = None,
        use_ais            = False,
        enc_catalog_url    = None  # SKIP ENC LOADING!
    )
    sys.exit(view.app.exec_())
