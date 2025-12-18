from PyQt5 import QtWidgets
import importlib

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
try:
    m = importlib.import_module('emergent.salmon_abm.viewer_v3.viewer_shim')
    SV = getattr(m, 'SalmonViewer')
    sv = SV(simulation=None)
    print('SalmonViewer instantiated')
    attrs = ['play_btn','pause_btn','rebuild_btn','speed_slider','agent_count_label','perim_toggle_btn','episode_label','reward_plot','per_episode_plot','gl_widget','last_mesh_payload','_pending_mesh']
    for a in attrs:
        print(a, hasattr(sv, a))
except Exception as e:
    print('ERROR', type(e).__name__, e)
finally:
    try:
        QtWidgets.QApplication.quit()
    except Exception:
        pass
