"""Real-time simulation runner for viewer_v3.

Runs the simulation stepping loop in a `QThread` and emits state updates
(such as agent positions) at a target framerate.
"""
from PyQt5 import QtCore
import numpy as np


class RealTimeSolver(QtCore.QThread):
    """Run `simulation.timestep()` in a background thread and emit updates.

    Signals:
        frame_ready(dict) — emitted each frame with keys like 'positions' and 'metrics'.
    """
    frame_ready = QtCore.pyqtSignal(object)

    def __init__(self, simulation, dt=0.1, target_fps=30, parent=None):
        super().__init__(parent=parent)
        self.sim = simulation
        self.dt = float(dt)
        self.target_fps = int(target_fps)
        self._running = False
        self._pause = False

    def run(self):
        import time
        self._running = True
        frame_time = 1.0 / max(1, self.target_fps)
        last = time.time()
        while self._running:
            if self._pause:
                time.sleep(0.01)
                last = time.time()
                continue
            now = time.time()
            elapsed = now - last
            if elapsed < frame_time:
                time.sleep(max(0.0, frame_time - elapsed))
                continue
            last = time.time()
            # step the simulation once (simulation must provide `timestep` method)
            try:
                # prefer simulation.timestep signature (t, dt, gravity, pid) — if not, attempt simple call
                try:
                    pid = getattr(self.sim, 'pid_controller', None)
                    tindex = getattr(self.sim, 'current_timestep', 0)
                    self.sim.timestep(tindex, self.dt, 9.81, pid)
                    try:
                        # increment sim counters if present
                        self.sim.current_timestep = tindex + 1
                    except Exception:
                        pass
                except Exception:
                    # fallback: call timestep with dt only
                    if hasattr(self.sim, 'timestep'):
                        self.sim.timestep(self.dt)
                # prepare frame payload
                positions = None
                try:
                    x = getattr(self.sim, 'X', None)
                    y = getattr(self.sim, 'Y', None)
                    if x is not None and y is not None:
                        # stack X,Y,0
                        pos = np.column_stack([np.asarray(x).astype('f4'), np.asarray(y).astype('f4'), np.zeros(len(x), dtype='f4')])
                        positions = pos
                except Exception:
                    positions = None
                payload = {'positions': positions}
                # emit frame
                self.frame_ready.emit(payload)
            except Exception:
                # if simulation throws, pause and emit no frames
                self._pause = True
                continue

    def stop(self):
        self._running = False
        self.wait(timeout=2000)

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False
*** End Patch