import importlib
import numpy as np

m = importlib.import_module('emergent.ship_abm.ship_model')
# instantiate ship with 1 agent
state0 = np.zeros((4,1))
pos0 = np.zeros((2,1))
psi0 = np.zeros(1)
goals = np.zeros((2,1))

s = m.ship(state0,pos0,psi0,goals)
# enable PID debug printing in config for this run
import emergent.ship_abm.config as cfg
cfg.PID_DEBUG = True
# desired heading 30 deg
hd = np.array([np.radians(30.0)])
dt = 0.5
psi = np.array([0.0])

print('max_rudder_deg=', np.degrees(s.max_rudder), 'max_rudder_rate(rad/s)=', s.max_rudder_rate)
print('step, rudder_deg, delta_deg')
prev = 0.0
for i in range(40):
    rud = s.pid_control(psi, hd, dt)
    rud_deg = float(np.degrees(rud[0]))
    delta = rud_deg - prev
    print(f"{i:2d}, {rud_deg:7.3f}, {delta:7.3f}")
    prev = rud_deg
    # mock simple heading change proportional to rudder command
    psi = psi + 0.005 * rud
