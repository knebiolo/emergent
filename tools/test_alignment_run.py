import numpy as np
from emergent.salmon_abm.behavior import behavior

class MockSim:
    def __init__(self):
        self.num_agents = 4
        self.X = np.array([0.,1.,2.,3.])
        self.Y = np.array([0.,0.,0.,0.])
        # headings in radians: 0, 0, pi/2, pi
        self.heading = np.array([0., 0., np.pi/2, np.pi])
        self.sog = np.ones(4)
        self.length = 100.0
        self.agents_within_buffers = [np.array([1,2]), np.array([0,2]), np.array([0,1,3]), np.array([2])]
        self.x_vel = np.cos(self.heading)
        self.y_vel = np.sin(self.heading)

mock = MockSim()
beh = behavior(1.0, mock)
arr = beh.alignment_cue(10.0)
print('alignment array:\n', arr)
