#!/usr/bin/env python3
"""Unit test harness to exercise cohesion, alignment, and rheo cues directly.

This builds a lightweight Simulation stub with minimal attributes consumed by
`emergent.salmon_abm.behavior.behavior` and prints the resulting vectors.
"""
import numpy as np
import os

from emergent.salmon_abm.behavior import behavior


class DummySim:
    def __init__(self, n):
        self.num_agents = n
        # simple positions in a line
        self.X = np.linspace(0.0, 9.0, n)
        self.Y = np.zeros(n)
        # simple headings (radians)
        self.heading = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
        # velocities
        self.x_vel = np.zeros(n)
        self.y_vel = np.zeros(n)
        # neighbor buffers: each agent sees the next agent (cyclic)
        awb = []
        for i in range(n):
            awb.append(np.array([(i+1) % n], dtype=int))
        self.agents_within_buffers = awb
        # simulation transforms (not used by cohesion/alignment in this test)
        self.depth_rast_transform = None
        self.vel_dir_rast_transform = None
        self.vel_x_rast_transform = None
        self.vel_y_rast_transform = None
        self.vel_mag_rast_transform = None
        # runtime attrs used by behavior
        self.sog = np.ones(n)*0.1
        self.length = 100
        self.in_eddy = np.zeros(n, dtype=int)
        self.swim_behav = np.ones(n, dtype=int)
        self.max_cue_magnitude = 5000.0
        self.current_step = 0
        self.model_dir = os.path.join('outputs','diagnostics')


def run_tests():
    sims = DummySim(5)
    b = behavior(dt=1.0, simulation_object=sims)
    print('Positions X:', sims.X)
    print('Neighbors buffers:', sims.agents_within_buffers)

    # cohesion cue (weight 1.0)
    coh = b.cohesion_cue(1.0)
    print('\nCohesion cue (1.0) result shape:', coh.shape)
    print(coh)

    # alignment cue: set neighbor headings to known values and test
    sims.heading = np.array([0.0, np.pi/2, np.pi, -np.pi/2, 0.0])
    align = b.alignment_cue(1.0)
    print('\nAlignment cue (1.0) result shape:', align.shape)
    print(align)

    # rheo cue: stub out sample_environment on simulation to return a uniform flow
    def sample_env(transform, key):
        if key == 'vel_x':
            return np.ones(sims.num_agents) * 0.5
        if key == 'vel_y':
            return np.zeros(sims.num_agents)
        return np.zeros(sims.num_agents)
    sims.sample_environment = sample_env
    rheo = b.rheo_cue(1.0)
    print('\nRheo cue (1.0) result shape:', rheo.shape)
    print(rheo)


if __name__ == '__main__':
    run_tests()
