"""Agent state allocation and helpers for the refactored Simulation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class SimulationState:
    n_agents: int
    X: np.ndarray
    Y: np.ndarray
    prev_X: np.ndarray
    prev_Y: np.ndarray
    heading: np.ndarray
    sog: np.ndarray
    battery: np.ndarray
    dead: np.ndarray

    @classmethod
    def allocate(cls, n_agents: int, init_X: Optional[float] = 0.0, init_Y: Optional[float] = 0.0) -> 'SimulationState':
        X = np.full((n_agents,), float(init_X), dtype=np.float64)
        Y = np.full((n_agents,), float(init_Y), dtype=np.float64)
        prev_X = X.copy()
        prev_Y = Y.copy()
        heading = np.zeros((n_agents,), dtype=np.float64)
        sog = np.zeros((n_agents,), dtype=np.float64)
        battery = np.ones((n_agents,), dtype=np.float64)
        dead = np.zeros((n_agents,), dtype=np.int8)
        return cls(n_agents=n_agents, X=X, Y=Y, prev_X=prev_X, prev_Y=prev_Y, heading=heading, sog=sog, battery=battery, dead=dead)

    def as_dict(self):
        return {
            'X': self.X,
            'Y': self.Y,
            'prev_X': self.prev_X,
            'prev_Y': self.prev_Y,
            'heading': self.heading,
            'sog': self.sog,
            'battery': self.battery,
            'dead': self.dead,
        }
"""Agent state initialization helpers extracted from sockeye.simulation.

Provide deterministic, testable implementations for: sim_sex, sim_length,
sim_weight, sim_body_depth. These are written to match the legacy
`sockeye.simulation` semantics and use a pluggable random module via `rng`.
"""
from typing import Optional
import numpy as np


def sim_sex(num_agents: int, basin: Optional[str], rng: np.random.Generator):
    """Return an array of sex codes (0=male, 1=female) of length `num_agents`.

    Matches `sockeye.simulation.sim_sex` logic: special-case "Nushagak River"
    probability, otherwise 50/50.
    """
    if basin == "Nushagak River":
        p = [0.503, 0.497]
    else:
        p = [0.5, 0.5]
    return rng.choice([0, 1], size=num_agents, p=p)


def sim_length(num_agents: int, sex: np.ndarray, basin: Optional[str], pid_tuning: bool, fish_length: Optional[float], rng: np.random.Generator):
    """Return an array of lengths in mm and derived speed arrays.

    Returns a dict with keys: length, sog, ideal_sog, opt_sog, school_sog, ucrit
    mirroring sockeye semantics.
    """
    if pid_tuning and fish_length is not None:
        length = np.repeat(float(fish_length), num_agents)
    else:
        if basin == "Nushagak River":
            # lognormal with sex-specific params
            male = rng.lognormal(mean=6.426, sigma=0.072, size=num_agents)
            female = rng.lognormal(mean=6.349, sigma=0.067, size=num_agents)
            length = np.where(sex == 0, male, female)
        else:
            length = rng.lognormal(mean=6.39, sigma=0.07, size=num_agents)

    # Ensure sensible defaults and minimum length
    if length is None or length.size == 0:
        length = np.repeat(475.0, num_agents)
    length = np.where(length < 475.0, 475.0, length)

    sog = length / 1000.0
    ideal_sog = sog.copy()
    opt_sog = sog.copy()
    school_sog = sog.copy()
    ucrit = sog * 1.6

    return {
        'length': length,
        'sog': sog,
        'ideal_sog': ideal_sog,
        'opt_sog': opt_sog,
        'school_sog': school_sog,
        'ucrit': ucrit,
    }


def sim_weight(length_mm: np.ndarray):
    """Estimate weight (grams) from length (mm).

    This is a simple placeholder matching previous behaviour in sockeye
    (if any). Here we use cubic scaling as a reasonable proxy.
    """
    # convert mm to cm for common length-weight approximations
    cm = length_mm / 10.0
    # simple weight proxy: a * L^3 with a small coefficient
    a = 0.01
    return a * (cm ** 3)


def sim_body_depth(length_mm: np.ndarray):
    """Estimate body depth (cm) from length (mm)."""
    # body depth roughly ~ 0.1 * length (in mm) converted to cm
    return (length_mm * 0.1) / 10.0
