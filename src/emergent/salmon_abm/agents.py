"""Agent property generation helpers extracted from sockeye.py.

These functions provide deterministic but configurable ways to generate
agent sex, length, weight, and body depth arrays for a given basin.
"""
import numpy as np


def generate_sex(num_agents, basin=None, seed=None):
    rng = np.random.default_rng(seed)
    # default 50/50 split unless basin-specific rules are added
    return rng.choice([0, 1], size=num_agents)


def generate_length(num_agents, sex=None, basin=None, fish_length=None, seed=None):
    rng = np.random.default_rng(seed)
    if fish_length is not None:
        arr = np.full(num_agents, fish_length, dtype=float)
    else:
        # simple lognormal-like distribution centered near 500 mm
        arr = rng.normal(loc=500.0, scale=50.0, size=num_agents)
    arr = np.where(arr < 475., 475., arr)
    return arr


def generate_weight(length_mm):
    # using W = a * L^b relationship; original used (0.0155*(L/10)^3)/1000
    return (0.0155 * (length_mm/10.0)**3)/1000.0


def generate_body_depth(num_agents, basin=None, seed=None):
    rng = np.random.default_rng(seed)
    # crude body depth in cm around 8-10 cm
    arr = rng.normal(loc=9.0, scale=1.0, size=num_agents)
    return arr


__all__ = [
    'generate_sex',
    'generate_length',
    'generate_weight',
    'generate_body_depth',
]
