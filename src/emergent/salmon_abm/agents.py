"""Exact agent initialization routines ported from sockeye.py.

Each function mutates the provided `sim` object (matching the original
methods' behavior). This preserves the original logic and any quirks so
the migration remains behaviorally identical.
"""
import numpy as np


def sim_sex(sim):
    """Simulate the sex distribution of agents based on the basin.

    This mirrors the `simulation.sim_sex` method in the original
    `sockeye.py` and sets `sim.sex`.
    """
    if sim.basin == "Nushagak River":
        sim.sex = sim.arr.random.choice([0, 1], size=sim.num_agents, p=[0.503, 0.497])


def sim_length(sim, fish_length=None):
    """Simulate the length distribution of agents based on the basin and sex.

    This mirrors the `simulation.sim_length` method in `sockeye.py` and sets
    `sim.length`, `sim.sog`, `sim.ideal_sog`, `sim.opt_sog`, `sim.school_sog`, and `sim.ucrit`.
    """
    # length in mm
    if getattr(sim, 'pid_tuning', False) == True:
        sim.length = np.repeat(fish_length, sim.num_agents)  # testing

    else:
        if sim.basin == "Nushagak River":
            sim.length = np.where(sim.sex == 'M',
                                  sim.arr.random.lognormal(mean=6.426, sigma=0.072, size=sim.num_agents),
                                  sim.arr.random.lognormal(mean=6.349, sigma=0.067, size=sim.num_agents))

    # we can also set these arrays that contain parameters that are a function of length
    sim.length = np.where(sim.length < 475., 475., sim.length)
    sim.sog = sim.length / 1000.  # sog = speed over ground - assume fish maintain 1 body length per second
    sim.ideal_sog = sim.length / 1000.
    sim.opt_sog = sim.length / 1000.
    sim.school_sog = sim.length / 1000.
    sim.ucrit = sim.sog * 1.6


def sim_weight(sim):
    """Simulate fish weight based on length using FishBase relationship.

    Sets `sim.weight` (in kg).
    """
    sim.weight = (0.0155 * (sim.length / 10.0) ** 3) / 1000.


def sim_body_depth(sim):
    """Simulate fish body depth (cm) — mirrors original method.

    Sets `sim.body_depth`, `sim.too_shallow`, and `sim.opt_wat_depth`.
    """
    if sim.basin == "Nushagak River":
        sim.body_depth = np.where(sim.sex == 'M',
                                 sim.arr.exp(-1.938 + np.log(sim.length) * 1.084 + 0.0435) / 10.,
                                 sim.arr.exp(-1.938 + np.log(sim.length) * 1.084) / 10.)

    sim.too_shallow = sim.body_depth / 100. / 2.  # m
    sim.opt_wat_depth = sim.body_depth / 100 * 3.0 + sim.too_shallow


__all__ = ['sim_sex', 'sim_length', 'sim_weight', 'sim_body_depth']


# Compatibility standalone generators (previous API)
def generate_sex(num_agents, basin=None, seed=None):
    rng = np.random.default_rng(seed)
    # original used probabilities for Nushagak River
    if basin == "Nushagak River":
        return rng.choice([0, 1], size=num_agents, p=[0.503, 0.497])
    return rng.choice([0, 1], size=num_agents)


def generate_length(num_agents, sex=None, basin=None, fish_length=None, seed=None):
    rng = np.random.default_rng(seed)
    if fish_length is not None:
        arr = np.full(num_agents, fish_length, dtype=float)
    else:
        # If sex not provided, sample it with the same basin rules
        if sex is None:
            sex = generate_sex(num_agents, basin=basin, seed=seed)

        # sex may be numeric (0=male,1=female) — map to the original conditional
        male_mask = (sex == 0) | (sex == 'M')

        arr = np.empty(num_agents, dtype=float)
        # Use the same lognormal parameters as the original code
        arr[male_mask] = rng.lognormal(mean=6.426, sigma=0.072, size=male_mask.sum())
        arr[~male_mask] = rng.lognormal(mean=6.349, sigma=0.067, size=(~male_mask).sum())

    arr = np.where(arr < 475., 475., arr)
    return arr


def generate_weight(length_mm):
    return (0.0155 * (length_mm / 10.0) ** 3) / 1000.0


def generate_body_depth(num_agents_or_lengths, sex=None, basin=None, seed=None):
    """If passed a scalar int, returns array of body depths; if passed lengths array, computes depths from lengths."""
    rng = np.random.default_rng(seed)
    if isinstance(num_agents_or_lengths, int):
        num = num_agents_or_lengths
        # crude default: generate lengths then compute depth
        lengths = generate_length(num, sex=sex, basin=basin, seed=seed)
    else:
        lengths = np.array(num_agents_or_lengths)

    if sex is None:
        # assume numeric sex unknown — treat as female for depth formula
        sex = np.zeros(len(lengths), dtype=int)

    male_mask = (sex == 0) | (sex == 'M')
    body_depth = np.empty_like(lengths, dtype=float)
    body_depth[male_mask] = np.exp(-1.938 + np.log(lengths[male_mask]) * 1.084 + 0.0435) / 10.
    body_depth[~male_mask] = np.exp(-1.938 + np.log(lengths[~male_mask]) * 1.084) / 10.
    return body_depth


__all__ = ['sim_sex', 'sim_length', 'sim_weight', 'sim_body_depth',
           'generate_sex', 'generate_length', 'generate_weight', 'generate_body_depth']
