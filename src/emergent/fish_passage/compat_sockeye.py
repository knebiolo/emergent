"""Compatibility wrappers exposing legacy `sockeye`-named functions from
the `fish_passage` canonical implementations.

These wrappers live in `fish_passage` so legacy callers can be updated later
via a single PR; we must not edit `sockeye.py` itself without explicit user
approval. The wrappers are thin re-exports to the canonical functions.
"""
from __future__ import annotations

from typing import Any

from . import metrics as _metrics
from . import fatigue as _fatigue
from . import projection as _projection

# Schooling / drafting
def compute_schooling_metrics_biological(positions, headings, body_lengths, behavioral_weights, alive_mask=None):
    return _metrics.compute_schooling_metrics_biological(positions, headings, body_lengths, behavioral_weights, alive_mask=alive_mask)


def compute_drafting_benefits(positions, headings, velocities, body_lengths, behavioral_weights, alive_mask=None):
    return _metrics.compute_drafting_benefits(positions, headings, velocities, body_lengths, behavioral_weights, alive_mask=alive_mask)


# Fatigue / battery
def calc_battery(battery, per_rec, ttf, mask_sustained, dt):
    return _fatigue.calc_battery(battery, per_rec, ttf, mask_sustained, dt)


def merged_battery(battery, per_rec, ttf, mask_sustained, dt):
    return _fatigue.merged_battery(battery, per_rec, ttf, mask_sustained, dt)


def time_to_fatigue(swim_speeds, mask_prolonged, mask_sprint, a_p, b_p, a_s, b_s):
    return _fatigue.time_to_fatigue(swim_speeds, mask_prolonged, mask_sprint, a_p, b_p, a_s, b_s)


def bout_distance(prev_X, X, prev_Y, Y):
    return _fatigue.bout_distance(prev_X, X, prev_Y, Y)


# Projection helpers
def project_points_onto_line(xs_line, ys_line, px, py):
    return _projection.project_points_onto_line(xs_line, ys_line, px, py)
