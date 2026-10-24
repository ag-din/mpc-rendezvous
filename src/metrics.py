"""
metrics.py

Performance metrics of a closed-loop rendezvous run.

"""

import numpy as np

from .config import ScenarioConfig
from .controllers import cone_matrix
from .simulation import SimulationResult

FIRING_THRESHOLD = 1e-4   # Impulses below this [m/s] count as "no firing"


def total_delta_v(res: SimulationResult) -> float:
    """Total executed dv, sum of per-axis magnitudes (1-norm) [m/s]."""
    return float(np.abs(res.u_applied).sum())


def n_firings(res: SimulationResult) -> int:
    """Number of sampling instants with a non-negligible impulse."""
    return int((np.abs(res.u_applied).max(axis=1) > FIRING_THRESHOLD).sum())


def settling_time(res: SimulationResult, scenario: ScenarioConfig) -> float:
    """First time after which the position stays within the tolerance.

    Returns NaN if the chaser is not settled at the end of the run.
    """
    err = np.abs(res.x[:, :3] - scenario.x_setpoint[:3]).max(axis=1)
    inside = err <= scenario.settle_tol
    if not inside[-1]:
        return float("nan")
    outside = np.flatnonzero(~inside)
    first = 0 if outside.size == 0 else outside[-1] + 1
    return float(res.t[first])


def final_position_error(res: SimulationResult,
                         scenario: ScenarioConfig) -> float:
    """Euclidean position error at the end of the run [m]."""
    return float(np.linalg.norm(res.x[-1, :3] - scenario.x_setpoint[:3]))


def max_cone_violation(states: np.ndarray,
                       scenario: ScenarioConfig) -> float:
    """Largest distance outside the visibility cone [m]."""
    gamma = np.radians(scenario.cone_half_angle_deg)
    g = cone_matrix(scenario.cone_half_angle_deg)
    viol = (states @ g.T).max() * np.cos(gamma)
    return float(max(viol, 0.0))


def evaluate(res: SimulationResult, scenario: ScenarioConfig) -> dict:
    """Collect all metrics of one run in a dictionary."""
    return {
        "controller": res.name,
        "delta_v": total_delta_v(res),
        "firings": n_firings(res),
        "settling_time": settling_time(res, scenario),
        "final_error": final_position_error(res, scenario),
        "cone_viol_sampled": max_cone_violation(res.x, scenario),
        "cone_viol_intersample": max_cone_violation(res.x_dense, scenario),
        "solve_time_mean": float(res.solve_time.mean()),
        "solve_time_max": float(res.solve_time.max()),
        "all_feasible": bool(res.feasible.all()),
    }
