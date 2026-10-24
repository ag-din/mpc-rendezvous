"""
test.py
-------

Unit tests for the dynamics, controllers and metrics.

"""

import numpy as np
import pytest

from rendezvous_mpc.config import (NOMINAL_DISTURBANCE, OrbitConfig,
                                   QuadraticCost, ScenarioConfig)
from rendezvous_mpc.controllers import FixedHorizonMPC, cone_matrix
from rendezvous_mpc.dynamics import (HCWPlant, TwoBodyPlant,
                                     discretise_impulsive, hcw_transition,
                                     inertial_to_lvlh, lvlh_to_inertial)
from rendezvous_mpc.metrics import settling_time
from rendezvous_mpc.simulation import run_closed_loop

ORBIT = OrbitConfig()
N = ORBIT.mean_motion


def test_v_bar_points_are_hcw_equilibria():
    x = np.array([-1000.0, 0, 0, 0, 0, 0])
    assert np.allclose(hcw_transition(N, 500.0) @ x, x)


def test_impulsive_discretisation_matches_definition():
    a, b = discretise_impulsive(N, 200.0)
    x, u = np.array([10.0, 5, -3, 0.1, 0, 0.2]), np.array([0.5, -0.1, 0.3])
    expected = hcw_transition(N, 200.0) @ (x + np.r_[0, 0, 0, u])
    assert np.allclose(a @ x + b @ u, expected)


def test_frame_round_trip():
    r_t, v_t = np.array([ORBIT.radius, 0, 0]), np.array([0, 3300.0, 0])
    x = np.array([-500.0, 20, 30, 0.1, -0.2, 0.05])
    assert np.allclose(inertial_to_lvlh(r_t, v_t,
                                        *lvlh_to_inertial(r_t, v_t, x)), x)


def test_two_body_plant_matches_hcw_at_short_range():
    x0 = np.array([-100.0, 10, 20, 0.01, 0, -0.02])
    lin, nl = HCWPlant(N), TwoBodyPlant(ORBIT.mu, ORBIT.radius)
    lin.reset(x0)
    nl.reset(x0)
    err = lin.propagate(ORBIT.period, 10) - nl.propagate(ORBIT.period, 10)
    assert np.abs(err[:, :3]).max() < 0.5      # metres after one orbit


def test_cone_contains_v_bar_approach():
    g = cone_matrix(20.0)
    assert np.all(g @ np.array([-1000.0, 0, 0, 0, 0, 0]) <= 0)
    assert np.any(g @ np.array([-1000.0, 0, 500, 0, 0, 0]) > 0)


@pytest.mark.slow
def test_quadratic_mpc_converges_nominal():
    scenario = ScenarioConfig()
    a, b = discretise_impulsive(N, scenario.ts)
    ctrl = FixedHorizonMPC("QP", a, b, scenario, QuadraticCost())
    res = run_closed_loop(ctrl, HCWPlant(N), scenario, NOMINAL_DISTURBANCE)
    assert res.feasible.all()
    assert np.abs(res.u_cmd).max() <= scenario.u_max + 1e-6
    assert np.isfinite(settling_time(res, scenario))
