"""
simulation.py
--------------

Closed-loop simulation of MPC rendezvous.

At every sampling instant the controller receives a (possibly noisy)
state estimate, returns an impulse, the impulse is executed with
actuator imperfections and the plant drifts freely for one period.
The trajectory is also stored on a fine grid between samples so that
inter-sample constraint violations can be measured.
"""

from dataclasses import dataclass
import numpy as np
from .config import DisturbanceConfig, ScenarioConfig


@dataclass
class SimulationResult:
    """Closed-loop trajectory of one run."""

    name: str
    t: np.ndarray            # Sample times, shape (K+1,)
    x: np.ndarray            # True states at samples, shape (K+1, 6)
    u_cmd: np.ndarray        # Commanded impulses, shape (K, 3)
    u_applied: np.ndarray    # Executed impulses, shape (K, 3)
    t_dense: np.ndarray      # Fine time grid
    x_dense: np.ndarray      # True states on the fine grid
    solve_time: np.ndarray   # Solver time per step [s]
    horizon: np.ndarray      # Horizon used per step
    feasible: np.ndarray     # Solver success flag per step


def measure(x: np.ndarray, dist: DisturbanceConfig, rng) -> np.ndarray:
    """Add Gaussian navigation noise to the true state."""
    std = np.r_[np.full(3, dist.pos_noise_std), np.full(3, dist.vel_noise_std)]
    return x + std * rng.standard_normal(6)


def actuate(u: np.ndarray, dist: DisturbanceConfig, rng) -> np.ndarray:
    """Apply the minimum impulse bit and a relative magnitude error."""
    u_out = np.where(np.abs(u) < dist.min_impulse_bit, 0.0, u)
    return u_out * (1.0 + dist.exec_error_std * rng.standard_normal(3))


def run_closed_loop(controller, plant, scenario: ScenarioConfig,
                    dist: DisturbanceConfig, rng=None,
                    n_dense: int = 20) -> SimulationResult:
    """Simulate one closed-loop rendezvous."""
    rng = rng if rng is not None else np.random.default_rng(0)
    k_max, ts = scenario.n_steps, scenario.ts
    plant.reset(scenario.x0)

    x_log = [plant.relative_state()]
    u_cmd, u_app, solve_t, horizons, feas = [], [], [], [], []
    dense = [x_log[0][None, :]]

    for _ in range(k_max):
        res = controller.control(measure(x_log[-1], dist, rng))
        u_exec = actuate(res.u, dist, rng)
        plant.apply_impulse(u_exec)
        seg = plant.propagate(ts, n_dense)
        dense.append(seg[1:])
        x_log.append(plant.relative_state())
        u_cmd.append(res.u)
        u_app.append(u_exec)
        solve_t.append(res.solve_time)
        horizons.append(res.horizon)
        feas.append(res.feasible)

    return SimulationResult(
        name=controller.name,
        t=np.arange(k_max + 1) * ts,
        x=np.array(x_log),
        u_cmd=np.array(u_cmd),
        u_applied=np.array(u_app),
        t_dense=np.linspace(0.0, k_max * ts, k_max * n_dense + 1),
        x_dense=np.vstack(dense),
        solve_time=np.array(solve_t),
        horizon=np.array(horizons),
        feasible=np.array(feas),
    )
