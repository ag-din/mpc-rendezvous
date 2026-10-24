"""
config.py
--------

Scenario, controller and disturbance parameters.

The scenario reproduces the set-up of Fig. 4 in Hartley (2015): a chaser
is transferred from 15 km to 1 km behind a target on a circular orbit
around Mars, with a 20 deg visibility cone, |dv| <= 5 m/s per axis,
Ts = 200 s and a prediction horizon of 30 steps.

This file contains the following configuration classes:

- OrbitConfig: Circular target orbit.
- ScenarioConfig: Rendezvous scenario (LVLH state).
- QuadraticCost: Weights of the fixed-horizon quadratic cost, eq. (10
- L1Cost: Weights of the fixed-horizon 1-norm cost, eq. (11).
- VariableHorizonCost: Weights of the variable-horizon costs, eq. (15
- DisturbanceConfig: Uncertainty used in the realistic (Monte Carlo) scenario.

Note: All quantities are in International System of Units.
"""

# Libraries
import numpy as np
from dataclasses import dataclass, field

# Contants
MARS_MU = 4.282837e13       # Gravitational parameter of Mars [m^3/s^2]
MARS_RADIUS = 3389.5e3      # Mean radius of Mars [m]


# Class definitions
@dataclass(frozen=True)
class OrbitConfig:
    """Circular target orbit."""

    mu: float = MARS_MU
    altitude: float = 500e3

    @property
    def radius(self) -> float:
        """Orbit radius [m]."""
        return MARS_RADIUS + self.altitude

    @property
    def mean_motion(self) -> float:
        """Mean motion n [rad/s]."""
        return float(np.sqrt(self.mu / self.radius ** 3))

    @property
    def period(self) -> float:
        """Orbital period [s]."""
        return 2.0 * np.pi / self.mean_motion


@dataclass(frozen=True)
class ScenarioConfig:
    """Rendezvous scenario (LVLH state: [x, y, z, vx, vy, vz]).

    LVLH convention (Fehse): x along-track (V-bar), y opposite to the
    orbital angular momentum (-H-bar), z towards the planet (R-bar).
    """

    x0: np.ndarray = field(default_factory=lambda: np.array(
        [-15e3, 200.0, 0.0, 0.0, 0.0, 0.0]))
    x_setpoint: np.ndarray = field(default_factory=lambda: np.array(
        [-1e3, 0.0, 0.0, 0.0, 0.0, 0.0]))
    ts: float = 200.0                 # Sampling period [s]
    n_steps: int = 40                 # Simulated steps (8000 s)
    u_max: float = 5.0                # Max impulse per axis [m/s]
    cone_half_angle_deg: float = 20.0
    cone_penalty: float = 1e2         # Exact-penalty weight on cone slacks
    settle_tol: float = 10.0          # Position tolerance for settling [m]


@dataclass(frozen=True)
class QuadraticCost:
    """Weights of the fixed-horizon quadratic cost, eq. (10)."""

    horizon: int = 30       # Prediction horizon
    q_pos: float = 1e-6     # [1/m^2]
    q_vel: float = 0.1      # [s^2/m^2]
    r: float = 10.0         # [s^2/m^2]


@dataclass(frozen=True)
class L1Cost:
    """Weights of the fixed-horizon 1-norm cost, eq. (11)."""

    horizon: int = 30       # Prediction horizon
    q_pos: float = 1e-3     # [1/m]
    q_vel: float = 0.0      # [s/m]
    r: float = 5.0          # [s/m]
    p_scale: float = 1.0    # Terminal weight P = p_scale * Q


@dataclass(frozen=True)
class VariableHorizonCost:
    """Weights of the variable-horizon costs, eq. (15).

    With `terminal_tol` all zero the terminal set is T = {x_s}, as in the
    paper. A positive tolerance turns T into a box around x_s, which keeps
    the N = 1 problem feasible when the state estimate is noisy.
    """

    max_horizon: int = 30   # Maximum prediction horizon
    r_l1: float = 1.0       # Steps per m/s of dv, eq. (15a)
    r_quad: float = 0.5     # Steps per (m/s)^2 of dv, eq. (15b)
    terminal_tol: np.ndarray = field(
        default_factory=lambda: np.zeros(6))


# Terminal box used with navigation noise: +-5 m, +-5 mm/s per axis.
TERMINAL_BOX = np.array([5.0, 5.0, 5.0, 5e-3, 5e-3, 5e-3])


@dataclass(frozen=True)
class DisturbanceConfig:
    """Uncertainty used in the realistic (Monte Carlo) scenario."""

    pos_noise_std: float = 2.0      # Navigation noise, position [m]
    vel_noise_std: float = 2e-3     # Navigation noise, velocity [m/s]
    min_impulse_bit: float = 0.01   # Impulses below this are lost [m/s]
    exec_error_std: float = 0.03    # Relative thrust magnitude error [-]
    n_runs: int = 20                # Number of Monte Carlo runs
    seed: int = 2015                # Random seed for reproducibility


# Nominal (no disturbance) scenario
NOMINAL_DISTURBANCE = DisturbanceConfig(
    pos_noise_std=0.0, vel_noise_std=0.0, min_impulse_bit=0.0,
    exec_error_std=0.0, n_runs=1)
