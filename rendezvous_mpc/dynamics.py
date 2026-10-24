"""
dynamics.py
-----------

Relative orbital dynamics: HCW prediction model and nonlinear plant.

The prediction model is the Hill-Clohessy-Wiltshire (HCW) equations with
an impulsive discretisation. An impulse is applied at the start of each
sampling interval and the chaser then drifts freely,

    x(k+1) = Phi(Ts) (x(k) + [0; I] u(k))  =>  A = Phi(Ts),  B = Phi(Ts)[0; I].

The nonlinear plant integrates the exact two-body relative motion in an
inertial frame and reports the state in the target LVLH frame.
"""

# Liraries
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# Constants
STATE_DIM = 6
INPUT_DIM = 3


def hcw_matrix(n: float) -> np.ndarray:
    """Continuous-time HCW state matrix in the Fehse LVLH convention."""
    a_c = np.zeros((STATE_DIM, STATE_DIM))
    a_c[0:3, 3:6] = np.eye(3)
    a_c[3, 5] = 2.0 * n                     # x'' = 2n z'
    a_c[4, 1] = -n ** 2                     # y'' = -n^2 y
    a_c[5, 2] = 3.0 * n ** 2                # z'' = 3n^2 z - 2n x'
    a_c[5, 3] = -2.0 * n
    return a_c


def hcw_transition(n: float, dt: float) -> np.ndarray:
    """State transition matrix Phi(dt) of the HCW equations."""
    return expm(hcw_matrix(n) * dt)


def discretise_impulsive(n: float, ts: float):
    """Return (A, B) of the impulsive HCW model for sampling period ts."""
    phi = hcw_transition(n, ts)
    b_vel = np.vstack([np.zeros((3, 3)), np.eye(3)])
    return phi, phi @ b_vel


# ---------------------------------------------------------------------------
# Frame transformations
# ---------------------------------------------------------------------------

def lvlh_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotation matrix whose columns are the LVLH axes in inertial frame."""
    h = np.cross(r, v)
    e_z = -r / np.linalg.norm(r)
    e_y = -h / np.linalg.norm(h)
    e_x = np.cross(e_y, e_z)
    return np.column_stack([e_x, e_y, e_z])


def lvlh_rate(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angular velocity of the LVLH frame, expressed in LVLH axes."""
    h = np.cross(r, v)
    return np.array([0.0, -np.linalg.norm(h) / np.dot(r, r), 0.0])


def inertial_to_lvlh(r_t, v_t, rho_i, rho_dot_i) -> np.ndarray:
    """Convert an inertial relative state to the LVLH relative state."""
    rot = lvlh_basis(r_t, v_t)
    omega = lvlh_rate(r_t, v_t)
    pos = rot.T @ rho_i
    vel = rot.T @ rho_dot_i - np.cross(omega, pos)
    return np.concatenate([pos, vel])


def lvlh_to_inertial(r_t, v_t, x_lvlh):
    """Convert an LVLH relative state to inertial (rho, rho_dot)."""
    rot = lvlh_basis(r_t, v_t)
    omega = lvlh_rate(r_t, v_t)
    pos, vel = x_lvlh[:3], x_lvlh[3:]
    return rot @ pos, rot @ (vel + np.cross(omega, pos))


# ---------------------------------------------------------------------------
# Plants
# ---------------------------------------------------------------------------

class HCWPlant:
    """Linear plant identical to the prediction model (nominal case)."""

    def __init__(self, n: float):
        self.n = n
        self.x = np.zeros(STATE_DIM)

    def reset(self, x0: np.ndarray) -> None:
        """Set the relative LVLH state."""
        self.x = np.asarray(x0, dtype=float).copy()

    def relative_state(self) -> np.ndarray:
        """Current relative LVLH state."""
        return self.x.copy()

    def apply_impulse(self, dv: np.ndarray) -> None:
        """Instantaneous velocity change expressed in LVLH axes."""
        self.x[3:] += dv

    def propagate(self, duration: float, n_dense: int) -> np.ndarray:
        """Free drift; returns the n_dense + 1 states on a uniform grid."""
        phi = hcw_transition(self.n, duration / n_dense)
        states = [self.x.copy()]
        for _ in range(n_dense):
            states.append(phi @ states[-1])
        self.x = states[-1].copy()
        return np.array(states)


class TwoBodyPlant:
    """Exact two-body relative motion around a point-mass planet.

    The internal state is [r_t, v_t, rho, rho_dot] in an inertial frame,
    where rho is the chaser position relative to the target. Integrating
    rho directly avoids loss of precision from subtracting large numbers.
    """

    def __init__(self, mu: float, radius: float, rtol: float = 1e-11):
        self.mu = mu
        self.rtol = rtol
        speed = np.sqrt(mu / radius)
        self._target0 = (np.array([radius, 0.0, 0.0]),
                         np.array([0.0, speed, 0.0]))
        self.state = np.zeros(12)

    def reset(self, x0: np.ndarray) -> None:
        """Place the target on its circular orbit and the chaser at x0."""
        r_t, v_t = self._target0
        rho, rho_dot = lvlh_to_inertial(r_t, v_t, np.asarray(x0, float))
        self.state = np.concatenate([r_t, v_t, rho, rho_dot])

    def relative_state(self) -> np.ndarray:
        """Current relative LVLH state."""
        s = self.state
        return inertial_to_lvlh(s[0:3], s[3:6], s[6:9], s[9:12])

    def apply_impulse(self, dv: np.ndarray) -> None:
        """Instantaneous velocity change expressed in LVLH axes."""
        rot = lvlh_basis(self.state[0:3], self.state[3:6])
        self.state[9:12] += rot @ dv

    def _rhs(self, _t, s):
        r_t, v_t, rho, rho_dot = s[0:3], s[3:6], s[6:9], s[9:12]
        r_c = r_t + rho
        acc_t = -self.mu * r_t / np.linalg.norm(r_t) ** 3
        acc_c = -self.mu * r_c / np.linalg.norm(r_c) ** 3
        return np.concatenate([v_t, acc_t, rho_dot, acc_c - acc_t])

    def propagate(self, duration: float, n_dense: int) -> np.ndarray:
        """Free drift; returns the n_dense + 1 LVLH states on a grid."""
        t_eval = np.linspace(0.0, duration, n_dense + 1)
        sol = solve_ivp(self._rhs, (0.0, duration), self.state,
                        method="DOP853", t_eval=t_eval,
                        rtol=self.rtol, atol=1e-9)
        self.state = sol.y[:, -1].copy()
        return np.array([
            inertial_to_lvlh(s[0:3], s[3:6], s[6:9], s[9:12])
            for s in sol.y.T
        ])
