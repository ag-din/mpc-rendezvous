"""
controllers.py
--------------

Model predictive controllers for spacecraft rendezvous.

Four controllers share the same prediction model and constraints:

* QP   - fixed horizon, quadratic cost, eq. (10), P from the DARE.
* LP   - fixed horizon, 1-norm cost, eq. (11).
* VHLP - variable horizon, cost 1 + ||R u||_1, eq. (15a).
* VHQP - variable horizon, cost 1 + u'Ru, eq. (15b).

Constraint set includes:

* Input bounds per axis, eq. (6b): |u_p| <= u_max.
* Visibility cone for an approach from behind, eq. (8), softened with
  non-negative slacks and an exact (1-norm) penalty.
* Variable-horizon controllers add a terminal set x_N in T.

"""

# Libraries
import time
import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are
from dataclasses import dataclass
from .config import L1Cost, QuadraticCost, ScenarioConfig, VariableHorizonCost
from .dynamics import INPUT_DIM, STATE_DIM

ACCEPTED_STATUS = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


@dataclass
class ControlResult:
    """Output of one controller call."""

    u: np.ndarray
    solve_time: float
    horizon: int
    feasible: bool


def cone_matrix(half_angle_deg: float) -> np.ndarray:
    """Matrix G of eq. (8) so that the cone reads G x <= 0."""
    t = np.tan(np.radians(half_angle_deg))
    g = np.zeros((4, STATE_DIM))
    g[0, [0, 2]] = [t, 1.0]     # +z + x tan(gamma) <= 0
    g[1, [0, 2]] = [t, -1.0]    # -z + x tan(gamma) <= 0
    g[2, [0, 1]] = [t, 1.0]     # +y + x tan(gamma) <= 0
    g[3, [0, 1]] = [t, -1.0]    # -y + x tan(gamma) <= 0
    return g


class _HorizonProblem:
    """One parametric MPC problem with a given horizon N."""

    def __init__(self, a, b, scenario: ScenarioConfig, horizon: int):
        self.horizon = horizon
        self.x0 = cp.Parameter(STATE_DIM)
        self.x = cp.Variable((STATE_DIM, horizon + 1))
        self.u = cp.Variable((INPUT_DIM, horizon))
        self.slack = cp.Variable((4, horizon), nonneg=True)
        g = cone_matrix(scenario.cone_half_angle_deg)

        self.constraints = [self.x[:, 0] == self.x0]
        for i in range(horizon):
            self.constraints += [
                self.x[:, i + 1] == a @ self.x[:, i] + b @ self.u[:, i],
                cp.abs(self.u[:, i]) <= scenario.u_max,
                g @ self.x[:, i + 1] <= self.slack[:, i],
            ]
        self.cone_penalty = scenario.cone_penalty * cp.sum(self.slack)
        self.problem = None

    def build(self, cost, extra_constraints=()) -> None:
        """Create the CVXPY problem from a cost expression."""
        self.problem = cp.Problem(
            cp.Minimize(cost + self.cone_penalty),
            self.constraints + list(extra_constraints))

    def solve(self, x0: np.ndarray):
        """Solve for the current state. Returns (u0, cost) or (None, inf)."""
        self.x0.value = x0
        try:
            self.problem.solve(solver=cp.SCS)
        except cp.SolverError:
            return None, np.inf
        if self.problem.status not in ACCEPTED_STATUS:
            return None, np.inf
        return self.u.value[:, 0].copy(), float(self.problem.value)


class FixedHorizonMPC:
    """Fixed-horizon MPC with quadratic (eq. 10) or 1-norm (eq. 11) cost."""

    def __init__(self, name, a, b, scenario: ScenarioConfig, weights):
        self.name = name
        x_s = scenario.x_setpoint
        n_h = weights.horizon
        self._prob = _HorizonProblem(a, b, scenario, n_h)
        x, u = self._prob.x, self._prob.u
        q = np.diag([weights.q_pos] * 3 + [weights.q_vel] * 3)
        r = weights.r * np.eye(INPUT_DIM)

        if isinstance(weights, QuadraticCost):
            p = solve_discrete_are(a, b, q, r)
            cost = sum(cp.quad_form(x[:, i] - x_s, q)
                       + cp.quad_form(u[:, i], r) for i in range(n_h))
            cost += cp.quad_form(x[:, n_h] - x_s, p)
        elif isinstance(weights, L1Cost):
            p = weights.p_scale * q
            cost = sum(cp.norm1(q @ (x[:, i] - x_s)) + cp.norm1(r @ u[:, i])
                       for i in range(n_h))
            cost += cp.norm1(p @ (x[:, n_h] - x_s))
        else:
            raise TypeError("weights must be QuadraticCost or L1Cost")
        self._prob.build(cost)

    def control(self, x_hat: np.ndarray) -> ControlResult:
        """Compute the first impulse of the optimal plan."""
        start = time.perf_counter()
        u0, _ = self._prob.solve(x_hat)
        elapsed = time.perf_counter() - start
        feasible = u0 is not None
        u0 = u0 if feasible else np.zeros(INPUT_DIM)
        return ControlResult(u0, elapsed, self._prob.horizon, feasible)


class VariableHorizonMPC:
    """Variable-horizon MPC, eq. (15).

    One problem is built for every N in 1..N_max. At each step all of
    them are solved and the feasible one with the lowest cost is used
    (the enumeration approach of [17], [29]).
    """

    def __init__(self, name, a, b, scenario: ScenarioConfig,
                 weights: VariableHorizonCost, norm: str = "l1"):
        if norm not in ("l1", "quadratic"):
            raise ValueError("norm must be 'l1' or 'quadratic'")
        self.name = name
        x_s = scenario.x_setpoint
        tol = weights.terminal_tol
        self._problems = []
        for n_h in range(1, weights.max_horizon + 1):
            prob = _HorizonProblem(a, b, scenario, n_h)
            u = prob.u
            if norm == "l1":
                fuel = weights.r_l1 * cp.sum(cp.abs(u))
            else:
                fuel = weights.r_quad * cp.sum_squares(u)
            if np.any(tol > 0):
                terminal = [cp.abs(prob.x[:, n_h] - x_s) <= tol]
            else:
                terminal = [prob.x[:, n_h] == x_s]
            prob.build(n_h + fuel, terminal)
            self._problems.append(prob)

    def control(self, x_hat: np.ndarray) -> ControlResult:
        """Solve every horizon and keep the cheapest feasible plan."""
        start = time.perf_counter()
        best_u, best_cost, best_n = None, np.inf, 0
        for prob in self._problems:
            u0, cost = prob.solve(x_hat)
            if cost < best_cost:
                best_u, best_cost, best_n = u0, cost, prob.horizon
        elapsed = time.perf_counter() - start
        feasible = best_u is not None
        best_u = best_u if feasible else np.zeros(INPUT_DIM)
        return ControlResult(best_u, elapsed, best_n, feasible)


def build_controllers(a, b, scenario: ScenarioConfig,
                      qp: QuadraticCost = QuadraticCost(),
                      lp: L1Cost = L1Cost(),
                      vh: VariableHorizonCost = VariableHorizonCost()):
    """Return the four controllers compared in the study."""
    return [
        FixedHorizonMPC("QP", a, b, scenario, qp),
        FixedHorizonMPC("LP", a, b, scenario, lp),
        VariableHorizonMPC("VHLP", a, b, scenario, vh, norm="l1"),
        VariableHorizonMPC("VHQP", a, b, scenario, vh, norm="quadratic"),
    ]
