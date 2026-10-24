"""
run_experiments.py
------------------

Run the full study: model check, nominal comparison and Monte Carlo.

Usage:
    python run_experiments.py            # full study (20 Monte Carlo runs)
    python run_experiments.py --runs 5   # quicker Monte Carlo

Outputs are written to ./results (CSV tables, JSON summary, figures).
"""

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from rendezvous_mpc import plotting
from rendezvous_mpc.config import (NOMINAL_DISTURBANCE, TERMINAL_BOX,
                                   DisturbanceConfig, OrbitConfig,
                                   ScenarioConfig, VariableHorizonCost)
from rendezvous_mpc.controllers import build_controllers
from rendezvous_mpc.dynamics import (HCWPlant, TwoBodyPlant,
                                     discretise_impulsive)
from rendezvous_mpc.metrics import evaluate
from rendezvous_mpc.simulation import run_closed_loop

RESULTS = Path(__file__).parent / "results"
FIGURES = RESULTS / "figures"


def write_csv(rows, path) -> None:
    """Write a list of metric dictionaries to CSV."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_validation(orbit, scenario) -> dict:
    """Compare free drift of the HCW model and the two-body plant."""
    n_dense = 200
    hcw, nl = HCWPlant(orbit.mean_motion), TwoBodyPlant(orbit.mu,
                                                        orbit.radius)
    hcw.reset(scenario.x0)
    nl.reset(scenario.x0)
    x_hcw = hcw.propagate(orbit.period, n_dense)
    x_nl = nl.propagate(orbit.period, n_dense)
    t = np.linspace(0, orbit.period, n_dense + 1)
    plotting.plot_model_validation(t, x_hcw, x_nl,
                                   FIGURES / "model_validation.png")
    err = np.linalg.norm(x_hcw[:, :3] - x_nl[:, :3], axis=1)
    step = n_dense * int(scenario.ts) // int(orbit.period)
    return {"error_after_one_orbit_m": float(err[-1]),
            "error_after_one_step_m": float(err[max(step, 1)])}


def nominal_study(a, b, orbit, scenario) -> list:
    """Linear plant, perfect measurements and actuation."""
    results, rows = [], []
    for ctrl in build_controllers(a, b, scenario):
        res = run_closed_loop(ctrl, HCWPlant(orbit.mean_motion), scenario,
                              NOMINAL_DISTURBANCE)
        results.append(res)
        rows.append(evaluate(res, scenario))
        print(f"  nominal {ctrl.name:5s} dv={rows[-1]['delta_v']:6.2f} m/s")
    write_csv(rows, RESULTS / "metrics_nominal.csv")
    plotting.plot_trajectories(results, scenario,
                               FIGURES / "nominal_trajectories.png")
    plotting.plot_impulses(results, FIGURES / "nominal_impulses.png")
    return rows


def monte_carlo_study(a, b, orbit, scenario, dist) -> dict:
    """Two-body plant, navigation noise, MIB and execution errors."""
    vh = VariableHorizonCost(terminal_tol=TERMINAL_BOX)
    controllers = build_controllers(a, b, scenario, vh=vh)
    runs = {c.name: [] for c in controllers}
    table = {c.name: [] for c in controllers}
    for ctrl in controllers:
        for i in range(dist.n_runs):
            rng = np.random.default_rng(dist.seed + i)   # paired noise
            res = run_closed_loop(
                ctrl, TwoBodyPlant(orbit.mu, orbit.radius), scenario,
                dist, rng)
            runs[ctrl.name].append(res)
            table[ctrl.name].append(evaluate(res, scenario))
        print(f"  monte carlo {ctrl.name:5s} done")

    rows = [dict(run=i, **m) for name in table
            for i, m in enumerate(table[name])]
    write_csv(rows, RESULTS / "metrics_monte_carlo_runs.csv")
    plotting.plot_monte_carlo_trajectories(
        runs, scenario, FIGURES / "monte_carlo_trajectories.png")
    plotting.plot_monte_carlo_metrics(
        table, FIGURES / "monte_carlo_metrics.png")
    return summarise(table)


def summarise(table) -> dict:
    """Mean and standard deviation of each metric per controller."""
    keys = ["delta_v", "firings", "settling_time", "final_error",
            "cone_viol_sampled", "cone_viol_intersample", "solve_time_mean",
            "solve_time_max"]
    summary = {}
    for name, runs in table.items():
        entry = {"settled_fraction": float(np.mean(
            [np.isfinite(m["settling_time"]) for m in runs]))}
        for key in keys:
            vals = np.array([m[key] for m in runs], dtype=float)
            vals = vals[np.isfinite(vals)]
            entry[key] = {"mean": float(vals.mean()) if vals.size else None,
                          "std": float(vals.std()) if vals.size else None}
        summary[name] = entry
    rows = [{"controller": n, "settled_fraction": s["settled_fraction"],
             **{f"{k}_mean": s[k]["mean"] for k in keys},
             **{f"{k}_std": s[k]["std"] for k in keys}}
            for n, s in summary.items()]
    write_csv(rows, RESULTS / "metrics_monte_carlo_summary.csv")
    return summary


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=None,
                        help="number of Monte Carlo runs per controller")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    orbit, scenario = OrbitConfig(), ScenarioConfig()
    dist = DisturbanceConfig()
    if args.runs:
        dist = replace(dist, n_runs=args.runs)
    a, b = discretise_impulsive(orbit.mean_motion, scenario.ts)

    start = time.time()
    print("\n1/3 model validation")
    validation = model_validation(orbit, scenario)
    print("\n2/3 nominal study")
    nominal = nominal_study(a, b, orbit, scenario)
    print(f"\n3/3 Monte Carlo study ({dist.n_runs} runs per controller)")
    mc = monte_carlo_study(a, b, orbit, scenario, dist)

    summary = {
        "orbit": {"mean_motion_rad_s": orbit.mean_motion,
                  "period_s": orbit.period},
        "model_validation": validation,
        "nominal": nominal,
        "monte_carlo": mc,
        "runtime_s": time.time() - start,
    }
    with open(RESULTS / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"done in {summary['runtime_s']:.0f} s -> {RESULTS}")


if __name__ == "__main__":
    main()
