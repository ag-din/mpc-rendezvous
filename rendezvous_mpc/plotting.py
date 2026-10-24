"""
plotting.py
-----------

Figures for the rendezvous study.

"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import ScenarioConfig  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
CONE = "#8a8985"
# Categorical slots 1-4, fixed order (one colour per controller).
COLORS = {"QP": "#2a78d6", "LP": "#eb6834", "VHLP": "#1baf7a",
          "VHQP": "#eda100"}
PAIR = ("#2a78d6", "#eb6834")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED,
    "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 10, "axes.titlecolor": TEXT, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.frameon": False, "lines.linewidth": 1.5,
    "axes.spines.top": True, "axes.spines.right": True,
    "font.family": "DejaVu Sans",
})


def _show_top_right(ax) -> None:
    """Display the top and right axes and their ticks."""
    ax.tick_params(top=True, right=True)
    for spine in (ax.spines["top"], ax.spines["right"]):
        spine.set_visible(True)


def _cone(ax, scenario: ScenarioConfig, x_min: float) -> None:
    """Draw the visibility cone boundary in the x-z plane (km)."""
    t = np.tan(np.radians(scenario.cone_half_angle_deg))
    xs = np.array([0.0, x_min])
    for sign in (1, -1):
        ax.plot(xs / 1e3, sign * -xs * t / 1e3, color=CONE, lw=1.0, ls="--")


def plot_model_validation(t, x_hcw, x_nl, path) -> None:
    """Free drift over one orbit: HCW prediction vs two-body plant."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax in axes:
        _show_top_right(ax)
    ax = axes[0]
    ax.plot(x_nl[:, 0] / 1e3, x_nl[:, 2], color=PAIR[1],
            label="Two-body plant")
    ax.plot(x_hcw[:, 0] / 1e3, x_hcw[:, 2], color=PAIR[0], marker="o",
            ms=8, label="HCW model (x0 is an equilibrium)")
    ax.invert_yaxis()
    ax.set(xlabel="x, along-track (km)", ylabel="z, radial (m)",
           title="Free drift from x0 over one orbit")
    ax.legend()
    ax = axes[1]
    err = np.linalg.norm(x_hcw[:, :3] - x_nl[:, :3], axis=1)
    ax.plot(t / 3600, err, color=PAIR[0])
    ax.set(xlabel="Time (h)", ylabel="Position error (m)",
           title="HCW prediction error")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trajectories(results, scenario: ScenarioConfig, path) -> None:
    """Trajectory in the x-z plane and time histories (cf. paper Fig. 4)."""
    fig = plt.figure(figsize=(11, 5.6))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(grid[:, 0])
    _show_top_right(ax)
    _cone(ax, scenario, -16e3)
    for res in results:
        ax.plot(res.x_dense[:, 0] / 1e3, res.x_dense[:, 2] / 1e3,
                color=COLORS[res.name], label=res.name)
    ax.plot(scenario.x0[0] / 1e3, scenario.x0[2] / 1e3, "o", color=TEXT,
            ms=5)
    ax.plot(scenario.x_setpoint[0] / 1e3, 0, "*", color=TEXT, ms=9)
    ax.set(xlim=(0.5, -16), ylim=(4, -4), xlabel="x, along-track (km)",
           ylabel="z, radial (km)",
           title="Chaser trajectory in the LVLH x-z plane "
                 "(dashed: 20° cone)")
    ax.legend(loc="lower left", ncols=4)

    inset = ax.inset_axes([0.05, 0.62, 0.36, 0.33])
    _show_top_right(inset)
    _cone(inset, scenario, -2e3)
    for res in results:
        inset.plot(res.x_dense[:, 0] / 1e3, res.x_dense[:, 2] / 1e3,
                   color=COLORS[res.name], lw=1.2)
    inset.set(xlim=(-0.95, -1.15), ylim=(0.4, -0.15))
    inset.set_title("Zoom near set-point", fontsize=8)
    inset.tick_params(labelsize=7)

    panels = [(0, "x (km)", lambda r: r.x[:, 0] / 1e3),
              (1, "z (km)", lambda r: r.x[:, 2] / 1e3),
              (2, "Cumulative Δv (m/s)",
               lambda r: np.r_[0, np.cumsum(np.abs(r.u_applied).sum(1))])]
    for row, label, getter in panels:
        sub = fig.add_subplot(grid[row, 1])
        _show_top_right(sub)
        for res in results:
            sub.plot(res.t, getter(res), color=COLORS[res.name])
        sub.set_ylabel(label)
        if row == 2:
            sub.set_xlabel("Time (s)")
        else:
            sub.tick_params(labelbottom=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_impulses(results, path) -> None:
    """Executed impulses per axis and controller (small multiples)."""
    fig, axes = plt.subplots(3, len(results), figsize=(11, 5.4),
                             sharex=True, sharey="row")
    for ax in axes.flat:
        _show_top_right(ax)
    k = np.arange(results[0].u_applied.shape[0])
    for col, res in enumerate(results):
        for row, axis_name in enumerate("xyz"):
            ax = axes[row, col]
            u = res.u_applied[:, row]
            ax.vlines(k, 0, u, color=COLORS[res.name], lw=2.5)
            ax.axhline(0, color=GRID, lw=0.8)
            if row == 0:
                ax.set_title(res.name)
            if col == 0:
                ax.set_ylabel(f"u_{axis_name} (m/s)")
            if row == 2:
                ax.set_xlabel("Sampling instant k")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monte_carlo_trajectories(mc_results, scenario, path) -> None:
    """All Monte Carlo trajectories, one panel per controller."""
    names = list(mc_results)
    fig, axes = plt.subplots(1, len(names), figsize=(12, 3.4),
                             sharey=True)
    for ax in axes:
        _show_top_right(ax)
    for ax, name in zip(axes, names):
        _cone(ax, scenario, -16e3)
        for res in mc_results[name]:
            ax.plot(res.x_dense[:, 0] / 1e3, res.x_dense[:, 2] / 1e3,
                    color=COLORS[name], lw=0.8, alpha=0.5)
        ax.set(xlim=(0.5, -16), ylim=(4, -4), title=name,
               xlabel="x (km)")
    axes[0].set_ylabel("z (km)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monte_carlo_metrics(table, path) -> None:
    """Per-run Δv and settling time (dots) with the mean (bar)."""
    names = list(table)
    keys = [("delta_v", "Total Δv (m/s)"),
            ("settling_time", "Settling time (s)"),
            ("final_error", "Final position error (m)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 3.2))
    for ax in axes:
        _show_top_right(ax)
    rng = np.random.default_rng(0)
    for ax, (key, label) in zip(axes, keys):
        tick_labels = []
        for i, name in enumerate(names):
            vals = np.array([m[key] for m in table[name]], dtype=float)
            n_total = vals.size
            vals = vals[np.isfinite(vals)]
            missing = n_total - vals.size
            tick_labels.append(
                f"{name}\n{missing}/{n_total}\nunsettled" if missing
                else name)
            jitter = rng.uniform(-0.15, 0.15, vals.size)
            ax.plot(i + jitter, vals, "o", ms=4, color=COLORS[name],
                    alpha=0.7)
            if vals.size:
                ax.hlines(vals.mean(), i - 0.3, i + 0.3, color=TEXT, lw=1.5)
        ax.set_xticks(range(len(names)), tick_labels)
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
