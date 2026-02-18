"""
Point Kinetics Equations — SciPy Reference Solver

Solves the 6-group point kinetics equations using SciPy's Radau (implicit
Runge-Kutta) integrator.  This solution serves as the ground-truth benchmark
for comparison with the PINN and X-TFC methods.

Equations:
    dn/dt  = [(rho - beta) / Lambda] * n  +  sum_i lambda_i * C_i
    dCi/dt = (beta_i / Lambda) * n  -  lambda_i * C_i     (i = 1..6)

Reference:
    Keepin, G.R. "Physics of Nuclear Kinetics" (1965)
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Nuclear parameters — Keepin (1957), U-235 thermal fission, 6 groups
# ---------------------------------------------------------------------------
BETA = np.array([0.000221, 0.001467, 0.001313, 0.002647, 0.000771, 0.000281])
LAMBDA = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
BETA_TOTAL = np.sum(BETA)        # ~0.0067
LAMBDA_GEN = 2e-5                # Prompt neutron generation time (s)

GRAPHICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphics")


# ---------------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------------
def point_kinetics_ode(t, y, rho_func):
    """RHS of the point kinetics ODE for ``solve_ivp``.

    Parameters
    ----------
    t : float
        Current time (s).
    y : ndarray, shape (7,)
        State vector [n, C1, C2, C3, C4, C5, C6].
    rho_func : callable
        Returns reactivity at time *t*.

    Returns
    -------
    ndarray, shape (7,)
        Time derivatives [dn/dt, dC1/dt, ..., dC6/dt].
    """
    n, C = y[0], y[1:]
    rho = rho_func(t)

    dn_dt = ((rho - BETA_TOTAL) / LAMBDA_GEN) * n + np.sum(LAMBDA * C)
    dC_dt = (BETA / LAMBDA_GEN) * n - LAMBDA * C

    return np.concatenate([[dn_dt], dC_dt])


# ---------------------------------------------------------------------------
# Reactivity profiles
# ---------------------------------------------------------------------------
def step_reactivity(t, rho_step=0.003, t_step=0.0):
    """Step reactivity insertion of *rho_step* at time *t_step*."""
    return rho_step if t >= t_step else 0.0


def ramp_reactivity(t, rho_max=0.003, t_ramp=1.0):
    """Linear ramp from 0 to *rho_max* over *t_ramp* seconds."""
    if t < 0:
        return 0.0
    elif t < t_ramp:
        return rho_max * (t / t_ramp)
    else:
        return rho_max


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def solve_point_kinetics(t_end=10.0, n_points=1000, rho_func=None):
    """Solve the point kinetics equations.

    Parameters
    ----------
    t_end : float
        End time (s).
    n_points : int
        Number of output time points.
    rho_func : callable, optional
        Reactivity function (default: step insertion of 0.003).

    Returns
    -------
    solution : OdeResult
        SciPy solution object with an extra ``rho_func`` attribute.
    """
    if rho_func is None:
        rho_func = step_reactivity

    n0 = 1.0
    C0 = (BETA / (LAMBDA * LAMBDA_GEN)) * n0
    y0 = np.concatenate([[n0], C0])

    t_eval = np.linspace(0, t_end, n_points)

    solution = solve_ivp(
        lambda t, y: point_kinetics_ode(t, y, rho_func),
        (0, t_end),
        y0,
        method="Radau",
        t_eval=t_eval,
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )
    solution.rho_func = rho_func
    return solution


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_residuals(solution, n_points=5000):
    """Re-evaluate the ODE RHS on the dense output and return residuals."""
    rho_func = getattr(solution, "rho_func", step_reactivity)
    t = np.linspace(solution.t[0], solution.t[-1], n_points)
    y = solution.sol(t)

    dt = t[1] - t[0]
    dydt_num = np.gradient(y, dt, axis=1)

    dydt_ode = np.zeros_like(y)
    for i, ti in enumerate(t):
        dydt_ode[:, i] = point_kinetics_ode(ti, y[:, i], rho_func)

    residuals = dydt_num - dydt_ode
    return {
        "max_residual": np.max(np.abs(residuals)),
        "mean_residual": np.mean(np.abs(residuals)),
        "residuals": residuals,
        "t": t,
        "y": y,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(solution, stats, save_path=None):
    """Three-panel plot: neutron density, n-residual, precursor residuals."""
    t_sol = solution.t
    t_val = stats["t"]
    residuals = stats["residuals"]

    skip = max(50, len(t_val) // 100)
    t_crop = t_val[skip:-skip]
    res_crop = residuals[:, skip:-skip]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Neutron density
    axes[0].plot(t_sol, solution.y[0], "b-", lw=2)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(r"Neutron density $n/n_0$")
    axes[0].set_title("Reactor Response to Step Reactivity Insertion")
    axes[0].grid(True, alpha=0.3)

    # n residual
    axes[1].plot(t_crop, res_crop[0], "r-", lw=1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"Residual $dn/dt$")
    axes[1].set_title("Neutron Density ODE Residual")
    axes[1].axhline(0, color="k", ls="--", alpha=0.4)
    axes[1].grid(True, alpha=0.3)
    ym = np.max(np.abs(res_crop[0]))
    axes[1].set_ylim(-ym, ym)

    # Precursor residuals
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i in range(6):
        axes[2].plot(t_crop, res_crop[i + 1], color=colors[i], lw=1,
                     label=f"Group {i+1}", alpha=0.7)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel(r"Residual $dC_i/dt$")
    axes[2].set_title("Precursor ODE Residuals")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].axhline(0, color="k", ls="--", alpha=0.4)
    axes[2].grid(True, alpha=0.3)
    ym = np.max(np.abs(res_crop[1:]))
    axes[2].set_ylim(-ym, ym)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Point Kinetics Benchmark (SciPy Radau)")
    print("=" * 45)

    solution = solve_point_kinetics(t_end=10.0)

    print(f"  Solver status : {solution.message}")
    print(f"  Time steps    : {len(solution.t)}")
    print(f"  n(0)          : {solution.y[0, 0]:.6f}")
    print(f"  n(10 s)       : {solution.y[0, -1]:.6f}")

    stats = validate_residuals(solution)
    print(f"  Max residual  : {stats['max_residual']:.2e}")
    print(f"  Mean residual : {stats['mean_residual']:.2e}")

    os.makedirs(GRAPHICS_DIR, exist_ok=True)
    plot_results(solution, stats,
                 save_path=os.path.join(GRAPHICS_DIR, "scipy_benchmark.png"))


if __name__ == "__main__":
    main()
