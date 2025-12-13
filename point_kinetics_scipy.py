"""
Point Kinetics Equations Solver using SciPy

Solves the 6-group point kinetics equations for nuclear reactor transient analysis.
This serves as a numerical benchmark for comparison with Physics-Informed Neural
Networks (PINNs).

Equations:
    dn/dt = [(ρ - β) / Λ] * n + Σ λᵢCᵢ
    dCᵢ/dt = (βᵢ / Λ) * n - λᵢCᵢ  (i = 1..6)

where:
    n = neutron density
    Cᵢ = delayed neutron precursor concentration for group i
    ρ = reactivity
    β = total delayed neutron fraction
    βᵢ = delayed neutron fraction for group i
    λᵢ = decay constant for precursor group i
    Λ = prompt neutron generation time

Reference:
    Keepin, G.R. "Physics of Nuclear Kinetics" (1965)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime


# Nuclear Parameters (U-235, thermal spectrum)

# Delayed neutron fractions by precursor group
BETA = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])

# Precursor decay constants (s^-1)
LAMBDA = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])

# Total delayed neutron fraction
BETA_TOTAL = np.sum(BETA)

# Prompt neutron generation time (s)
LAMBDA_GEN = 2e-5



def point_kinetics_ode(t, y, rho_func):
    """
    Point kinetics ODE system for scipy.integrate.solve_ivp.

    Args:
        t: Current time (s)
        y: State vector [n, C1, C2, C3, C4, C5, C6]
        rho_func: Callable returning reactivity at time t

    Returns:
        Derivatives [dn/dt, dC1/dt, ..., dC6/dt]
    """
    n = y[0]
    C = y[1:]
    rho = rho_func(t)

    # Neutron density: prompt production + delayed neutron source
    dn_dt = ((rho - BETA_TOTAL) / LAMBDA_GEN) * n + np.sum(LAMBDA * C)

    # Precursor concentrations: production from fission - decay
    dC_dt = (BETA / LAMBDA_GEN) * n - LAMBDA * C

    return np.concatenate([[dn_dt], dC_dt])



def step_reactivity(t, rho_step=0.003, t_step=0.0):
    """
    Step reactivity insertion.

    Args:
        t: Current time (s)
        rho_step: Reactivity magnitude (default: 0.003 ≈ 0.5β)
        t_step: Time of insertion (s)

    Returns:
        Reactivity value at time t
    """
    return rho_step if t >= t_step else 0.0



def solve_point_kinetics(t_end=10.0, n_points=1000, rho_func=None):
    """
    Solve point kinetics equations with specified parameters.

    Args:
        t_end: Simulation end time (s)
        n_points: Number of output time points
        rho_func: Reactivity function (default: step_reactivity)

    Returns:
        scipy.integrate.OdeResult object containing solution
    """
    if rho_func is None:
        rho_func = step_reactivity

    # Steady-state initial conditions (n0 = 1, equilibrium precursors)
    n0 = 1.0
    C0 = (BETA / (LAMBDA * LAMBDA_GEN)) * n0
    y0 = np.concatenate([[n0], C0])

    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, n_points)

    solution = solve_ivp(
        lambda t, y: point_kinetics_ode(t, y, rho_func),
        t_span,
        y0,
        method="Radau",
        t_eval=t_eval,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10
    )
    
    # Store rho_func for validation
    solution.rho_func = rho_func

    return solution

def validate_residuals(solution, n_validation_points=5000):
    """
    Validate solution using dense output interpolation.
    
    Uses the solver's continuous solution to compute residuals at many points,
    avoiding discretization errors from np.gradient.

    Args:
        solution: OdeResult from solve_point_kinetics (with dense_output=True)
        n_validation_points: Number of points for validation

    Returns:
        Dict with residual statistics and validation data
    """
    rho_func = getattr(solution, 'rho_func', step_reactivity)
    
    # Use dense output for smooth interpolation
    t_val = np.linspace(solution.t[0], solution.t[-1], n_validation_points)
    y_val = solution.sol(t_val)  # Continuous interpolation
    
    # Compute derivative using small finite difference on dense output
    dt = t_val[1] - t_val[0]
    dydt_numerical = np.gradient(y_val, dt, axis=1)
    
    # Expected derivatives from ODE
    dydt_ode = np.zeros_like(y_val)
    for i, ti in enumerate(t_val):
        dydt_ode[:, i] = point_kinetics_ode(ti, y_val[:, i], rho_func)
    
    # Residuals
    residuals = dydt_numerical - dydt_ode
    
    return {
        "max_residual": np.max(np.abs(residuals)),
        "mean_residual": np.mean(np.abs(residuals)),
        "residuals": residuals,
        "t": t_val,
        "y": y_val
    }

def plot_results(solution, stats, save_path=None):
    """Combined plot: solution and residual validation."""
    t_sol = solution.t
    t_val = stats["t"]
    residuals = stats["residuals"]
    
    # Exclude boundary effects (first and last 1% of points)
    skip = max(50, len(t_val) // 100)
    t_crop = t_val[skip:-skip]
    res_crop = residuals[:, skip:-skip]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Top: Neutron density solution (full time range)
    axes[0].plot(t_sol, solution.y[0], 'b-', linewidth=2)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Neutron Density (n/n₀)")
    axes[0].set_title("Reactor Response to Step Reactivity")
    axes[0].grid(True, alpha=0.3)

    # Middle: Neutron density residual (excluding boundaries)
    axes[1].plot(t_crop, res_crop[0], 'r-', linewidth=1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual (dn/dt)")
    axes[1].set_title("Neutron Density Residual")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    # Center y-axis around zero
    ymax = max(abs(res_crop[0].min()), abs(res_crop[0].max()))
    axes[1].set_ylim(-ymax, ymax)

    # Bottom: Precursor residuals (excluding boundaries)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(6):
        axes[2].plot(t_crop, res_crop[i+1], color=colors[i], linewidth=1,
                     label=f'Group {i+1}', alpha=0.7)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Residual (dC/dt)")
    axes[2].set_title("Precursor Concentration Residuals")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    # Center y-axis around zero
    all_res = res_crop[1:].flatten()
    ymax = max(abs(all_res.min()), abs(all_res.max()))
    axes[2].set_ylim(-ymax, ymax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()



def main():
    """Run benchmark simulation and display results."""
    print("Point Kinetics Benchmark (SciPy)")
    print("=" * 40)

    solution = solve_point_kinetics(t_end=10.0)

    print(f"Solver: {solution.message}")
    print(f"Time steps computed: {len(solution.t)}")
    print(f"Initial neutron density: {solution.y[0, 0]:.4f}")
    print(f"Final neutron density: {solution.y[0, -1]:.4f}")

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%H%M%S_%d%m%y")
    
    # Validate and plot combined results
    stats = validate_residuals(solution)
    print(f"Max residual: {stats['max_residual']:.2e}")
    print(f"Mean residual: {stats['mean_residual']:.2e}")
    plot_results(solution, stats, save_path=f"point_kinetics_{timestamp}.png")


if __name__ == "__main__":
    main()