"""
Generate additional poster-quality graphics for the SIAM poster.

Produces:
    - Reactivity profile diagram (step function)
    - Individual precursor group dynamics (6-panel)
    - Prompt jump visualization (early-time zoom)
    - IC satisfaction comparison (PINN soft vs X-TFC hard)
    - Phase portrait (n vs total precursor)
    - Architecture schematic comparison (text-based)
    - Sensitivity to neuron count (X-TFC)
    - Ramp reactivity comparison

All outputs go to graphics/.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from point_kinetics_scipy import (
    solve_point_kinetics, step_reactivity, ramp_reactivity,
    BETA, LAMBDA, BETA_TOTAL, LAMBDA_GEN, point_kinetics_ode,
)
from point_kinetics_xtfc import XTFC, N0, C0
from point_kinetics_pinn import PINN, predict as pinn_predict, train as pinn_train, device

GRAPHICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphics")
os.makedirs(GRAPHICS_DIR, exist_ok=True)

# Common style
COLORS_6 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_reactivity_profile():
    """Step reactivity insertion diagram."""
    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.linspace(-0.5, 5, 1000)
    rho = np.array([step_reactivity(ti, rho_step=0.003) for ti in t])

    ax.plot(t, rho * 1000, "b-", lw=2.5)
    ax.axhline(0, color="k", ls="-", lw=0.8, alpha=0.5)
    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.4)

    ax.fill_between(t, 0, rho * 1000, alpha=0.15, color="blue")
    ax.annotate(r"$\rho_0 = 0.003 \approx 0.45\beta$",
                xy=(2.5, 3.0), fontsize=13, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"Reactivity $\rho$ ($\times 10^{-3}$)", fontsize=12)
    ax.set_title("Step Reactivity Insertion", fontsize=14)
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 4.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "reactivity_profile.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: reactivity_profile.png")


def plot_precursor_panels():
    """Six-panel plot showing each precursor group individually with parameters."""
    sol = solve_point_kinetics(t_end=10.0, n_points=1000)
    t = sol.t

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    half_lives = np.log(2) / LAMBDA

    for i in range(6):
        ax = axes[i]
        ax.plot(t, sol.y[i + 1], color=COLORS_6[i], lw=2)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(r"$C_%d$" % (i + 1), fontsize=12)
        ax.set_title(
            f"Group {i+1}: "
            r"$\beta_%d$" % (i + 1) + f" = {BETA[i]:.4e},  "
            r"$t_{1/2}$" + f" = {half_lives[i]:.2f} s",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle("Delayed Neutron Precursor Groups (SciPy Reference)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "precursor_panels.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: precursor_panels.png")


def plot_prompt_jump():
    """Early-time zoom showing the prompt jump + delayed rise."""
    sol = solve_point_kinetics(t_end=0.5, n_points=5000)
    t = sol.t
    n = sol.y[0]

    # Analytical prompt jump: n_prompt = n0 / (1 - rho/beta)
    rho = 0.003
    n_prompt = N0 / (1 - rho / BETA_TOTAL)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, n, "b-", lw=2.5, label="SciPy solution")
    ax.axhline(n_prompt, color="r", ls="--", lw=1.5, alpha=0.7,
               label=f"Prompt jump level = {n_prompt:.3f}")
    ax.axhline(N0, color="gray", ls=":", lw=1, alpha=0.5)

    ax.annotate("Prompt jump\n(microseconds)",
                xy=(0.002, n_prompt * 0.97), fontsize=11,
                color="red", ha="left")
    ax.annotate("Delayed rise\n(seconds timescale)",
                xy=(0.25, n[len(n)//2]), fontsize=11,
                color="blue", ha="center")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"Neutron density $n/n_0$", fontsize=12)
    ax.set_title("Prompt Jump and Delayed Neutron Rise (early time)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "prompt_jump.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: prompt_jump.png")


def plot_ic_satisfaction():
    """Compare IC satisfaction: PINN (soft) vs X-TFC (hard)."""
    # X-TFC
    xtfc = XTFC(n_neurons=100, t_max=10.0, seed=42)
    xtfc.train(n_collocation=500)

    # PINN (quick train)
    pinn = PINN(4, 64).to(device)
    pinn_train(pinn, epochs=2000, verbose=False)

    # Evaluate near t=0
    t_near = np.linspace(0, 0.05, 200)

    xtfc_pred = xtfc.predict(t_near)
    pinn_pred_vals = pinn_predict(pinn, t_near)

    # Reference IC
    C0_vals = C0.numpy() if hasattr(C0, 'numpy') else C0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Neutron density near t=0
    axes[0].plot(t_near, pinn_pred_vals["n"], ":", color="#d62728", lw=2, label="PINN (soft IC)")
    axes[0].plot(t_near, xtfc_pred["n"], "--", color="#1f77b4", lw=2, label="X-TFC (hard IC)")
    axes[0].axhline(N0, color="k", ls="-", lw=1, alpha=0.4, label=f"True IC = {N0}")
    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel(r"$n(t)$", fontsize=12)
    axes[0].set_title("Neutron Density Near t = 0", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Total precursor concentration near t=0
    pinn_C_total = np.sum(pinn_pred_vals["C"], axis=1)
    xtfc_C_total = np.sum(xtfc_pred["C"], axis=1)
    C0_total = np.sum(C0_vals)

    axes[1].plot(t_near, pinn_C_total, ":", color="#d62728", lw=2, label="PINN (soft IC)")
    axes[1].plot(t_near, xtfc_C_total, "--", color="#1f77b4", lw=2, label="X-TFC (hard IC)")
    axes[1].axhline(C0_total, color="k", ls="-", lw=1, alpha=0.4, label=f"True IC = {C0_total:.1f}")
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel(r"$\sum C_i(t)$", fontsize=12)
    axes[1].set_title("Total Precursor Conc. Near t = 0", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Initial Condition Satisfaction: Soft (PINN) vs Hard (X-TFC)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "ic_satisfaction.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: ic_satisfaction.png")


def plot_phase_portrait():
    """Phase portrait: neutron density vs total precursor concentration."""
    sol = solve_point_kinetics(t_end=10.0, n_points=2000)
    n = sol.y[0]
    C_total = np.sum(sol.y[1:], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(C_total, n, c=sol.t, cmap="viridis", s=3, zorder=2)
    ax.plot(C_total[0], n[0], "ro", ms=10, zorder=3, label="t = 0 (steady state)")
    ax.plot(C_total[-1], n[-1], "r^", ms=10, zorder=3, label=f"t = {sol.t[-1]:.0f} s")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time (s)", fontsize=11)

    ax.set_xlabel(r"Total precursor concentration $\sum C_i$", fontsize=12)
    ax.set_ylabel(r"Neutron density $n/n_0$", fontsize=12)
    ax.set_title("Phase Portrait: Reactor Transient Trajectory", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "phase_portrait.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: phase_portrait.png")


def plot_neuron_sensitivity():
    """X-TFC accuracy vs number of ELM neurons."""
    scipy_sol = solve_point_kinetics(t_end=10.0, n_points=500)
    t = scipy_sol.t
    ref_n = scipy_sol.y[0]
    ref_C = scipy_sol.y[1:].T

    neuron_counts = [10, 25, 50, 75, 100, 150, 200]
    max_errors_n = []
    max_errors_C = []
    residual_norms = []

    for nn_count in neuron_counts:
        model = XTFC(n_neurons=nn_count, t_max=10.0, seed=42)
        model.train(n_collocation=500)
        pred = model.predict(t)

        err_n = np.max(np.abs(pred["n"] - ref_n) / (np.abs(ref_n) + 1e-10)) * 100
        err_C = np.max(np.abs(pred["C"] - ref_C) / (np.abs(ref_C) + 1e-10)) * 100

        max_errors_n.append(err_n)
        max_errors_C.append(err_C)
        residual_norms.append(model.residual_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(neuron_counts, max_errors_n, "o-", color="#1f77b4", lw=2, ms=8, label=r"$n(t)$")
    axes[0].semilogy(neuron_counts, max_errors_C, "s-", color="#d62728", lw=2, ms=8, label=r"$C_i(t)$")
    axes[0].set_xlabel("Number of ELM Neurons", fontsize=12)
    axes[0].set_ylabel("Max Relative Error (%)", fontsize=12)
    axes[0].set_title("X-TFC Accuracy vs Network Size", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(neuron_counts, residual_norms, "o-", color="#2ca02c", lw=2, ms=8)
    axes[1].set_xlabel("Number of ELM Neurons", fontsize=12)
    axes[1].set_ylabel("Training Residual Norm", fontsize=12)
    axes[1].set_title("X-TFC Training Residual vs Network Size", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "neuron_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: neuron_sensitivity.png")


def plot_ramp_vs_step():
    """Compare reactor response to step vs ramp reactivity insertion."""
    sol_step = solve_point_kinetics(t_end=10.0, n_points=1000,
                                     rho_func=step_reactivity)
    sol_ramp = solve_point_kinetics(t_end=10.0, n_points=1000,
                                     rho_func=ramp_reactivity)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Reactivity profiles
    t = np.linspace(0, 10, 1000)
    rho_s = [step_reactivity(ti) for ti in t]
    rho_r = [ramp_reactivity(ti) for ti in t]
    axes[0].plot(t, np.array(rho_s) * 1000, "-", color="#d62728", lw=2, label="Step")
    axes[0].plot(t, np.array(rho_r) * 1000, "--", color="#1f77b4", lw=2, label="Ramp (1 s)")
    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel(r"$\rho$ ($\times 10^{-3}$)", fontsize=12)
    axes[0].set_title("Reactivity Profiles", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Neutron density response
    axes[1].plot(sol_step.t, sol_step.y[0], "-", color="#d62728", lw=2, label="Step response")
    axes[1].plot(sol_ramp.t, sol_ramp.y[0], "--", color="#1f77b4", lw=2, label="Ramp response")
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel(r"Neutron density $n/n_0$", fontsize=12)
    axes[1].set_title("Neutron Density Response", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "ramp_vs_step.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: ramp_vs_step.png")


def plot_method_summary_table():
    """Visual summary table comparing the three methods."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    headers = ["", "Standard PINN", "X-TFC", "SciPy Radau"]
    rows = [
        ["IC Handling", "Soft (penalty term)", "Hard (exact, via TFC)", "Exact (numerical)"],
        ["Training", "Gradient descent\n(Adam, 5000 epochs)", "Least-squares\n(single solve)", "Adaptive Radau\n(implicit RK)"],
        ["Parameters", "~17k (4×64 NN)", "700 (ELM output weights)", "N/A (direct solve)"],
        ["Accuracy", "~90% error", "~0.06% error", "Machine precision"],
        ["Speed", "~60 s (CPU)", "~0.5 s", "~0.01 s"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Style row labels
    for i in range(len(rows)):
        cell = table[i + 1, 0]
        cell.set_facecolor("#ecf0f1")
        cell.set_text_props(fontweight="bold")

    # Alternate row colors
    for i in range(len(rows)):
        for j in range(1, len(headers)):
            cell = table[i + 1, j]
            if i % 2 == 0:
                cell.set_facecolor("#f8f9fa")
            else:
                cell.set_facecolor("#ffffff")

    ax.set_title("Method Comparison Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "method_summary_table.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: method_summary_table.png")


def main():
    print("Generating additional poster graphics")
    print("=" * 45)

    plot_reactivity_profile()
    plot_precursor_panels()
    plot_prompt_jump()
    plot_ic_satisfaction()
    plot_phase_portrait()
    plot_neuron_sensitivity()
    plot_ramp_vs_step()
    plot_method_summary_table()

    print(f"\nAll graphics saved to {GRAPHICS_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(GRAPHICS_DIR)):
        if f.endswith(".png"):
            size_kb = os.path.getsize(os.path.join(GRAPHICS_DIR, f)) / 1024
            print(f"  {f:<35s} {size_kb:>6.0f} KB")


if __name__ == "__main__":
    main()
