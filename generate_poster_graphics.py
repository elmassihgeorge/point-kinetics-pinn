"""
Generate all poster-quality graphics for the SIAM poster.

Produces the following in graphics/:
    - precursor_comparison.png       PINN vs SciPy precursor vertical stack
    - xtfc_precursor_comparison.png  X-TFC vs SciPy precursor vertical stack
    - pinn_training.png              Two-panel PINN training loss
    - precursor_panels.png           Individual precursor group dynamics (6-panel)
    - ramp_vs_step.png               Step vs ramp reactivity response
    - method_summary_table.png       Visual comparison table

The remaining graphics (neutron_comparison, neutron_error, error_over_time,
pinn_solution, xtfc_solution, scipy_benchmark) are produced by
compare_methods.py, point_kinetics_pinn.py, point_kinetics_xtfc.py,
and point_kinetics_scipy.py respectively.

All outputs go to graphics/ at 300 DPI.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from point_kinetics_scipy import (
    solve_point_kinetics, step_reactivity, ramp_reactivity,
    BETA, LAMBDA, BETA_TOTAL, LAMBDA_GEN,
)
from point_kinetics_xtfc import XTFC, N0, C0
from point_kinetics_pinn import PINN, predict as pinn_predict, train as pinn_train, device

GRAPHICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphics")
os.makedirs(GRAPHICS_DIR, exist_ok=True)

COLORS_6 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def plot_precursor_comparison():
    """Vertical stack: SciPy reference vs Standard PINN precursors."""
    sol = solve_point_kinetics(t_end=10.0, n_points=1000)
    t = sol.t
    ref_C = sol.y[1:].T

    print("  Training PINN (2000 epochs)...")
    model = PINN(4, 64).to(device)
    pinn_train(model, epochs=2000, verbose=False)
    pinn = pinn_predict(model, t)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharey=True, sharex=True)

    for i in range(6):
        axes[0].plot(t, ref_C[:, i], color=COLORS_6[i], lw=1.5, label=f"Group {i+1}")
    axes[0].set_ylabel("Precursor Concentration", fontsize=12)
    axes[0].set_title("SciPy Radau (Reference)", fontsize=14)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    for i in range(6):
        axes[1].plot(t, pinn["C"][:, i], color=COLORS_6[i], lw=1.5)
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Precursor Concentration", fontsize=12)
    axes[1].set_title("Standard PINN", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "precursor_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: precursor_comparison.png")


def plot_xtfc_precursor_comparison():
    """Vertical stack: SciPy reference vs X-TFC precursors."""
    sol = solve_point_kinetics(t_end=10.0, n_points=1000)
    t = sol.t
    ref_C = sol.y[1:].T

    xtfc = XTFC(n_neurons=100, t_max=10.0, seed=42)
    xtfc.train(n_collocation=500)
    pred = xtfc.predict(t)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharey=True, sharex=True)

    for i in range(6):
        axes[0].plot(t, ref_C[:, i], color=COLORS_6[i], lw=1.5, label=f"Group {i+1}")
    axes[0].set_ylabel("Precursor Concentration", fontsize=12)
    axes[0].set_title("SciPy Radau (Reference)", fontsize=14)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    for i in range(6):
        axes[1].plot(t, pred["C"][:, i], color=COLORS_6[i], lw=1.5)
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Precursor Concentration", fontsize=12)
    axes[1].set_title("X-TFC", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "xtfc_precursor_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: xtfc_precursor_comparison.png")


def plot_pinn_training():
    """Two-panel PINN training loss: total on top, components on bottom."""
    print("  Training PINN (5000 epochs)...")
    model = PINN(4, 64).to(device)
    history = pinn_train(model, epochs=5000, n_collocation=1000,
                         lr=1e-3, lambda_ic=10.0)

    epochs = np.array(history["epoch"])
    phys = np.array(history["loss_physics"])
    ic = np.array(history["loss_ic"])
    total = np.array(history["loss"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].semilogy(epochs, total, "-", color="#2c3e50", lw=2.5)
    axes[0].set_ylabel(r"$\mathcal{L}_{\mathrm{total}}$", fontsize=13)
    axes[0].set_title("PINN Training Loss", fontsize=15)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=11)

    axes[1].semilogy(epochs, ic, "-", color="#d62728", lw=2,
                     label=r"$\mathcal{L}_{\mathrm{IC}}$")
    axes[1].semilogy(epochs, phys, "-", color="#1f77b4", lw=2,
                     label=r"$\mathcal{L}_{\mathrm{phys}}$")
    axes[1].set_xlabel("Epoch", fontsize=13)
    axes[1].set_ylabel("Component Loss", fontsize=13)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "pinn_training.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: pinn_training.png")


def plot_precursor_panels():
    """Six-panel plot showing each precursor group individually."""
    sol = solve_point_kinetics(t_end=10.0, n_points=1000)
    t = sol.t
    half_lives = np.log(2) / LAMBDA

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

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

    fig.suptitle("Delayed Neutron Precursor Groups (SciPy Reference)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "precursor_panels.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: precursor_panels.png")


def plot_ramp_vs_step():
    """Compare reactor response to step vs ramp reactivity insertion."""
    sol_step = solve_point_kinetics(t_end=10.0, n_points=1000,
                                    rho_func=step_reactivity)
    sol_ramp = solve_point_kinetics(t_end=10.0, n_points=1000,
                                    rho_func=ramp_reactivity)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

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

    axes[1].plot(sol_step.t, sol_step.y[0], "-", color="#d62728", lw=2, label="Step response")
    axes[1].plot(sol_ramp.t, sol_ramp.y[0], "--", color="#1f77b4", lw=2, label="Ramp response")
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel(r"Neutron density $n/n_0$", fontsize=12)
    axes[1].set_title("Neutron Density Response", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "ramp_vs_step.png"),
                dpi=300, bbox_inches="tight")
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
        ["Parameters", "~17k (4\u00d764 NN)", "700 (ELM output weights)", "N/A (direct solve)"],
        ["Accuracy", "~90% error", "~0.06% error", "Machine precision"],
        ["Speed", "~60 s (CPU)", "~0.5 s", "~0.01 s"],
    ]

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        table[i + 1, 0].set_facecolor("#ecf0f1")
        table[i + 1, 0].set_text_props(fontweight="bold")
        for j in range(1, len(headers)):
            table[i + 1, j].set_facecolor("#f8f9fa" if i % 2 == 0 else "#ffffff")

    ax.set_title("Method Comparison Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, "method_summary_table.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: method_summary_table.png")


def main():
    print("Generating poster graphics")
    print("=" * 45)

    plot_precursor_comparison()
    plot_xtfc_precursor_comparison()
    plot_pinn_training()
    plot_precursor_panels()
    plot_ramp_vs_step()
    plot_method_summary_table()

    print(f"\nAll graphics saved to {GRAPHICS_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(GRAPHICS_DIR)):
        if f.endswith(".png"):
            size_kb = os.path.getsize(os.path.join(GRAPHICS_DIR, f)) / 1024
            print(f"  {f:<40s} {size_kb:>6.0f} KB")


if __name__ == "__main__":
    main()
