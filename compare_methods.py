"""
Compare PINN vs X-TFC vs SciPy Solutions

Generates comparison plots and error metrics for all three methods.
Outputs are saved to the graphics/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from point_kinetics_scipy import solve_point_kinetics
from point_kinetics_xtfc import XTFC
from point_kinetics_pinn import PINN, predict as pinn_predict, train as pinn_train, device

GRAPHICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphics")


def get_scipy_reference(t_max=10.0, n_points=1000):
    solution = solve_point_kinetics(t_end=t_max, n_points=n_points)
    return {"t": solution.t, "n": solution.y[0], "C": solution.y[1:].T}


def get_xtfc_solution(t, n_neurons=100):
    model = XTFC(n_neurons=n_neurons, t_max=t[-1], seed=42)
    model.train(n_collocation=500)
    return model.predict(t)


def get_pinn_solution(t, model_path=None):
    from model_utils import load_model
    if model_path:
        model, _, _ = load_model(model_path, PINN, device)
    else:
        print("  Training quick PINN (2000 epochs)...")
        model = PINN(4, 64).to(device)
        pinn_train(model, epochs=2000, verbose=False)
    return pinn_predict(model, t)


def compute_errors(pred, ref):
    """Absolute and relative errors on precursor concentrations."""
    C_err = np.abs(pred["C"] - ref["C"])
    C_rel = C_err / (np.abs(ref["C"]) + 1e-10) * 100
    return {
        "abs_error": C_err,
        "rel_error": C_rel,
        "max_abs": np.max(C_err),
        "max_rel_pct": np.max(C_rel),
        "mean_rel_pct": np.mean(C_rel),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_neutron_comparison(t, scipy_ref, pinn_pred, xtfc_pred, save_path=None):
    """Overlay neutron density from all three methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, scipy_ref["n"], "k-", lw=2.5, label="SciPy Radau (reference)")
    ax.plot(t, xtfc_pred["n"], "--", color="#1f77b4", lw=2, label="X-TFC")
    ax.plot(t, pinn_pred["n"], ":", color="#d62728", lw=2, label="Standard PINN")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"Neutron density $n/n_0$", fontsize=12)
    ax.set_title("Neutron Density — Method Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_precursor_comparison(t, scipy_ref, pinn_pred, xtfc_pred, save_path=None):
    """Side-by-side precursor plots for all three methods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for ax, data, title in zip(
        axes,
        [scipy_ref, pinn_pred, xtfc_pred],
        ["SciPy Radau (Reference)", "Standard PINN", "X-TFC"],
    ):
        for i in range(6):
            ax.plot(t, data["C"][:, i], color=colors[i], lw=1.5, label=f"Group {i+1}")
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Precursor Concentration", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_error_comparison(pinn_errors, xtfc_errors, save_path=None):
    """Bar chart comparing max relative error by precursor group."""
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = np.arange(1, 7)
    width = 0.35

    pinn_by_group = np.max(pinn_errors["rel_error"], axis=0)
    xtfc_by_group = np.max(xtfc_errors["rel_error"], axis=0)

    ax.bar(groups - width / 2, pinn_by_group, width,
           label="Standard PINN", color="#d62728", alpha=0.8)
    ax.bar(groups + width / 2, xtfc_by_group, width,
           label="X-TFC", color="#1f77b4", alpha=0.8)

    ax.set_yscale("log")
    ax.set_xlabel("Precursor Group", fontsize=12)
    ax.set_ylabel("Max Relative Error (%)", fontsize=12)
    ax.set_title("Error Comparison by Precursor Group", fontsize=14)
    ax.set_xticks(groups)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_error_over_time(t, pinn_errors, xtfc_errors, save_path=None):
    """Mean relative error across all precursor groups vs time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pinn_mean = np.mean(pinn_errors["rel_error"], axis=1)
    xtfc_mean = np.mean(xtfc_errors["rel_error"], axis=1)

    ax.semilogy(t, pinn_mean, "-", color="#d62728", lw=2, label="Standard PINN")
    ax.semilogy(t, xtfc_mean, "-", color="#1f77b4", lw=2, label="X-TFC")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Mean Relative Error (%)", fontsize=12)
    ax.set_title("Error Evolution Over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_neutron_error(t, scipy_ref, pinn_pred, xtfc_pred, save_path=None):
    """Absolute error in neutron density for both methods."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pinn_err = np.abs(pinn_pred["n"] - scipy_ref["n"])
    xtfc_err = np.abs(xtfc_pred["n"] - scipy_ref["n"])

    ax.semilogy(t, pinn_err, "-", color="#d62728", lw=2, label="Standard PINN")
    ax.semilogy(t, xtfc_err, "-", color="#1f77b4", lw=2, label="X-TFC")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"$|n_{\mathrm{pred}} - n_{\mathrm{ref}}|$", fontsize=12)
    ax.set_title("Neutron Density — Absolute Error", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def print_summary(pinn_errors, xtfc_errors):
    print("\n" + "=" * 55)
    print("ACCURACY COMPARISON (vs SciPy reference)")
    print("=" * 55)
    print(f"{'Metric':<30} {'PINN':>10} {'X-TFC':>10}")
    print("-" * 55)
    print(f"{'Max Relative Error (%)':<30} {pinn_errors['max_rel_pct']:>10.2f} {xtfc_errors['max_rel_pct']:>10.4f}")
    print(f"{'Mean Relative Error (%)':<30} {pinn_errors['mean_rel_pct']:>10.2f} {xtfc_errors['mean_rel_pct']:>10.4f}")
    print(f"{'Max Absolute Error':<30} {pinn_errors['max_abs']:>10.2f} {xtfc_errors['max_abs']:>10.4f}")
    print("=" * 55)
    improvement = pinn_errors["max_rel_pct"] / max(xtfc_errors["max_rel_pct"], 1e-12)
    print(f"\n  X-TFC is ~{improvement:.0f}x more accurate than standard PINN")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Comparing PINN vs X-TFC vs SciPy")
    print("=" * 45)
    os.makedirs(GRAPHICS_DIR, exist_ok=True)

    print("\n1. SciPy reference ...")
    scipy_ref = get_scipy_reference()
    t = scipy_ref["t"]

    print("\n2. X-TFC ...")
    xtfc_pred = get_xtfc_solution(t)

    print("\n3. Standard PINN ...")
    pinn_pred = get_pinn_solution(t)

    print("\n4. Computing errors ...")
    xtfc_errors = compute_errors(xtfc_pred, scipy_ref)
    pinn_errors = compute_errors(pinn_pred, scipy_ref)
    print_summary(pinn_errors, xtfc_errors)

    print("\n5. Generating graphics ...")
    gfx = GRAPHICS_DIR
    plot_neutron_comparison(t, scipy_ref, pinn_pred, xtfc_pred,
                            os.path.join(gfx, "neutron_comparison.png"))
    plot_precursor_comparison(t, scipy_ref, pinn_pred, xtfc_pred,
                              os.path.join(gfx, "precursor_comparison.png"))
    plot_error_comparison(pinn_errors, xtfc_errors,
                          os.path.join(gfx, "error_by_group.png"))
    plot_error_over_time(t, pinn_errors, xtfc_errors,
                         os.path.join(gfx, "error_over_time.png"))
    plot_neutron_error(t, scipy_ref, pinn_pred, xtfc_pred,
                       os.path.join(gfx, "neutron_error.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
