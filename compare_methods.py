"""
Compare PINN vs X-TFC vs SciPy Solutions

Generates comparison plots and error metrics for all three methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch

from point_kinetics_scipy import solve_point_kinetics
from point_kinetics_xtfc import XTFC
from point_kinetics_pinn import PINN, predict as pinn_predict, device


def get_scipy_reference(t_max=10.0, n_points=1000):
    """Get SciPy reference solution."""
    solution = solve_point_kinetics(t_end=t_max, n_points=n_points)
    return {
        't': solution.t,
        'n': solution.y[0],
        'C': solution.y[1:].T
    }


def get_xtfc_solution(t, n_neurons=100):
    """Train and evaluate X-TFC model."""
    model = XTFC(n_neurons=n_neurons, t_max=t[-1])
    model.train(n_collocation=500)
    return model.predict(t)


def get_pinn_solution(t, model_path=None):
    """Load PINN model and evaluate, or train a quick one."""
    from model_utils import load_model
    
    if model_path:
        model, _, _ = load_model(model_path, PINN, device)
    else:
        # Train a quick PINN for comparison
        print("  Training quick PINN (2000 epochs)...")
        model = PINN(4, 64).to(device)
        from point_kinetics_pinn import train
        train(model, epochs=2000, verbose=False)
    
    return pinn_predict(model, t)


def compute_errors(pred, ref):
    """Compute error metrics between prediction and reference."""
    C_error = np.abs(pred['C'] - ref['C'])
    C_rel_error = C_error / (np.abs(ref['C']) + 1e-10) * 100
    
    return {
        'abs_error': C_error,
        'rel_error': C_rel_error,
        'max_abs': np.max(C_error),
        'max_rel_pct': np.max(C_rel_error),
        'mean_rel_pct': np.mean(C_rel_error)
    }


def plot_solutions(t, scipy_ref, pinn_pred, xtfc_pred, save_path=None):
    """Plot 3 separate panels: SciPy, PINN, X-TFC."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    
    # Panel 1: SciPy (reference)
    ax = axes[0]
    for i in range(6):
        ax.plot(t, scipy_ref['C'][:, i], color=colors[i], lw=1.5, label=f'Group {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Precursor Concentration')
    ax.set_title('SciPy (Reference)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')
    
    # Panel 2: Standard PINN
    ax = axes[1]
    for i in range(6):
        ax.plot(t, pinn_pred['C'][:, i], color=colors[i], lw=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Precursor Concentration')
    ax.set_title('Standard PINN')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: X-TFC
    ax = axes[2]
    for i in range(6):
        ax.plot(t, xtfc_pred['C'][:, i], color=colors[i], lw=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Precursor Concentration')
    ax.set_title('X-TFC')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_error_comparison(pinn_errors, xtfc_errors, save_path=None):
    """Bar chart comparing errors by precursor group."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    groups = np.arange(1, 7)
    width = 0.35
    
    pinn_by_group = np.max(pinn_errors['rel_error'], axis=0)
    xtfc_by_group = np.max(xtfc_errors['rel_error'], axis=0)
    
    bars1 = ax.bar(groups - width/2, pinn_by_group, width, label='Standard PINN', color='#d62728', alpha=0.8)
    bars2 = ax.bar(groups + width/2, xtfc_by_group, width, label='X-TFC', color='#1f77b4', alpha=0.8)
    
    ax.set_yscale('log')
    ax.set_xlabel('Precursor Group', fontsize=12)
    ax.set_ylabel('Max Relative Error (%)', fontsize=12)
    ax.set_title('Error Comparison by Precursor Group', fontsize=14)
    ax.set_xticks(groups)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def print_summary(xtfc_errors, pinn_errors):
    """Print summary comparison."""
    print("\n" + "=" * 50)
    print("ACCURACY COMPARISON (vs SciPy reference)")
    print("=" * 50)
    print(f"{'Metric':<30} {'PINN':>10} {'X-TFC':>10}")
    print("-" * 50)
    print(f"{'Max Relative Error (%)':<30} {pinn_errors['max_rel_pct']:>10.2f} {xtfc_errors['max_rel_pct']:>10.4f}")
    print(f"{'Mean Relative Error (%)':<30} {pinn_errors['mean_rel_pct']:>10.2f} {xtfc_errors['mean_rel_pct']:>10.4f}")
    print(f"{'Max Absolute Error':<30} {pinn_errors['max_abs']:>10.2f} {xtfc_errors['max_abs']:>10.4f}")
    print("=" * 50)
    
    improvement = pinn_errors['max_rel_pct'] / xtfc_errors['max_rel_pct']
    print(f"\nX-TFC is {improvement:.0f}x more accurate than standard PINN")


def main():
    print("Comparing PINN vs X-TFC vs SciPy")
    print("=" * 40)
    
    # Get reference solution
    print("\n1. Generating SciPy reference...")
    scipy_ref = get_scipy_reference()
    t = scipy_ref['t']
    
    # Train and evaluate X-TFC
    print("\n2. Training X-TFC...")
    xtfc_pred = get_xtfc_solution(t)
    
    # Train and evaluate PINN
    print("\n3. Training PINN...")
    pinn_pred = get_pinn_solution(t)
    
    # Compute errors
    print("\n4. Computing errors...")
    xtfc_errors = compute_errors(xtfc_pred, scipy_ref)
    pinn_errors = compute_errors(pinn_pred, scipy_ref)
    
    # Print summary
    print_summary(xtfc_errors, pinn_errors)
    
    # Generate plots
    timestamp = datetime.now().strftime("%H%M%S_%d%m%y")
    plot_solutions(t, scipy_ref, pinn_pred, xtfc_pred, 
                   save_path=f"solutions_{timestamp}.png")
    plot_error_comparison(pinn_errors, xtfc_errors,
                         save_path=f"errors_{timestamp}.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

