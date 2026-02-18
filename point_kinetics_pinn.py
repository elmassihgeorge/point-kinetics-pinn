"""
Standard Physics-Informed Neural Network for Point Kinetics Equations

Baseline PINN using soft constraints on initial conditions.
Serves as comparison for the X-TFC method.

Equations:
    dn/dt = [(rho - beta) / Lambda] * n + sum_i lambda_i * C_i
    dCi/dt = (beta_i / Lambda) * n - lambda_i * C_i  (i = 1..6)

Usage:
    python point_kinetics_pinn.py --epochs 5000 --save model.pt
    python point_kinetics_pinn.py --load model.pt
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Nuclear parameters — Keepin (1957), U-235 thermal fission, 6 groups
# ---------------------------------------------------------------------------
BETA = torch.tensor([0.000221, 0.001467, 0.001313, 0.002647, 0.000771, 0.000281],
                    dtype=torch.float32, device=device)
LAMBDA = torch.tensor([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01],
                      dtype=torch.float32, device=device)
BETA_TOTAL = torch.sum(BETA)
LAMBDA_GEN = 2e-5
RHO_STEP = 0.003

# Steady-state initial conditions
N0 = 1.0
C0 = BETA / (LAMBDA * LAMBDA_GEN)

GRAPHICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphics")


class PINN(nn.Module):
    """Fully-connected neural network for the standard PINN approach."""

    def __init__(self, hidden_layers=4, neurons_per_layer=64):
        super().__init__()
        layers = [nn.Linear(1, neurons_per_layer), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons_per_layer, neurons_per_layer), nn.Tanh()]
        layers.append(nn.Linear(neurons_per_layer, 7))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        t_norm = t / 10.0
        y = self.network(t_norm)
        return nn.functional.softplus(y) + 1e-10


def compute_residuals(model, t, rho_func):
    """Compute physics residuals (how well the NN satisfies the ODE)."""
    t = t.clone().requires_grad_(True)
    y = model(t)
    n, C = y[:, 0:1], y[:, 1:7]

    dn_dt = torch.autograd.grad(n, t, torch.ones_like(n), create_graph=True)[0]
    dC_dt = torch.zeros_like(C)
    for i in range(6):
        dC_dt[:, i:i+1] = torch.autograd.grad(
            C[:, i:i+1], t, torch.ones_like(C[:, i:i+1]), create_graph=True
        )[0]

    rho = rho_func(t)
    dn_dt_ode = ((rho - BETA_TOTAL) / LAMBDA_GEN) * n + torch.sum(LAMBDA * C, dim=1, keepdim=True)
    dC_dt_ode = (BETA / LAMBDA_GEN) * n - LAMBDA * C

    return dn_dt - dn_dt_ode, dC_dt - dC_dt_ode


def compute_ic_loss(model):
    """Initial condition loss at t=0 (soft constraint)."""
    t0 = torch.zeros((1, 1), dtype=torch.float32, device=device)
    y0 = model(t0)
    return (y0[0, 0] - N0) ** 2 + torch.mean((y0[0, 1:7] - C0) ** 2)


def step_reactivity(t):
    """Step reactivity insertion at t=0."""
    return torch.where(t >= 0, RHO_STEP, torch.zeros_like(t))


def train(model, epochs=5000, n_collocation=1000, t_max=10.0,
          lr=1e-3, lambda_ic=10.0, verbose=True):
    """Train the PINN.  Loss = L_physics + lambda * L_IC."""
    optimizer = Adam(model.parameters(), lr=lr)
    history = {"epoch": [], "loss": [], "loss_physics": [], "loss_ic": []}
    t_colloc = torch.rand((n_collocation, 1), device=device) * t_max

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        res_n, res_C = compute_residuals(model, t_colloc, step_reactivity)
        loss_phys = torch.mean(res_n ** 2) + torch.mean(res_C ** 2)
        loss_ic = compute_ic_loss(model)
        loss = loss_phys + lambda_ic * loss_ic

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["loss_physics"].append(loss_phys.item())
            history["loss_ic"].append(loss_ic.item())
            if verbose and epoch % 500 == 0:
                print(f"  Epoch {epoch:5d} | Loss {loss.item():.2e} | "
                      f"Phys {loss_phys.item():.2e} | IC {loss_ic.item():.2e}")

    return history


def predict(model, t):
    """Generate predictions from a trained model."""
    model.eval()
    if isinstance(t, np.ndarray):
        t = torch.tensor(t, dtype=torch.float32, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(1)
    with torch.no_grad():
        y = model(t)
    return {"n": y[:, 0].cpu().numpy(), "C": y[:, 1:7].cpu().numpy()}


def plot_training(history, save_path=None):
    """Plot training convergence."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = history["epoch"]
    ax.semilogy(epochs, history["loss_physics"], "-", color="#1f77b4", lw=2,
                label=r"$\mathcal{L}_{\mathrm{phys}}$")
    ax.semilogy(epochs, history["loss_ic"], "-", color="#d62728", lw=2,
                label=r"$\mathcal{L}_{\mathrm{IC}}$")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (log scale)", fontsize=12)
    ax.set_title("Standard PINN Training Convergence", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_solution(model, t_max=10.0, save_path=None):
    """Plot neutron density and precursor concentrations."""
    t = np.linspace(0, t_max, 1000)
    pred = predict(model, t)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Neutron density
    axes[0].plot(t, pred["n"], "b-", lw=2)
    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel(r"Neutron density $n/n_0$", fontsize=12)
    axes[0].set_title("Standard PINN: Neutron Density", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Precursors
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i in range(6):
        axes[1].plot(t, pred["C"][:, i], color=colors[i], lw=1.5, label=f"Group {i+1}")
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Precursor Concentration", fontsize=12)
    axes[1].set_title("Standard PINN: Delayed Neutron Precursors", fontsize=14)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Standard PINN for Point Kinetics")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--collocation", type=int, default=1000)
    parser.add_argument("--lambda-ic", type=float, default=10.0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    from model_utils import save_model, load_model

    if args.load:
        print(f"Loading model from: {args.load}")
        model, history, _ = load_model(args.load, PINN, device)
        os.makedirs(GRAPHICS_DIR, exist_ok=True)
        plot_training(history, os.path.join(GRAPHICS_DIR, "pinn_training.png"))
        plot_solution(model, save_path=os.path.join(GRAPHICS_DIR, "pinn_solution.png"))
        return

    print("Standard PINN for Point Kinetics")
    print("=" * 45)
    print(f"  Device       : {device}")
    print(f"  Architecture : {args.layers} layers x {args.neurons} neurons")
    print(f"  Training     : {args.epochs} epochs, lr={args.lr}")
    print(f"  IC weight    : lambda={args.lambda_ic}")
    print()

    model = PINN(args.layers, args.neurons).to(device)
    history = train(model, epochs=args.epochs, n_collocation=args.collocation,
                    lr=args.lr, lambda_ic=args.lambda_ic)

    hyperparams = {
        "hidden_layers": args.layers,
        "neurons_per_layer": args.neurons,
        "epochs": args.epochs,
        "lr": args.lr,
        "collocation": args.collocation,
        "lambda_ic": args.lambda_ic,
    }

    save_path = args.save or "pinn_standard.pt"
    save_model(model, history, save_path, hyperparams)

    os.makedirs(GRAPHICS_DIR, exist_ok=True)
    plot_training(history, os.path.join(GRAPHICS_DIR, "pinn_training.png"))
    plot_solution(model, save_path=os.path.join(GRAPHICS_DIR, "pinn_solution.png"))

    print(f"\n  Final loss: {history['loss'][-1]:.2e}")


if __name__ == "__main__":
    main()
