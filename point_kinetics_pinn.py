"""
Standard Physics-Informed Neural Network for Point Kinetics Equations

Baseline PINN using soft constraints on initial conditions.
Serves as comparison for more advanced methods like X-TFC.

Equations:
    dn/dt = [(ρ - β) / Λ] * n + Σ λᵢCᵢ
    dCᵢ/dt = (βᵢ / Λ) * n - λᵢCᵢ  (i = 1..6)

Usage:
    python point_kinetics_pinn.py --epochs 5000 --save model.pt
    python point_kinetics_pinn.py --load model.pt
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nuclear Parameters (U-235, thermal spectrum)
BETA = torch.tensor([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273],
                    dtype=torch.float32, device=device)
LAMBDA = torch.tensor([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01],
                      dtype=torch.float32, device=device)
BETA_TOTAL = torch.sum(BETA)
LAMBDA_GEN = 2e-5  # Prompt neutron generation time (s)
RHO_STEP = 0.003   # Reactivity step magnitude

# Initial conditions (steady state)
N0 = 1.0
C0 = BETA / (LAMBDA * LAMBDA_GEN)


class PINN(nn.Module):
    """Fully-connected neural network for PINN."""
    
    def __init__(self, hidden_layers=4, neurons_per_layer=64):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(1, neurons_per_layer))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
        
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
        y = nn.functional.softplus(y) + 1e-10
        return y


def compute_residuals(model, t, rho_func):
    """Compute physics residuals (ODE error)."""
    t = t.clone().requires_grad_(True)
    y = model(t)
    
    n = y[:, 0:1]
    C = y[:, 1:7]
    
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
    """Compute initial condition loss at t=0."""
    t0 = torch.zeros((1, 1), dtype=torch.float32, device=device)
    y0 = model(t0)
    loss_n = (y0[0, 0] - N0) ** 2
    loss_C = torch.mean((y0[0, 1:7] - C0) ** 2)
    return loss_n + loss_C


def step_reactivity(t):
    """Step reactivity insertion at t=0."""
    return torch.where(t >= 0, RHO_STEP, torch.zeros_like(t))


def train(model, epochs=5000, n_collocation=1000, t_max=10.0, 
          lr=1e-3, lambda_ic=10.0, verbose=True):
    """Train the PINN. Loss = L_physics + λ * L_IC"""
    optimizer = Adam(model.parameters(), lr=lr)
    
    history = {'epoch': [], 'loss': [], 'loss_physics': [], 'loss_ic': []}
    t_colloc = torch.rand((n_collocation, 1), device=device) * t_max
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        res_n, res_C = compute_residuals(model, t_colloc, step_reactivity)
        loss_physics = torch.mean(res_n ** 2) + torch.mean(res_C ** 2)
        loss_ic = compute_ic_loss(model)
        loss = loss_physics + lambda_ic * loss_ic
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['loss_physics'].append(loss_physics.item())
            history['loss_ic'].append(loss_ic.item())
            
            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.2e} | "
                      f"Physics: {loss_physics.item():.2e} | IC: {loss_ic.item():.2e}")
    
    return history


def predict(model, t):
    """Generate predictions from trained model."""
    model.eval()
    if isinstance(t, np.ndarray):
        t = torch.tensor(t, dtype=torch.float32, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(1)
    with torch.no_grad():
        y = model(t)
    return {'n': y[:, 0].cpu().numpy(), 'C': y[:, 1:7].cpu().numpy()}


def plot_training(history, save_path=None):
    """Plot training convergence (3 panels)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = history['epoch']
    
    axes[0].semilogy(epochs, history['loss'], 'b-', lw=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_title('Total Loss')
    axes[0].set_xlim(0, max(epochs))
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(epochs, history['loss_physics'], 'r-', lw=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Physics Loss')
    axes[1].set_xlim(0, max(epochs))
    axes[1].grid(True, alpha=0.3)
    
    axes[2].semilogy(epochs, history['loss_ic'], 'g-', lw=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss (log scale)')
    axes[2].set_title('IC Loss')
    axes[2].set_xlim(0, max(epochs))
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_solution(model, t_max=10.0, save_path=None):
    """Plot precursor concentrations."""
    t = np.linspace(0, t_max, 1000)
    pred = predict(model, t)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    for i in range(6):
        ax.plot(t, pred['C'][:, i], color=colors[i], lw=1.5, label=f'Group {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Precursor Concentration')
    ax.set_title('Standard PINN: Delayed Neutron Precursors')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Standard PINN for Point Kinetics')
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--layers', type=int, default=4, help='Hidden layers')
    parser.add_argument('--neurons', type=int, default=64, help='Neurons per layer')
    parser.add_argument('--collocation', type=int, default=1000, help='Collocation points')
    parser.add_argument('--lambda-ic', type=float, default=10.0, help='IC loss weight')
    parser.add_argument('--save', type=str, default=None, help='Save model path')
    parser.add_argument('--load', type=str, default=None, help='Load model path')
    args = parser.parse_args()
    
    from model_utils import save_model, load_model
    
    if args.load:
        print(f"Loading model from: {args.load}")
        model, history, _ = load_model(args.load, PINN, device)
        plot_training(history)
        plot_solution(model)
        return
    
    print("Standard PINN for Point Kinetics")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"Architecture: {args.layers} layers × {args.neurons} neurons")
    print(f"Training: {args.epochs} epochs, lr={args.lr}")
    print(f"IC weight: λ={args.lambda_ic}")
    print()
    
    model = PINN(args.layers, args.neurons).to(device)
    history = train(model, epochs=args.epochs, n_collocation=args.collocation,
                    lr=args.lr, lambda_ic=args.lambda_ic)
    
    hyperparams = {
        'hidden_layers': args.layers,
        'neurons_per_layer': args.neurons,
        'epochs': args.epochs,
        'lr': args.lr,
        'collocation': args.collocation,
        'lambda_ic': args.lambda_ic
    }
    
    timestamp = datetime.now().strftime("%H%M%S_%d%m%y")
    save_path = args.save or f"pinn_standard_{timestamp}.pt"
    save_model(model, history, save_path, hyperparams)
    
    plot_training(history, f"pinn_training_{timestamp}.png")
    plot_solution(model, save_path=f"pinn_solution_{timestamp}.png")
    
    print(f"\nFinal Loss: {history['loss'][-1]:.2e}")


if __name__ == "__main__":
    main()
