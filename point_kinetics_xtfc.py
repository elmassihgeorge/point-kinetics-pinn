"""
X-TFC: Extreme Theory of Functional Connections for Point Kinetics

Uses hard constraints on initial conditions via TFC and
Extreme Learning Machines (ELM) with least-squares training.

Key features:
    - ICs satisfied exactly (not via penalty)
    - Single-layer neural network with random hidden weights
    - Training via least-squares (no gradient descent)

Reference:
    Schiassi et al. "Physics-informed neural networks for the 
    point kinetics equations" (2022)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Nuclear Parameters (U-235, thermal spectrum)
BETA = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
LAMBDA = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
BETA_TOTAL = np.sum(BETA)
LAMBDA_GEN = 2e-5  # Prompt neutron generation time (s)
RHO_STEP = 0.003   # Reactivity step magnitude

# Initial conditions (steady state)
N0 = 1.0
C0 = BETA / (LAMBDA * LAMBDA_GEN)


class XTFC:
    """
    Extreme Theory of Functional Connections solver.
    
    Uses TFC constrained expression: y(t) = y0 + t * g(t)
    where g(t) is an ELM (Extreme Learning Machine).
    """
    
    def __init__(self, n_neurons=100, t_max=10.0):
        self.n_neurons = n_neurons
        self.t_max = t_max
        self.n_outputs = 7  # n + 6 precursors
        
        # ELM: random hidden layer weights (fixed, never trained)
        self.W_hidden = np.random.randn(1, n_neurons) * 2.0
        self.b_hidden = np.random.randn(n_neurons) * 2.0
        
        # Output weights (trained via least-squares)
        self.W_out = None
        
        # Training history
        self.residual_norm = None
    
    def _activation(self, t):
        """Compute hidden layer activation: tanh(W*t + b)"""
        t = t.reshape(-1, 1)
        z = t @ self.W_hidden + self.b_hidden
        return np.tanh(z)
    
    def _activation_derivative(self, t):
        """Derivative of tanh activation: 1 - tanh^2"""
        t = t.reshape(-1, 1)
        z = t @ self.W_hidden + self.b_hidden
        tanh_z = np.tanh(z)
        # d/dt[tanh(W*t + b)] = W * (1 - tanh^2)
        return self.W_hidden * (1 - tanh_z**2)
    
    def _tfc_expression(self, t, g):
        """
        TFC constrained expression: y(t) = y0 + t * g(t)
        
        Guarantees y(0) = y0 exactly.
        """
        t = t.reshape(-1, 1)
        y0 = np.concatenate([[N0], C0])  # (7,)
        return y0 + t * g
    
    def _compute_g(self, t):
        """Compute free function g(t) = activation(t) @ W_out"""
        H = self._activation(t)  # (N, n_neurons)
        return H @ self.W_out    # (N, 7)
    
    def _compute_g_derivative(self, t):
        """Compute dg/dt"""
        dH = self._activation_derivative(t)  # (N, n_neurons)
        return dH @ self.W_out  # (N, 7)
    
    def train(self, n_collocation=1000):
        """
        Train via least-squares.
        
        Solve: A @ W_out = b
        where A encodes the physics residual equations.
        """
        # Collocation points (exclude t=0 to avoid division issues)
        t = np.linspace(0.01, self.t_max, n_collocation)
        t_col = t.reshape(-1, 1)
        
        # Activation and its derivative
        H = self._activation(t)         # (N, n_neurons)
        dH = self._activation_derivative(t)  # (N, n_neurons)
        
        # For y = y0 + t*g, we have dy/dt = g + t*dg/dt
        # We need to find W_out such that dy/dt = f(y) (the ODE RHS)
        
        # Build the linear system for each output
        # This is a coupled system, solve jointly
        
        N = len(t)
        n_neurons = self.n_neurons
        n_out = self.n_outputs
        
        # Large matrix for all equations
        A = np.zeros((N * n_out, n_neurons * n_out))
        b = np.zeros(N * n_out)
        
        y0 = np.concatenate([[N0], C0])
        
        for i, ti in enumerate(t):
            # Current state from constrained expression
            # y = y0 + t * H @ W_out (need to iterate or linearize)
            
            # For linear solve, use collocation approach:
            # dy/dt - f(y) = 0
            # (g + t*dg/dt) - f(y0 + t*g) = 0
            
            Hi = H[i:i+1, :]    # (1, n_neurons)
            dHi = dH[i:i+1, :]  # (1, n_neurons)
            
            rho = RHO_STEP if ti >= 0 else 0.0
            
            # For neutron equation: row 0
            # dy[0]/dt = (rho-beta)/Lambda * y[0] + sum(lambda_j * y[j+1])
            # y[0] = N0 + t * (H @ W_out[:, 0])
            # dy[0]/dt = H @ W_out[:, 0] + t * dH @ W_out[:, 0]
            
            # LHS coefficient for W_out[:, 0] from dy/dt term
            row = i * n_out
            
            # Neutron equation
            A[row, 0:n_neurons] = Hi.flatten() + ti * dHi.flatten()  # dy/dt term
            A[row, 0:n_neurons] -= ((rho - BETA_TOTAL) / LAMBDA_GEN) * ti * Hi.flatten()  # -(rho-beta)/L * t*g term
            for j in range(6):
                A[row, (j+1)*n_neurons:(j+2)*n_neurons] -= LAMBDA[j] * ti * Hi.flatten()
            b[row] = ((rho - BETA_TOTAL) / LAMBDA_GEN) * N0 + np.sum(LAMBDA * C0)
            
            # Precursor equations
            for j in range(6):
                row = i * n_out + j + 1
                A[row, (j+1)*n_neurons:(j+2)*n_neurons] = Hi.flatten() + ti * dHi.flatten()
                A[row, (j+1)*n_neurons:(j+2)*n_neurons] += LAMBDA[j] * ti * Hi.flatten()
                A[row, 0:n_neurons] -= (BETA[j] / LAMBDA_GEN) * ti * Hi.flatten()
                b[row] = (BETA[j] / LAMBDA_GEN) * N0 - LAMBDA[j] * C0[j]
        
        # Solve least-squares
        W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Reshape to (n_neurons, n_outputs)
        self.W_out = W_flat.reshape(n_neurons, n_out, order='F')
        
        # Compute actual residual norm
        self.residual_norm = np.linalg.norm(A @ W_flat - b) / len(b)
        
        print(f"X-TFC Training complete")
        print(f"  Neurons: {self.n_neurons}")
        print(f"  Collocation points: {n_collocation}")
        print(f"  Residual norm: {self.residual_norm:.2e}")
    
    def predict(self, t):
        """Generate predictions at time points t."""
        if self.W_out is None:
            raise ValueError("Model not trained. Call train() first.")
        
        t = np.asarray(t)
        g = self._compute_g(t)
        y = self._tfc_expression(t, g)
        
        return {'n': y[:, 0], 'C': y[:, 1:7]}
    
    def verify_ic(self):
        """Verify initial conditions are satisfied exactly."""
        pred = self.predict(np.array([0.0]))
        ic_error_n = abs(pred['n'][0] - N0)
        ic_error_C = np.max(np.abs(pred['C'][0] - C0))
        
        print(f"IC Verification:")
        print(f"  n(0) error: {ic_error_n:.2e}")
        print(f"  C(0) max error: {ic_error_C:.2e}")
        
        return ic_error_n, ic_error_C


def plot_solution(model, t_max=10.0, save_path=None):
    """Plot X-TFC solution."""
    t = np.linspace(0, t_max, 1000)
    pred = model.predict(t)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    for i in range(6):
        ax.plot(t, pred['C'][:, i], color=colors[i], lw=1.5, label=f'Group {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Precursor Concentration')
    ax.set_title('X-TFC: Delayed Neutron Precursors')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='X-TFC for Point Kinetics')
    parser.add_argument('--neurons', type=int, default=100, help='ELM neurons')
    parser.add_argument('--collocation', type=int, default=1000, help='Collocation points')
    parser.add_argument('--t-max', type=float, default=10.0, help='Simulation time')
    args = parser.parse_args()
    
    print("X-TFC for Point Kinetics")
    print("=" * 40)
    
    model = XTFC(n_neurons=args.neurons, t_max=args.t_max)
    model.train(n_collocation=args.collocation)
    model.verify_ic()
    
    timestamp = datetime.now().strftime("%H%M%S_%d%m%y")
    plot_solution(model, t_max=args.t_max, save_path=f"xtfc_solution_{timestamp}.png")


if __name__ == "__main__":
    main()
