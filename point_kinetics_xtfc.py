"""
X-TFC: Extreme Theory of Functional Connections for Point Kinetics

Hard-constrains initial conditions via TFC and trains a single-layer
Extreme Learning Machine (ELM) with least-squares — no gradient descent.

Key advantages over standard PINNs:
    - ICs satisfied *exactly* (not via penalty term)
    - Training via linear least-squares (fast, deterministic)
    - Orders-of-magnitude better accuracy

Reference:
    Schiassi et al., "Physics-informed neural networks for the
    point kinetics equations" (2022)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

# ---------------------------------------------------------------------------
# Nuclear parameters — Keepin (1957), U-235 thermal fission, 6 groups
# ---------------------------------------------------------------------------
BETA = np.array([0.000221, 0.001467, 0.001313, 0.002647, 0.000771, 0.000281])
LAMBDA = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
BETA_TOTAL = np.sum(BETA)
LAMBDA_GEN = 2e-5
RHO_STEP = 0.003

N0 = 1.0
C0 = BETA / (LAMBDA * LAMBDA_GEN)

GRAPHICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphics")


class XTFC:
    """Extreme Theory of Functional Connections solver.

    Constrained expression:  y(t) = y0 + t * g(t)
    where g(t) is an ELM with random fixed hidden weights.
    """

    def __init__(self, n_neurons=100, t_max=10.0, seed=None):
        self.n_neurons = n_neurons
        self.t_max = t_max
        self.n_outputs = 7  # n + 6 precursors

        rng = np.random.default_rng(seed)
        self.W_hidden = rng.standard_normal((1, n_neurons)) * 2.0
        self.b_hidden = rng.standard_normal(n_neurons) * 2.0
        self.W_out = None
        self.residual_norm = None

    # -- hidden layer ---------------------------------------------------------
    def _activation(self, t):
        z = t.reshape(-1, 1) @ self.W_hidden + self.b_hidden
        return np.tanh(z)

    def _activation_derivative(self, t):
        z = t.reshape(-1, 1) @ self.W_hidden + self.b_hidden
        return self.W_hidden * (1 - np.tanh(z) ** 2)

    # -- TFC constrained expression ------------------------------------------
    def _tfc_expression(self, t, g):
        """y(t) = y0 + t * g(t)  =>  y(0) = y0 exactly."""
        y0 = np.concatenate([[N0], C0])
        return y0 + t.reshape(-1, 1) * g

    def _compute_g(self, t):
        return self._activation(t) @ self.W_out

    def _compute_g_derivative(self, t):
        return self._activation_derivative(t) @ self.W_out

    # -- training -------------------------------------------------------------
    def train(self, n_collocation=1000):
        """Assemble and solve the linearised physics system via lstsq."""
        t = np.linspace(0.01, self.t_max, n_collocation)
        H = self._activation(t)
        dH = self._activation_derivative(t)

        N = len(t)
        n_neu = self.n_neurons
        n_out = self.n_outputs

        A = np.zeros((N * n_out, n_neu * n_out))
        b = np.zeros(N * n_out)
        y0 = np.concatenate([[N0], C0])

        for i, ti in enumerate(t):
            Hi = H[i]
            dHi = dH[i]
            rho = RHO_STEP if ti >= 0 else 0.0
            row = i * n_out

            # Neutron equation
            A[row, :n_neu] = (Hi + ti * dHi
                              - ((rho - BETA_TOTAL) / LAMBDA_GEN) * ti * Hi)
            for j in range(6):
                A[row, (j+1)*n_neu:(j+2)*n_neu] = -LAMBDA[j] * ti * Hi
            b[row] = ((rho - BETA_TOTAL) / LAMBDA_GEN) * N0 + np.sum(LAMBDA * C0)

            # Precursor equations
            for j in range(6):
                r = row + j + 1
                A[r, (j+1)*n_neu:(j+2)*n_neu] = Hi + ti * dHi + LAMBDA[j] * ti * Hi
                A[r, :n_neu] = -(BETA[j] / LAMBDA_GEN) * ti * Hi
                b[r] = (BETA[j] / LAMBDA_GEN) * N0 - LAMBDA[j] * C0[j]

        W_flat, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        self.W_out = W_flat.reshape(n_neu, n_out, order="F")
        self.residual_norm = np.linalg.norm(A @ W_flat - b) / len(b)

        print(f"  X-TFC training complete")
        print(f"    Neurons         : {self.n_neurons}")
        print(f"    Collocation pts : {n_collocation}")
        print(f"    Residual norm   : {self.residual_norm:.2e}")

    # -- prediction -----------------------------------------------------------
    def predict(self, t):
        if self.W_out is None:
            raise ValueError("Model not trained — call train() first.")
        t = np.asarray(t)
        g = self._compute_g(t)
        y = self._tfc_expression(t, g)
        return {"n": y[:, 0], "C": y[:, 1:7]}

    def verify_ic(self):
        pred = self.predict(np.array([0.0]))
        err_n = abs(pred["n"][0] - N0)
        err_C = np.max(np.abs(pred["C"][0] - C0))
        print(f"  IC verification — n(0) err: {err_n:.2e}, max C(0) err: {err_C:.2e}")
        return err_n, err_C


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_solution(model, t_max=10.0, save_path=None):
    t = np.linspace(0, t_max, 1000)
    pred = model.predict(t)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(t, pred["n"], "b-", lw=2)
    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel(r"Neutron density $n/n_0$", fontsize=12)
    axes[0].set_title("X-TFC: Neutron Density", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i in range(6):
        axes[1].plot(t, pred["C"][:, i], color=colors[i], lw=1.5, label=f"Group {i+1}")
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Precursor Concentration", fontsize=12)
    axes[1].set_title("X-TFC: Delayed Neutron Precursors", fontsize=14)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="X-TFC for Point Kinetics")
    parser.add_argument("--neurons", type=int, default=100)
    parser.add_argument("--collocation", type=int, default=1000)
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("X-TFC for Point Kinetics")
    print("=" * 45)

    model = XTFC(n_neurons=args.neurons, t_max=args.t_max, seed=args.seed)
    model.train(n_collocation=args.collocation)
    model.verify_ic()

    os.makedirs(GRAPHICS_DIR, exist_ok=True)
    plot_solution(model, t_max=args.t_max,
                  save_path=os.path.join(GRAPHICS_DIR, "xtfc_solution.png"))


if __name__ == "__main__":
    main()
