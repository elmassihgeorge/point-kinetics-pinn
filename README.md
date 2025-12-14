# Point Kinetics PINN

Physics-Informed Neural Network for solving the point kinetics equations in nuclear reactor dynamics.

## Equations

```
dn/dt = [(ρ - β) / Λ] × n + Σ λᵢCᵢ
dCᵢ/dt = (βᵢ / Λ) × n - λᵢCᵢ  (i = 1..6)
```

## Project Structure

```
├── point_kinetics_pinn.py    # Standard PINN (PyTorch)
├── point_kinetics_scipy.py   # SciPy benchmark solver
├── model_utils.py            # Save/load utilities
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Train a new model:**
```bash
python point_kinetics_pinn.py --epochs 5000 --save model.pt
```

**Load and visualize:**
```bash
python point_kinetics_pinn.py --load model.pt
```

**Run SciPy benchmark:**
```bash
python point_kinetics_scipy.py
```

## Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 5000 | Training epochs |
| `--lr` | 0.001 | Learning rate |
| `--layers` | 4 | Hidden layers |
| `--neurons` | 64 | Neurons per layer |
| `--lambda-ic` | 10 | IC loss weight |

## References

- Keepin, G.R. "Physics of Nuclear Kinetics" (1965)
- Raissi, M. et al. "Physics-informed neural networks" (2019)
- Schiassi, E. et al. "Physics-informed neural networks for the point kinetics equations" (2022)

## License

MIT