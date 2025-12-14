# Point Kinetics PINN

Physics-Informed Neural Networks for solving the point kinetics equations in nuclear reactor dynamics.

## Methods

| Method | IC Handling | Training | Accuracy |
|--------|-------------|----------|----------|
| Standard PINN | Soft (penalty) | Gradient descent | ~70-90% error |
| **X-TFC** | **Hard (exact)** | **Least-squares** | **0.06% error** |

## Equations

```
dn/dt = [(ρ - β) / Λ] × n + Σ λᵢCᵢ
dCᵢ/dt = (βᵢ / Λ) × n - λᵢCᵢ  (i = 1..6)
```

## Project Structure

```
├── point_kinetics_pinn.py    # Standard PINN (PyTorch)
├── point_kinetics_xtfc.py    # X-TFC (NumPy)
├── point_kinetics_scipy.py   # SciPy benchmark
├── compare_methods.py        # Comparison visualization
├── model_utils.py            # Save/load utilities
└── requirements.txt
```

## Usage

**Standard PINN:**
```bash
python point_kinetics_pinn.py --epochs 5000 --save model.pt
```

**X-TFC (recommended):**
```bash
python point_kinetics_xtfc.py --neurons 100
```

**Compare methods:**
```bash
python compare_methods.py
```

## References

- Keepin, G.R. "Physics of Nuclear Kinetics" (1965)
- Raissi, M. et al. "Physics-informed neural networks" (2019)
- Schiassi, E. et al. "Physics-informed neural networks for the point kinetics equations" (2022)

## License

MIT