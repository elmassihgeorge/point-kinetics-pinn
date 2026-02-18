# Point Kinetics PINN

Physics-Informed Neural Networks for solving the point kinetics equations in nuclear reactor dynamics. Compares a standard PINN (soft IC constraints, gradient descent) against X-TFC (hard IC constraints, least-squares) with a SciPy Radau benchmark.

## Equations

The six-group point kinetics equations model neutron density *n(t)* coupled with six delayed-neutron precursor groups *C_i(t)*:

```
dn/dt  = [(ρ − β) / Λ] · n  +  Σ λᵢCᵢ
dCᵢ/dt = (βᵢ / Λ) · n  −  λᵢCᵢ      (i = 1…6)
```

Parameters use the Keepin (1957) six-group data for U-235 thermal fission with Λ = 2×10⁻⁵ s.

## Methods

| Method | IC Handling | Training | Accuracy (vs SciPy) |
|--------|-------------|----------|----------------------|
| Standard PINN | Soft (penalty) | Gradient descent (Adam) | ~90% error |
| **X-TFC** | **Hard (exact via TFC)** | **Least-squares (single solve)** | **~0.06% error** |
| SciPy Radau | Exact (numerical) | Implicit Runge-Kutta | Reference |

## Project Structure

```
├── point_kinetics_pinn.py         # Standard PINN (PyTorch)
├── point_kinetics_xtfc.py         # X-TFC solver (NumPy)
├── point_kinetics_scipy.py        # SciPy Radau benchmark
├── compare_methods.py             # Side-by-side comparison + error analysis
├── generate_poster_graphics.py    # Additional poster-quality figures
├── model_utils.py                 # Save/load PINN checkpoints
├── requirements.txt
└── graphics/                      # All generated figures (see below)
```

## Quick Start

```bash
pip install -r requirements.txt

# Run individual solvers
python point_kinetics_scipy.py
python point_kinetics_xtfc.py --neurons 100 --seed 42
python point_kinetics_pinn.py --epochs 5000

# Run full comparison (trains both methods, generates error plots)
python compare_methods.py

# Generate all poster graphics
python generate_poster_graphics.py
```

## Graphics

All figures are saved to `graphics/` at 300 DPI. Key plots include:

| File | Description |
|------|-------------|
| `neutron_comparison.png` | Overlay of all three methods |
| `precursor_comparison.png` | Side-by-side precursor dynamics |
| `error_by_group.png` | Max relative error per precursor group |
| `error_over_time.png` | Error evolution over simulation |
| `neutron_error.png` | Absolute error in neutron density |
| `prompt_jump.png` | Early-time prompt jump + delayed rise |
| `phase_portrait.png` | n vs ΣCᵢ trajectory with time colormap |
| `ic_satisfaction.png` | Soft vs hard IC enforcement |
| `neuron_sensitivity.png` | X-TFC accuracy vs ELM neuron count |
| `ramp_vs_step.png` | Step vs ramp reactivity response |
| `precursor_panels.png` | Individual precursor group dynamics |
| `reactivity_profile.png` | Step reactivity diagram |
| `method_summary_table.png` | Visual comparison table |

## References

- Keepin, G.R. *Physics of Nuclear Kinetics* (1965)
- Raissi, M. et al. "Physics-informed neural networks" (2019)
- Schiassi, E. et al. "Physics-informed neural networks for the point kinetics equations" (2022)

## License

MIT
