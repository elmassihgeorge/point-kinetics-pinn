# Point Kinetics PINN

A Physics-Informed Neural Network (PINN) for solving the point kinetics equations in nuclear reactor transient analysis.

## Overview

This project implements the 6-group point kinetics equations that describe neutron population dynamics in nuclear reactors:

```
dn/dt = [(ρ - β) / Λ] × n + Σ λᵢCᵢ
dCᵢ/dt = (βᵢ / Λ) × n - λᵢCᵢ  (i = 1..6)
```

Where:
- `n` = neutron density
- `Cᵢ` = delayed neutron precursor concentration (6 groups)
- `ρ` = reactivity
- `β` = total delayed neutron fraction
- `Λ` = prompt neutron generation time

## Project Structure

```
├── point_kinetics_scipy.py   # SciPy numerical benchmark
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## Features

- **SciPy Benchmark**: High-accuracy numerical solution using the Radau solver
- **6-Group Delayed Neutrons**: Full delayed neutron precursor dynamics (U-235)
- **Validation**: Residual analysis comparing numerical derivatives to ODE expectations
- **Visualization**: Combined plots of solution and validation metrics

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy, SciPy, Matplotlib

### Installation

```bash
git clone https://github.com/yourusername/point-kinetics-pinn.git
cd point-kinetics-pinn
pip install -r requirements.txt
```

### Run the Benchmark

```bash
python point_kinetics_scipy.py
```

This will:
1. Solve the point kinetics equations for a step reactivity insertion
2. Validate the solution using residual analysis
3. Generate a combined visualization plot

## Physics Background

The point kinetics equations model the time-dependent behavior of neutron population in a nuclear reactor. Key physical phenomena captured:

- **Prompt Jump**: Instantaneous response to reactivity changes (~µs timescale)
- **Delayed Neutron Rise**: Slower exponential growth governed by precursor decay (~s timescale)
- **Multi-group Dynamics**: Six delayed neutron precursor groups with different decay constants

### Parameters (U-235, Thermal Spectrum)

| Parameter | Value | Description |
|-----------|-------|-------------|
| β (total) | 0.0065 | Total delayed neutron fraction |
| Λ | 2×10⁻⁵ s | Prompt neutron generation time |
| ρ | 0.003 | Step reactivity insertion (~0.5β) |

## Roadmap

- [x] SciPy numerical benchmark
- [x] Validation and residual analysis
- [ ] PINN implementation
- [ ] Benchmark comparison (PINN vs SciPy)
- [ ] Interactive visualization

## References

- Keepin, G.R. "Physics of Nuclear Kinetics" (1965)
- Raissi, M. et al. "Physics-informed neural networks" (2019)

## License

MIT License - see [LICENSE](LICENSE) for details.