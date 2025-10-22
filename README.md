# Harbor Bridge Optimization

Multi-Criteria Decision Analysis (MCDA) framework for optimizing harbor bridge design considering multiple stakeholder preferences using genetic algorithms.

## Project Structure

```
u1_harbor_bridge/
├── harbor_bridge_optimization.ipynb  # Main analysis notebook
├── src/
│   ├── mcda/                          # MCDA API library
│   │   ├── __init__.py
│   │   └── api.py                     # Problem & Result classes
│   └── genetic_algorithm_pfm/         # Genetic algorithm implementation
│       ├── algorithm.py
│       ├── weighted_minmax/
│       └── tetra_pfm/
├── plots/                             # Generated visualizations
│   ├── svg/                           # Vector format plots
│   └── png/                           # Raster format plots
├── docs/                              # Documentation
├── examples/                          # Example use cases
└── week 3- MCDA.xlsx                  # Problem specification data
```

## Installation

### Prerequisites
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Install uv package manager:

```bash
# macOS
brew install uv

# Windows
winget install uv

# or via pip
pip install uv
```

2. Clone and setup environment:

```bash
# Navigate to project directory
cd u1_harbor_bridge

# Install dependencies and create virtual environment
uv sync

# Activate virtual environment
# Unix/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

## Usage

### Running the Main Analysis

Open and run the main Jupyter notebook:

```bash
jupyter notebook harbor_bridge_optimization.ipynb
```

### Using the MCDA API

```python
from src.mcda import Problem
import numpy as np

# 1. Define the optimization problem
problem = Problem(
    variables=['x1', 'x2', 'x3', 'x4'],
    bounds=[(10, 70), (10, 400), (2, 10), (350, 750)],
    variable_descriptions={
        'x1': 'Vertical clearance [meters]',
        'x2': 'Distance between pillars [meters]',
        'x3': 'Number of traffic lanes',
        'x4': 'Span of bridge [meters]'
    }
)

# 2. Add stakeholders with their objectives and preferences
problem.add_stakeholder(
    name="Contractor",
    weight=0.25,
    objective="(0.4 / x2**0.8) + (0.002 * x1**1.1) + (0.05 * x3**1.2) + (0.0008 * x4**1.05)",
    preference_points=([0.5186, 1.0695, 1.7192], [100, 40, 0]),
    x_label="Cost [billion dollars]"
)

# 3. Add constraints
problem.add_constraint("x2 >= 35")
problem.add_constraint("x2 <= x4")

# 4. Solve using different paradigms
result = problem.solve(paradigm='minmax')  # or 'tetra'

# 5. Visualize and analyze results
result.plot(paradigm='minmax')
result.print_summary()
```

## Features

### MCDA API (`src/mcda/`)
- Simple, intuitive interface for multi-stakeholder optimization
- String-based objective function expressions
- Flexible preference curve definition using PCHIP interpolation
- Support for multiple aggregation paradigms (minmax, tetra)
- Automatic visualization of results

### Genetic Algorithm Library (`src/genetic_algorithm_pfm/`)
- Custom genetic algorithm implementation
- Support for real and integer variables
- Constraint handling
- Multiple aggregation methods for preference scores

## Problem Description

This project optimizes the design of a harbor bridge considering conflicting objectives from multiple stakeholders:

### Stakeholders
1. **Contractor** (25%): Minimize construction cost
2. **Port Authority** (25%): Maximize vertical clearance for ships
3. **TxDOT** (30%): Maximize traffic capacity
4. **Environmental Agency** (10%): Minimize CO2 emissions
5. **Local Community** (10%): Minimize traffic incidents

### Design Variables
- `x1`: Vertical clearance (10-70 meters)
- `x2`: Distance between pillars (10-400 meters)
- `x3`: Number of traffic lanes (2-10)
- `x4`: Span of bridge (350-750 meters)

### Optimization Paradigms
- **Minmax**: Maximizes the minimum stakeholder satisfaction
- **Tetra**: Balanced approach with multiple satisfaction levels

## Results

Results are automatically saved in the `plots/` directory:
- Combined stakeholder preference plots
- Individual stakeholder visualizations
- Available in both SVG (vector) and PNG (raster) formats

## Development

### Project Dependencies
See [pyproject.toml](pyproject.toml) for full dependency list:
- numpy: Numerical computations
- scipy: Interpolation and scientific computing
- matplotlib: Visualization
- requests, urllib3: HTTP utilities

### Running Tests
```bash
# If tests are added in the future
pytest src/genetic_algorithm_pfm/tests/
```

## License

Educational project for University coursework.

## Authors

Tilburg University - Systems Engineering Course

## Acknowledgments

Built on the genetic algorithm framework with preference-based multi-criteria optimization capabilities.
