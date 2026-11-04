# Getting started

## Installation

Python 3.12–3.14 is supported by `pyproject.toml`. The reference development path uses `uv` and `uv.lock`; the core runtime needs NumPy. From a fresh checkout:

```sh
git clone https://github.com/nyameko/photonicslab.git
cd photonicslab
uv sync --frozen --extra dev --extra notebooks
uv run --no-sync photonicslab polarization --angle-deg 22.5
uv run --no-sync jupyter lab labs/01-polarization/lesson.ipynb
```

Alternatively create a virtual environment and run `python -m pip install -e '.[dev,notebooks]'`. Do not install both legacy and modern package sources into the same environment.

## First Python experiment

```python
import numpy as np
from photonicslab import Grid, GaussianBeam, OpticalBench, HalfWavePlate, LinearPolarizer, PowerMeter

grid = Grid()
beam = GaussianBeam(wavelength=632.8e-9, waist=0.35e-3, power=1e-3).emit(grid)
bench = OpticalBench([HalfWavePlate(angle=np.pi/8), LinearPolarizer(angle=0)])
result = bench.run(beam)
print(PowerMeter().measure(result.output))  # about 0.0005 W for ideal elements
for observation in result.observations:
    print(observation.element, observation.power_w, observation.edge_power_fraction)
```

This is an **idealized** coherent Gaussian beam, waveplate and absorbing polarizer. Units are metres, radians and watts; field data is not a calibrated electric-field measurement.

## Explore next

Run the event-level detector/QKD demonstration with `uv run --no-sync python examples/quantum_cape_tabletop.py`. This model uses **simulated photon arrival events**, not optical power converted automatically into clicks. It does not generate a cryptographic key.

See [current 0.6 quickstart](https://github.com/nyameko/photonicslab/blob/main/README.md) and [first lesson](https://github.com/nyameko/photonicslab/blob/main/labs/01-polarization/lesson.ipynb).