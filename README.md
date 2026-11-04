# PhotonicsLab

> **Documentation:** [Wiki home](docs/wiki/Home.md) · [Getting started](docs/wiki/Getting-started.md) · [Optical bench](docs/wiki/Optical-bench-and-components.md) · [Photodetection](docs/wiki/Photodetection-and-coincidences.md) · [QKD scope](docs/wiki/BB84-and-QKD-limitations.md) · [Developer guide](docs/wiki/Developer-guide.md). Reviewed sources are versioned in `docs/wiki/`; [native GitHub Wiki publication](docs/WIKI_PUBLISHING.md) is a separate reviewable action.


A Python virtual optics bench for learning, teaching and reproducible experiments.

The new **0.6 alpha** starts with coherent classical light: Gaussian beams, ideal
polarizers and wave plates, a phase SLM, lenses, apertures, and Fresnel / scalar
angular-spectrum propagation. Its reference models are small enough to inspect
alongside the equations. Python is the teaching API; specialised numerical engines
can later provide validated CPU/GPU implementations.

This is the first working foundation, **not yet a general photonics or quantum-optics
simulator**. Cavities, pumped lasers, anisotropic volume propagation and SPDC are
planned, not implemented. The [architecture and staged roadmap](docs/ARCHITECTURE.md)
explain the separate physical models and acceptance criteria.

## Try a virtual table

Python 3.12–3.14; development defaults to 3.14, matching Quantum Platform / Workflows.
The core needs only NumPy. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra dev --extra notebooks
uv run --no-sync photonicslab polarization --angle-deg 22.5
uv run --no-sync jupyter lab labs/01-polarization/lesson.ipynb
```

Or install with `python -m pip install -e '.[dev,notebooks]'` in a virtual environment.

```python
import numpy as np
from photonicslab import (
    Grid, GaussianBeam, OpticalBench, HalfWavePlate, LinearPolarizer, PowerMeter,
)

beam = GaussianBeam(wavelength=632.8e-9, waist=0.35e-3, power=1e-3).emit(Grid())
bench = OpticalBench([
    HalfWavePlate(angle=np.pi / 8),
    LinearPolarizer(angle=0),
])
result = bench.run(beam)
print(PowerMeter().measure(result.output))  # 0.0005 watts
```

Lengths are metres, angles radians, power watts, and wavelength is the vacuum
wavelength. Field arrays have shape `(2, ny, nx)` for `(Ex, Ey)`. They are power-normalised
Jones envelopes in `sqrt(W)/m`, rather than electric fields in V/m. Elements return
new fields. Absorbed power is never silently renormalised.

The [first notebook](labs/01-polarization/lesson.ipynb) connects theory, numerical
assertions, plots, exercises and the corresponding real teaching bench. It also
demonstrates diffraction, a focusing lens and SLM beam steering. The lesson is
development material awaiting independent teaching/science review.

## Quantum Cape tabletop detectors and QKD (experimental)

The new event-level module is deliberately separate from coherent Jones-field
optics: photon arrival timestamps are not derived from laser beam power.
The conference-inspired teaching demo explores threshold photon detectors,
coincidence histograms and ideal single-photon BB84 (including intercept-resend).
It does **not** produce a calibrated entangled source or secure cryptographic key.

```bash
uv run --no-sync python examples/quantum_cape_tabletop.py
uv run --no-sync pytest -q tests/test_quantum_photonics.py
```

See [the Quantum Cape research roadmap](docs/QUANTUM-CAPE-PHOTONICS.md) for planned
APD quenching electronics, neutral-density/weak-coherent pulsed lasers, HBT,
HOM, FWM/SPDC, E91, CHSH and optomechanics.

## Validate

```bash
uv run --no-sync pytest
uv run --no-sync ruff check src tests examples
uv run --no-sync ruff format --check src tests examples
uv run --no-sync python examples/execute_lesson.py
uv build
```

Tests compare against Malus’ law, signed Stokes parameters, Gaussian diffraction,
an analytical lens spot size, aperture transmission, a plane-wave transfer factor,
SLM steering and mesh/window refinement. Conservation alone is insufficient.
FFT propagation uses periodic windows and requires sampling/convergence checks.
Two transverse Jones components do not constitute a full-vector anisotropic solver.

## Relationship to the Quantum Platform family

PhotonicsLab owns optical models and their validation. `quantum-workflows` owns
production scientific runners, notebooks, scheduler entry points and canonical
workflow provenance. Quantum Platform owns identity, programmes, access and results;
ACP owns durable agent context; infrastructure owns Kubernetes, Slurm and GPU placement.

The local CLI writes a distinct `photonicslab.lesson-run/1` manifest with inputs,
source revision/dirty state, versions, result hashes, numerical checks and field data.
The [adapter example](examples/quantum_workflows_polarization.py) uses the actual
`quantum_workflows.artifacts.RunArtifacts` API when that package is installed.
It is not yet registered as a `qw` command or a portal execution offering.

## Review and original work

- [2021 code review and migration notes](docs/REVIEW-2021.md)
- [Comparable tools and reuse recommendations](docs/ECOSYSTEM.md)
- [Physics architecture, integration plan and roadmap](docs/ARCHITECTURE.md)

The original source, tests and setup files are preserved under `legacy/`. Original
notebooks in `notebooks/` and generated documentation in `docs/html/` describe that
historical API; they have not been validated with 0.6. This is an intentional API
break. Historical modules are excluded from the new wheel and test suite, rather
than presented as passing modern models. Git history remains intact.

GPL-3.0-only, preserving the repository’s existing GPLv3 licence. No third-party
solver or teaching material is copied into the new implementation.
