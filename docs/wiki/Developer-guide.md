# Developer guide

## Source modules and separation

`fields.py`: sampled Jones envelopes and Gaussian sources. `elements.py`: thin optical transformations and propagation. `bench.py`: sequential element orchestration and observations. `photodetection.py`: event-level source, detector and coincidence statistics. `qkd.py`: ideal BB84. `artifacts.py`: a versioned local lesson result; `cli.py`: minimal teaching runner.

Python should remain the stable, simple teaching API. Use NumPy CPU reference implementations first. Consider compiled/C++/CUDA kernels only after a numerical bottleneck is measured and the independent CPU reference/tests exist. Reuse maintained upstream optics and quantum libraries through opt-in adapters.

## Implementing an element

Specify physical regime, input/output `Field` type, SI units, approximations and energy/phase conventions. Implement an `apply(field) -> Field` transformation that does not mutate its argument. Validate parameters early and reject unsupported regimes. Add a textbook/analytic baseline test, energy or passivity checks where appropriate, and grid/window convergence. Add a short educational notebook exercise. Do not couple a pure optical element to GitHub, Slurm, network access or a GPU resource.

## New source types

**Do not** convert arbitrary laser power directly into photon pair events. An entangled-state source requires a quantum optical state representation and measurement theory; anisotropic propagation and pulsed nonlinear optics need separate solver contracts.

## Local quality gate

```sh
uv sync --frozen --extra dev --extra notebooks
uv run --no-sync pytest
uv run --no-sync ruff check src tests examples
uv run --no-sync ruff format --check src tests examples
uv build
```

Add integration checks only when both packages are installed in a compatible test environment. Preserve the `legacy/` archive for reference instead of importing old interfaces into the modern API.

See [architecture](https://github.com/nyameko/photonicslab/blob/main/docs/ARCHITECTURE.md), [tests](https://github.com/nyameko/photonicslab/blob/main/docs/TESTING.md) and [existing ecosystem tools](https://github.com/nyameko/photonicslab/blob/main/docs/ECOSYSTEM.md).