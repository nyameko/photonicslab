# Verification, testing and scientific acceptance

## Standalone 0.6 alpha checks

```sh
uv sync --frozen --extra dev --extra notebooks
uv run --no-sync pytest --collect-only -q
uv run --no-sync pytest --cov=photonicslab
uv run --no-sync ruff check src tests examples
uv run --no-sync ruff format --check src tests examples
MPLBACKEND=Agg uv run --no-sync python examples/execute_lesson.py
uv run --no-sync python examples/quantum_cape_tabletop.py
uv build
```

The modern test suite currently includes `tests/test_physics.py`, `tests/test_artifacts.py` and `tests/test_quantum_photonics.py`. The separate `integration_tests/` require a compatible Quantum Workflows installation; historical `legacy/tests/` are not evidence of 0.6 correctness.

## Independent physics evidence

- Polarizers: Malus' law, non-increasing absorbed power.
- Waveplates: Jones matrix unitary limits and signed Stokes parameters.
- Diffraction: Gaussian beam width/phase and plane-wave transfer comparison.
- Lens: analytical focal spot prediction in the paraxial regime.
- Apertures/SLMs: reference transmission/steering and sampling requirements.
- Detectors: Poisson counts, acceptance/dead time/jitter, accidental coincidence scaling.
- BB84: loss, basis mismatch, expected QBER/intercept-resend statistics.

A passing test suite does **not** establish physical validity outside declared approximations. Check independent theory, multiple grids/step sizes, uncertainties and reproducible input/seed/version. A solver that conserves power can still be wrong about phase.

See [testing policy](https://github.com/nyameko/photonicslab/blob/main/docs/TESTING.md) and [architecture](https://github.com/nyameko/photonicslab/blob/main/docs/ARCHITECTURE.md).