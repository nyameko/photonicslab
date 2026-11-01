# Testing the modern PhotonicsLab physics core

The 0.6 alpha test suite verifies the supported NumPy-based physics and provenance
APIs, not the historical 2021 implementation.

## Clean checkout

```bash
uv sync --frozen --extra dev --extra notebooks
uv run --no-sync pytest --collect-only -q
uv run --no-sync pytest --cov=photonicslab
uv run --no-sync ruff check src tests examples
uv run --no-sync ruff format --check src tests examples
MPLBACKEND=Agg uv run --no-sync python examples/execute_lesson.py
uv build
```

The authoritative pytest configuration lives in `pyproject.toml`. Only
`tests/test_physics.py` and `tests/test_artifacts.py` constitute the
standalone modern test suite at this point. `integration_tests/` requires a
separately installed compatible `quantum-workflows` package and is intentionally
excluded from the standalone core suite.

## Historical archive

The old source and test files are retained in `legacy/photonicslab/` and
`legacy/tests/`. These tests depend on the old solver APIs and optional modules
such as `h5py`; they are *not* an indicator of 0.6 readiness and should not be
silently collected as the package's CI suite. `legacy/pytest.ini` preserves the
old benchmark configuration for reference. Do not reinstall obsolete dependencies
in the modern development environment just to make these tests collect.

## Scientific acceptance

Passing unit tests does not prove a model is valid outside its declared
approximations. Before publishing each new lesson, check analytical predictions,
conservation where applicable, mesh/window convergence, optical phase conventions,
and at least one independent package or published numerical reference when feasible.
See `docs/ARCHITECTURE.md` and `docs/ECOSYSTEM.md` for validity domains and
prospective external solver adapters.
