# Legacy 2021 source versus modern 0.6 alpha

## Two distinct generations

The repository preserves substantial original research code, notebooks and generated Sphinx documentation under `legacy/`, `notebooks/` and parts of `docs/`. Those files describe historical APIs for beam propagation, thermal lensing, birefringence, laser resonators and other experimental work; **their existence does not imply those features are implemented or passing in the modern 0.6 package**.

The current shipped Python package is `src/photonicslab/` with explicit exports in `__init__.py`. It intentionally avoids silently importing the old package or running the original test suite as if it were modern acceptance. The old notebooks may require obsolete dependencies, different field conventions or incomplete source methods.

## Safe migration strategy

1. Pin the original commit and identify the exact old notebook/solver and scientific problem.
2. Record governing equations, physical assumptions, unit conventions, code inputs, expected outputs and any existing independent validation.
3. Determine whether the 0.6 reference API already represents that regime. A thin Jones waveplate is **not** a 3D anisotropic solver; a single-pass lens is not a resonator.
4. Preserve historical notebooks as read-only reproducibility references. Create a modern lesson under the current `labs/` with explicit API changes, numerical checks and source references.
5. Add an optional adapter to a maintained external solver for regimes outside the small NumPy core. C++/CUDA optimization is justified only after a verified reference.
6. Run new independent tests; never claim legacy test results as proof of 0.6 correctness.

## A practical compatibility check

```sh
uv run --no-sync python -c 'import photonicslab; print(photonicslab.__version__, photonicslab.__file__)'
uv run --no-sync pytest tests
```

The package version should reflect the modern alpha; ensure imports resolve into `src/photonicslab/`. Historical tests in `legacy/tests/` must be kept separately and may require archival environment recreation, not automatic current CI.

## Contribution policy

Do not erase the 2021 archive or misrepresent its algorithms as freshly validated; scientific reuse must cite original authors and adhere to GPL-3.0-only licensing/compatible dependencies. When preparing a new laser/optomechanics/quantum optics feature, include a staged architecture and independent convergence/accuracy gate.

See [migration review](https://github.com/nyameko/photonicslab/blob/main/docs/REVIEW-2021.md), [modern package exports](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/__init__.py), [test policy](https://github.com/nyameko/photonicslab/blob/main/docs/TESTING.md) and [architecture roadmap](https://github.com/nyameko/photonicslab/blob/main/docs/ARCHITECTURE.md).