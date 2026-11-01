# Review of the historical PhotonicsLab

Reviewed 1 November 2026 against commit `739a3f4e9cffc93c0ee4e378bf5b98bba6f57f8f`.
Paths below refer to preserved files in `legacy/`. This is a source review and
modern-environment baseline, not a scientific certification of historical notebooks.

## Assessment

The strongest reusable assets are the experimental questions: spatially shaped
heat sources, thermo-optic lenses, structured beams, amplification and comparisons
with experimental profiles. The HDF5-backed notebook approach anticipated reproducible
simulation, but storage, plotting, material properties, meshes and solvers are tightly
coupled. Inheritance often expresses file/data reuse rather than physical substitutability.
The package is a useful research archive, not a validated universal virtual bench.

## Findings, ordered by consequence

| Priority | Evidence | Consequence / required action |
| --- | --- | --- |
| High: FTCS update aliases its input | `photonicslab/fdsolver.py:89–106`: `fsol = self.finit`, `temp = self.finit`, then in-place updates | New values contaminate the same time step and mutate the initial condition. This is not the documented explicit FTCS method. Port with independent buffers and a manufactured-solution convergence test. |
| High: spatial coefficients exchanged | `heateqn.py:658–659` defines `sx` from `dx`, `sy` from `dy`; `fdsolver.py:100–101` applies `sx` along j and `sy` along i, despite mesh indexing `(x,y)` | Square isotropic examples can conceal the error. Test rectangular anisotropic manufactured solutions. |
| High: volume solver allocates dense matrices | `fdsolver.py:312–313` allocates two `(nx*ny*nz)²` arrays before sparse use | A 50³ grid needs about 233 GiB for those two float64 arrays alone. Use sparse/matrix-free operators before HPC scaling. |
| High: amplifier calls arrays as functions | `amplifier.py:268` makes a NumPy `gain` array; `:310` uses `gain(i,j,z_step)`, with similar coefficient calls | The affected ADI path cannot execute as written. Static confirmation; it was not GPU-benchmarked. Port equations rather than this implementation wholesale. |
| High: physical implementations are missing | `anisotropy.py:31–50` has empty constructors for birefringence/stress subclasses; `demultiplex.py` and several quality classes are placeholders | Names do not establish capability. These are not validated anisotropic or polarization models. |
| Medium: explicit loops omit an interior row/column | `fdsolver.py:97–98` uses `range(1, n-2)` | Index `n-2` remains unupdated. Boundary ownership must be explicit and tested. |
| Medium: Maxwell-module import is broken | `maxwelleqn.py:31` imports `photoncislab.config` | A typo prevents direct import. Default package import does not exercise it. Its nominal test file has no collected `test_*` function. |
| Medium: alternate sparse solver imports are invalid | `fdsolver.py:273–289` imports from top-level `sps.linalg`, with incorrect `lmres` names in some branches | Iterative paths need correction and residual-based acceptance tests. |
| Medium: normalization obscures physics | `beamprop.py:139–172` normalises fields/intensities separately; convolution lacks an explicit sampled Fresnel prefactor/integration measure | Shape visualisation is insufficient to establish absolute irradiance, loss or gain. Adopt a declared normalization and retain power through every element. |
| Medium: mixed units | `beamprop.py:22–52` documents both cm and m and a consolidation TODO; defaults elsewhere differ | Use SI and document vacuum/in-medium wavelength and material conventions. |
| Medium: numerical accuracy is not established | The Gaussian propagation comparison in `tests/beamprop/test_beamprop.py:13` is skipped; many tests check shapes or inequality | Changing an array does not establish a correct solver. Require refinement studies and independently calculated references. |
| Setup: obsolete tooling | README targets Python 3.6/CUDA 9.1; requirements include `cupy-cuda91`; tests import `distutils` and `skimage.measure.compare_*`; setup imports setuptools' removed test command | Recreate only in an isolated archival environment. Do not impose these constraints on the new teaching environment. |

Original collection was attempted on Python 3.14.8 with current numerical, HDF5,
plotting and test dependencies: **15 tests collected, three collection errors**
(`distutils` imports). No claim is made that those tests passed. Further removed
scikit-image interfaces are visible in source but were not reached in that attempt.
The table distinguishes baseline evidence from code-level defects; historical
experimental results have not been independently reproduced.

## Migration decision

Preserve the old source and lessons. Develop `src/photonicslab` as an incompatible
alpha with a small validated domain. Storage is a result adapter, not a prerequisite
for constructing a wavefront. Thin optics use composition; distributed media,
cavities, thermodynamics and quantum states need separate models.

Before migrating each notebook, record its question, units, geometry, material /
temperature / wavelength assumptions, boundary conditions, observable and independent
reference. Compare analytical limits, mesh/time refinement and experimental data
where available. Existing plots are evidence to investigate, not golden truth to
reproduce by tuning a solver.

The replacement tests cover polarization, homogeneous propagation, ideal thin elements
and local provenance. They do **not** validate archived thermal, amplifier, resonator
or anisotropic solvers.
