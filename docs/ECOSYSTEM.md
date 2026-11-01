# Optics software survey and reuse decisions

Primary project documentation inspected on 1 November 2026. This is a capability
survey, not a head-to-head accuracy benchmark, security audit or claim that each
package supports our Python/CUDA versions. No listed solver has been installed
as a production adapter in this first slice.

## Where existing projects fit

| Project / primary source | Documented strengths | Proposed role and limits |
| --- | --- | --- |
| [HCIPy](https://github.com/ehpor/hcipy) | Object-oriented wavefronts/optical elements, Fresnel/Fraunhofer propagation, Jones polarization, wave plates, adaptive optics | First candidate for an independent reference and a substantial classical-optics adapter. Its astronomy focus does not prevent tabletop lessons. Closest conceptual match to a Python virtual bench. |
| [LightPipes](https://github.com/opticspy/lightpipes) | Two-dimensional complex fields; apertures, lenses, splitters, free-space diffraction and examples | Strong undergraduate teaching comparison and scalar reference. Do not equate scalar diffraction with anisotropic vector Maxwell propagation. |
| [Diffractio](https://github.com/optbrea/diffractio) | Scalar and vector diffraction; BPM/WPM, vector Rayleigh–Sommerfeld, Jones masks, polarized propagation methods | Investigate vector and volumetric cases before writing custom high-NA/crystal engines. Its [GPLv3-or-later notice](https://github.com/optbrea/diffractio/blob/master/LICENSE) must accompany relevant distribution choices. Algorithm validity domains still matter. |
| [POPPY](https://github.com/spacetelescope/poppy) | Physical-optics system composition; Fresnel/Fraunhofer diffraction and PSFs | Reference for pupil/image planes, aberrations and propagation conventions; oriented toward astronomical systems. [BSD-3-Clause](https://github.com/spacetelescope/poppy/blob/develop/LICENSE.md). |
| [Finesse](https://gitlab.com/ifosim/finesse/finesse3) | Python interferometer simulation; linked Finesse/Pykat ecosystem | Candidate for modal, resonant and interferometric systems. The inspected README describes an early-development state and links [release documentation](https://finesse.ifosim.org/docs/); check current release docs before selecting a version. Do not infer maturity from the package name alone. |
| [OSCAR](https://github.com/Jerome-LMA/OSCAR) | MATLAB FFT optical fields in Fabry–Perot cavities; arbitrary mirror profiles, thermal lensing, diffraction loss and eigenmodes | Direct answer to the MATLAB comparison: study cavity benchmarks and experimental pedagogy. MATLAB is not a mandatory runtime for this free Python laboratory. |
| [Meep](https://github.com/NanoComp/meep) | FDTD Maxwell solver, anisotropic/dispersive/nonlinear media, flux and mode analysis | Specialised full-wave reference for small structures/interfaces; not the default for a metre-scale table resolved at optical wavelength. Computational scaling and GPL distribution require a deliberate runner choice. |
| [SPDCalc](https://github.com/spdcalc/spdcalc) | Rust SPDC source model with Python binding; phase matching, joint spectra, walk-off, fibre coupling, HOM calculations | Strong first SPDC integration candidate. It already addresses physics that a Jones-only bench cannot represent. Validate against published crystal/temperature/dispersion datasets before teaching quantitative predictions. |
| [Perceval](https://github.com/Quandela/Perceval) | Object-oriented discrete-variable photonic circuits, Fock states, sources/error models, C-backed simulators | Preferred investigation for linear-optical quantum circuits and HOM lessons. A spatial classical wavefront is not a Fock state. Its README reports Python 3.10–3.14; runner compatibility still needs execution tests. [MIT](https://github.com/Quandela/Perceval/blob/main/LICENSE). Remote QPU use stays opt-in. |
| [QuTiP](https://github.com/qutip/qutip) | Quantum dynamics/open systems and quantum-optics examples | Cavity QED, Lindblad dynamics, emitter/detector models and independent quantum-state references. It does not replace a macroscopic diffraction engine. |
| [Photontorch](https://github.com/flaport/photontorch) | PyTorch-based parallel photonic circuit simulation and optimization, CUDA support | Useful design reference for differentiable circuits. The inspected README includes old tensor/API limitations; test current releases before choosing it. Its documented AGPLv3 licensing differs from permissive numerical kernels. |
| [PyVISA](https://github.com/pyvisa/pyvisa), [PyMeasure](https://github.com/pymeasure/pymeasure) | Instrument communication; measurement procedures/data collection and drivers | Future hardware adapters. A virtual lens and a physical motion stage must not share an unguarded side-effecting `apply` method. Vendor-specific SDKs and calibration remain necessary. |

MATLAB has useful optical packages and an instrument ecosystem, but the appropriate
comparison depends on whether the problem is ray tracing, coherent diffraction,
resonant cavities, full-wave electromagnetism or quantum light. OSCAR is a specific
verified example; this review does not claim one universal MATLAB photonics toolbox
or validate MATLAB hardware interfaces. Commercial optical design software is not
required to run the open laboratory.

## What to build, reuse and measure

Build the cohesive learning experience: simple experiment composition, visible
assumptions, independent numerical checks, experiment-to-hardware explanations,
versioned lessons, result/provenance adapters and continuity across notebook/HPC runs.
Keep the initial NumPy kernels as transparent reference models. Do not try to
outperform every upstream solver or duplicate specialist nonlinear/quantum engines.

Next, run identical Gaussian propagation and wave-plate cases in HCIPy or LightPipes;
compare signed conventions, absolute powers and complex fields up to a declared
global phase. For a new solver, verify its release/licence, supported platform,
error/convergence, memory use and appropriate benchmark before exposing it as an
interchangeable backend. A readable API match is not evidence of scientific equivalence.

Maintain separate optional environments for tools with incompatible Python/CUDA
requirements; do not force every future laboratory into one enormous environment.
The current research fetched primary README/licence material, not an exhaustive
literature search. Publication should add peer-reviewed method references and
version-pinned numerical comparisons.

### First independent numerical check

`examples/compare_hcipy.py` was executed separately with HCIPy 0.7.1, NumPy 2.5.4
and Python 3.12.14. Identical 128² sampled Jones inputs at 632.8 nm were compared
for a 22.5° half-wave plate plus horizontal analyser, and a 0.25 mm waist Gaussian
propagating 0.2 m. With point-sampled transfer functions (`num_oversampling=1` in
HCIPy, padding 2), relative intensity L2 difference was about `3.7e-15`; transmitted
and propagated powers agreed to floating-point precision.

HCIPy's default transfer supersampling of 2 gave a relative intensity difference
of `0.002393` and relative power difference of `0.001950` for that case. Its source
averages supersampled transfer factors; that is a different discrete operator.
Both settings are retained in the comparison script and report. This is not evidence
that all defaults or physical regimes are interchangeable, and no backend was
modified or acceptance tolerance relaxed to hide that difference. No specialised
volume, nonlinear, quantum or GPU engine was benchmarked.

## Rights and attribution

The existing PhotonicsLab GPLv3 licence is retained. Apache-2.0 in the workflow
repository does not automatically relicense a combined distribution containing
GPL components; document the chosen packaging/runner boundary before release.
No upstream source or textbook figures were copied here. The new lesson is original;
the larger academy notebook collection referenced by Quantum Workflows is still a
proposal and was not imported or represented as approved content.
