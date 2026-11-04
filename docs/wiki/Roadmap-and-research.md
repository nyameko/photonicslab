# Research roadmap and staged capabilities

## Available now versus planned

**Implemented modern alpha:** sampled coherent Jones fields, Gaussian beam, ideal polarizers, thin waveplates, thin lens, phase SLM, circular aperture, Fresnel/Angular-spectrum free-space propagation, ideal power measurement, local artifact runner, photon-arrival statistics, time-tagged detectors, coincidence counter and idealized BB84.

**Not implemented as validated 0.6 solver:** optical cavities/laser resonator mode solving, pump/gain/thermal lasing, 3D birefringent volume propagation, waveguides/fibres with modal dispersion, full-vector high-NA Maxwell solver, nonlinear SPDC/FWM, quantum Fock/Gaussian entangled light, realistic APD/SPAD circuits, HBT/HOM interference, CHSH/E91 security experiments, hardware drivers or experimental calibration.

## Proposed staged order

1. Validate the teaching bench as independent numerical physics.
2. Add partially coherent light and realistic device transfer functions.
3. Introduce waveguide/birefringent propagation with benchmark field solvers.
4. Build resonator, gain/thermal and laser-cavity specialist adapters.
5. Introduce separately validated quantum optical states/sources, photon statistics and measurement.
6. Compose interferometers, HBT/HOM, SPDC/FWM and entanglement experiments.
7. Add QKD security postprocessing and realistic device noise with expert review.
8. Optimize verified bottlenecks with optional HPC/GPU engines and reproducible platform workflows.

A future capability must not be advertised as working because a similarly named 2021 class exists in `legacy/` or a generated Sphinx page. For each new physical regime require analytical limits, independent solver comparison and uncertainty estimates.

References: [roadmap](https://github.com/nyameko/photonicslab/blob/main/docs/ARCHITECTURE.md), [2021 review](https://github.com/nyameko/photonicslab/blob/main/docs/REVIEW-2021.md), [conference-inspired roadmap](https://github.com/nyameko/photonicslab/blob/main/docs/QUANTUM-CAPE-PHOTONICS.md).