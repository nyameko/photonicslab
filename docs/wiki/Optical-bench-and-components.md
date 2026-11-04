# Composing an optical bench

## Object model

`OpticalBench(elements)` applies an ordered sequence of components, with an observation after the source and each element. The `BenchResult` has `output` and `observations`; use `PowerMeter().measure(field)` to get ideal power in watts. The modern API is defined in [package exports](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/__init__.py).

| Component | Example construction | Role |
|---|---|---|
| Grid | `Grid(nx=256, ny=256, width=0.006, height=0.006)` | numerical transverse window |
| Gaussian source | `GaussianBeam(wavelength=632.8e-9, waist=0.35e-3, power=1e-3)` | coherent input |
| Linear polarizer | `LinearPolarizer(angle=0.0)` | ideal absorbing Jones projector |
| Half-wave plate | `HalfWavePlate(angle=np.pi/8)` | ideal polarization rotation/retardance |
| Quarter-wave plate | `QuarterWavePlate(angle=np.pi/4)` | ideal ellipticity |
| Thin lens | `ThinLens(...)` | scalar sampled lens phase (inspect signature) |
| Aperture | `CircularAperture(...)` | geometric amplitude loss |
| Phase SLM | `PhaseSLM(...)` | sampled prescribed phase |
| Free space | `FreeSpace(...)` | scalar propagation algorithm |

For optional parameters of lens, aperture, SLM and propagation, inspect the actual dataclass signatures in `src/photonicslab/elements.py` rather than assuming a specific device configuration.

## Example workflow

1. Select wavelength, physical beam waist, polarization, power and a grid large enough to resolve it.
2. Build the ordered list of elements and explain the real-table arrangement.
3. Run once and record intermediate observables after each component.
4. Compare Malus' law, expected waveplate polarization and diffraction against analytical baselines.
5. Double resolution/window and compare results; report numerical convergence and uncertainty.
6. Save the experiment's inputs, versions and numerical checks using the lesson runner when reproducibility matters.

A linear chain is not a branch, an interferometer with two spatial arms or a resonator feedback loop. Those are future experiment abstractions requiring additional physics and state representation.

See [bench implementation](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/bench.py) and [component implementation](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/elements.py).