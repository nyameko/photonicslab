# Troubleshooting

## Import fails or legacy symbol is missing

Check `python -c "import photonicslab; print(photonicslab.__version__)"`; use 0.6.0a1 module exports and installed editable package. The archived `legacy/photonicslab` and old Sphinx references are intentionally incompatible and should not be mixed into modern imports.

## Strange diffraction patterns

Check FFT periodic wraparound, beam power near grid edges, phase mask resolution, wavelength/length SI units and transverse sampling. Change both field grid density and physical width to separate aliasing from clipping. Verify with a plane wave or Gaussian analytic limit.

## Unexpected polarization power

Check radians versus degrees, fast-axis convention, Jones vector ordering, absorbing polarizers and whether the measurement is **watts** rather than normalized probability. A lossless waveplate should not create power; an ideal polarizer may remove it.

## Coincidence or QBER changes between runs

Use an explicit `np.random.default_rng(seed)`. With small event counts shot/Poisson fluctuations can dominate; check detector efficiency, jitter/dead time and coincidence full-window semantics. QBER can be `None` when there is no sifted bit. Paired time-tags are not proof of entanglement.

## Tests pass but scientific claim is disputed

Report physical assumptions, equations/reference, units, grid/seed, numerical convergence, data/error bars, version and minimal reproduction. Do not claim calibrated single-photon electronics, full SPDC or secure QKD from educational demonstrations.

See [test guide](Validation-and-testing.md) and [issue tracker](https://github.com/nyameko/photonicslab/issues).