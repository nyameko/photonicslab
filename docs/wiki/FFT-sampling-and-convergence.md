# Fourier propagation, finite windows and numerical convergence

## Why FFT results need checks

The current `FreeSpace` element implements homogeneous scalar Fresnel or angular-spectrum propagation separately for the two Jones components. The sampled transverse plane is a *periodic numerical FFT window*; a field leaving its edge can re-enter from the opposite side. Simulations can conserve discrete power and still be physically incorrect because a periodic copy contaminates the optical region.

The default `Grid(nx=128,ny=128,width=4e-3,height=4e-3)` has `dx=width/nx` and `dy=height/ny`, without a duplicated endpoint. A hard aperture or spatial phase mask introduces higher Fourier frequencies than a smooth Gaussian beam. Insufficient sampling causes aliasing; a too-small window clips/diffracts into its periodic neighbours.

## Convergence strategy

1. State wavelength, vacuum refractive index assumption, distance, initial beam waist, polarization and optical elements.
2. Estimate beam width and expected diffraction pattern analytically before the numerical run.
3. Choose a physical grid window larger than the region of meaningful field energy. Inspect `edge_power_fraction()` before and after propagation.
4. Refine grid resolution with **fixed physical window**; compare transmitted power, beam width, focal spot, phase and Stokes values.
5. Increase physical window with **fixed or appropriately refined spatial sampling**; compare to detect periodic wraparound.
6. Vary propagation length and lens/SLM mask spatial features to test method validity and phase aliasing.
7. Record convergence curves and threshold; a single 128×128 image cannot establish a solver's accuracy.

## Analytical references

Use Gaussian free-space beam spreading, a plane-wave transfer factor, ideal thin-lens paraxial focus and known aperture transmission. Fresnel is a paraxial approximation; scalar angular spectrum can capture nonparaxial scalar dispersion for propagating modes but is not automatically full-vector Maxwell physics or an evanescent Poynting flux treatment. Do not use a 2-component transverse Jones propagation result to claim validated 3D birefringent walk-off or high-NA vector focusing.

## Performance and reproducibility

Record NumPy/Python versions, grid shape, dimensions, FFT method, wavelength, distance, source normalization and independent reference error. Only optimize CPU/GPU kernels after the reference is correct and test tolerances are frozen. Use the small educational bench for normal teaching; schedule large GPU work as an independent, approved Quantum Workflows job, not as an import-side effect.

Sources: [grid and field implementation](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/fields.py), [propagation implementation](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/elements.py), [scientific architecture](https://github.com/nyameko/photonicslab/blob/main/docs/ARCHITECTURE.md).