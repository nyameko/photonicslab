# Physical units and numerical validity

## Modern coherent field representation

`Field` stores a monochromatic transverse Jones envelope with array shape `(2, ny, nx)` for `(Ex, Ey)`. It is power normalized in **sqrt(W)/m**; it is *not* an electric field in V/m. The carrier convention is `Re[E(x,y,z) exp(i*k*z - i*omega*t)]`. The grid is an evenly sampled periodic FFT window with transverse coordinates `(y,x)`; the default grid is 128×128 over a 4 mm square.

Optical elements transform the field and produce **new** field objects. Lossy polarizers and apertures reduce power; the bench does not renormalize away absorption. Inspect `field.power`, `field.intensity()`, `field.stokes()` and `field.edge_power_fraction()` for meaningful checks.

## Implemented approximation domains

| Model | Supported physical meaning | Explicit limitation |
|---|---|---|
| Thin polarizer/waveplate | ideal Jones projector/retarder | no depolarization, dispersion, nonideal coating |
| GaussianBeam | coherent paraxial spatial envelope | not a pulsed multimode laser or cavity |
| Thin lens/phase SLM | sampled scalar phase mask | no device electronics or pixel calibration |
| Circular aperture | ideal amplitude mask | finite FFT window/mesh bias |
| Fresnel/Angular spectrum free space | homogeneous scalar diffraction per Jones component | periodic boundaries; not high-NA full-vector Maxwell propagation |
| Event-level detector | threshold clicks and time stamps | cannot be inferred from coherent field power without a separately calibrated source/model |

## Convergence rules

A field can appear plausible but be corrupted by FFT wraparound, undersampling, nonphysical high spatial frequencies, aperture aliasing, or a crop that removes energy. Change `nx`, `ny`, width and height **independently**; check phase resolution, edge power and expected analytical limits. Verify units and the sign conventions of retardance/Stokes parameters.

Do not equate a single fixed waveplate to 3D birefringence; do not equate a thin lens to a pumped optical cavity or a single-pass resonator. Those require separate numerical engines.

See [physics source](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/fields.py), [element source](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/elements.py) and [architecture validity table](https://github.com/nyameko/photonicslab/blob/main/docs/ARCHITECTURE.md).