# Jones optics, Stokes parameters and physical units

## Representation and sign conventions

The modern PhotonicsLab `Field` is a sampled, complex transverse Jones envelope `(Ex,Ey)` with array shape `(2,ny,nx)`. It uses the carrier convention `Re[E(x,y,z)exp(i*k*z-i*omega*t)]`, where `E` in the package is a *power-normalized envelope* in `sqrt(W)/m`, **not** the full physical electric field in `V/m`. Coordinates and wavelengths use metres, angles use radians, and integrated intensity yields watts. Do not apply Maxwell energy-density formulas to the envelope without the correct physical normalization.

The transverse Jones vector represents a fully coherent monochromatic state of polarization at each grid point. It cannot alone represent unpolarized/depolarized light, stochastic partial coherence, high numerical aperture longitudinal fields, or an arbitrary mixed quantum polarization state.

## Implemented element operators

`LinearPolarizer(angle)` is an ideal absorbing projector onto a transverse axis. `WavePlate(retardance,angle)` is an ideal uniform Jones retarder with zero phase on the fast axis and `+retardance` on the orthogonal axis. `HalfWavePlate` and `QuarterWavePlate` are fixed idealizations; they are not wavelength-dispersive crystal models. These operators create new fields and preserve or reduce power according to their physical idealization.

## Checks against analytic physics

For a linearly polarized input at angle `theta` relative to an ideal polarizer's axis, transmitted power satisfies Malus' law `P_out = P_in cos²(theta)`. A lossless waveplate conserves integrated power while changing polarization phase and Stokes components. A polarizer can decrease power, never amplify it. For Stokes sign and handedness, derive the result using this package's stated complex phase convention rather than importing an incompatible convention from another optics package.

## Practical lesson

Create a `GaussianBeam`, apply an ideal half-wave plate and polarizer, inspect `result.observations`, and plot the signed Stokes quantities as a function of angle. Repeat with a larger physical grid and a finer grid while keeping optical parameters fixed to identify truncation and normalization artifacts. Include a short derivation and expected tolerance in the notebook.

Do not infer photon counts or quantum entanglement from Jones `Field.power`. Those are distinct models, represented in `photodetection.py` and `qkd.py` with different assumptions.

Sources: [field definitions](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/fields.py), [element Jones operators](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/elements.py), [modern physics tests](https://github.com/nyameko/photonicslab/blob/main/tests/test_physics.py).