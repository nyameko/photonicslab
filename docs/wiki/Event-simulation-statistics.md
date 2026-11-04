# Photon arrivals, detector noise and Monte Carlo statistics

## Source and detection assumptions

`PhotonPairSource` emits an ideal Poisson number of pairs over an observation interval, with uniform emission times and optional Gaussian relative timing skew. This provides **classically correlated arrival timestamps**, not a fully simulated SPDC/FWM optical field or entangled quantum state.

`SinglePhotonDetector` models each incident arrival with a Bernoulli detection efficiency, independently adds Poisson dark events, applies **non-paralyzable** dead time to physical candidate events, then applies Gaussian electronic timestamp jitter. It does not resolve photon number or provide a calibrated wavelength/temperature response, afterpulsing or detector quenching circuit. All times and jitter are seconds; rates are Hz.

## Coincidence measurement

The `CoincidenceCounter` counts one-to-one coincident timestamps inside a defined full gate and can histogram time delays `t_B-t_A`. A narrow timing correlation is not a Bell violation or proof of nonclassical optical states. The independent accidental approximation is of order `(N_A/T)(N_B/T)*W*T` where `W` is full coincidence window width and `T` acquisition duration; it is unreliable under severe saturation, strongly correlated streams or large finite gate effects.

## Statistical validation sequence

Generate multiple seeded datasets, record singles counts, coincidence counts, dark rate, lag peak and confidence interval. Change source pair rate, detector efficiency, dead time, timing jitter and gate width one at a time to distinguish source statistics from instrument effects. For convergence, increase acquisition duration and repeat with independent seeds; do not use one seed as a scientific uncertainty estimate.

At low rates, observed singles roughly scale with efficiency and duration. At high rates, dead time reduces counts. Increased jitter broadens lag histograms and can reduce narrow-gate coincidences even when total pairs are unchanged. Dark counts create accidental coincidences; a histogram baseline alone does not identify the physical noise mechanism.

## What cannot be claimed

This model cannot calibrate a specific SPAD/APD, recover true photon-pair spectral entanglement, demonstrate HBT/HOM interference, estimate Bell-state fidelity or guarantee QKD security. Add such claims only with a separately validated quantum-optical source, measurement operators and independent experimental/numerical evidence.

Sources: [event-level module](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/photodetection.py), [Quantum Cape tabletop example](https://github.com/nyameko/photonicslab/blob/main/examples/quantum_cape_tabletop.py), [quantum photonics tests](https://github.com/nyameko/photonicslab/blob/main/tests/test_quantum_photonics.py).