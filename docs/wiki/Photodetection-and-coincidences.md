# Event-level photodetection and coincidences

## Model and limitations

The `PhotonPairSource` generates paired Poisson arrival times and optional relative Gaussian timing skew. **It is not an SPDC/FWM quantum state, entangled source or calibrated heralded photon source.** `SinglePhotonDetector` independently applies efficiency, Poisson dark counts, non-paralyzable dead time and timestamp jitter. `CoincidenceCounter` counts gate coincidences and computes lag histograms and accidental estimates.

```python
import numpy as np
from photonicslab import PhotonPairSource, SinglePhotonDetector, CoincidenceCounter

rng = np.random.default_rng(20261102)
duration = 0.2
a, b = PhotonPairSource(pair_rate_hz=8000, relative_jitter_s=250e-12).emit(duration, rng)
det = SinglePhotonDetector(efficiency=0.65, dark_count_rate_hz=80,
                           dead_time_s=40e-9, timing_jitter_s=120e-12)
tags_a = det.detect(a, duration, rng)
tags_b = det.detect(b, duration, rng)
counter = CoincidenceCounter(2e-9)
print(counter.count(tags_a, tags_b))
print(counter.expected_accidentals(len(tags_a), len(tags_b), duration))
```

All times are **seconds**, rates **Hz** and coincidence gate width **seconds**. A histogram peak shows time correlation, not entanglement or Bell inequality violation. For independent streams, accidentals roughly scale with both singles rates, coincidence window and observation time, but true pair correlations and detector saturation violate the simple approximation.

## Validation

With a fixed RNG, test efficient detection limits, deterministic zero-dark cases, dead-time saturation, jitter recovery and gate-width sensitivity. Compare measured distributions against analytic/Monte Carlo expectations across multiple seeds. Separate detector effects from source statistics. There is not yet a device-specific APD quenching, afterpulsing, wavelength-dependent efficiency or electronics model.

See [photodetection source](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/photodetection.py), [tabletop example](https://github.com/nyameko/photonicslab/blob/main/examples/quantum_cape_tabletop.py) and [Quantum Cape design](https://github.com/nyameko/photonicslab/blob/main/docs/QUANTUM-CAPE-PHOTONICS.md).