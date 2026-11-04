# BB84: a teaching protocol, not a secure key product

## Current model

`BB84Experiment` simulates idealised single-photon polarization in two conjugate bases, Bob's randomly chosen bases, channel/detector loss, bit errors, dark clicks and an optional intercept-resend fraction. `BB84Result` includes sent/received counts, sifted bits, observed errors and quantum bit error rate (`qber`).

```python
import numpy as np
from photonicslab import BB84Experiment

rng = np.random.default_rng(20261102)
baseline = BB84Experiment(channel_transmission=0.8,
                          detector_efficiency=0.7).run(50_000, rng)
eve = BB84Experiment(intercept_resend_fraction=1.0).run(50_000, rng)
print("Baseline sifted/QBER:", baseline.sifted_bits, baseline.qber)
print("Eve sifted/QBER:", eve.sifted_bits, eve.qber)
```

For symmetric ideal conditions the sifted fraction approaches half detected pulses; ideal full intercept-resend tends to **25% QBER**, modulo finite-sample randomness. If no bits are sifted, `qber` is `None`.

## Critical security boundaries

This does **not** establish a secret key. The simulator omits authenticated classical reconciliation, practical light-source attacks, decoy states, error correction, privacy amplification, composable finite-key analysis, detector side channels and security proof assumptions. Do not export its bit arrays as encryption keys or interpret low QBER as unconditional confidentiality.

A meaningful future QKD lab should introduce weak coherent pulses versus true single photons, quantum channel loss, leakage/side channels, authenticated postprocessing and realistic key-rate bounds before any secure-system claims.

See [QKD source](https://github.com/nyameko/photonicslab/blob/main/src/photonicslab/qkd.py) and [conference tabletop tutorial](https://github.com/nyameko/photonicslab/blob/main/docs/QUANTUM-CAPE-PHOTONICS.md).