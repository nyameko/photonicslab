"""Conference-inspired tabletop experiment: photon counting -> coincidences -> BB84.

Run: python examples/quantum_cape_tabletop.py
Only NumPy required; demo seed is fixed. No cryptographic key is generated.
"""

import numpy as np

from photonicslab import BB84Experiment, CoincidenceCounter, PhotonPairSource, SinglePhotonDetector


def main():
    rng = np.random.default_rng(20261102)
    duration_s = 0.2
    arm_a, arm_b = PhotonPairSource(8_000, 250e-12).emit(duration_s, rng)
    alice = SinglePhotonDetector(0.65, 80, 40e-9, 120e-12)
    bob = SinglePhotonDetector(0.72, 80, 40e-9, 120e-12)
    tags_a = alice.detect(arm_a, duration_s, rng)
    tags_b = bob.detect(arm_b, duration_s, rng)
    counter = CoincidenceCounter(2e-9)
    coincidences = counter.count(tags_a, tags_b)
    accidentals = counter.expected_accidentals(len(tags_a), len(tags_b), duration_s)
    edges = np.linspace(-5e-9, 5e-9, 41)
    hist = counter.histogram(tags_a, tags_b, edges)
    peak_lag_s = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2

    print("Bench A: ideal paired source -> two imperfect detectors -> time tagger")
    print(f"Singles A/B: {len(tags_a)} / {len(tags_b)}")
    print(f"Coincidences (2 ns full-width window): {coincidences}")
    print(f"Independent-stream accidental estimate: {accidentals:.4g}")
    print(f"Cross-correlation histogram peak lag: {peak_lag_s * 1e9:.3f} ns")

    print("\nBench B: ideal single-photon polarization BB84, 50,000 pulses")
    for label, fraction in [("no Eve", 0.0), ("intercept-resend Eve", 1.0)]:
        result = BB84Experiment(
            channel_transmission=0.8,
            detector_efficiency=0.7,
            optical_bit_error_probability=0.01,
            intercept_resend_fraction=fraction,
        ).run(50_000, rng)
        print(
            f"{label}: detected={result.received}, sifted={result.sifted_bits}, "
            f"QBER={result.qber:.2%}, secure key NOT estimated"
        )


if __name__ == "__main__":
    main()
