"""Idealized single-photon polarization BB84, with explicit security limits.

This is an event-level qubit/Born-rule lesson, not a calibrated QKD security
implementation. No decoy states, finite-key proof, authentication, privacy
amplification, detector side channels or multi-photon weak-coherent pulses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _probability(value: float, name: str) -> None:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must lie within [0, 1]")


@dataclass(frozen=True)
class BB84Result:
    alice_bits: np.ndarray
    alice_bases: np.ndarray
    bob_bases: np.ndarray
    bob_bits: np.ndarray
    detected: np.ndarray
    sifted: np.ndarray

    @property
    def sent(self) -> int:
        return len(self.alice_bits)

    @property
    def received(self) -> int:
        return int(np.count_nonzero(self.detected))

    @property
    def sifted_bits(self) -> int:
        return int(np.count_nonzero(self.sifted))

    @property
    def errors(self) -> int:
        return int(np.count_nonzero(self.alice_bits[self.sifted] != self.bob_bits[self.sifted]))

    @property
    def qber(self) -> float | None:
        """Observed quantum bit error rate (None for empty sifted key)."""
        return self.errors / self.sifted_bits if self.sifted_bits else None


@dataclass(frozen=True)
class BB84Experiment:
    """Prepare-and-measure BB84 with two ideal conjugate polarization bases.

    Basis 0: H/V, basis 1: D/A. If preparation and measurement bases differ,
    Born's rule gives unbiased random outcomes. Intercept-resend uses a random
    Eve basis on a specified fraction of pulses. Simulated 'bits' are not a
    cryptographically secure key, even when QBER is small.
    """

    channel_transmission: float = 1.0
    detector_efficiency: float = 1.0
    optical_bit_error_probability: float = 0.0
    dark_click_probability_per_gate: float = 0.0
    intercept_resend_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "channel_transmission",
            "detector_efficiency",
            "optical_bit_error_probability",
            "dark_click_probability_per_gate",
            "intercept_resend_fraction",
        ):
            _probability(getattr(self, name), name)

    def run(self, pulses: int, rng: np.random.Generator) -> BB84Result:
        if isinstance(pulses, (bool, np.bool_)) or not isinstance(pulses, (int, np.integer)):
            raise ValueError("pulses must be a positive integer")
        if pulses <= 0:
            raise ValueError("pulses must be a positive integer")

        alice_bits = rng.integers(0, 2, pulses, dtype=np.int8)
        alice_bases = rng.integers(0, 2, pulses, dtype=np.int8)
        bob_bases = rng.integers(0, 2, pulses, dtype=np.int8)

        intercepted = rng.random(pulses) < self.intercept_resend_fraction
        eve_bases = rng.integers(0, 2, pulses, dtype=np.int8)
        eve_random = rng.integers(0, 2, pulses, dtype=np.int8)
        eve_bits = np.where(eve_bases == alice_bases, alice_bits, eve_random)
        received_bases = np.where(intercepted, eve_bases, alice_bases)
        received_bits = np.where(intercepted, eve_bits, alice_bits)

        bob_random = rng.integers(0, 2, pulses, dtype=np.int8)
        measured_bits = np.where(bob_bases == received_bases, received_bits, bob_random)
        noisy_bits = measured_bits ^ (
            rng.random(pulses) < self.optical_bit_error_probability
        ).astype(np.int8)

        photon_click = rng.random(pulses) < (
            self.channel_transmission * self.detector_efficiency
        )
        dark_click = (~photon_click) & (
            rng.random(pulses) < self.dark_click_probability_per_gate
        )
        detected = photon_click | dark_click
        dark_bit = rng.integers(0, 2, pulses, dtype=np.int8)
        observed_bits = np.where(photon_click, noisy_bits, dark_bit)
        sifted = detected & (alice_bases == bob_bases)

        arrays = (alice_bits, alice_bases, bob_bases, observed_bits, detected, sifted)
        for array in arrays:
            array.flags.writeable = False
        return BB84Result(*arrays)
