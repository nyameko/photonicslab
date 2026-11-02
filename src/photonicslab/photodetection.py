"""Event-level educational photodetection; distinct from coherent optical power fields.

All times are seconds. Models are idealized non-paralyzable threshold detectors:
no afterpulsing, saturation recovery curve, wavelength dependence or electronics
pulse shaping. Random numbers are passed explicitly for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _nonnegative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _duration(value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError("duration_s must be finite and positive")


def _tags(values: np.ndarray) -> np.ndarray:
    tags = np.asarray(values, dtype=float)
    if tags.ndim != 1 or not np.all(np.isfinite(tags)):
        raise ValueError("time tags must be a finite one-dimensional array")
    return np.sort(tags)


@dataclass(frozen=True)
class PhotonPairSource:
    """Ideal paired arrivals, NOT an SPDC joint spectral amplitude or entangled state.

    Pair creation follows a Poisson process. Photon B has a Gaussian timing skew
    relative to A. Events whose times fall outside the observation interval are
    discarded as pairs; optical transmission loss belongs in detector/channel models.
    """

    pair_rate_hz: float = 20_000.0
    relative_jitter_s: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative(self.pair_rate_hz, "pair_rate_hz")
        _nonnegative(self.relative_jitter_s, "relative_jitter_s")

    def emit(self, duration_s: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        _duration(duration_s)
        n_pairs = rng.poisson(self.pair_rate_hz * duration_s)
        a = rng.uniform(0.0, duration_s, n_pairs)
        skew = (
            rng.normal(0.0, self.relative_jitter_s, n_pairs)
            if self.relative_jitter_s
            else np.zeros(n_pairs)
        )
        b = a + skew
        inside = (b >= 0) & (b < duration_s)
        return np.sort(a[inside]), np.sort(b[inside])


@dataclass(frozen=True)
class SinglePhotonDetector:
    """Threshold click detector with efficiency, dark events, dead time and jitter.

    Apply dead time to physical detection events BEFORE electronic timestamp jitter.
    No photon-number resolution, afterpulsing, spectral efficiency or gated recovery.
    """

    efficiency: float = 0.65
    dark_count_rate_hz: float = 100.0
    dead_time_s: float = 40e-9
    timing_jitter_s: float = 100e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.efficiency) or not 0 <= self.efficiency <= 1:
            raise ValueError("efficiency must be within [0, 1]")
        for name in ("dark_count_rate_hz", "dead_time_s", "timing_jitter_s"):
            _nonnegative(getattr(self, name), name)

    def detect(
        self, arrivals_s: np.ndarray, duration_s: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Return sorted timestamps within [0, duration_s); new, read-only array."""
        _duration(duration_s)
        arrivals = _tags(arrivals_s)
        if np.any((arrivals < 0) | (arrivals >= duration_s)):
            raise ValueError("incident photon arrivals must lie within the observation interval")

        accepted = arrivals[rng.random(len(arrivals)) < self.efficiency]
        n_dark = rng.poisson(self.dark_count_rate_hz * duration_s)
        dark = rng.uniform(0, duration_s, n_dark)
        candidate = np.sort(np.concatenate((accepted, dark)))

        # Non-paralyzable: events during dead time do NOT extend it.
        physical_clicks = []
        last_click = -np.inf
        for t in candidate:
            if t >= last_click + self.dead_time_s:
                physical_clicks.append(t)
                last_click = t
        recorded = np.asarray(physical_clicks, dtype=float)
        if self.timing_jitter_s and recorded.size:
            recorded = recorded + rng.normal(0, self.timing_jitter_s, recorded.size)
        recorded = np.sort(recorded[(recorded >= 0) & (recorded < duration_s)])
        recorded.flags.writeable = False
        return recorded


@dataclass(frozen=True)
class CoincidenceCounter:
    """Two-channel time tag correlation with a full-width coincidence gate.

    count() uses a one-to-one chronological match. histogram() intentionally
    counts ALL A/B tag pairs in each lag bin for correlation analysis, and
    thus does not impose one-to-one matching.
    """

    window_s: float = 2e-9

    def __post_init__(self) -> None:
        if not np.isfinite(self.window_s) or self.window_s <= 0:
            raise ValueError("window_s must be finite and positive")

    def count(self, tags_a_s: np.ndarray, tags_b_s: np.ndarray, *, delay_s: float = 0.0) -> int:
        if not np.isfinite(delay_s):
            raise ValueError("delay_s must be finite")
        a = _tags(tags_a_s)
        b = _tags(tags_b_s) - delay_s
        i = j = matched = 0
        half_window = self.window_s / 2
        while i < len(a) and j < len(b):
            lag = b[j] - a[i]
            if abs(lag) <= half_window:
                matched += 1
                i += 1
                j += 1
            elif lag < -half_window:
                j += 1
            else:
                i += 1
        return matched

    def histogram(
        self, tags_a_s: np.ndarray, tags_b_s: np.ndarray, bin_edges_s: np.ndarray
    ) -> np.ndarray:
        """Count all cross-channel time differences t_B - t_A in each lag bin."""
        a = _tags(tags_a_s)
        b = _tags(tags_b_s)
        edges = np.asarray(bin_edges_s, dtype=float)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("bin_edges_s must be finite and strictly increasing")
        counts = np.zeros(len(edges) - 1, dtype=np.int64)
        for t in a:
            lo = np.searchsorted(b, t + edges[0], side="left")
            hi = np.searchsorted(b, t + edges[-1], side="right")
            if lo < hi:
                counts += np.histogram(b[lo:hi] - t, bins=edges)[0]
        return counts

    def expected_accidentals(self, n_a: int, n_b: int, duration_s: float) -> float:
        """Rare-event, independent-stream approximation R_A R_B W T.

        Boundary effects, bunching, pile-up, dead time and real pair correlations
        are excluded: never subtract this as an exact coincidence correction.
        """
        _duration(duration_s)
        if isinstance(n_a, (bool, np.bool_)) or isinstance(n_b, (bool, np.bool_)):
            raise ValueError("tag counts must be non-negative integers")
        if int(n_a) != n_a or int(n_b) != n_b or n_a < 0 or n_b < 0:
            raise ValueError("tag counts must be non-negative integers")
        return float(n_a * n_b * self.window_s / duration_s)
