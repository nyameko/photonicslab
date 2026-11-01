"""Compose optical elements without coupling physics to storage or a scheduler."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .elements import OpticalElement
from .fields import Field


@dataclass(frozen=True)
class Observation:
    index: int
    element: str
    power_w: float
    edge_power_fraction: float


@dataclass(frozen=True)
class BenchResult:
    output: Field
    observations: tuple[Observation, ...]


class OpticalBench:
    """A sequential, single-pass optical train. Branches and cavities need other models."""

    def __init__(self, elements: Sequence[OpticalElement]):
        self.elements = tuple(elements)
        if any(not callable(getattr(element, "apply", None)) for element in self.elements):
            raise TypeError("each optical element must provide apply(field)")

    def run(self, beam: Field) -> BenchResult:
        observations = [Observation(0, "source", beam.power, beam.edge_power_fraction())]
        current = beam
        for index, element in enumerate(self.elements, 1):
            current = element.apply(current)
            observations.append(
                Observation(
                    index,
                    type(element).__name__,
                    current.power,
                    current.edge_power_fraction(),
                )
            )
        return BenchResult(current, tuple(observations))


@dataclass(frozen=True)
class PowerMeter:
    """Ideal noiseless detector; returns watts without modifying the field."""

    def measure(self, beam: Field) -> float:
        return beam.power
