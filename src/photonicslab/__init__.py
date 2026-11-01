"""Small, explicit classical-optics models. Distances are metres; angles are radians."""

from .bench import BenchResult, OpticalBench, PowerMeter
from .elements import (
    CircularAperture,
    FreeSpace,
    HalfWavePlate,
    LinearPolarizer,
    PhaseSLM,
    QuarterWavePlate,
    ThinLens,
    WavePlate,
)
from .fields import Field, GaussianBeam, Grid

__version__ = "0.6.0a1"

__all__ = [
    "BenchResult",
    "CircularAperture",
    "Field",
    "FreeSpace",
    "GaussianBeam",
    "Grid",
    "HalfWavePlate",
    "LinearPolarizer",
    "OpticalBench",
    "PhaseSLM",
    "PowerMeter",
    "QuarterWavePlate",
    "ThinLens",
    "WavePlate",
]
