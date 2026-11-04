"""Virtual optics laboratory: classical fields and separate event-level models."""

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
from .photodetection import CoincidenceCounter, PhotonPairSource, SinglePhotonDetector
from .qkd import BB84Experiment, BB84Result

__version__ = "0.6.0a1"

__all__ = [
    "BB84Experiment",
    "BB84Result",
    "BenchResult",
    "CircularAperture",
    "CoincidenceCounter",
    "Field",
    "FreeSpace",
    "GaussianBeam",
    "Grid",
    "HalfWavePlate",
    "LinearPolarizer",
    "OpticalBench",
    "PhaseSLM",
    "PhotonPairSource",
    "PowerMeter",
    "QuarterWavePlate",
    "SinglePhotonDetector",
    "ThinLens",
    "WavePlate",
]
