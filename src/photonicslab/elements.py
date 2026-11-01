"""Ideal thin optical elements and scalar diffraction of transverse Jones fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from .fields import Field, finite, positive


class OpticalElement(Protocol):
    def apply(self, beam: Field) -> Field: ...


def _axes(angle: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c, s]), np.array([-s, c])


def _jones(beam: Field, matrix: np.ndarray) -> Field:
    return beam.with_amplitude(np.einsum("ab,byx->ayx", matrix, beam.amplitude))


@dataclass(frozen=True)
class LinearPolarizer:
    """Ideal absorbing polarizer; angle is its transmission axis from +x in radians."""

    angle: float = 0.0

    def __post_init__(self) -> None:
        finite(self.angle, "angle")

    def apply(self, beam: Field) -> Field:
        u, _ = _axes(self.angle)
        return _jones(beam, np.outer(u, u))


@dataclass(frozen=True)
class WavePlate:
    """Uniform ideal retarder: fast-axis phase 0, orthogonal-axis phase +retardance.

    A fixed retardance is a single-wavelength model, not a dispersive crystal model.
    """

    retardance: float
    angle: float = 0.0

    def __post_init__(self) -> None:
        finite(self.retardance, "retardance")
        finite(self.angle, "angle")

    def apply(self, beam: Field) -> Field:
        u, v = _axes(self.angle)
        return _jones(beam, np.outer(u, u) + np.exp(1j * self.retardance) * np.outer(v, v))


class HalfWavePlate(WavePlate):
    def __init__(self, angle: float = 0.0):
        super().__init__(retardance=np.pi, angle=angle)


class QuarterWavePlate(WavePlate):
    def __init__(self, angle: float = 0.0):
        super().__init__(retardance=np.pi / 2, angle=angle)


@dataclass(frozen=True)
class ThinLens:
    """Ideal paraxial lens; positive focal_length focuses in the +z direction."""

    focal_length: float

    def __post_init__(self) -> None:
        finite(self.focal_length, "focal_length")
        if self.focal_length == 0:
            raise ValueError("focal_length must be nonzero")

    def apply(self, beam: Field) -> Field:
        x, y = beam.grid.mesh
        k = 2 * np.pi * beam.refractive_index / beam.wavelength
        phase = np.exp(-1j * k * (x**2 + y**2) / (2 * self.focal_length))
        return beam.with_amplitude(beam.amplitude * phase)


@dataclass(frozen=True)
class CircularAperture:
    radius: float

    def __post_init__(self) -> None:
        positive(self.radius, "radius")

    def apply(self, beam: Field) -> Field:
        x, y = beam.grid.mesh
        return beam.with_amplitude(beam.amplitude * (x**2 + y**2 <= self.radius**2))


@dataclass(frozen=True, eq=False)
class PhaseSLM:
    """Ideal phase-only SLM acting on the axis at angle; other axis unchanged.

    Phase is a scalar or a map (ny, nx), in radians. No pixel calibration,
    fill-factor, quantisation, wavelength response or hardware control is implied.
    """

    phase: ArrayLike = field(repr=False)
    angle: float = 0.0

    def __post_init__(self) -> None:
        finite(self.angle, "angle")
        phase = np.array(self.phase, dtype=float, copy=True)
        if phase.ndim not in (0, 2) or not np.isfinite(phase).all():
            raise ValueError("phase must be a finite scalar or 2D map in radians")
        phase.flags.writeable = False
        object.__setattr__(self, "phase", phase)

    def apply(self, beam: Field) -> Field:
        if self.phase.ndim == 2 and self.phase.shape != (beam.grid.ny, beam.grid.nx):
            raise ValueError("phase map must match (grid.ny, grid.nx)")
        u, v = _axes(self.angle)
        along = np.einsum("a,ayx->yx", u, beam.amplitude)
        across = np.einsum("a,ayx->yx", v, beam.amplitude)
        data = u[:, None, None] * along * np.exp(1j * self.phase)
        data += v[:, None, None] * across
        return beam.with_amplitude(data)


@dataclass(frozen=True)
class FreeSpace:
    """Forward homogeneous propagation on a periodic FFT window.

    Fresnel is paraxial. Angular spectrum retains the scalar Helmholtz dispersion
    and decays evanescent components. Neither couples Ex/Ey or solves anisotropic
    Maxwell equations. Both remove the uniform longitudinal carrier phase.
    Sampling and window convergence remain the caller's responsibility.
    """

    distance: float
    method: str = "fresnel"

    def __post_init__(self) -> None:
        finite(self.distance, "distance")
        if self.distance < 0:
            raise ValueError("distance must be nonnegative; inverse propagation is not supported")
        if self.method not in {"fresnel", "angular-spectrum"}:
            raise ValueError("method must be 'fresnel' or 'angular-spectrum'")

    def apply(self, beam: Field) -> Field:
        if self.distance == 0:
            return beam.with_amplitude(beam.amplitude)
        fx, fy = np.meshgrid(
            np.fft.fftfreq(beam.grid.nx, beam.grid.dx),
            np.fft.fftfreq(beam.grid.ny, beam.grid.dy),
            indexing="xy",
        )
        f2 = fx**2 + fy**2
        if self.method == "fresnel":
            phase = -np.pi * beam.wavelength / beam.refractive_index * self.distance * f2
            transfer = np.exp(1j * phase)
        else:
            k = 2 * np.pi * beam.refractive_index / beam.wavelength
            transverse_k2 = (2 * np.pi) ** 2 * f2
            kz = np.sqrt((k**2 - transverse_k2).astype(complex))
            # Algebraic form avoids subtracting nearly equal numbers near the axis.
            delta_k = -transverse_k2 / (kz + k)
            transfer = np.exp(1j * self.distance * delta_k)
        transformed = np.fft.fft2(beam.amplitude, axes=(-2, -1))
        return beam.with_amplitude(np.fft.ifft2(transformed * transfer, axes=(-2, -1)))
