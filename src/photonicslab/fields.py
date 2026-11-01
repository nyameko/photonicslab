"""Sampled transverse Jones envelopes, normalised so integral intensity is watts.

The two amplitudes have units sqrt(W)/m, rather than electric-field units V/m.
Convention: Re[E(x,y,z) exp(i*k*z - i*omega*t)]. The carrier is factored out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def positive(value: float, name: str) -> None:
    finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Grid:
    """Uniform, periodic FFT window with array axes (y, x); no duplicate endpoint."""

    nx: int = 128
    ny: int = 128
    width: float = 4e-3
    height: float = 4e-3

    def __post_init__(self) -> None:
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        positive(self.width, "width")
        positive(self.height, "height")

    @property
    def dx(self) -> float:
        return self.width / self.nx

    @property
    def dy(self) -> float:
        return self.height / self.ny

    @property
    def x(self) -> NDArray[np.float64]:
        return (np.arange(self.nx) - self.nx // 2) * self.dx

    @property
    def y(self) -> NDArray[np.float64]:
        return (np.arange(self.ny) - self.ny // 2) * self.dy

    @property
    def mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.meshgrid(self.x, self.y, indexing="xy")


@dataclass(frozen=True, eq=False)
class Field:
    """Coherent monochromatic transverse field; owned, read-only complex128 samples.

    Jones components propagate independently in a homogeneous isotropic medium.
    This is not a full-vector Maxwell field or a quantum state.
    """

    grid: Grid
    wavelength: float
    amplitude: ArrayLike = field(repr=False)
    refractive_index: float = 1.0

    def __post_init__(self) -> None:
        positive(self.wavelength, "vacuum wavelength")
        positive(self.refractive_index, "refractive_index")
        data = np.array(self.amplitude, dtype=np.complex128, copy=True)
        if data.shape != (2, self.grid.ny, self.grid.nx):
            raise ValueError("amplitude must have shape (2, grid.ny, grid.nx)")
        if not np.isfinite(data).all():
            raise ValueError("amplitude must be finite")
        data.flags.writeable = False
        object.__setattr__(self, "amplitude", data)

    @property
    def intensity(self) -> NDArray[np.float64]:
        """Total irradiance in W/m², summed over both transverse polarizations."""
        return np.sum(np.abs(self.amplitude) ** 2, axis=0)

    @property
    def power(self) -> float:
        return float(np.sum(self.intensity) * self.grid.dx * self.grid.dy)

    @property
    def stokes(self) -> NDArray[np.float64]:
        """Integrated S0..S3 in watts; S3 = 2 Im(conj(Ex) Ey), explicitly signed."""
        ex, ey = self.amplitude
        cross = np.sum(np.conj(ex) * ey) * self.grid.dx * self.grid.dy
        px, py = np.sum(np.abs(self.amplitude) ** 2, axis=(1, 2))
        px, py = np.array([px, py]) * self.grid.dx * self.grid.dy
        return np.array([px + py, px - py, 2 * cross.real, 2 * cross.imag])

    def with_amplitude(self, amplitude: ArrayLike) -> Field:
        return Field(self.grid, self.wavelength, amplitude, self.refractive_index)

    def edge_power_fraction(self) -> float:
        """Power in the outer ~5% border; a clipping/wrap-around diagnostic only."""
        intensity = self.intensity
        total = float(np.sum(intensity))
        if total == 0:
            return 0.0
        bx, by = max(1, self.grid.nx // 20), max(1, self.grid.ny // 20)
        interior = float(np.sum(intensity[by:-by, bx:-bx]))
        return max(0.0, min(1.0, 1.0 - interior / total))


@dataclass(frozen=True)
class GaussianBeam:
    """TEM00 source at its waist; waist is the 1/e² intensity radius in metres.

    The sampled window is normalised to power. Always check window convergence:
    renormalisation alone cannot diagnose a truncated physical beam.
    """

    wavelength: float = 632.8e-9
    waist: float = 0.35e-3
    power: float = 1e-3
    polarization: tuple[complex, complex] = (1, 0)
    refractive_index: float = 1.0

    def __post_init__(self) -> None:
        for name in ("wavelength", "waist", "refractive_index"):
            positive(getattr(self, name), name)
        finite(self.power, "power")
        if self.power < 0:
            raise ValueError("power must be nonnegative")
        jones = np.asarray(self.polarization, dtype=complex)
        if jones.shape != (2,) or not np.isfinite(jones).all() or np.linalg.norm(jones) == 0:
            raise ValueError("polarization must be a finite nonzero Jones pair")
        object.__setattr__(self, "polarization", tuple(jones))

    def emit(self, grid: Grid) -> Field:
        x, y = grid.mesh
        profile = np.exp(-(x**2 + y**2) / self.waist**2)
        norm = float(np.sum(profile**2) * grid.dx * grid.dy)
        jones = np.asarray(self.polarization) / np.linalg.norm(self.polarization)
        amplitude = jones[:, None, None] * profile * np.sqrt(self.power / norm)
        return Field(grid, self.wavelength, amplitude, self.refractive_index)
