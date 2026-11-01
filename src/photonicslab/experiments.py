"""Small credential-free teaching experiments, usable without a notebook server."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .bench import OpticalBench
from .elements import HalfWavePlate, LinearPolarizer
from .fields import GaussianBeam, Grid, finite


def polarization_experiment(
    *,
    angle: float = np.pi / 8,
    samples: int = 64,
) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    """Horizontal laser -> half-wave plate at angle -> horizontal analyser.

    The independent prediction is T = cos²(2 angle), not cos²(angle).
    """
    finite(angle, "angle")
    grid = Grid(samples, samples)
    source = GaussianBeam().emit(grid)
    result = OpticalBench([HalfWavePlate(angle), LinearPolarizer()]).run(source)
    transmission = result.output.power / source.power
    expected = float(np.cos(2 * angle) ** 2)
    error = abs(transmission - expected)
    summary = {
        "workflow": "photonics-polarization",
        "model": "coherent-monochromatic-jones",
        "input_power_w": source.power,
        "output_power_w": result.output.power,
        "transmission": transmission,
        "expected_transmission": expected,
        "absolute_error": error,
        "tolerance": 1e-12,
        "passed": error < 1e-12,
    }
    arrays = {
        "x_m": grid.x,
        "y_m": grid.y,
        "amplitude": result.output.amplitude,
        "intensity_w_m2": result.output.intensity,
    }
    return summary, arrays, [asdict(item) for item in result.observations]


def polarization_parameters(*, angle: float, samples: int) -> dict:
    """Complete inputs for this fixed, versioned experiment, in SI units."""
    finite(angle, "angle")
    source = GaussianBeam()
    return {
        "lesson_id": "photonics.polarization.01",
        "lesson_version": 1,
        "half_wave_plate_angle_rad": angle,
        "analyser_angle_rad": 0.0,
        "grid": asdict(Grid(samples, samples)),
        "source": {
            "vacuum_wavelength_m": source.wavelength,
            "waist_m": source.waist,
            "power_w": source.power,
            "polarization_jones": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in source.polarization
            ],
            "refractive_index": source.refractive_index,
        },
    }
