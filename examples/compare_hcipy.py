"""Optional independent reference check: install HCIPy in a separate environment."""

import json

import hcipy as hp
import numpy as np

from photonicslab import FreeSpace, GaussianBeam, Grid, HalfWavePlate, LinearPolarizer


def main() -> None:
    grid = Grid(128, 128)
    beam = GaussianBeam(waist=0.25e-3).emit(grid)
    reference_grid = hp.make_pupil_grid(grid.nx, grid.width)
    # Identical input samples and spacing. Propagation is translation-invariant;
    # no lens/coordinate mask is compared on the grids' different centre offsets.
    reference_beam = hp.Wavefront(
        hp.Field(beam.amplitude.reshape(2, -1), reference_grid),
        beam.wavelength,
    )
    reference_output = hp.LinearPolarizer(0)(hp.HalfWavePlate(np.pi / 8)(reference_beam))
    output = LinearPolarizer().apply(HalfWavePlate(np.pi / 8).apply(beam))
    polarization_error = abs(reference_output.total_power / output.power - 1)
    assert polarization_error < 1e-12

    propagated = FreeSpace(0.2).apply(beam)
    errors = {}
    for oversampling in (1, 2):
        reference = hp.FresnelPropagator(
            reference_grid,
            0.2,
            num_oversampling=oversampling,
            zero_padding=2,
        )(reference_beam)
        intensity = np.asarray(reference.intensity).reshape(grid.ny, grid.nx)
        errors[str(oversampling)] = {
            "relative_intensity_l2": float(
                np.linalg.norm(intensity - propagated.intensity) / np.linalg.norm(intensity)
            ),
            "relative_power_difference": float(abs(reference.total_power / propagated.power - 1)),
        }
    # Point-sampled transfer factors match our FFT model. HCIPy's default of two
    # supersamples averages transfer factors and is a different discrete operator.
    assert errors["1"]["relative_intensity_l2"] < 1e-10
    print(
        json.dumps(
            {
                "hcipy_version": hp.__version__,
                "numpy_version": np.__version__,
                "polarization_relative_power_difference": float(polarization_error),
                "fresnel_by_hcipy_transfer_oversampling": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
