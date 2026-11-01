"""Analytical acceptance checks, independent of the implementation's matrix helpers."""

import numpy as np
import pytest

from photonicslab import (
    CircularAperture,
    Field,
    FreeSpace,
    GaussianBeam,
    Grid,
    HalfWavePlate,
    LinearPolarizer,
    OpticalBench,
    PhaseSLM,
    PowerMeter,
    QuarterWavePlate,
    ThinLens,
    WavePlate,
)


def radius_x(beam):
    """For a Gaussian, the 1/e² intensity radius is twice its standard deviation."""
    weights = beam.intensity.sum(axis=0)
    centroid = np.sum(beam.grid.x * weights) / weights.sum()
    return 2 * np.sqrt(np.sum((beam.grid.x - centroid) ** 2 * weights) / weights.sum())


@pytest.fixture
def source():
    return GaussianBeam(power=2e-3).emit(Grid(128, 96, 4e-3, 3e-3))


@pytest.mark.parametrize("angle", np.linspace(-np.pi, np.pi, 17))
def test_malus_law_with_absolute_power(source, angle):
    transmitted = LinearPolarizer(angle).apply(source)
    assert transmitted.power == pytest.approx(source.power * np.cos(angle) ** 2, abs=1e-17)


@pytest.mark.parametrize("angle", [0, np.pi / 8, np.pi / 4, np.pi / 2, -np.pi / 6])
def test_half_wave_plate_rotates_linear_polarization(source, angle):
    result = HalfWavePlate(angle).apply(source)
    np.testing.assert_allclose(
        result.stokes / source.power,
        [1, np.cos(4 * angle), np.sin(4 * angle), 0],
        atol=1e-14,
    )


def test_quarter_wave_plate_circular_stokes_convention(source):
    result = QuarterWavePlate(np.pi / 4).apply(source)
    np.testing.assert_allclose(result.stokes / source.power, [1, 0, 0, -1], atol=1e-14)
    assert LinearPolarizer().apply(result).power == pytest.approx(source.power / 2)


def test_inverse_retarders_restore_elliptical_field():
    beam = GaussianBeam(polarization=(1, 0.4j)).emit(Grid())
    result = OpticalBench([WavePlate(1.7, 0.35), WavePlate(-1.7, 0.35)]).run(beam)
    np.testing.assert_allclose(result.output.amplitude, beam.amplitude, atol=1e-13)


def test_components_do_not_mutate_or_renormalize_source(source):
    before = source.amplitude.copy()
    result = OpticalBench([HalfWavePlate(np.pi / 8), LinearPolarizer()]).run(source)
    np.testing.assert_array_equal(source.amplitude, before)
    assert not source.amplitude.flags.writeable
    assert not result.output.amplitude.flags.writeable
    assert [o.power_w for o in result.observations] == pytest.approx([2e-3, 2e-3, 1e-3])
    assert PowerMeter().measure(result.output) == pytest.approx(1e-3)


@pytest.mark.parametrize("method", ["fresnel", "angular-spectrum"])
@pytest.mark.parametrize("index", [1.0, 1.5])
def test_gaussian_diffraction_matches_rayleigh_law(method, index):
    wavelength, waist, distance = 632.8e-9, 0.25e-3, 0.4
    beam = GaussianBeam(wavelength, waist, refractive_index=index).emit(Grid(256, 192))
    propagated = FreeSpace(distance, method).apply(beam)
    rayleigh = np.pi * index * waist**2 / wavelength
    predicted = waist * np.sqrt(1 + (distance / rayleigh) ** 2)
    assert radius_x(propagated) == pytest.approx(predicted, rel=3e-5)
    assert propagated.power == pytest.approx(beam.power, rel=1e-13)
    assert propagated.edge_power_fraction() < 1e-8


def test_lens_focus_sign_and_diffraction_spot_size():
    wavelength, waist, focal = 632.8e-9, 0.35e-3, 0.2
    beam = GaussianBeam(wavelength, waist).emit(Grid(256, 256))
    result = OpticalBench([ThinLens(focal), FreeSpace(focal)]).run(beam)
    predicted_at_focal_plane = wavelength * focal / (np.pi * waist)
    assert radius_x(result.output) == pytest.approx(predicted_at_focal_plane, rel=1e-5)
    assert result.output.power == pytest.approx(beam.power, rel=1e-13)


def test_aperture_transmission_matches_gaussian_integral():
    waist, radius = 0.5e-3, 0.35e-3
    beam = GaussianBeam(waist=waist).emit(Grid(512, 512))
    predicted = 1 - np.exp(-2 * radius**2 / waist**2)
    assert CircularAperture(radius).apply(beam).power / beam.power == pytest.approx(
        predicted,
        abs=0.002,
    )


def test_slm_modulates_only_selected_polarization():
    grid = Grid(64, 48)
    beam = GaussianBeam(polarization=(1, 1j)).emit(grid)
    phase = np.full((grid.ny, grid.nx), 0.7)
    slm = PhaseSLM(phase)
    phase[:] = 0  # The device owns its phase map.
    output = slm.apply(beam)
    np.testing.assert_allclose(output.amplitude[0], beam.amplitude[0] * np.exp(0.7j))
    np.testing.assert_allclose(output.amplitude[1], beam.amplitude[1])
    assert output.power == pytest.approx(beam.power)


def test_slm_phase_ramp_steers_beam_in_positive_x():
    grid = Grid(256, 192)
    beam = GaussianBeam(waist=0.12e-3).emit(grid)
    fx, distance = 10 / grid.width, 0.2
    phase = 2 * np.pi * fx * grid.mesh[0]
    output = OpticalBench([PhaseSLM(phase), FreeSpace(distance)]).run(beam).output
    weights = output.intensity.sum(axis=0)
    measured = np.sum(grid.x * weights) / weights.sum()
    assert measured == pytest.approx(distance * beam.wavelength * fx, abs=1e-9)


@pytest.mark.parametrize("method", ["fresnel", "angular-spectrum"])
def test_plane_wave_eigenvalue_and_propagation_composition(method):
    grid = Grid(48, 32, 1e-3, 0.8e-3)
    wavelength, z, fx, fy = 532e-9, 0.03, 2 / grid.width, -3 / grid.height
    x, y = grid.mesh
    plane = np.exp(2j * np.pi * (fx * x + fy * y))
    beam = Field(grid, wavelength, [plane, 0.3j * plane])
    if method == "fresnel":
        phase = -np.pi * wavelength * z * (fx**2 + fy**2)
    else:
        k = 2 * np.pi / wavelength
        phase = z * (np.sqrt(k**2 - (2 * np.pi) ** 2 * (fx**2 + fy**2)) - k)
    result = FreeSpace(z, method).apply(beam)
    np.testing.assert_allclose(result.amplitude, beam.amplitude * np.exp(1j * phase), atol=1e-10)
    twice = FreeSpace(z / 2, method).apply(FreeSpace(z / 2, method).apply(beam))
    np.testing.assert_allclose(result.amplitude, twice.amplitude, atol=1e-13)


def test_angular_spectrum_evanescent_mode_decays_instead_of_growing():
    wavelength = 1e-6
    grid = Grid(64, 48, 4 * wavelength, 4 * wavelength)
    fx, z = 6 / grid.width, wavelength / 10
    plane = np.exp(2j * np.pi * fx * grid.mesh[0])
    beam = Field(grid, wavelength, [plane, np.zeros_like(plane)])
    alpha = 2 * np.pi * np.sqrt(fx**2 - 1 / wavelength**2)
    expected_norm_ratio = np.exp(-2 * alpha * z)
    assert FreeSpace(z, "angular-spectrum").apply(beam).power / beam.power == pytest.approx(
        expected_norm_ratio,
        rel=1e-12,
    )


def test_grid_refinement_and_window_convergence():
    widths = []
    for samples, extent in [(128, 4e-3), (256, 4e-3), (384, 6e-3)]:
        beam = GaussianBeam(waist=0.25e-3).emit(Grid(samples, samples, extent, extent))
        widths.append(radius_x(FreeSpace(0.3).apply(beam)))
    np.testing.assert_allclose(widths, widths[0], rtol=1e-8)


def test_zero_power_and_zero_distance_are_well_defined():
    beam = GaussianBeam(power=0).emit(Grid())
    result = FreeSpace(0).apply(beam)
    assert result.power == result.edge_power_fraction() == 0
    np.testing.assert_array_equal(result.amplitude, beam.amplitude)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nx": 1},
        {"ny": 4.5},
        {"nx": True},
        {"width": 0},
        {"height": np.inf},
    ],
)
def test_invalid_grid_rejected(kwargs):
    with pytest.raises(ValueError):
        Grid(**kwargs)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GaussianBeam(wavelength=-1),
        lambda: GaussianBeam(waist=0),
        lambda: GaussianBeam(power=-1),
        lambda: GaussianBeam(polarization=(0, 0)),
        lambda: GaussianBeam(polarization=(1, np.nan)),
        lambda: ThinLens(0),
        lambda: FreeSpace(-1),
        lambda: FreeSpace(np.nan),
        lambda: FreeSpace(1, "gpu"),
        lambda: WavePlate(np.inf),
        lambda: LinearPolarizer(np.nan),
        lambda: PhaseSLM([1, 2]),
        lambda: PhaseSLM(np.nan),
        lambda: CircularAperture(0),
    ],
)
def test_invalid_physics_parameters_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_invalid_field_and_slm_shape_rejected(source):
    with pytest.raises(ValueError, match="shape"):
        Field(Grid(), 532e-9, np.ones((128, 128)))
    with pytest.raises(ValueError, match="finite"):
        Field(Grid(2, 2), 532e-9, np.full((2, 2, 2), np.nan))
    with pytest.raises(ValueError, match="match"):
        PhaseSLM(np.zeros((2, 2))).apply(source)
    with pytest.raises(TypeError, match="apply"):
        OpticalBench([object()])
