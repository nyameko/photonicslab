"""Unit tests for display semantics, no optics simulation implied."""

import math

import pytest

from photonicslab.visualization import OpticalPath, OpticGlyph, TabletopDiagram, wavelength_to_rgb


@pytest.mark.parametrize("wavelength, dominant", [(450, 2), (532, 1), (650, 0)])
def test_visible_spectral_color(wavelength, dominant):
    rgb = wavelength_to_rgb(wavelength)
    assert len(rgb) == 3
    assert all(0 <= channel <= 1 for channel in rgb)
    assert rgb[dominant] == max(rgb)


@pytest.mark.parametrize("wavelength", [200, 810, 1064, 1550])
def test_nonvisible_wavelengths_never_faked_as_visible(wavelength):
    assert wavelength_to_rgb(wavelength) is None


@pytest.mark.parametrize("wavelength", [0, -633, float("nan"), float("inf")])
def test_invalid_wavelength_rejected(wavelength):
    with pytest.raises(ValueError, match="wavelength"):
        wavelength_to_rgb(wavelength)


def test_layout_model_geometry_and_units():
    scene = TabletopDiagram("A demo")
    item = scene.place("hwp", (1, 2), label="HWP", angle_deg=22.5)
    path = scene.route((0, 2), (1, 2), (3, 2), wavelength_nm=633)
    assert scene.optics == [item]
    assert path.points == ((0, 2), (1, 2), (3, 2))
    assert path.wavelength_nm == 633
    assert scene.title == "A demo"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "fake", "at": (0, 0)},
        {"kind": "lens", "at": (float("nan"), 2)},
        {"kind": "lens", "at": (0, 0), "angle_deg": float("inf")},
        {"kind": "lens", "at": (0, 0), "size": -1},
    ],
)
def test_invalid_optical_element_geometry(kwargs):
    with pytest.raises(ValueError):
        OpticGlyph(**kwargs)


def test_path_validation_and_separation_of_electronics():
    with pytest.raises(ValueError):
        OpticalPath(((0, 0), (1, 0)))
    with pytest.raises(ValueError):
        OpticalPath(((0, 0), (0, 0)), 532)
    with pytest.raises(ValueError):
        OpticalPath(((0, 0), (1, 0)), 532, "electrical")
    with pytest.raises(ValueError):
        OpticalPath(((0, 0), (1, math.inf)), 532)
    assert OpticalPath(((0, 0), (1, 0)), kind="electrical").wavelength_nm is None


def test_headless_export_svg_png(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    table = TabletopDiagram("Export")
    for kind in [
        "laser",
        "mirror",
        "beamsplitter",
        "lens",
        "polarizer",
        "hwp",
        "qwp",
        "slm",
        "aperture",
        "crystal",
        "fiber",
        "detector",
        "spd",
        "counter",
    ]:
        table.place(kind, (len(table.optics) * 2.0, 0), label=kind)
    table.route((0, 0), (26, 0), wavelength_nm=532)
    table.route((1, 1), (4, 1), wavelength_nm=810, kind="photon")
    table.route((2, 2), (6, 2), kind="electrical")
    before = set(plt.get_fignums())
    for extension in ("svg", "png"):
        output = tmp_path / f"table.{extension}"
        table.savefig(output)
        assert output.is_file()
        assert output.stat().st_size > 1000
    assert set(plt.get_fignums()) == before
