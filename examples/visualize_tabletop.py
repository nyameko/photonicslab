"""Illustrated geometry for two *schematic*, not numerically traced, optical tables.

Run: python examples/visualize_tabletop.py
Outputs: mach_zehnder.svg, photon_coincidences.svg in the current directory.
"""

from photonicslab.visualization import TabletopDiagram


def mach_zehnder() -> TabletopDiagram:
    table = TabletopDiagram("Mach–Zehnder interferometer | optical geometry only")
    table.place("laser", (-4, 0), label="Laser 532 nm", size=1.0)
    table.place("beamsplitter", (0, 0), label="BS1", size=1.0)
    table.place("mirror", (0, 4), label="M1", angle_deg=45, size=1.0)
    table.place("mirror", (8, 0), label="M2", angle_deg=45, size=1.0)
    table.place("beamsplitter", (8, 4), label="BS2", size=1.0)
    table.place("detector", (12, 4), label="D0", size=0.9)
    table.place("detector", (8, 7.3), label="D1", angle_deg=-90, size=0.9)
    for route in [
        ((-3.1, 0), (0, 0)),
        ((0, 0), (0, 4), (8, 4)),
        ((0, 0), (8, 0), (8, 4)),
        ((8, 4), (11.2, 4)),
        ((8, 4), (8, 6.5)),
    ]:
        table.route(*route, wavelength_nm=532)
    return table


def photon_coincidences() -> TabletopDiagram:
    table = TabletopDiagram("Coincidence counting | illustrative paths, 810 nm photons")
    table.place("crystal", (0, 0), label="Pair source (concept)", size=1.1)
    table.place("spd", (6, 3), label="SPD A", size=1.0)
    table.place("spd", (6, -3), label="SPD B", size=1.0)
    table.place("counter", (10, 0), label="Coincidence counter", size=1.0)
    table.route((0, 0), (5.2, 3), wavelength_nm=810, kind="photon")
    table.route((0, 0), (5.2, -3), wavelength_nm=810, kind="photon")
    table.route((6.8, 3), (8.4, 3), (8.4, 0.2), (9.2, 0.2), kind="electrical")
    table.route((6.8, -3), (8.4, -3), (8.4, -0.2), (9.2, -0.2), kind="electrical")
    return table


if __name__ == "__main__":
    mach_zehnder().savefig("mach_zehnder.svg")
    photon_coincidences().savefig("photon_coincidences.svg")
