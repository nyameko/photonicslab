"""Lightweight tabletop *schematics*, separate from numerical optics engines.

No optical propagation, Jones calculus or photon detection is performed here.
All positions are in arbitrary diagram units; wavelengths are vacuum nanometres.
Requires matplotlib only when rendering. SVG/PNG output works headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal


Glyph = Literal[
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
]
PathKind = Literal["beam", "photon", "electrical"]
KNOWN_GLYPHS = frozenset(Glyph.__args__)
KNOWN_PATHS = frozenset(PathKind.__args__)


def _point(value: tuple[float, float]) -> tuple[float, float]:
    if len(value) != 2 or not all(isfinite(v) for v in value):
        raise ValueError("points must have two finite coordinates")
    return (float(value[0]), float(value[1]))


def wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float] | None:
    """Approximate display RGB for visible vacuum wavelengths; not colorimetry.

    Returns None outside 380–780 nm: IR/UV must not be falsely shown as visible.
    Does not represent beam power, spectral linewidth, perceived brightness or a
    calibrated monitor's color response.
    """
    if not isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be finite and positive")
    w = wavelength_nm
    if not 380 <= w <= 780:
        return None
    if w < 440:
        rgb = (-(w - 440) / 60, 0.0, 1.0)
    elif w < 490:
        rgb = (0.0, (w - 440) / 50, 1.0)
    elif w < 510:
        rgb = (0.0, 1.0, -(w - 510) / 20)
    elif w < 580:
        rgb = ((w - 510) / 70, 1.0, 0.0)
    elif w < 645:
        rgb = (1.0, -(w - 645) / 65, 0.0)
    else:
        rgb = (1.0, 0.0, 0.0)
    strength = (
        0.3 + 0.7 * (w - 380) / 40 if w < 420 else 0.3 + 0.7 * (780 - w) / 80 if w > 700 else 1.0
    )
    return tuple(float((channel * strength) ** 0.8) for channel in rgb)


@dataclass(frozen=True)
class OpticGlyph:
    kind: Glyph
    at: tuple[float, float]
    label: str = ""
    angle_deg: float = 0.0
    size: float = 0.8

    def __post_init__(self) -> None:
        if self.kind not in KNOWN_GLYPHS:
            raise ValueError(f"unknown glyph: {self.kind}")
        object.__setattr__(self, "at", _point(self.at))
        if not isfinite(self.angle_deg):
            raise ValueError("angle_deg must be finite")
        if not isfinite(self.size) or self.size <= 0:
            raise ValueError("size must be finite and positive")


@dataclass(frozen=True)
class OpticalPath:
    points: tuple[tuple[float, float], ...]
    wavelength_nm: float | None = None
    kind: PathKind = "beam"
    label: str = ""

    def __post_init__(self) -> None:
        points = tuple(_point(point) for point in self.points)
        if len(points) < 2 or all(a == b for a, b in zip(points, points[1:], strict=False)):
            raise ValueError("a path needs at least two distinct points")
        object.__setattr__(self, "points", points)
        if self.kind not in KNOWN_PATHS:
            raise ValueError(f"unknown path kind: {self.kind}")
        if self.kind != "electrical" and self.wavelength_nm is None:
            raise ValueError("optical paths require an explicit vacuum wavelength in nm")
        if self.kind == "electrical" and self.wavelength_nm is not None:
            raise ValueError("electrical signal paths have no optical wavelength")
        if self.wavelength_nm is not None:
            wavelength_to_rgb(self.wavelength_nm)


@dataclass
class TabletopDiagram:
    """User-defined optical table layout (not a ray-tracing or quantum solver)."""

    title: str = "Optical table"
    optics: list[OpticGlyph] = field(default_factory=list)
    paths: list[OpticalPath] = field(default_factory=list)

    def place(
        self,
        kind: Glyph,
        at: tuple[float, float],
        *,
        label: str = "",
        angle_deg: float = 0.0,
        size: float = 0.8,
    ) -> OpticGlyph:
        optic = OpticGlyph(kind, at, label, angle_deg, size)
        self.optics.append(optic)
        return optic

    def route(
        self,
        *points: tuple[float, float],
        wavelength_nm: float | None = None,
        kind: PathKind = "beam",
        label: str = "",
    ) -> OpticalPath:
        path = OpticalPath(points, wavelength_nm, kind, label)
        self.paths.append(path)
        return path

    def plot(self, *, ax=None, figsize: tuple[float, float] = (11, 6), grid: bool = False):
        """Return a Matplotlib Figure. Call fig.savefig('bench.svg') if desired."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle
            from matplotlib.transforms import Affine2D
        except ImportError as exc:
            raise ImportError("Install 'photonicslab[notebooks]' for Matplotlib rendering") from exc

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        else:
            fig = ax.figure
        ink, muted, housing, edge = "#e2e8f0", "#94a3b8", "#17243a", "#7e90a8"
        ax.set_facecolor("#0b1324")
        fig.set_facecolor("#0b1324")
        ax.set_aspect("equal", adjustable="box")
        extents = []
        legend = set()
        for path in self.paths:
            xs, ys = zip(*path.points, strict=True)
            extents.extend(path.points)
            if path.kind == "electrical":
                color, style, width = muted, ":", 1.8
            else:
                color = wavelength_to_rgb(path.wavelength_nm)
                style = "--" if path.kind == "photon" or color is None else "-"
                color = color or "#b8c3d1"
                width = 2.4 if path.kind == "beam" else 1.9
                legend.add((path.wavelength_nm, wavelength_to_rgb(path.wavelength_nm) is None))
            ax.plot(xs, ys, color=color, ls=style, lw=width, zorder=2, solid_capstyle="round")
            if path.label:
                mid = path.points[len(path.points) // 2]
                ax.annotate(
                    path.label,
                    mid,
                    (0, 9),
                    textcoords="offset points",
                    ha="center",
                    color=ink,
                    fontsize=8,
                    zorder=5,
                )

        for item in self.optics:
            x, y = item.at
            s = item.size
            extents.extend(((x - s, y - s), (x + s, y + s)))
            transform = Affine2D().rotate_deg_around(x, y, item.angle_deg) + ax.transData

            def patch(p, *, transform=transform):
                p.set_transform(transform)
                p.set_zorder(4)
                ax.add_patch(p)

            if item.kind in {"laser", "detector", "spd", "counter"}:
                patch(
                    FancyBboxPatch(
                        (x - 0.8 * s, y - 0.45 * s),
                        1.6 * s,
                        0.9 * s,
                        boxstyle=f"round,pad=0,rounding_size={0.14 * s}",
                        facecolor=housing,
                        edgecolor=edge,
                        lw=1.5,
                    )
                )
                if item.kind == "laser":
                    patch(
                        Circle(
                            (x + 0.73 * s, y),
                            0.105 * s,
                            facecolor="#f8fafc",
                            edgecolor=edge,
                            lw=0.8,
                        )
                    )
                elif item.kind in {"detector", "spd"}:
                    patch(
                        Circle(
                            (x - 0.72 * s, y),
                            0.15 * s,
                            facecolor="#38bdf8",
                            edgecolor="#bae6fd",
                            lw=0.8,
                        )
                    )
                else:
                    patch(
                        Rectangle(
                            (x - 0.25 * s, y - 0.15 * s),
                            0.5 * s,
                            0.3 * s,
                            facecolor="#0f766e",
                            edgecolor="#5eead4",
                        )
                    )
            elif item.kind == "beamsplitter":
                patch(
                    Rectangle(
                        (x - s * 0.39, y - s * 0.39),
                        s * 0.78,
                        s * 0.78,
                        facecolor="#22435e",
                        edgecolor="#9cc8e4",
                        lw=1.4,
                    )
                )
                ax.plot(
                    [x - 0.37 * s, x + 0.37 * s],
                    [y - 0.37 * s, y + 0.37 * s],
                    transform=transform,
                    color="#d9f2ff",
                    lw=2,
                    zorder=5,
                )
            elif item.kind == "mirror":
                patch(
                    Rectangle(
                        (x - 0.45 * s, y - 0.08 * s),
                        0.9 * s,
                        0.16 * s,
                        facecolor="#80bff0",
                        edgecolor="#eefaff",
                        lw=1.0,
                    )
                )
            elif item.kind == "lens":
                patch(
                    Ellipse(
                        (x, y),
                        width=0.3 * s,
                        height=1.35 * s,
                        facecolor="#3c89bd",
                        edgecolor="#bde9fd",
                        lw=1.3,
                    )
                )
            elif item.kind in {"hwp", "qwp", "polarizer"}:
                patch(
                    Circle(
                        (x, y), radius=0.52 * s, facecolor="#215b66", edgecolor="#b2f5ea", lw=1.5
                    )
                )
                ax.plot(
                    [x - 0.33 * s, x + 0.33 * s],
                    [y, y],
                    transform=transform,
                    color="#d1fae5",
                    lw=1.5,
                    zorder=5,
                )
            elif item.kind in {"slm", "crystal"}:
                patch(
                    Rectangle(
                        (x - 0.29 * s, y - 0.57 * s),
                        0.58 * s,
                        1.14 * s,
                        facecolor="#7658a0" if item.kind == "slm" else "#b28d35",
                        edgecolor="#ddd6fe" if item.kind == "slm" else "#fde68a",
                    )
                )
            elif item.kind == "aperture":
                patch(
                    Rectangle(
                        (x - 0.12 * s, y - 0.65 * s),
                        0.24 * s,
                        1.30 * s,
                        facecolor="#475569",
                        edgecolor=ink,
                    )
                )
                patch(Circle((x, y), 0.12 * s, facecolor="#0b1324", edgecolor=muted))
            elif item.kind == "fiber":
                patch(
                    Polygon(
                        [
                            (x - 0.5 * s, y - 0.18 * s),
                            (x + 0.5 * s, y - 0.18 * s),
                            (x + 0.5 * s, y + 0.18 * s),
                            (x - 0.5 * s, y + 0.18 * s),
                        ],
                        facecolor="#1d4ed8",
                        edgecolor="#93c5fd",
                    )
                )
            if item.label:
                ax.annotate(
                    item.label,
                    (x, y),
                    (0, -item.size * 23 - 9),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    color=ink,
                    fontsize=8,
                    zorder=6,
                )
        if extents:
            left, right = min(p[0] for p in extents), max(p[0] for p in extents)
            bottom, top = min(p[1] for p in extents), max(p[1] for p in extents)
            dx, dy = max(1, right - left), max(1, top - bottom)
            ax.set_xlim(left - 0.12 * dx - 0.4, right + 0.12 * dx + 0.4)
            ax.set_ylim(bottom - 0.22 * dy - 0.3, top + 0.20 * dy + 0.3)
        ax.set_title(self.title, color=ink, fontsize=14, loc="left", pad=16)
        ax.tick_params(colors=muted)
        if grid:
            ax.grid(color="#26364d", lw=0.6, alpha=0.7)
            ax.set_xlabel("Diagram x (arbitrary units)", color=muted)
            ax.set_ylabel("Diagram y (arbitrary units)", color=muted)
        else:
            ax.axis("off")
        for w, invisible in sorted(legend):
            label = f"{w:g} nm" + (" (outside visible range)" if invisible else "")
            color = wavelength_to_rgb(w) or "#b8c3d1"
            ax.plot([], [], label=label, color=color, ls="--" if invisible else "-")
        if legend:
            ax.legend(
                loc="upper right",
                facecolor="#17243a",
                edgecolor="#44546b",
                labelcolor=ink,
                fontsize=8,
            )
        return fig

    def savefig(self, filename: str | Path, **kwargs) -> None:
        """Save headless-compatible PNG/SVG/PDF and close the created Figure."""
        import matplotlib.pyplot as plt

        fig = self.plot()
        try:
            fig.savefig(filename, bbox_inches="tight", **kwargs)
        finally:
            plt.close(fig)
