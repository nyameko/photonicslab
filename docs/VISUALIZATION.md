# Tabletop schematic visualization (Phase 1)

This lightweight Matplotlib presentation layer describes a 2D arrangement of optical
components, explicitly routed beams, photon paths and electronic connections. **It does
not trace beams, solve a Mach–Zehnder interferometer, infer path phase or simulate photon
statistics.** The numerical models in `photonicslab` remain the sole physics authority.

## Design and deployment

- Core remains NumPy-only: `from photonicslab import ...` does not require Matplotlib.
- `photonicslab.visualization` imports Matplotlib lazily on `plot`/`savefig`.
- `pip install 'photonicslab[notebooks]'` provides Matplotlib in JupyterHub/OCI.
- Headless PNG/SVG/PDF export requires no GUI/X11, GPU, Slurm or running web server.
- Coordinates are **diagram units**, not real distances or automatically computed ray
  geometry. Beam paths must be specified explicitly until validated tracing adapters exist.
- Wavelengths are **vacuum nanometres**, unlike the SI metres of `GaussianBeam`.
- RGB is a **qualitative visual approximation** only. 380–780 nm is rendered in a
  rough spectral color; UV/IR are **neutral dashed**, explicitly labelled by wavelength.
  Brightness/color cannot be used to infer optical power or calibrate a detector.
- Electrical connections are dotted; photon/event paths are dashed. Legend differentiates
  visible and non-visible wavelengths; this is explanatory, not a quantum state plot.
- Glyph selection is decorative and independent of whether the corresponding optical
  component has an implemented simulation model.

## Minimal notebook example

```python
from photonicslab.visualization import TabletopDiagram

table = TabletopDiagram("Polarization demonstration")
table.place("laser", (0, 0), label="HeNe 632.8 nm")
table.place("hwp", (3, 0), label="HWP 22.5°", angle_deg=22.5)
table.place("polarizer", (6, 0), label="Analyzer")
table.place("detector", (9, 0), label="Power meter")
table.route((0.6, 0), (8.4, 0), wavelength_nm=632.8)
fig = table.plot()
table.savefig("polarization.svg")
```

Run `python examples/visualize_tabletop.py` for illustrative Mach–Zehnder and
photon-coincidence schematics. Optional notebook interactive controls can use
`ipywidgets` to modify the scene and redraw the existing Matplotlib figure; avoid
introducing a second UI framework in Phase 1.

## Next phases (not included)

1. Map source/element *IDs* and physical coordinate transforms to solver objects;
   display traced geometry only when obtained from an explicit, verified ray/ABCD model.
2. Add output overlays via explicit observations (power, Jones/Stokes, detector rate,
   timing histogram) using proven solver data instead of interpolating visual brightness.
3. Add optional Plotly `Scatter` adapter for hover, dragging and browser rendering,
   separate from the default Matplotlib/SVG backend. Any drag must update model
   coordinates, validate constraints, and deliberately rerun a compatible solver.
4. For complex optical networks, model ports, propagation, coherent splitting/
   interference, polarization, timing and quantum states in separate validated kernels.

See `docs/ARCHITECTURE.md` and `docs/ECOSYSTEM.md` for physical validity boundaries.
