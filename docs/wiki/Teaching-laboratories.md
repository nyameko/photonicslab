# Teaching laboratories

## Lesson architecture

Every teaching experiment should state prerequisites, physical learning outcomes, equipment layout, equations, numerical implementation, predictions, error sources, exercises, evaluation and how to reproduce results locally. A computer demonstration should explicitly explain what would be different on a real tabletop bench.

## Implemented laboratory

`labs/01-polarization/lesson.ipynb` covers Gaussian field initialization, half-wave plate/polarizer transmission, diffraction, a focusing lens and SLM steering in a deliberately simple NumPy core. The lesson is available for development/teaching trials and awaits independent pedagogical/scientific review.

## Candidate lesson progression

| Stage | Physical question | Observable / criterion |
|---|---|---|
| P01 | How does polarization set transmitted power? | Malus' law with unit conventions |
| P02 | How does retardance change Stokes components? | signed Stokes parameters/unitarity |
| P03 | What controls beam diffraction? | Gaussian analytical propagation and mesh convergence |
| P04 | How do lenses/SLMs change spatial phase? | far-field steering, sampling and energy |
| P05 | How does a threshold detector record clicks? | efficiency, jitter, dead time, dark counts |
| P06 | Can two time-taggers detect correlations? | coincidence window/histogram/accidentals |
| P07 | How does BB84 basis sifting work? | sift fraction and QBER; no secure key |
| Future | HOM, HBT, SPDC/FWM, E91, cavities, fibre, anisotropy | separate validated engines, experimental comparison |

Only the first notebook and identified event-level demonstrations are implemented in the current modern package. The stage labels above are a **suggested teaching organization**, not filenames of already delivered labs.

Sources: [lesson](https://github.com/nyameko/photonicslab/blob/main/labs/01-polarization/lesson.ipynb), [conference experiment](https://github.com/nyameko/photonicslab/blob/main/docs/QUANTUM-CAPE-PHOTONICS.md).