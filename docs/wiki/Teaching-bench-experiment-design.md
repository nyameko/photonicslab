# Designing a physical-to-virtual tabletop lesson

## Teaching objective

PhotonicsLab should help students connect laboratory components, an explanatory mathematical model and quantitative observations. A virtual bench does not replace laser safety training, experimental calibration or a physical detector's actual noise/response. Keep the lesson's primary question narrow enough that its predictions can be checked before a user modifies parameters.

## Suggested lesson template

| Section | Required material |
|---|---|
| Audience and prerequisites | level; optics/linear algebra/programming preparation |
| Physical apparatus | real bench diagram, source, wavelengths, alignments, detector |
| Model | state representation, governing equations, idealizations and omissions |
| Predictions | expected numerical power, polarization, fringe or count with units |
| Implementation | runnable versioned Python API, known seeds, grid, parameters |
| Validation | analytical limit, numerical convergence, uncertainty and tests |
| Exercises | independent variations and analysis/questions |
| Limitations | what it cannot teach or physically certify |
| Provenance | source commit, package versions, lesson/environment, data outputs |

## Example polarization bench

A linearly polarized coherent laser drives a half-wave plate then a polarizer and ideal power meter. Predict Malus' law; vary the waveplate angle; compare ideal output to source-normalized result and record any FFT/grid errors when spatial propagation is added. An extension can insert an SLM/lens and discuss diffraction/polarization coupling limitations. The existing lesson at `labs/01-polarization/lesson.ipynb` is a development reference awaiting independent teaching review.

## Event-level bench sequence

Treat photon source, two threshold detectors, time taggers and coincidence counter as a **separate** experiment from continuous Jones-field propagation. First simulate singles, dark counts and dead time. Then examine coincidence peaks versus accidentals and timing gate. Finally model ideal BB84 basis sifting and intercept/resend with explicit statement that no secure key is produced.

## Research integrity

Use explanatory units and error bars, reject outcomes based solely on visual plausibility and label synthetic/experimental results. An LLM can help learners diagnose commands and assumptions but cannot certify a numerical model as calibrated or a student as competent without verified assessments.

References: [first lab notebook](https://github.com/nyameko/photonicslab/blob/main/labs/01-polarization/lesson.ipynb), [architecture](https://github.com/nyameko/photonicslab/blob/main/docs/ARCHITECTURE.md), [conference QKD lesson design](https://github.com/nyameko/photonicslab/blob/main/docs/QUANTUM-CAPE-PHOTONICS.md).