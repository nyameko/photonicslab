# Quantum Cape: tabletop single-photon detection and QKD

Scope: educational event-level Monte Carlo demonstrators, not hardware control,
a calibrated detector response, a full SPDC/FWM source, or a secure-key product.

## Conference research connections (2–3 November 2026)

- Rory Pentz (CPUT): single-photon detector development; build an APD/SPAD
  statistical detection bench, then investigate bias/temperature/quencher/circuit
  effects ONLY after measured device data or validated electronic models exist.
  Research context: https://doi.org/10.21203/rs.3.rs-7752182/v1
- Yaseera Ismail (SU QuPhotonics): connect polarization preparation, channels,
  detection and quantum communication to BB84 and future entanglement-based E91.
- Mark Tame (SU Quantum NanoPhotonics): later quantum interference, integrated
  photonic elements and nanophotonic quantum sensing models.
- Partha Nandi (SU): later optomechanical/interferometric cavity testbeds.
- These are thematic curriculum connections, NOT accounts of what each person
  said in their individual conference presentation.

## Virtual optical benches and predictions

### A: Paired-photon time tagging and coincidences

    Ideal paired arrivals --+--> Detector A --+--> time tags --+
                            |                 |               |
                            +--> Detector B --+--> time tags --+--> CoincidenceCounter
                                                            --> lag histogram

1. Draw a Poisson-distributed number of pairs and uniform emission times; add
   Gaussian relative timing skew. This is a simple coincidence *source*, not
   SPDC/FWM entanglement, heralding or quantum-state generation.
2. Independently thin photon arrivals by detection efficiency and add Poisson
   dark clicks. Apply non-paralyzable dead time; then add Gaussian electronic
   timing jitter. There is currently no wavelength, afterpulsing or gate model.
3. Acquire singles, one-to-one coincidences (full gate width W), and the
   all-pairs histogram of differences t_B - t_A.
4. Predict accidental matches for independent rare-event streams:
   N_acc ~ (N_A / T) (N_B / T) W T.
   This is an approximation; true pair correlations, rate saturation and
   finite-window effects violate its assumptions.
5. Explore detection efficiency, timestamp jitter, dark counts, and gate width;
   verify the histogram peak does not imply entanglement.

### B: BB84 QKD, an idealized prepare-measure lesson

    Alice (H/V or D/A) -> photon channel -> Bob (H/V or D/A)
                                         -> threshold outcome -> sifting -> QBER

- Two orthogonal choices within each of two conjugate bases are modeled as
  ideal single-photon qubit states. Measurement in the other basis is uniform.
- Bob detects photons with probability transmission x detection efficiency;
  if no true click occurs, a simulated dark click may report a random bit.
- Compare noise-free QBER = 0 and a full intercept-resend Eve: sifted QBER
  tends to 25%. Sifted key length approaches half the detections with unbiased
  random basis choices.
- The simulator reports **QBER, NOT a secure key rate**: it has no multi-photon
  source security, basis reconciliation authentication, error correction,
  privacy amplification, finite-key proof, or composable security analysis.
  Real QKD requires classical authenticated communication.

Run both demonstrations:

    python examples/quantum_cape_tabletop.py
    pytest -q tests/test_quantum_photonics.py

## Next reviewable work, ordered by evidence

1. Weak-coherent pulsed laser source: Poisson photon number, neutral-density
   attenuators, beam splitters, gating and distinction from on-demand Fock
   single-photon emitters. Start with the reported 780 nm attenuated diode-laser
   context, while labeling lab values as illustrative unless calibrated.
2. Time-to-digital converter: finite timestamp resolution, clock drift,
   channel offsets, multi-channel coincidence histograms, live/dead-time
   corrections, afterpulsing and SPAD quenching electronics.
3. HBT g2(0), nonclassical antibunching, HOM visibility and uncertainty
   intervals; separate the physical inputs for classical and quantum sources.
4. FWM and SPDC entangled pair generation, complex JSA with spectral phase
   and quadrature weights, filtering, heralding and purity via Schmidt
   analysis (never infer purity from intensity alone). Cross-check against
   QuTiP/Perceval/SPDCalc as appropriate.
5. E91 and CHSH with spacelike-setting and detection loophole caveats;
   entanglement-based QKD with a full adversary/error model.
6. Classical-channel free-space/fibre transmission, polarization drift,
   decoy-state BB84, device calibration, finite-size key estimates,
   finite-detector/side-channel effects and research instrument adapters.
7. Cavity-enhanced sources, microresonators, quantum sensing and
   optomechanical interferometry with independent scientific gates.

**Repository boundaries:** PhotonicsLab implements validated physical components
and observables. quantum-workflows owns versioned notebooks and experimental
provenance; quantum-platform provides workbench and access; ACP retains approved
research assistant context; infra-hpc-qc-k8s handles scheduling. CPU NumPy is
adequate for these small demos; no GPU scheduling is justified yet.
