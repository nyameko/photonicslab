"""Independent analytic and statistical checks for the event-level demo."""

import numpy as np
import pytest

from photonicslab import BB84Experiment, CoincidenceCounter, PhotonPairSource, SinglePhotonDetector


def test_detector_identity_and_dead_time():
    arrivals = np.array([0.1, 0.2, 0.25, 0.4])
    ideal = SinglePhotonDetector(1.0, 0.0, 0.0, 0.0)
    np.testing.assert_array_equal(ideal.detect(arrivals, 1.0, np.random.default_rng(1)), arrivals)
    detector = SinglePhotonDetector(1.0, 0.0, 0.15, 0.0)
    np.testing.assert_allclose(
        detector.detect(arrivals, 1.0, np.random.default_rng(1)),
        [0.1, 0.25, 0.4],
        atol=1e-15,
    )


def test_detector_dark_counts_follow_poisson_mean():
    detector = SinglePhotonDetector(0.0, 2_000.0, 0.0, 0.0)
    n_clicks = len(detector.detect([], 5.0, np.random.default_rng(28)))
    assert 9_500 < n_clicks < 10_500


def test_pair_source_is_correlated_and_counted_once():
    source = PhotonPairSource(pair_rate_hz=5_000, relative_jitter_s=0.0)
    rng = np.random.default_rng(11)
    a, b = source.emit(0.5, rng)
    np.testing.assert_array_equal(a, b)
    counter = CoincidenceCounter(window_s=2e-9)
    assert counter.count(a, b) == len(a)
    assert counter.count(a, b, delay_s=1e-6) < 4
    assert counter.expected_accidentals(100, 120, 1.0) == pytest.approx(24e-6)


def test_coincidence_histogram_and_one_to_one_semantics():
    counter = CoincidenceCounter(window_s=1e-9)
    a = np.array([0.0, 5e-9])
    b = np.array([0.25e-9, 5.25e-9])
    assert counter.count(a, b) == 2
    assert counter.count([0], [0, 0]) == 1
    hist = counter.histogram(a, b, [-1e-9, 0, 0.5e-9, 1e-9])
    np.testing.assert_array_equal(hist, [0, 2, 0])


def test_bb84_ideal_same_basis_has_zero_errors():
    result = BB84Experiment().run(20_000, np.random.default_rng(42))
    assert result.received == 20_000
    assert 9_700 < result.sifted_bits < 10_300
    assert result.qber == 0
    assert not result.alice_bits.flags.writeable


def test_bb84_intercept_resend_causes_one_quarter_qber():
    result = BB84Experiment(intercept_resend_fraction=1.0).run(100_000, np.random.default_rng(2))
    assert 0.24 < result.qber < 0.26


def test_bb84_loss_and_dark_clicks_do_not_claim_security():
    result = BB84Experiment(channel_transmission=0.0, dark_click_probability_per_gate=1.0).run(
        10_000, np.random.default_rng(9)
    )
    assert result.received == result.sent
    assert 0.47 < result.qber < 0.53
    assert BB84Experiment(channel_transmission=0.0).run(100, np.random.default_rng(1)).qber is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SinglePhotonDetector(efficiency=1.1),
        lambda: SinglePhotonDetector(dark_count_rate_hz=-1),
        lambda: SinglePhotonDetector(timing_jitter_s=float("nan")),
        lambda: PhotonPairSource(pair_rate_hz=-5),
        lambda: CoincidenceCounter(window_s=0),
        lambda: BB84Experiment(intercept_resend_fraction=-0.1),
        lambda: BB84Experiment(dark_click_probability_per_gate=1.1),
    ],
)
def test_invalid_parameters_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_bad_time_tags_and_duration_rejected():
    detector = SinglePhotonDetector()
    with pytest.raises(ValueError):
        detector.detect([1.0], 1.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        detector.detect([[0.1]], 1.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        PhotonPairSource().emit(0.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        CoincidenceCounter().histogram([], [], [0, 0])


def test_bb84_reproducible_for_same_seed():
    experiment = BB84Experiment(
        channel_transmission=0.7,
        detector_efficiency=0.8,
        dark_click_probability_per_gate=0.01,
        optical_bit_error_probability=0.02,
    )
    a = experiment.run(1000, np.random.default_rng(3))
    b = experiment.run(1000, np.random.default_rng(3))
    np.testing.assert_array_equal(a.bob_bits, b.bob_bits)
    np.testing.assert_array_equal(a.sifted, b.sifted)
    with pytest.raises(ValueError):
        experiment.run(0, np.random.default_rng(0))
