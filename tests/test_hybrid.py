import numpy as np
import pytest

from brna.hybrid import (
    ClosedLoopBRNA,
    DigitalDecoder,
    DigitalEncoder,
    SimulatedReservoir,
)
from brna.hybrid_experiment import build_record
from brna.proto import NodeNet, make_data


@pytest.mark.req("R-HYBRID-01")
def test_closed_loop_uses_bounded_feedback_and_is_deterministic_without_noise():
    """R-HYBRID-01: connect translators, simulated reservoir, and critic-bounded loop."""
    rng = np.random.default_rng(12)
    inputs, _ = make_data(8, rng)
    model = ClosedLoopBRNA(
        np.random.default_rng(13),
        outer_loops=3,
        noise_std=0.0,
    )

    first = model.run(inputs)
    second = model.run(inputs)

    assert first["rounds"] == 3
    assert first["feedback_edges"] == 2
    assert first["node_roles"] == model.node_roles
    assert len(set(first["node_roles"])) == 6
    np.testing.assert_array_equal(first["logits"], second["logits"])


@pytest.mark.req("R-HYBRID-01")
def test_translator_and_reservoir_gradients_match_finite_differences():
    """R-HYBRID-01: verify gradients through both digital translators and reservoir."""
    rng = np.random.default_rng(31)
    inputs = rng.normal(size=(1, 12, 4))
    encoder = DigitalEncoder(
        np.random.default_rng(32), code="latency", adaptive=False
    )
    stimuli, encoder_cache = encoder.encode(inputs)
    stimulus_gradient = rng.normal(size=stimuli.shape)
    input_gradient = encoder.backward(
        encoder_cache, stimulus_gradient, learning_rate=0.0
    )

    def encoded_objective(candidate):
        return float(np.sum(encoder.encode(candidate)[0] * stimulus_gradient))

    input_index = (0, 3, 2)
    epsilon = 1e-6
    plus = inputs.copy()
    minus = inputs.copy()
    plus[input_index] += epsilon
    minus[input_index] -= epsilon
    numerical_input_gradient = (
        encoded_objective(plus) - encoded_objective(minus)
    ) / (2 * epsilon)
    assert input_gradient[input_index] == pytest.approx(
        numerical_input_gradient, rel=1e-5, abs=1e-7
    )

    decoder = DigitalDecoder(np.random.default_rng(33), time_bins=3, adaptive=False)
    activity = rng.normal(size=(3, 1, 12, 4))
    decoded, decoder_cache = decoder.decode(activity)
    decoded_gradient = rng.normal(size=decoded.shape)
    activity_gradient = decoder.backward(
        decoder_cache, decoded_gradient, learning_rate=0.0
    )
    activity_index = (1, 0, 4, 2)

    def decoded_objective(candidate):
        return float(np.sum(decoder.decode(candidate)[0] * decoded_gradient))

    plus_activity = activity.copy()
    minus_activity = activity.copy()
    plus_activity[activity_index] += epsilon
    minus_activity[activity_index] -= epsilon
    numerical_activity_gradient = (
        decoded_objective(plus_activity) - decoded_objective(minus_activity)
    ) / (2 * epsilon)
    assert activity_gradient[activity_index] == pytest.approx(
        numerical_activity_gradient, rel=1e-5, abs=1e-7
    )

    culture = SimulatedReservoir(np.random.default_rng(34), noise_std=0.0)
    output, culture_cache = culture.respond(stimuli)
    output_gradient = rng.normal(size=output.shape)
    culture_input_gradient = culture.backward(culture_cache, output_gradient)
    culture_index = (1, 0, 5, 1)

    def culture_objective(candidate):
        return float(np.sum(culture.respond(candidate)[0] * output_gradient))

    plus_stimuli = stimuli.copy()
    minus_stimuli = stimuli.copy()
    plus_stimuli[culture_index] += epsilon
    minus_stimuli[culture_index] -= epsilon
    numerical_culture_gradient = (
        culture_objective(plus_stimuli) - culture_objective(minus_stimuli)
    ) / (2 * epsilon)
    assert culture_input_gradient[culture_index] == pytest.approx(
        numerical_culture_gradient, rel=1e-5, abs=1e-7
    )


@pytest.mark.req("R-HYBRID-02")
def test_node_network_returns_correct_input_gradient_for_translator_training():
    """R-HYBRID-02: provide a verified downstream gradient to the encoder."""
    rng = np.random.default_rng(35)
    model = NodeNet(rng, d=3, T=2)
    inputs, labels = make_data(2, np.random.default_rng(36))
    input_index = (0, 4, 2)
    epsilon = 1e-6

    def loss(candidate):
        logits = model.fwd(candidate)[4]
        return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))

    plus = inputs.copy()
    minus = inputs.copy()
    plus[input_index] += epsilon
    minus[input_index] -= epsilon
    numerical_gradient = (loss(plus) - loss(minus)) / (2 * epsilon)
    analytic_gradient = model.train_step(
        inputs, labels, return_input_gradient=True
    )

    assert analytic_gradient is not None
    assert analytic_gradient[input_index] == pytest.approx(
        numerical_gradient, rel=1e-5, abs=1e-7
    )


@pytest.mark.req("R-HYBRID-02")
def test_adaptive_translators_train_while_fixed_translators_remain_fixed():
    """R-HYBRID-02: optimize the digital encoder and decoder from task feedback."""
    inputs, labels = make_data(8, np.random.default_rng(21))
    adaptive = ClosedLoopBRNA(
        np.random.default_rng(22),
        adaptive_translators=True,
        noise_std=0.0,
    )
    fixed = ClosedLoopBRNA(
        np.random.default_rng(22),
        adaptive_translators=False,
        noise_std=0.0,
    )
    adaptive_encoder = adaptive.encoder.node_weights.copy()
    adaptive_profile = adaptive.encoder.profile_logits.copy()
    adaptive_decoder = adaptive.decoder.feature_weights.copy()
    fixed_encoder = fixed.encoder.node_weights.copy()
    fixed_decoder = fixed.decoder.feature_weights.copy()

    assert np.isfinite(adaptive.train_step(inputs, labels))
    assert np.isfinite(fixed.train_step(inputs, labels))
    assert adaptive.network.opt.t == 1
    assert fixed.network.opt.t == 1

    assert not np.array_equal(adaptive.encoder.node_weights, adaptive_encoder)
    assert not np.array_equal(adaptive.encoder.profile_logits, adaptive_profile)
    assert not np.array_equal(adaptive.decoder.feature_weights, adaptive_decoder)
    np.testing.assert_array_equal(fixed.encoder.node_weights, fixed_encoder)
    np.testing.assert_array_equal(fixed.decoder.feature_weights, fixed_decoder)


@pytest.mark.req("R-HYBRID-03")
def test_experiment_pairs_task_data_and_marks_biological_hypothesis_unmeasured():
    """R-HYBRID-03: compare same-seed software variants and disclose H3 limits."""
    record = build_record(
        seeds=1,
        steps=2,
        batch_size=4,
        train_samples=8,
        test_samples=16,
        variants=(
            "hybrid_adaptive_latency",
            "hybrid_fixed_latency",
            "silicon_equal_budget",
            "random_reservoir_control",
        ),
    )

    assert len(record["results"]) == 4
    assert record["comparisons"]["H1_adaptive_vs_fixed_latency"] is not None
    assert record["comparisons"]["H2_hybrid_vs_silicon_equal_parameter_budget"] is not None
    assert record["comparisons"]["H3_culture_contribution"]["status"].startswith(
        "not_evaluable"
    )
    assert all(result["train_updates"] == 2 for result in record["results"])
    assert all(np.isfinite(result["test_accuracy"]) for result in record["results"])


@pytest.mark.req("R-HYBRID-04")
def test_silicon_control_matches_hybrid_total_parameter_budget_exactly():
    """R-HYBRID-04: enforce exact total parameter slots in the silicon comparator."""
    record = build_record(
        seeds=1,
        steps=1,
        batch_size=4,
        train_samples=4,
        test_samples=8,
        variants=("hybrid_adaptive_latency", "silicon_equal_budget"),
    )
    hybrid, silicon = record["results"]

    assert hybrid["total_system_parameters"] == silicon["total_system_parameters"]
    assert silicon["trainable_parameters"] == silicon["total_system_parameters"]
