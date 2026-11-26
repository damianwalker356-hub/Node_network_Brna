import numpy as np
import pytest

from brna.language_model import CharacterLanguageModel


@pytest.mark.req("R-STUDIO-01")
def test_training_reduces_loss_and_generates_known_characters():
    model = CharacterLanguageModel(hidden_size=8, seed=3)
    losses = model.train(
        "hello brna\nassistant learns 🧠\n" * 3,
        epochs=8,
        learning_rate=0.1,
        sequence_length=32,
    )

    generated = model.generate("hello", max_new_tokens=24)
    assert losses[-1] < losses[0]
    assert 0 < len(generated) <= 24
    assert set(generated) <= set(model.vocabulary)


@pytest.mark.req("R-STUDIO-01")
def test_model_round_trip_preserves_parameters_and_generation(tmp_path):
    model = CharacterLanguageModel(hidden_size=8, seed=7)
    model.train("an original local corpus\n" * 2, epochs=2, sequence_length=16)
    path = tmp_path / "model.npz"
    model.save(path)

    restored = CharacterLanguageModel.load(path, seed=11)

    assert restored.trained_characters == model.trained_characters
    for actual, expected in zip(restored._parameters(), model._parameters(), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert restored.generate("hello", max_new_tokens=24) == model.generate(
        "hello", max_new_tokens=24
    )


@pytest.mark.req("R-STUDIO-01")
def test_model_rejects_untrained_and_invalid_training_requests():
    model = CharacterLanguageModel(hidden_size=8)
    with pytest.raises(RuntimeError, match="not been trained"):
        model.generate("hello")
    with pytest.raises(ValueError, match="at least two"):
        model.train("x")
    with pytest.raises(ValueError, match="epochs"):
        model.train("valid text", epochs=0)
