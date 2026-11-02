"""Tests for abliterix.data — trial formatting and model card generation.

Skips load_prompt_dataset (requires real dataset files).
"""

from types import SimpleNamespace

from abliterix.data import format_trial_params, generate_model_card
from abliterix.types import ChatMessage


# ---------------------------------------------------------------------------
# format_trial_params
# ---------------------------------------------------------------------------


def test_format_vector_index_float(mock_trial):
    result = format_trial_params(mock_trial)
    assert result["vector_index"] == "5.50"


def test_format_vector_index_none():
    trial = SimpleNamespace(
        user_attrs={
            "vector_index": None,
            "parameters": {},
        }
    )
    result = format_trial_params(trial)
    assert result["vector_index"] == "per layer"


def test_format_component_params(mock_trial):
    result = format_trial_params(mock_trial)
    assert "q_proj.max_weight" in result
    assert result["q_proj.max_weight"] == "1.20"
    assert "v_proj.min_weight" in result
    assert result["v_proj.min_weight"] == "0.05"


# ---------------------------------------------------------------------------
# generate_model_card
# ---------------------------------------------------------------------------


def test_model_card_contains_kl(mock_trial, abliterix_config):
    target = [ChatMessage(system="", user=f"Prompt {i}") for i in range(10)]
    card = generate_model_card(abliterix_config, mock_trial, 10, target)
    assert "0.0123" in card


def test_model_card_hub_model_link(mock_trial, abliterix_config):
    """Non-local model ID should produce a HuggingFace link."""
    target = [ChatMessage(system="", user=f"P{i}") for i in range(5)]
    card = generate_model_card(abliterix_config, mock_trial, 5, target)
    assert "huggingface.co/" in card


def test_model_card_contains_parameters_table(mock_trial, abliterix_config):
    target = [ChatMessage(system="", user="test")]
    card = generate_model_card(abliterix_config, mock_trial, 1, target)
    assert "q_proj.max_weight" in card
    assert "v_proj.max_weight" in card


def test_model_card_refusal_counts(mock_trial, abliterix_config):
    target = [ChatMessage(system="", user=f"P{i}") for i in range(10)]
    card = generate_model_card(abliterix_config, mock_trial, 10, target)
    # trial_refusals = 3, baseline = 10, len(target) = 10
    assert "3/10" in card
    assert "10/10" in card


def test_model_card_method_settings_fall_back_for_legacy_trials(
    mock_trial, abliterix_config
):
    from abliterix.types import VectorMethod

    abliterix_config.steering.vector_method = VectorMethod.SRA
    card = generate_model_card(abliterix_config, mock_trial, 1, [])

    assert "| **steering.vector_method** | sra |" in card
    assert "| **steering.steering_mode** | lora |" in card
    assert "| **steering.n_directions** | 1 |" in card
    assert "| **steering.weight_normalization** | full |" in card
    assert "active export configuration" in card


def test_model_card_prefers_selected_trial_recipe_over_current_settings(
    mock_trial, abliterix_config
):
    mock_trial.user_attrs["steering_recipe"] = {
        "schema_version": 1,
        "steering": {
            "vector_method": "pca",
            "steering_mode": "direct",
            "n_directions": 2,
            "weight_normalization": "none",
        },
    }
    before = abliterix_config.steering.model_dump()

    card = generate_model_card(abliterix_config, mock_trial, 1, [])

    assert "| **steering.vector_method** | pca |" in card
    assert "| **steering.steering_mode** | direct |" in card
    assert "| **steering.n_directions** | 2 |" in card
    assert "| **steering.weight_normalization** | none |" in card
    assert abliterix_config.steering.model_dump() == before
