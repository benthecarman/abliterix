"""Independent-reproduction contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from abliterix.reproducibility import (
    SCHEMA_VERSION,
    assess_reproducibility,
    build_manifest,
    canonical_manifest_sha256,
    compare_reproduction_metrics,
    load_manifest,
    manifest_trial,
    pin_remote_sources,
    validate_manifest,
    write_reproduce_artifacts,
)
from abliterix.scriptlib import build_trial_recipe, extract_trial_artifact
from abliterix.settings import AbliterixConfig


PIN = "a" * 40


def _pinned_config(*, llm_judge: bool = False) -> AbliterixConfig:
    config = AbliterixConfig.model_validate(
        {
            "model": {"model_id": "owner/model", "revision": PIN},
            "seed": 7,
            "detection": {"llm_judge": llm_judge},
        }
    )
    for source in (
        config.benign_prompts,
        config.target_prompts,
        config.benign_eval_prompts,
        config.target_eval_prompts,
    ):
        source.revision = PIN
    return config


def _trial(*, scope: str = "global") -> SimpleNamespace:
    config = _pinned_config()
    return SimpleNamespace(
        user_attrs={
            "index": 3,
            "vector_index": None if scope == "per layer" else 0.42,
            "parameters": {
                "attn.o_proj": {
                    "min_weight": 0.2,
                    "max_weight": 1.1,
                    "max_weight_position": 0.5,
                    "min_weight_distance": 0.3,
                }
            },
            "moe_parameters": None,
            "decay_kernel": "linear",
            "direct_transform": None,
            "steering_variant": "single",
            "steering_recipe": build_trial_recipe(
                config,
                direct_transform=config.steering.direct_transform,
                steering_variant="single",
                vector_scope=scope,
            ),
            "kl_divergence": 0.125,
            "refusals": 2,
        }
    )


def test_reproducibility_requires_pinned_inputs_and_builtin_evaluator(monkeypatch):
    monkeypatch.setattr("abliterix.reproducibility._git_commit", lambda: None)
    eligible, reasons = assess_reproducibility(_pinned_config())
    assert eligible is True
    assert reasons == []

    config = _pinned_config(llm_judge=True)
    config.target_prompts.revision = None
    eligible, reasons = assess_reproducibility(config)
    assert eligible is False
    assert any("external LLM judge" in reason for reason in reasons)
    assert any("target_prompts" in reason for reason in reasons)


def test_remote_sources_are_resolved_once_to_commit_pins():
    config = AbliterixConfig.model_validate(
        {"model": {"model_id": "owner/model"}, "detection": {"llm_judge": False}}
    )

    class Api:
        def model_info(self, model_id, revision=None):
            assert revision is None
            return SimpleNamespace(sha="1" * 40)

        def dataset_info(self, dataset_id, revision=None):
            assert revision is None
            return SimpleNamespace(sha="2" * 40)

    resolved = pin_remote_sources(config, api=Api())
    assert len(resolved) == 5
    assert config.model.revision == "1" * 40
    assert config.benign_prompts.revision == "2" * 40


def test_evaluation_model_is_pinned_independently():
    config = AbliterixConfig.model_validate(
        {
            "model": {
                "model_id": "owner/model",
                "revision": "1" * 40,
                "evaluate_model_id": "owner/evaluator",
            },
            "detection": {"llm_judge": False},
        }
    )
    for source in (
        config.benign_prompts,
        config.target_prompts,
        config.benign_eval_prompts,
        config.target_eval_prompts,
    ):
        source.revision = "2" * 40

    class Api:
        def model_info(self, model_id, revision=None):
            assert model_id == "owner/evaluator"
            return SimpleNamespace(sha="3" * 40)

    pin_remote_sources(config, api=Api())
    assert config.model.evaluate_model_revision == "3" * 40


def test_manifest_contains_exact_trial_and_detects_tampering(monkeypatch):
    monkeypatch.setattr(
        "abliterix.reproducibility.collect_environment", lambda: {"python": "x"}
    )
    monkeypatch.setattr("abliterix.reproducibility.collect_packages", lambda: {})
    monkeypatch.setattr("abliterix.reproducibility._git_commit", lambda: None)
    manifest = build_manifest(
        _pinned_config(),
        _trial(),
        weight_shas={"model.safetensors": "a" * 64},
    )

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["reproducible"] is True
    assert manifest["trial"]["steering_recipe"]["schema_version"] == 1
    assert manifest_trial(manifest).user_attrs["vector_index"] == 0.42
    validate_manifest(manifest)

    manifest["trial"]["vector_index"] = 0.7
    with pytest.raises(ValueError, match="integrity"):
        validate_manifest(manifest)


def test_metric_reverification_is_strict_for_counts_and_tolerant_for_float():
    manifest = {
        "metrics": {"kl_divergence": 0.1, "refusals": 2},
    }
    assert (
        compare_reproduction_metrics(manifest, kl_divergence=0.10000001, refusals=2)
        == []
    )
    findings = compare_reproduction_metrics(manifest, kl_divergence=0.2, refusals=3)
    assert len(findings) == 2
    assert any("refusals" in finding for finding in findings)


def test_manifest_without_weight_hashes_is_not_reproducible(monkeypatch):
    monkeypatch.setattr("abliterix.reproducibility.collect_environment", lambda: {})
    monkeypatch.setattr("abliterix.reproducibility.collect_packages", lambda: {})
    monkeypatch.setattr("abliterix.reproducibility._git_commit", lambda: None)
    manifest = build_manifest(_pinned_config(), _trial(), weight_shas={})
    assert manifest["reproducible"] is False
    assert any("SHA256" in reason for reason in manifest["reproducibility_reasons"])
    with pytest.raises(ValueError, match="SHA256"):
        validate_manifest(manifest)


@pytest.fixture
def manifest_factory(monkeypatch):
    monkeypatch.setattr("abliterix.reproducibility.collect_environment", lambda: {})
    monkeypatch.setattr("abliterix.reproducibility.collect_packages", lambda: {})
    monkeypatch.setattr("abliterix.reproducibility._git_commit", lambda: None)

    def make(scope="global"):
        return build_manifest(
            _pinned_config(),
            _trial(scope=scope),
            weight_shas={"model.safetensors": "a" * 64},
        )

    return make


@pytest.mark.parametrize("scope", ["global", "per layer"])
def test_manifest_round_trips_exact_vector_scope(manifest_factory, scope, tmp_path):
    manifest = manifest_factory(scope)
    write_reproduce_artifacts(tmp_path, manifest)
    restored = load_manifest(tmp_path / "reproduce.json")
    artifact = extract_trial_artifact(manifest_trial(restored))

    assert manifest["reproducible"] is True
    assert artifact.vector_index == (None if scope == "per layer" else 0.42)
    assert artifact.recipe["steering"]["fixed_vector_scope"] == scope
    assert artifact.profiles["attn.o_proj"].max_weight == 1.1


@pytest.mark.parametrize(
    ("scope", "vector_index"),
    [
        ("global", None),
        ("global", True),
        ("global", "0.42"),
        ("global", float("nan")),
        ("global", float("inf")),
        ("per layer", 0.42),
        (None, None),
        ("invalid", None),
    ],
)
def test_manifest_rejects_inconsistent_vector_scope(
    manifest_factory, scope, vector_index
):
    manifest = manifest_factory()
    manifest["trial"]["steering_recipe"]["steering"]["fixed_vector_scope"] = scope
    manifest["trial"]["vector_index"] = vector_index
    manifest["integrity"]["canonical_manifest"] = canonical_manifest_sha256(manifest)

    with pytest.raises(ValueError, match="vector_index|fixed_vector_scope"):
        manifest_trial(manifest)


@pytest.mark.parametrize("scope", ["global", "per layer"])
def test_manifest_requires_explicit_vector_index_key(manifest_factory, scope):
    manifest = manifest_factory(scope)
    del manifest["trial"]["vector_index"]
    manifest["integrity"]["canonical_manifest"] = canonical_manifest_sha256(manifest)

    with pytest.raises(ValueError, match="missing 'vector_index'"):
        validate_manifest(manifest)
