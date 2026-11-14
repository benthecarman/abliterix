"""Steering plans without loading the model (core/plan_export.py): the plan a
trial records on a skeleton built from the checkpoint's config, with the
search's saved hidden states and safety experts, is the plan it records on
the loaded model; and its bake edits the checkpoint as that plan says."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from abliterix.core import fp4_repack as rp
from abliterix.core import plan_export as pe
from abliterix.core.engine import SteeringEngine
from abliterix.settings import AbliterixConfig
from abliterix.types import ExpertRoutingConfig

PROFILE = {"max_weight": 1.2, "max_weight_position": 1.0, "min_weight": 0.4, "min_weight_distance": 2.0}


def _qwen3_moe(path: Path) -> int:
    from transformers import Qwen3MoeConfig

    cfg = Qwen3MoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)
    model.save_pretrained(path)  # experts stored one tensor each, under other keys
    return cfg.num_hidden_layers


def _config(model: Path, checkpoints: Path, **steering) -> AbliterixConfig:
    return AbliterixConfig(
        model={"model_id": str(model), "backend": "vllm", "use_in_place_editing": True,
               "plans_without_hf_model": True, "dtype_fallback_order": ["bfloat16"]},
        steering={"steering_mode": "direct", **steering},
        optimization={"checkpoint_dir": str(checkpoints)},
    )


def _loaded_engine(config: AbliterixConfig) -> SteeringEngine:
    """The engine's post-load state on the vLLM backend (no PEFT), with the
    weights loaded: what the HF path records plans on."""
    engine = SteeringEngine.without_model(config, tokenizer=False)
    engine.model = AutoModelForCausalLM.from_pretrained(config.model.model_id, dtype=torch.bfloat16)
    engine._fused_down_proj_transposed = False
    engine._lora_b_weights = []
    engine.peft_config = None
    engine._init_expert_routing()
    return engine


def _same_plans(a, b):
    assert len(a) == len(b) and a
    for x, y in zip(a, b):
        for f in ("logical_name", "kind", "strength", "preserve_row_norm", "hidden_dim", "transposed",
                  "projection_side", "direct_transform", "rows", "scale", "expert"):
            assert getattr(x, f) == getattr(y, f), (x.logical_name, f)
        for f in ("direction", "basis", "benign_dir"):
            u, v = getattr(x, f), getattr(y, f)
            assert (u is None) == (v is None) and (u is None or torch.equal(u, v)), (x.logical_name, f)


def _states(n_layers: int, hidden: int):
    g = torch.Generator().manual_seed(1)
    return (torch.randn(6, n_layers + 1, hidden, generator=g),
            torch.randn(5, n_layers + 1, hidden, generator=g) + 0.3)


def _trial():
    return SimpleNamespace(number=7, user_attrs={
        "vector_index": None,
        "parameters": {"attn.o_proj": PROFILE, "mlp.down_proj": PROFILE},
        "moe_parameters": {"n_suppress": 2, "router_bias": -4.0, "expert_ablation_weight": 0.8},
    })


def test_skeleton_from_config_matches_the_loaded_model(tmp_path):
    n = _qwen3_moe(tmp_path / "m")
    config = _config(tmp_path / "m", tmp_path / "ck")
    skel = pe.config_skeleton(config)
    loaded = _loaded_engine(config)
    params = {k: (v.shape, v.dtype) for k, v in loaded.model.named_parameters()}
    assert {k: (v.shape, v.dtype) for k, v in skel.model.named_parameters()} == params
    assert all(p.is_meta for p in skel.model.parameters())
    assert len(skel.transformer_layers) == n and skel.has_expert_routing()
    assert skel.list_steerable_components() == loaded.list_steerable_components()
    assert rp.plan_meta(skel) == rp.plan_meta(loaded)


def test_skeleton_ignores_the_checkpoint_quantization_config(tmp_path):
    _qwen3_moe(tmp_path / "m")
    cfg_path = tmp_path / "m" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    del cfg["transformers_version"]
    cfg["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    cfg_path.write_text(json.dumps(cfg))
    skel = pe.config_skeleton(_config(tmp_path / "m", tmp_path / "ck"))
    assert all(p.dtype == torch.bfloat16 for p in skel.model.parameters())
    assert json.loads(cfg_path.read_text()) == cfg  # the checkpoint is not touched


def test_export_records_the_searched_plan_and_bakes_it(tmp_path):
    n = _qwen3_moe(tmp_path / "m")
    config = _config(tmp_path / "m", tmp_path / "ck")
    benign, target = _states(n, 32)
    safety = {i: [(e, 1.0 - 0.1 * e) for e in (2, 0, 3, 1)] for i in range(n)}
    pe.save_search_state(config, benign, target, safety, safety_experts_method="hooks")

    out = tmp_path / "out"
    result = pe.export_plan_without_model(
        _config(tmp_path / "m", tmp_path / "ck"), _trial(), out, plan_file=tmp_path / "plan.pt",
        use_cuda=False)

    # The plan the loaded model gives for the same trial and inputs.
    from abliterix.scriptlib import compute_trial_vectors, extract_trial_artifact

    artifact = extract_trial_artifact(_trial())
    vectors = compute_trial_vectors(artifact, benign, target, config)
    want = rp.record_steering_plan_from_trial(
        _loaded_engine(config), vectors, artifact.vector_index, artifact.profiles, config,
        benign_states=benign, target_states=target, safety_experts=safety,
        routing_config=ExpertRoutingConfig(n_suppress=2, router_bias=-4.0, expert_ablation_weight=0.8))
    _same_plans(result.edits, want)
    assert {e.kind for e in want} == {"direct", "ega", "row_scale", "expert_project"}
    saved, meta = rp.load_plan_with_meta(tmp_path / "plan.pt")
    _same_plans(saved, want)
    assert meta == rp.plan_meta(_loaded_engine(config))

    # The bake: the checkpoint with exactly the plan's tensors edited.
    mapped = rp.map_plan_to_checkpoint(want, tmp_path / "m", model_class=meta["model_class"])
    edited = {e.logical_name for e in mapped}
    with safe_open(tmp_path / "m" / "model.safetensors", "pt") as src, \
            safe_open(out / "model.safetensors", "pt") as dst:
        assert set(src.keys()) == set(dst.keys())
        for key in src.keys():
            a, b = src.get_tensor(key), dst.get_tensor(key)
            if key in edited:
                group = [e for e in mapped if e.logical_name == key]
                expect = rp.apply_tensor_edits(a.float(), group).to(a.dtype)
                assert torch.equal(b, expect), key
            else:
                assert torch.equal(a, b), key
    assert any(".experts.2.down_proj" in k for k in edited)  # one expert's own tensor


def test_export_needs_the_search_state_of_its_prompts(tmp_path):
    n = _qwen3_moe(tmp_path / "m")
    config = _config(tmp_path / "m", tmp_path / "ck")
    with pytest.raises(FileNotFoundError):
        pe.export_plan_without_model(config, _trial(), None)
    pe.save_search_state(config, *_states(n, 32), None)
    other = _config(tmp_path / "m", tmp_path / "ck")
    other.system_prompt = "something else"
    assert pe.load_search_state(other) is None
    with pytest.raises(ValueError, match="other direction prompts"):
        pe.load_search_state(other, required=True)
    assert pe.load_search_state(config)["safety_experts"] is None
    plan_only = pe.export_plan_without_model(config, _trial(), None)
    assert plan_only.stats is None and plan_only.edits
    assert not any(e.kind in ("row_scale", "expert_project") for e in plan_only.edits)


def test_export_takes_the_revision_the_search_pinned(tmp_path):
    n = _qwen3_moe(tmp_path / "m")
    searched = _config(tmp_path / "m", tmp_path / "ck")
    searched.model.revision = "a" * 40  # as pin_remote_sources leaves it
    pe.save_search_state(searched, *_states(n, 32), None)
    config = _config(tmp_path / "m", tmp_path / "ck")
    assert config.model.revision is None
    assert pe.export_plan_without_model(config, _trial(), None).edits
    assert config.model.revision == "a" * 40
    other = _config(tmp_path / "m", tmp_path / "ck")
    other.system_prompt = "something else"
    pe.adopt_search_revision(other)
    assert other.model.revision is None
