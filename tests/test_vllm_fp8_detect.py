"""``_detect_fp8_and_kv_dtype`` reads the checkpoint's config.json, not a config
class: transformers' Inkling2Config adds a default FP8 quantization_config that
vLLM's class for the same model lacks, so a class-based answer flipped with
whether vLLM had been loaded first (a resumed search asked vLLM for fp8)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from abliterix.core.vllm_backend import _detect_fp8_and_kv_dtype


def _config():
    return SimpleNamespace(model=SimpleNamespace(quant_method=None, revision=None, kv_cache_dtype=None))


def _checkpoint(tmp_path, **extra):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "abliterix_fp8_probe", **extra}))
    return str(tmp_path)


class _ProbeConfig(PretrainedConfig):
    model_type = "abliterix_fp8_probe"

    def __init__(self, quantization_config=None, **kwargs):
        # Like Inkling2Config: an FP8 default when the checkpoint names none.
        self.quantization_config = quantization_config or {"quant_method": "fp8"}
        super().__init__(**kwargs)


def test_a_config_class_default_is_not_the_checkpoint(tmp_path):
    AutoConfig.register("abliterix_fp8_probe", _ProbeConfig, exist_ok=True)
    try:
        model = _checkpoint(tmp_path)
        assert AutoConfig.from_pretrained(model).quantization_config["quant_method"] == "fp8"
        assert _detect_fp8_and_kv_dtype(_config(), model_id=model, trust_remote_code=False) == (False, None)
    finally:
        CONFIG_MAPPING._extra_content.pop("abliterix_fp8_probe", None)


def test_a_native_fp8_checkpoint_is_still_fp8(tmp_path):
    model = _checkpoint(tmp_path, quantization_config={"quant_method": "fp8"})
    assert _detect_fp8_and_kv_dtype(_config(), model_id=model, trust_remote_code=False)[0] is True
    nested = tmp_path / "nested"
    nested.mkdir()
    model = _checkpoint(nested, text_config={"quantization_config": {"quant_method": "fp8"}})
    assert _detect_fp8_and_kv_dtype(_config(), model_id=model, trust_remote_code=False)[0] is True
