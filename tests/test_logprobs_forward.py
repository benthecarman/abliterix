"""Single-token scoring must bound logits and preserve metric precision."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from abliterix.core.engine import SteeringEngine


class _LimitedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(
            torch.tensor(
                [[[1.0, -1.0, 0.0], [12.0, 3.125, -8.5]]],
                dtype=torch.bfloat16,
            )
        )
        self.calls = []

    def forward(self, input_ids, *, logits_to_keep=0, use_cache=True):
        self.calls.append((logits_to_keep, use_cache, torch.is_grad_enabled()))
        logits = self.scores * 1
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        self.logits_shape = logits.shape
        return SimpleNamespace(logits=logits)


class _PeftWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def get_base_model(self):
        return self.base

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)


class _CompiledWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self._orig_mod = base

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


class _LegacyModel(nn.Module):
    def forward(self, input_ids):
        return SimpleNamespace(logits=input_ids.float().unsqueeze(-1))


class _CustomModel(nn.Module):
    def forward(self, input_ids, **kwargs):
        assert not kwargs, "Generic **kwargs does not promise logits/cache support"
        return SimpleNamespace(logits=input_ids.float().unsqueeze(-1))


def _engine(model):
    engine = object.__new__(SteeringEngine)
    engine.model = model
    engine._tokenize = lambda _messages: {"input_ids": torch.tensor([[1, 2]])}
    return engine


@pytest.mark.parametrize("wrapper", [lambda x: x, _PeftWrapper, _CompiledWrapper])
def test_forward_scores_only_final_token_without_cache_or_autograd(wrapper):
    model = _LimitedModel()
    engine = _engine(wrapper(model))

    with torch.enable_grad():
        actual = engine._logprobs_forward_pass([])

    expected = F.log_softmax(model.scores[:, -1, :].detach().float(), dim=-1)
    assert model.calls == [(1, False, False)]
    assert model.logits_shape == (1, 1, 3)
    assert actual.dtype == torch.float32
    assert actual.shape == (1, 3)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_type", [_LegacyModel, _CustomModel])
def test_forward_preserves_legacy_and_remote_model_compatibility(model_type):
    engine = _engine(_PeftWrapper(model_type()))

    actual = engine._logprobs_forward_pass([])

    assert actual.shape == (1, 1)
    assert actual.dtype == torch.float32
    assert actual.item() == 0.0


def test_forward_rechecks_supported_options_when_model_is_replaced():
    engine = _engine(_LimitedModel())
    engine._logprobs_forward_pass([])
    engine.model = _LegacyModel()

    actual = engine._logprobs_forward_pass([])

    assert actual.shape == (1, 1)


def test_forward_does_not_retry_an_internal_type_error():
    class BrokenModel(nn.Module):
        calls = 0

        def forward(self, input_ids, *, logits_to_keep=0, use_cache=True):
            self.calls += 1
            raise TypeError("internal model failure")

    model = BrokenModel()
    with pytest.raises(TypeError, match="internal model failure"):
        _engine(model)._logprobs_forward_pass([])

    assert model.calls == 1


@pytest.mark.parametrize("wrapped", [False, True])
def test_real_tiny_llama_matches_full_forward_without_downloading_weights(wrapped):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        ).eval()
        if wrapped:
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "v_proj"],
                    r=2,
                    lora_alpha=2,
                ),
            ).eval()
        input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        engine = _engine(model)
        engine._tokenize = lambda _messages: {"input_ids": input_ids}
        with torch.no_grad():
            expected = F.log_softmax(
                model(input_ids, use_cache=False).logits[:, -1, :],
                dim=-1,
                dtype=torch.float32,
            )
        actual = engine._logprobs_forward_pass([])

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert not actual.requires_grad
