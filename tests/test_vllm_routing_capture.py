"""Routing capture by hooks (core/vllm_moe_editor.py) for MoEs that route
outside FusedMoE, as Inkling 2's vLLM MoE does: the ranking equals
``SteeringEngine.identify_safety_experts`` on the same MoE inputs.

No vLLM: each fake rank's MoE module routes from ``gate.weight`` /
``gate.bias`` without calling the gate, like ``Inkling2MoE``; the HF side is
transformers' ``Inkling2TopKRouter`` math on the same inputs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from abliterix.core import vllm_moe_editor as me
from abliterix.core.engine import SteeringEngine
from abliterix.types import ChatMessage

H, E, TOP_K, LAYERS = 16, 12, 3, 3


@pytest.fixture(autouse=True)
def _no_vllm(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: None))
BF16 = torch.bfloat16


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = [(torch.randn(E, H, generator=g) * 0.5).to(BF16) for _ in range(LAYERS)]
    b = [torch.randn(E, generator=g) * 0.3 for _ in range(LAYERS)]
    return w, b


def _moe_inputs(prompt: str, layer: int) -> torch.Tensor:
    """The MoE input of ``layer`` for a prompt, ``[tokens, H]`` in bf16."""
    n = 2 + len(prompt) % 5
    g = torch.Generator().manual_seed(hash((prompt, layer)) % (2**31))
    shift = 0.8 if prompt.startswith("harm") else 0.0
    return (torch.randn(n, H, generator=g) + shift).to(BF16)


# --- vLLM side -------------------------------------------------------------


class _VllmGate(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.weight = nn.Parameter(w.clone(), requires_grad=False)
        self.bias = nn.Parameter(b.clone(), requires_grad=False)
        self.topk = TOP_K

    def forward(self, *a):  # Inkling2MoE never calls its gate
        raise AssertionError("the gate module is not called")


class _VllmMoE(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.gate = _VllmGate(w, b)

    def forward(self, x):
        return x * 0  # routes in its own kernel; the output does not matter here


def _vllm_rank(w, b):
    layers = nn.ModuleList(nn.Module() for _ in range(LAYERS))
    for i, layer in enumerate(layers):
        layer.mlp = _VllmMoE(w[i], b[i])
    decoder = nn.Module()
    decoder.layers = layers
    top = nn.Module()
    top.model = decoder
    return top


class _Tok:
    def apply_chat_template(self, chat, **kw):
        return chat[-1]["content"]


class _FakeLLM:
    """``collective_rpc`` over in-process workers; ``generate`` runs each
    rank's MoE modules on the prompts' inputs (all of them, or with
    ``shard`` a slice of the tokens per rank, like sequence parallelism)."""

    def __init__(self, ranks, shard=False):
        self.workers = [SimpleNamespace(model_runner=SimpleNamespace(model=r)) for r in ranks]
        self.llm_engine = SimpleNamespace(collective_rpc=self._rpc)
        self.shard = shard

    def _rpc(self, fn, args=()):
        return [fn(w, *args) for w in self.workers]

    def generate(self, prompts, params, use_tqdm=False):
        outs = []
        for text in prompts:
            for i in range(LAYERS):
                x = _moe_inputs(text, i)
                for r, w in enumerate(self.workers):
                    rows = x[r::len(self.workers)] if self.shard else x
                    w.model_runner.model.model.layers[i].mlp(rows)
            outs.append(SimpleNamespace(prompt_token_ids=list(range(_moe_inputs(text, 0).shape[0]))))
        return outs


# --- HF side ---------------------------------------------------------------


class _HfRouter(nn.Module):
    """transformers' Inkling2TopKRouter.forward."""

    def __init__(self, w, b):
        super().__init__()
        self.top_k = TOP_K
        self.hidden_dim = H
        self.weight = nn.Parameter(w.float())  # _keep_in_fp32_modules_strict
        self.score_fn = torch.sigmoid
        self.register_buffer("e_score_correction_bias", b.clone())

    def forward(self, hidden_states):
        hidden_states = hidden_states.float()
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights, indices


def _hf_engine(w, b):
    layers = nn.ModuleList(nn.Module() for _ in range(LAYERS))
    for i, layer in enumerate(layers):
        layer.mlp = nn.Module()
        layer.mlp.gate = _HfRouter(w[i], b[i])
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = layers
    engine = SteeringEngine.__new__(SteeringEngine)
    engine.model = model

    def extract(msgs):
        for m in msgs:
            for i in range(LAYERS):
                layers[i].mlp.gate(_moe_inputs(m.user, i))

    engine.extract_hidden_states_batched = extract
    return engine


BENIGN = [ChatMessage(system="", user=u) for u in ("bake bread", "a joke", "two plus two", "name a colour")]
TARGET = [ChatMessage(system="", user=u) for u in ("harm one", "harm the second", "harm 3", "harm iv", "harm v!")]


@pytest.mark.parametrize("shard", [False, True])
def test_hooks_rank_experts_as_hf_does(shard):
    w, b = _weights()
    llm = _FakeLLM([_vllm_rank(w, b), _vllm_rank(w, b)], shard=shard)
    got = me.profile_safety_experts_vllm_hooks(llm, BENIGN, TARGET, _Tok(), top_k=TOP_K)
    want = _hf_engine(w, b).identify_safety_experts(BENIGN, TARGET)
    assert got == want
    assert sorted(got) == list(range(LAYERS)) and all(len(v) == E for v in got.values())
    assert any(score != 0 for v in got.values() for _, score in v)
    # The hooks are gone afterwards and never replaced the MoE's input.
    assert all(not hasattr(wk, me._ROUTING_STATE) for wk in llm.workers)
    for wk in llm.workers:
        for layer in wk.model_runner.model.model.layers:
            assert not layer.mlp._forward_pre_hooks


def test_hooks_that_miss_tokens_are_not_used():
    w, b = _weights()
    llm = _FakeLLM([_vllm_rank(w, b)])
    real = llm.generate

    def graphs(prompts, params, use_tqdm=False):  # as if a CUDA graph ran the last prompt
        outs = real(prompts[:-1], params)
        return outs + [SimpleNamespace(prompt_token_ids=[0, 1, 2])]

    llm.generate = graphs
    assert me.profile_safety_experts_vllm_hooks(llm, BENIGN, TARGET, _Tok(), top_k=TOP_K) == {}
    assert not hasattr(llm.workers[0], me._ROUTING_STATE)


def test_no_gates_no_profile():
    top = nn.Module()
    top.model = nn.Module()
    top.model.layers = nn.ModuleList([nn.Module()])
    top.model.layers[0].mlp = nn.Linear(H, H)  # dense: no gate
    llm = _FakeLLM([top])
    assert me.profile_safety_experts_vllm_hooks(llm, BENIGN, TARGET, _Tok(), top_k=TOP_K) == {}


def test_merge_rank_routing():
    a = {"counts": {0: [1, 2]}, "tokens": {0: 3}}
    c = {"counts": {0: [2, 1]}, "tokens": {0: 3}}
    assert me.merge_rank_routing([a, a], 3) == a          # replicated: rank 0
    assert me.merge_rank_routing([a, c], 6) == {"counts": {0: [3, 3]}, "tokens": {0: 6}}  # shards
    assert me.merge_rank_routing([a, c], 5) is None       # neither
    assert me.merge_rank_routing([None], 3) is None


# --- the cli's profiling chain ---------------------------------------------


@pytest.mark.parametrize("routed, hooks, want", [
    ({0: [(1, 0.5)]}, {0: [(2, 0.5)]}, "routed_experts"),
    ({}, {0: [(2, 0.5)]}, "routing_hooks"),
    ({}, {}, "router_weight"),
])
def test_cli_profiles_routed_then_hooks_then_weights(monkeypatch, routed, hooks, want):
    from abliterix import cli
    from abliterix.core import vllm_hidden_states as vhs
    from abliterix.settings import AbliterixConfig

    class Probe:
        def __init__(self, llm, safety):
            self._router_layers = [0]

        def probe(self):
            pass

    calls = []
    monkeypatch.setattr(me, "VLLMMoEEditor", Probe)
    monkeypatch.setattr(vhs, "_load_text_config_data", lambda *a: {"num_experts_per_tok": 8})
    monkeypatch.setattr(me, "profile_safety_experts_vllm", lambda *a, **k: calls.append("routed") or routed)

    def by_hooks(*a, **k):
        calls.append(("hooks", k["top_k"], k["scoring_func"]))
        return hooks

    monkeypatch.setattr(me, "profile_safety_experts_vllm_hooks", by_hooks)
    monkeypatch.setattr(me, "profile_safety_experts_by_weight", lambda *a: {0: [(3, 0.1)]})
    tp_gen = SimpleNamespace(llm=None, tokenizer=_Tok())
    safety, method = cli._profile_safety_experts_on_vllm(AbliterixConfig(), tp_gen, BENIGN, TARGET, torch.zeros(2, H))
    assert method == want
    assert safety == {"routed_experts": routed, "routing_hooks": hooks, "router_weight": {0: [(3, 0.1)]}}[want]
    if want != "routed_experts":
        assert calls[1] == ("hooks", 8, "sigmoid")


def test_plans_without_hf_model_is_opt_in():
    from abliterix import cli
    from abliterix.settings import AbliterixConfig

    on = {"backend": "vllm", "use_in_place_editing": True, "plans_without_hf_model": True}
    assert cli._wants_plans_without_hf_model(AbliterixConfig(model=on, steering={"steering_mode": "direct"}))
    assert not cli._wants_plans_without_hf_model(
        AbliterixConfig(model={**on, "plans_without_hf_model": False}, steering={"steering_mode": "direct"}))
    assert not cli._wants_plans_without_hf_model(
        AbliterixConfig(model={**on, "backend": "hf"}, steering={"steering_mode": "direct"}))
