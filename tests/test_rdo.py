"""Tests for abliterix.rdo — gradient-based Refusal Direction Optimization.

Verifies the optimization-based extractor from Wollschläger et al., ICML 2025
(arXiv:2502.17420).  A tiny end-to-end differentiable causal LM stands in for
a real model so the intervention hooks, teacher-forced losses, and gradient
flow to the single direction parameter can be exercised on CPU with no GPU
and no model download.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from abliterix.settings import AbliterixConfig
from abliterix.types import VectorMethod
from abliterix.vectors import compute_steering_vectors


# ---------------------------------------------------------------------------
# Tiny end-to-end differentiable LM + mock engine
# ---------------------------------------------------------------------------


class _TinyLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, h):
        # Returns a bare tensor (RDO's hook must handle both tensor & tuple).
        return h + torch.tanh(self.lin(h))


class _TinyLayerTuple(nn.Module):
    """Layer returning a tuple ``(hidden, aux)`` — the real HF decoder shape."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, h):
        return (h + torch.tanh(self.lin(h)), None)


class _TinyLM(nn.Module):
    def __init__(self, vocab: int, hidden: int, n_layers: int, tuple_out: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        layer_cls = _TinyLayerTuple if tuple_out else _TinyLayer
        self.layers = nn.ModuleList([layer_cls(hidden) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, vocab)
        self.config = SimpleNamespace(hidden_size=hidden)
        self._tuple_out = tuple_out

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        for layer in self.layers:
            # A forward hook returning a value replaces the layer output, so
            # the ablation/addition intervention lands here.
            out = layer(h)
            h = out[0] if self._tuple_out else out
        return SimpleNamespace(logits=self.head(h))


class _TinyTokenizer:
    def __init__(self, vocab: int):
        self.vocab = vocab

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = [(ord(c) % self.vocab) or 1 for c in text][:6] or [1]
        return SimpleNamespace(input_ids=torch.tensor([ids]))


def _make_engine(vocab=40, hidden=32, n_layers=4, seq=5, seed=0, tuple_out=False):
    torch.manual_seed(seed)
    model = _TinyLM(vocab, hidden, n_layers, tuple_out=tuple_out)
    tokenizer = _TinyTokenizer(vocab)

    def _tokenize(messages):
        n = len(messages)
        # Deterministic pseudo-tokens per prompt; all-ones attention (no pad).
        ids = torch.randint(1, vocab, (n, seq))
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(n, seq, dtype=torch.long),
        }

    return SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        transformer_layers=model.layers,
        _tokenize=_tokenize,
    )


def _rdo_config(**overrides) -> AbliterixConfig:
    cfg = AbliterixConfig()
    cfg.steering.vector_method = VectorMethod.RDO
    cfg.steering.rdo_steps = 4
    cfg.steering.rdo_batch_size = 2
    cfg.steering.rdo_max_prompts = 4
    cfg.steering.rdo_seed = 123
    for k, v in overrides.items():
        setattr(cfg.steering, k, v)
    return cfg


def _msgs(n):
    return [f"prompt number {i} about something" for i in range(n)]


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_rdo_returns_correct_shape_and_unit_norm():
    from abliterix.rdo import optimize_rdo_direction

    engine = _make_engine(n_layers=4, hidden=32)
    cfg = _rdo_config()
    benign = torch.randn(6, 5, 32)
    target = benign + 0.4

    vectors = optimize_rdo_direction(
        engine, _msgs(6), _msgs(6), cfg, benign_states=benign, target_states=target
    )

    assert vectors.shape == (5, 32)  # (n_layers+1, hidden)
    norms = vectors.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert torch.isfinite(vectors).all()
    # Must return on the states' device (convention: CPU when offloaded), not
    # the model/train device — else downstream projection/steering mismatches.
    assert vectors.device == benign.device


def test_rdo_broadcasts_single_direction_across_layers():
    """RDO learns ONE direction; every layer row is that same unit vector."""
    from abliterix.rdo import optimize_rdo_direction

    engine = _make_engine()
    cfg = _rdo_config(projected_abliteration=False, winsorize_vectors=False)
    vectors = optimize_rdo_direction(engine, _msgs(4), _msgs(4), cfg)

    first = vectors[0]
    for row in vectors[1:]:
        assert torch.allclose(row, first, atol=1e-6)


def test_rdo_actually_optimizes_direction():
    """The returned direction should move away from the mean-diff warm start."""
    from abliterix.rdo import optimize_rdo_direction

    engine = _make_engine()
    cfg = _rdo_config(rdo_steps=15, rdo_lr=0.05)
    benign = torch.randn(6, 5, 32)
    target = benign + torch.randn(1, 5, 32) * 0.5

    # Warm-start seed = mean-diff at the addition layer + 1.
    add_layer = round(cfg.steering.rdo_add_layer_frac * (4 - 1))
    seed = torch.nn.functional.normalize(
        target[:, add_layer + 1, :].mean(0) - benign[:, add_layer + 1, :].mean(0),
        p=2,
        dim=0,
    )

    vectors = optimize_rdo_direction(
        engine, _msgs(6), _msgs(6), cfg, benign_states=benign, target_states=target
    )
    learned = vectors[0]
    cos = torch.dot(learned, seed).abs().item()
    # Optimization changed the direction (not identical to the seed).
    assert cos < 0.999


def test_rdo_handles_tuple_layer_output():
    """Real HF decoder layers return tuples; the hook must intervene on out[0]."""
    from abliterix.rdo import optimize_rdo_direction

    engine = _make_engine(tuple_out=True)
    cfg = _rdo_config()
    benign = torch.randn(6, 5, 32)
    target = benign + 0.4
    vectors = optimize_rdo_direction(
        engine, _msgs(6), _msgs(6), cfg, benign_states=benign, target_states=target
    )
    assert vectors.shape == (5, 32)
    assert torch.isfinite(vectors).all()


def test_rdo_works_with_grad_globally_disabled():
    """Regression: abliterix's cli.configure_libraries calls
    torch.set_grad_enabled(False) at startup, so RDO must re-enable autograd
    locally (else backward raises 'element 0 ... does not require grad')."""
    from abliterix.rdo import optimize_rdo_direction

    engine = _make_engine()
    cfg = _rdo_config()
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)  # simulate the global startup disable
    try:
        vectors = optimize_rdo_direction(engine, _msgs(4), _msgs(4), cfg)
        assert vectors.shape == (5, 32)
        assert torch.isfinite(vectors).all()
        # RDO must restore the global grad flag to what it found (False here).
        assert torch.is_grad_enabled() is False
    finally:
        torch.set_grad_enabled(prev)


def test_rdo_random_init_without_states():
    from abliterix.rdo import optimize_rdo_direction

    engine = _make_engine()
    cfg = _rdo_config(rdo_init="random")
    vectors = optimize_rdo_direction(engine, _msgs(4), _msgs(4), cfg)
    assert vectors.shape == (5, 32)
    assert torch.isfinite(vectors).all()


# ---------------------------------------------------------------------------
# Side-effect hygiene
# ---------------------------------------------------------------------------


def test_rdo_restores_model_grad_and_mode():
    from abliterix.rdo import optimize_rdo_direction

    engine = _make_engine()
    model = engine.model
    model.train()
    before = {id(p): p.requires_grad for p in model.parameters()}

    optimize_rdo_direction(engine, _msgs(4), _msgs(4), _rdo_config())

    assert model.training is True  # restored to train mode
    after = {id(p): p.requires_grad for p in model.parameters()}
    assert before == after  # requires_grad restored


def test_rdo_removes_all_hooks():
    from abliterix.rdo import optimize_rdo_direction

    engine = _make_engine()
    optimize_rdo_direction(engine, _msgs(4), _msgs(4), _rdo_config())
    for layer in engine.transformer_layers:
        assert len(layer._forward_hooks) == 0


def test_rdo_projected_abliteration_composes():
    """With projected_abliteration on, output is orthogonal to the benign mean."""
    from abliterix.rdo import optimize_rdo_direction

    engine = _make_engine()
    cfg = _rdo_config(projected_abliteration=True)
    benign = torch.randn(6, 5, 32)
    target = benign + 0.4
    vectors = optimize_rdo_direction(
        engine, _msgs(6), _msgs(6), cfg, benign_states=benign, target_states=target
    )
    benign_dir = torch.nn.functional.normalize(benign.mean(0), p=2, dim=1)
    dots = (vectors * benign_dir).sum(dim=1).abs()
    assert (dots < 1e-4).all()


# ---------------------------------------------------------------------------
# Guards & validation
# ---------------------------------------------------------------------------


def test_rdo_requires_loaded_model():
    from abliterix.rdo import optimize_rdo_direction

    engine = _make_engine()
    engine.model = None
    with pytest.raises(RuntimeError, match="requires a loaded HuggingFace model"):
        optimize_rdo_direction(engine, _msgs(4), _msgs(4), _rdo_config())


def test_compute_steering_vectors_rejects_rdo():
    benign = torch.randn(6, 5, 32)
    target = benign + 0.4
    with pytest.raises(ValueError, match="cannot be computed from cached"):
        compute_steering_vectors(benign, target, VectorMethod.RDO, False)


def test_config_rejects_rdo_with_multi_direction():
    cfg = AbliterixConfig()
    with pytest.raises(ValueError, match="single direction"):
        cfg.steering.__class__(vector_method="rdo", n_directions=3)


def test_config_rejects_bad_rdo_init():
    with pytest.raises(ValueError, match="rdo_init must be"):
        AbliterixConfig().steering.__class__(vector_method="rdo", rdo_init="bogus")
