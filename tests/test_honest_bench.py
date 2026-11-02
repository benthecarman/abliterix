"""Frozen benchmark comparisons require identical token meanings and contexts."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from abliterix.core.engine import SteeringEngine
from abliterix.types import ChatMessage
from scripts import honest_bench as bench


class _Batch(dict):
    def to(self, _device):
        return self


class _Tokenizer:
    def __init__(self, *, vocabulary=None, template="", reverse=False):
        self.vocabulary = vocabulary or {"<pad>": 0, "a": 1, "b": 2, "!": 3}
        self.template = template
        self.reverse = reverse

    def get_vocab(self):
        return dict(self.vocabulary)

    def apply_chat_template(self, chats, **kwargs):
        assert kwargs == {
            "add_generation_prompt": True,
            "tokenize": False,
            "enable_thinking": False,
        }
        return [self.template + chat[-1]["content"] for chat in chats]

    def __call__(self, texts, **kwargs):
        assert kwargs == {
            "return_tensors": "pt",
            "padding": True,
            "return_token_type_ids": False,
        }
        rows = [
            [self.vocabulary.get(character, 1) for character in text] for text in texts
        ]
        if self.reverse:
            rows = [row[::-1] for row in rows]
        width = max(map(len, rows))
        return _Batch(
            input_ids=torch.tensor([[0] * (width - len(row)) + row for row in rows]),
            attention_mask=torch.tensor(
                [[0] * (width - len(row)) + [1] * len(row) for row in rows]
            ),
        )


def _config():
    return SimpleNamespace(
        inference=SimpleNamespace(max_gen_tokens=150, min_gen_tokens=100, batch_size=2),
        kl=SimpleNamespace(token_count=1),
        optimization=SimpleNamespace(checkpoint_dir="unused"),
    )


def _engine(tokenizer=None, *, batch_size=2, prefix=""):
    engine = object.__new__(SteeringEngine)
    engine.tokenizer = tokenizer or _Tokenizer()
    engine.model = SimpleNamespace(device=torch.device("cpu"))
    engine.config = _config()
    engine.config.inference.batch_size = batch_size
    engine.response_prefix = prefix
    return engine


def _messages():
    return [ChatMessage(system="", user="ab"), ChatMessage(system="", user="b")]


def test_context_accepts_identical_token_meanings_with_different_batch_padding():
    target = bench._capture_kl_context(_engine(batch_size=2, prefix="!"), _messages())
    reordered_vocab = {"!": 3, "b": 2, "a": 1, "<pad>": 0}
    base = bench._capture_kl_context(
        _engine(_Tokenizer(vocabulary=reordered_vocab), batch_size=1, prefix="!"),
        _messages(),
    )

    bench._require_matching_kl_context(target, base)

    assert target.rendered_prompts == ("ab!", "b!")
    assert target.input_ids == ((1, 2, 3), (2, 3))


def test_context_rejects_same_size_vocabulary_with_permuted_token_ids():
    target = bench._capture_kl_context(_engine(), _messages())
    base = bench._capture_kl_context(
        _engine(_Tokenizer(vocabulary={"<pad>": 0, "a": 2, "b": 1, "!": 3})),
        _messages(),
    )

    with pytest.raises(SystemExit, match="token-to-ID vocabularies differ"):
        bench._require_matching_kl_context(target, base)


@pytest.mark.parametrize(
    "different_engine,reason",
    [
        (lambda: _engine(_Tokenizer(template="a")), "render different"),
        (lambda: _engine(prefix="!"), "render different"),
        (lambda: _engine(_Tokenizer(reverse=True)), "different input IDs"),
    ],
)
def test_context_rejects_different_template_prefix_or_tokenization(
    different_engine, reason
):
    target = bench._capture_kl_context(_engine(), _messages())
    base = bench._capture_kl_context(different_engine(), _messages())

    with pytest.raises(SystemExit, match=reason):
        bench._require_matching_kl_context(target, base)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("kl", "token_count", 2),
        ("inference", "min_gen_tokens", 99),
        ("inference", "max_gen_tokens", 151),
    ],
)
def test_config_drift_is_rejected_before_model_construction(
    monkeypatch, tmp_path, section, field, value
):
    config = _config()
    setattr(getattr(config, section), field, value)
    calls = []
    monkeypatch.setenv("AX_CONFIG", "unused")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "unused")
    monkeypatch.setattr(
        bench, "_build_temp_config", lambda *_args: str(tmp_path / "cfg.toml")
    )
    monkeypatch.setattr("abliterix.settings.AbliterixConfig", lambda: config)
    monkeypatch.setattr(
        "abliterix.core.engine.SteeringEngine", lambda cfg: calls.append(cfg)
    )

    with pytest.raises(SystemExit, match="Frozen bench config drifted"):
        bench._make_engine("unused/model", "cpu", False)

    assert calls == []


def _main_fixture(monkeypatch, tmp_path, *, base_tokenizer=None):
    import abliterix.eval.detector

    args = SimpleNamespace(
        model="org/candidate",
        base_model="org/base",
        device_map="cpu",
        trust_remote_code=False,
        dry_run=False,
        tier="self_reported",
        output=str(tmp_path / "result.json"),
    )
    entries = [
        {"prompt": "ab", "language": "en", "category": "test"},
        {"prompt": "b", "language": "en", "category": "test"},
    ]
    current = F.log_softmax(torch.tensor([[0.0, 1.0, 2.0, 0.0]]).expand(2, -1), dim=-1)
    baseline = F.log_softmax(torch.tensor([[2.0, 0.0, 1.0, 0.0]]).expand(2, -1), dim=-1)
    events = []
    engines = {"org/candidate": _engine(), "org/base": _engine(base_tokenizer)}

    def make_engine(model_id, *_args):
        engine = engines[model_id]
        engine.name = model_id
        logits = current if model_id == "org/candidate" else baseline

        def compute(_messages):
            events.append((model_id, "score"))
            return logits

        engine.compute_logprobs_batched = compute
        return engine, engine.config, "unused"

    def generate(engine, messages):
        events.append((engine.name, "generate"))
        return ["some text"] * len(messages)

    def free(engine, _path):
        events.append((engine.name, "free"))
        engine.model = None
        engine.tokenizer = None

    class Detector:
        def __init__(self, _config):
            self.config = SimpleNamespace(detection=SimpleNamespace())
            self._judge_prompt_hash = "prompt-hash"

        def _is_degenerate(self, _response):
            return False

        def close(self):
            pass

    monkeypatch.setattr(bench, "parse_args", lambda: args)
    monkeypatch.setattr(
        bench, "_load_dataset", lambda *_args: (entries, "dataset-hash")
    )
    monkeypatch.setattr(bench, "_make_engine", make_engine)
    monkeypatch.setattr(bench, "_generate_responses", generate)
    monkeypatch.setattr(bench, "_free_engine", free)
    monkeypatch.setattr(
        bench, "_judge_responses", lambda _det, prompts, _resp: [False] * len(prompts)
    )
    monkeypatch.setattr(bench, "_judge_cache_sha", lambda _path: "cache-hash")
    monkeypatch.setattr(bench, "_git_commit", lambda: "test-commit")
    monkeypatch.setattr(bench, "_gpu_name", lambda: "cpu")
    monkeypatch.setattr(bench, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(abliterix.eval.detector, "RefusalDetector", Detector)
    return events, current, baseline


def test_main_rejects_incompatible_context_before_base_scoring_and_frees_models(
    monkeypatch, tmp_path
):
    events, _, _ = _main_fixture(
        monkeypatch,
        tmp_path,
        base_tokenizer=_Tokenizer(vocabulary={"<pad>": 0, "a": 2, "b": 1, "!": 3}),
    )

    with pytest.raises(SystemExit, match="token-to-ID vocabularies differ"):
        bench.main()

    assert ("org/base", "score") not in events
    assert ("org/base", "generate") not in events
    assert ("org/candidate", "free") in events
    assert events[-1] == ("org/base", "free")
    assert not (tmp_path / "result.json").exists()


def test_main_uses_shared_kl_with_baseline_to_candidate_orientation(
    monkeypatch, tmp_path
):
    import json

    import abliterix.eval.kl

    _, current, baseline = _main_fixture(monkeypatch, tmp_path)
    calls = []
    original = abliterix.eval.kl.mean_token_kl

    def measure(candidate_logprobs, baseline_logprobs):
        calls.append((candidate_logprobs, baseline_logprobs))
        return original(candidate_logprobs, baseline_logprobs)

    monkeypatch.setattr(abliterix.eval.kl, "mean_token_kl", measure)

    assert bench.main() == 0

    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], current)
    torch.testing.assert_close(calls[0][1], baseline)
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    expected = (baseline.exp() * (baseline - current)).sum(dim=-1).mean().item()
    assert result["spec_version"] == "1.1"
    assert result["damage_metric"] == {
        "name": "full_distribution_kl",
        "estimator": "mean_per_token_kl",
        "units": "nats/token",
        "value": pytest.approx(expected, abs=1e-6),
    }
