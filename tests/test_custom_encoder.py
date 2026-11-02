"""Local encoder signatures work without downloading model code or weights."""

import pytest

from abliterix.core.engine import SteeringEngine


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": [[1, 2]] if isinstance(texts, list) else [1, 2]}


def _install(tmp_path, source, kwargs=None):
    path = tmp_path / "custom_encoder.py"
    path.write_text(source, encoding="utf-8")
    engine = object.__new__(SteeringEngine)
    engine.tokenizer = _Tokenizer()
    engine._install_custom_encoder(str(path), kwargs or {})
    return engine.tokenizer


def test_encoder_without_generation_prompt_parameter(tmp_path):
    # The official DeepSeek-V4 encoder takes thinking_mode, not a generation flag.
    tokenizer = _install(
        tmp_path,
        "def encode_messages(messages, thinking_mode, context=None):\n"
        "    return thinking_mode + ':' + messages[0]['content']\n",
        {"thinking_mode": "chat"},
    )
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "你好"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        == "chat:你好"
    )


@pytest.mark.parametrize(
    "signature, expression, options",
    [
        ("messages, add_generation_prompt=False", "add_generation_prompt", {}),
        ("messages, *, add_generation_prompt=False", "add_generation_prompt", {}),
        ("messages, **kwargs", "kwargs['add_generation_prompt']", {}),
        (
            "messages, **add_generation_prompt",
            "add_generation_prompt['add_generation_prompt']",
            {},
        ),
        ("messages, add_generation_prompt=False, /", "add_generation_prompt", {}),
        (
            "messages, mode, add_generation_prompt=False, /",
            "add_generation_prompt",
            {"mode": "chat"},
        ),
    ],
)
def test_encoder_honors_runtime_flag_for_supported_signatures(
    tmp_path, signature, expression, options
):
    options = {**options, "add_generation_prompt": False}
    original = dict(options)
    tokenizer = _install(
        tmp_path,
        f"def encode_messages({signature}):\n"
        f"    return messages[0]['content'] + ':' + str({expression})\n",
        options,
    )
    messages = [{"role": "user", "content": "hello"}]
    for flag in (True, False, True):
        assert (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=flag
            )
            == f"hello:{flag}"
        )
    assert options == original


def test_encoder_supports_variadic_messages_argument(tmp_path):
    tokenizer = _install(
        tmp_path,
        "def encode_messages(*args, **kwargs):\n"
        "    return args[0][0]['content'] + ':' + str(kwargs['add_generation_prompt'])\n",
    )
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        == "hello:True"
    )


def test_encoder_batch_and_tokenization_options(tmp_path):
    tokenizer = _install(
        tmp_path,
        "def encode_messages(messages):\n    return '<BOS>' + messages[0]['content']\n",
    )
    batch = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
    ]
    assert tokenizer.apply_chat_template(batch, tokenize=False) == [
        "<BOS>one",
        "<BOS>two",
    ]
    assert tokenizer.apply_chat_template(
        batch, return_tensors="pt", padding=True, truncation=True
    ) == [[1, 2]]
    assert tokenizer.calls == [
        (
            ["<BOS>one", "<BOS>two"],
            {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "add_special_tokens": False,
            },
        )
    ]


@pytest.mark.parametrize("options", [{}, {"mode": "chat", "typo": True}])
def test_encoder_reports_bad_options_at_installation(tmp_path, options):
    with pytest.raises(TypeError, match="Invalid options for custom encoder"):
        _install(
            tmp_path,
            "def encode_messages(messages, mode):\n    return 'prompt'\n",
            options,
        )
