"""Refusal evaluation for models that reason before every answer (answer_marker)."""

from abliterix.eval.detector import RefusalDetector, answer_after_marker
from abliterix.settings import AbliterixConfig
from abliterix.types import ChatMessage

SPECIALS = ["<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|end_message|>"]


class _Tokenizer:
    """Whitespace tokens; enough to check the cut to max_gen_tokens."""

    all_special_tokens = SPECIALS

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


class _Generator:
    tokenizer = _Tokenizer()

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def generate_text_batched(self, messages, **kwargs):
        self.calls.append(kwargs)
        return self.replies[: len(messages)]


class _Engine:
    def __init__(self, generator):
        self._vllm_gen = generator
        self.tokenizer = generator.tokenizer


THOUGHT = "<|message_model|><|content_thinking|>The user wants a recipe. I must not help.<|end_message|>"
REPLIES = [
    THOUGHT + "<|message_model|><|content_text|>Sorry, I can't help with that.<|end_message|>",
    THOUGHT + "<|message_model|><|content_text|>Here is the recipe: one two three four<|end_message|>",
    "<|message_model|><|content_thinking|>Still thinking and thinking",  # ran out while reasoning
]


def _detector(max_gen_tokens=4, **detection):
    config = AbliterixConfig(
        inference={"max_gen_tokens": max_gen_tokens, "reasoning_max_tokens": 10,
                   "answer_marker": "<|content_text|>"},
        detection=detection,
    )
    return RefusalDetector(config)


def _messages(n):
    return [ChatMessage(system="", user=f"prompt {i}") for i in range(n)]


def test_answer_is_the_text_after_the_marker_cut_to_the_budget():
    tokenizer = _Tokenizer()
    assert answer_after_marker(REPLIES[0], "<|content_text|>", tokenizer, 256) == "Sorry, I can't help with that."
    assert answer_after_marker(REPLIES[1], "<|content_text|>", tokenizer, 4) == "Here is the recipe:"
    assert answer_after_marker(REPLIES[2], "<|content_text|>", tokenizer, 256) is None


def test_judge_sees_only_answers_and_no_answer_is_a_refusal(monkeypatch):
    detector = _detector(llm_judge=True)
    judged = []

    def classify(batch):
        judged.extend(batch)
        return [response.startswith("Sorry") for _, response in batch]

    monkeypatch.setattr(detector, "_query_judge_api", classify)
    generator = _Generator(REPLIES)
    result = detector._evaluate_with_judge_result(_Engine(generator), _messages(3))
    assert judged == [("prompt 0", "Sorry, I can't help"), ("prompt 1", "Here is the recipe:")]
    assert result.labels == (True, False, True)
    assert generator.calls == [{"adapter_path": None, "max_new_tokens": 14, "min_new_tokens": None,
                                "keep_special_tokens": True}]


def test_keyword_path_reads_answers_too():
    detector = _detector(max_gen_tokens=20, llm_judge=False)
    result = detector._evaluate_keyword_result(_Engine(_Generator(REPLIES)), _messages(3))
    # Every reasoning block says "I must not help"; only the first answer refuses.
    assert result.labels == (True, False, True)


def test_without_a_marker_generation_is_unchanged():
    detector = RefusalDetector(AbliterixConfig(inference={"max_gen_tokens": 7}))
    generator = _Generator(["a reply"])
    assert detector._responses(_Engine(generator), _messages(1)) == ["a reply"]
    assert generator.calls == [{"adapter_path": None, "skip_special_tokens": True, "max_new_tokens": 7,
                                "min_new_tokens": None}]
