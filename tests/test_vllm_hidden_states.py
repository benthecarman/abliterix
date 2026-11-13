"""vLLM hidden-state extraction indexes layers as HF does: aux id ``k`` (the
state after ``k`` layers) lands at index ``k``, HF's ``hidden_states[k]``."""

import sys
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

import abliterix.core.vllm_compat as compat
import abliterix.core.vllm_hidden_states as hs
from abliterix.settings import AbliterixConfig
from abliterix.types import ChatMessage

N, H = 5, 8


def _reference(prompt: int, length: int) -> torch.Tensor:
    """HF's ``hidden_states`` for a prompt, ``[N + 1, length, H]``: row ``k`` is
    the state after ``k`` layers, every value tagged with ``(prompt, k, pos)``."""
    k = torch.arange(N + 1).view(-1, 1, 1)
    pos = torch.arange(length).view(1, -1, 1)
    return (1000.0 * prompt + 10.0 * k + 0.01 * pos).expand(N + 1, length, H).clone()


class _Tokenizer:
    def apply_chat_template(self, chat, **kwargs):
        return chat[-1]["content"]


def _fake_vllm(tmp_path, seen):
    class LLM:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.ids = kwargs["speculative_config"]["draft_model_config"]["hf_config"][
                "eagle_aux_hidden_state_layer_ids"]

        def generate(self, prompts, sampling_params):
            outs = []
            for p, text in enumerate(prompts):
                ref = _reference(int(text), 3 + p)
                # The connector stores [prompt_len, len(ids), H], aux id k is ref[k].
                stored = torch.stack([ref[k] for k in self.ids], dim=1)
                path = tmp_path / f"{p}.safetensors"
                save_file({"hidden_states": stored.contiguous()}, str(path))
                outs.append(NS(request_id=str(p), kv_transfer_params={"hidden_states_path": str(path)}))
            return outs

    return NS(LLM=LLM, SamplingParams=lambda **kw: None)


def test_aux_id_k_lands_at_hf_index_k(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(compat, "install_gemma4_transformers_compat", lambda: None)
    monkeypatch.setitem(sys.modules, "vllm", _fake_vllm(tmp_path, seen))
    monkeypatch.setattr(hs, "_load_text_config_data", lambda *a: {"num_hidden_layers": N})
    monkeypatch.setattr(hs.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    monkeypatch.setenv("AX_HIDDEN_STATES_DIR", str(tmp_path))
    sets = {"benign": [ChatMessage(system="", user="0"), ChatMessage(system="", user="1")],
            "target": [ChatMessage(system="", user="2")]}

    out = hs.extract_hidden_states_vllm(AbliterixConfig(), sets)

    ids = seen["speculative_config"]["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"]
    assert ids == list(range(1, N + 1))
    assert out["benign"].shape == (2, N + 1, H) and out["target"].shape == (1, N + 1, H)
    for name, prompts in (("benign", [0, 1]), ("target", [2])):
        for row, p in enumerate(prompts):
            got = out[name][row]
            assert torch.equal(got[0], torch.zeros(H))  # the embeddings are not extracted
            last = _reference(p, 3 + (p if name == "benign" else 2))[:, -1]
            assert torch.equal(got[1:], last[1:])


def test_layer_rows_reads_either_layout():
    ref = _reference(0, 4)[1:]  # aux ids 1..N, [N, len, H]
    want = torch.cat([torch.zeros(1, H), ref[:, -1]])
    assert torch.equal(hs.layer_rows(ref.transpose(0, 1).contiguous(), N), want)
    assert torch.equal(hs.layer_rows(ref, N), want)
    assert torch.equal(hs.layer_rows(ref.transpose(0, 1).contiguous(), N, 1)[1:], ref[:, 1])


def test_a_model_without_the_last_state_is_refused():
    # deepseek_v2 in vLLM 0.20 captures id k before layer k: id N never comes.
    with pytest.raises(RuntimeError, match="does not capture the state after its last layer"):
        hs.layer_rows(torch.zeros(4, N - 1, H), N)


@pytest.mark.parametrize("text_type, top_type, ok", [
    ("inkling2_text", "inkling2", True),
    (None, "inkling2", True),       # the release's text_config has no model_type
    (None, "unknown", False),
    ("unknown", "inkling2", False),
])
def test_inkling2_is_supported(monkeypatch, tmp_path, text_type, top_type, ok):
    import json

    text = {"num_hidden_layers": N} | ({"model_type": text_type} if text_type else {})
    (tmp_path / "config.json").write_text(json.dumps({"model_type": top_type, "text_config": text}))
    monkeypatch.setattr(hs, "_load_text_config_data", lambda *a: text)
    config = AbliterixConfig(model={"model_id": str(tmp_path)})
    assert hs.is_model_supported(config) is ok
