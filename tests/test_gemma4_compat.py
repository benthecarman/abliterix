"""The Gemma4 config shim must not replace a Transformers' own Gemma4 classes:
vLLM then rejects the config ("Invalid type of HuggingFace config")."""

import transformers

import abliterix.core.vllm_compat as compat


def test_shim_leaves_native_gemma4_alone(monkeypatch):
    registered = []
    monkeypatch.setattr(compat, "_transformers_knows", lambda model_type: True)
    monkeypatch.setattr(transformers.AutoConfig, "register",
                        lambda *args, **kwargs: registered.append(args[0]))
    compat.install_gemma4_transformers_compat()
    assert registered == []


def test_shim_registers_gemma4_where_missing(monkeypatch):
    registered = []
    monkeypatch.setattr(compat, "_transformers_knows", lambda model_type: False)
    monkeypatch.setattr(transformers.AutoConfig, "register",
                        lambda *args, **kwargs: registered.append(args[0]))
    compat.install_gemma4_transformers_compat()
    assert registered == ["gemma4", "gemma4_text"]


def test_knows_a_model_type_transformers_ships():
    assert compat._transformers_knows("llama")
    assert not compat._transformers_knows("no_such_model_type")
