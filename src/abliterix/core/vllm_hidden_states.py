# Abliterix — vLLM native hidden state extraction
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Extract per-layer hidden states using vLLM 0.19's native API.

Replaces the ``speculators`` library which is incompatible with vLLM 0.19.
Uses vLLM's built-in ``extract_hidden_states`` speculative method and
``ExampleHiddenStatesConnector`` to extract hidden states with full tensor
parallelism — all GPUs compute simultaneously.

Requires vLLM >= 0.17.0 (PR #33736).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer

from ..settings import AbliterixConfig
from ..types import ChatMessage
from ..util import flush_memory, print, chat_template_kwargs


_SUPPORTED_MODEL_TYPES = {
    "llama",
    "qwen",
    "minicpm",
    "gpt_oss",
    "hunyuan_vl",
    "hunyuan_v1_dense",
    "afmoe",
    "nemotron_h",
    "deepseek_v2",
    "deepseek_v3",
    "kimi_k2",
    "kimi_k25",
    "gemma4",
    "gemma4_text",
    "inkling2",
    "inkling2_text",
    # "minimax_m2" — disabled: vLLM 0.19.1's extract_hidden_states path on
    # MiniMax-M2 (62 layers × eagle_aux_hidden_state hooks) deadlocks after
    # NCCL init on Blackwell PCIe (sm_120 RTX PRO 6000) — workers spin at
    # 100% CPU / idle GPU, no weight loading. disable_custom_all_reduce=True
    # does not help. Falls back to HF pipeline parallelism for Phase 1
    # (1-GPU-busy, ~30 min). Re-enable when vLLM upstream fixes this
    # (tracked in issue #33041 + related MiniMax extract_hidden_states).
    # "step3p5" — similar incompatibility; falls back to HF PP.
}


def _config_json(model_id: str, revision: str | None = None) -> dict[str, Any]:
    path = os.path.join(model_id, "config.json")
    if not os.path.isfile(path):
        path = hf_hub_download(model_id, "config.json", revision=revision)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_text_config_data(
    model_id: str,
    trust_remote_code: bool,
    revision: str | None = None,
) -> dict[str, Any]:
    """Return text config fields even when Transformers lacks a fresh model type."""
    try:
        auto_cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        text_cfg = getattr(auto_cfg, "text_config", auto_cfg)
        if hasattr(text_cfg, "to_dict"):
            return text_cfg.to_dict()
        return vars(text_cfg)
    except ValueError as exc:
        if "does not recognize this architecture" not in str(exc):
            raise

    cfg = _config_json(model_id, revision)
    text_cfg = cfg.get("text_config") or cfg
    return text_cfg


def _engine_accepts(name: str) -> bool:
    """Whether the installed vLLM's engine arguments include ``name``."""
    import dataclasses

    try:
        from vllm.engine.arg_utils import EngineArgs
    except ImportError:
        return False
    return name in {f.name for f in dataclasses.fields(EngineArgs)}


def _top_level_model_type(config: AbliterixConfig) -> str:
    return _config_json(config.model.model_id, config.model.revision).get("model_type", "")


def is_model_supported(config: AbliterixConfig) -> bool:
    """Check if the model's architecture is in extract_hidden_states whitelist."""
    try:
        text_cfg = _load_text_config_data(
            config.model.model_id,
            config.model.trust_remote_code or False,
            config.model.revision,
        )
        model_type = text_cfg.get("model_type", "")
        if model_type in _SUPPORTED_MODEL_TYPES:
            return True
        # Release configs may leave text_config's model_type out (Inkling 2).
        return not model_type and _top_level_model_type(config) in _SUPPORTED_MODEL_TYPES
    except Exception:
        return False


def aux_layer_ids(num_layers: int) -> list[int]:
    """The ``eagle_aux_hidden_state_layer_ids`` to request: ``1..N``.

    vLLM's aux id ``k`` is the residual stream after ``k`` decoder layers
    (llama/gemma4 capture id ``k + 1`` after layer ``k``; Inkling 2 captures
    id ``i`` before layer ``i`` and appends its final output as id ``N``),
    which is HF's ``hidden_states[k]``. Id 0, the embeddings, is left out:
    HF steering never uses it and we keep index 0 a zero placeholder."""
    return list(range(1, num_layers + 1))


def layer_rows(hs: Tensor, num_layers: int, token_offset: int = -1) -> Tensor:
    """One prompt's connector tensor to ``[N + 1, hidden]`` indexed like HF's
    ``hidden_states``: row 0 a zero placeholder for the embeddings, row ``k``
    the state after ``k`` layers (aux id ``k``, see :func:`aux_layer_ids`).

    The connector stores ``[prompt_len, N, hidden]`` (observed on gpt-oss /
    vLLM 0.19); older builds stored ``[N, prompt_len, hidden]``. The layer
    axis is the one of length ``N``."""
    if hs.dim() != 3:
        raise RuntimeError(
            f"Expected 3-D hidden_states tensor, got shape {list(hs.shape)}"
        )
    if hs.shape[1] == num_layers:
        layer_vecs = hs[token_offset, :, :]
    elif hs.shape[0] == num_layers:
        layer_vecs = hs[:, token_offset, :]
    elif num_layers - 1 in (hs.shape[0], hs.shape[1]):
        # Models that capture id k before layer k (deepseek_v2 in vLLM
        # 0.20) never reach id N; refuse rather than shift the layers.
        raise RuntimeError(
            f"hidden_states shape {list(hs.shape)} has {num_layers - 1} layers "
            f"of {num_layers}: this vLLM model does not capture the state "
            "after its last layer."
        )
    else:
        raise RuntimeError(
            f"Unexpected hidden_states shape {list(hs.shape)} "
            f"(num_layers={num_layers}); neither dim 0 nor dim 1 "
            f"matches the layer count."
        )
    placeholder = torch.zeros(1, layer_vecs.shape[1], dtype=layer_vecs.dtype)
    return torch.cat([placeholder, layer_vecs], dim=0)


def _worker_aux_model(worker: Any):
    """The module whose forward returns ``(out, aux_hidden_states)`` once
    its ``aux_hidden_state_layers`` are set (vLLM's Eagle-3 interface)."""
    for module in worker.model_runner.model.modules():
        if hasattr(module, "aux_hidden_state_layers") and hasattr(module, "layers"):
            return module
    raise RuntimeError("this vLLM model has no aux_hidden_state_layers to capture")


def _worker_install_aux_capture(worker: Any, layer_ids: list[int], token_offset: int) -> None:
    """Set the aux layers and wrap the model's forward so it keeps one
    ``[len(layer_ids), hidden]`` row per forward (the token at
    ``token_offset``) and returns only its output, as without aux layers."""
    model = _worker_aux_model(worker)
    model.aux_hidden_state_layers = tuple(layer_ids)
    rows: list[Tensor] = []
    inner = model.forward

    def forward(*args, **kwargs):
        out, aux = inner(*args, **kwargs)
        rows.append(torch.stack([a[token_offset] for a in aux]).float().cpu())
        return out

    model.forward = forward
    worker._abliterix_aux_rows = rows


def _worker_take_aux_rows(worker: Any) -> list[Tensor]:
    rows = list(worker._abliterix_aux_rows)
    worker._abliterix_aux_rows.clear()
    return rows


def _capture_aux_rows(llm, prompts, sampling_params, layer_ids, num_layers, token_offset) -> list[Tensor]:
    """Each prompt's ``[N + 1, hidden]`` rows (see :func:`layer_rows`), read
    from the model's aux hidden states, one prompt per forward."""
    rpc = llm.llm_engine.collective_rpc
    rpc(_worker_install_aux_capture, args=(layer_ids, token_offset))
    results = []
    for prompt in prompts:
        llm.generate([prompt], sampling_params, use_tqdm=False)
        # Every rank holds the same rows (gathered under sequence parallel).
        rows = rpc(_worker_take_aux_rows)[0]
        if len(rows) != 1:
            raise RuntimeError(f"expected one forward for the prompt, got {len(rows)}")
        results.append(layer_rows(rows[0].unsqueeze(0), num_layers, 0))
    return results


@contextlib.contextmanager
def _without_expandable_segments():
    """vLLM refuses the hidden-states KV connector when PyTorch's allocator
    may remap memory (``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``,
    which rented-machine drivers often set): drop that option while the
    engine starts, so its processes inherit the environment without it, and
    put it back after."""
    key = "PYTORCH_CUDA_ALLOC_CONF"
    before = os.environ.get(key)
    if before is None or "expandable_segments" not in before:
        yield
        return
    kept = [o for o in before.split(",") if o.strip() and not o.strip().startswith("expandable_segments")]
    if kept:
        os.environ[key] = ",".join(kept)
    else:
        del os.environ[key]
    try:
        yield
    finally:
        os.environ[key] = before


def extract_hidden_states_vllm(
    config: AbliterixConfig,
    prompt_sets: dict[str, list[ChatMessage]],
    token_offset: int = -1,
) -> dict[str, Tensor]:
    """Extract per-layer hidden states using vLLM's native extraction API.

    Loads the model once and extracts hidden states for every prompt set in
    a single ``llm.generate`` call — a reload per set would pay the MooseFS
    shard pull (~2.5 min for 15-shard models) for each one.

    Parameters
    ----------
    config : AbliterixConfig
        Model and inference configuration.
    prompt_sets : dict[str, list[ChatMessage]]
        Named prompt sets (e.g. ``{"benign": [...], "target": [...]}``).
        All sets are tokenized and submitted together; outputs are split
        back by name on return.
    token_offset : int
        Position in the sequence to extract from.  ``-1`` (default) extracts
        the final token.

    Returns
    -------
    dict[str, Tensor]
        Same keys as ``prompt_sets``.  Each tensor has shape
        ``(batch, layers+1, hidden_dim)``.  Index 0 along the layer axis is
        a zero placeholder for the embedding layer; indices 1..N are decoder
        layer outputs.
    """
    from .vllm_compat import install_gemma4_transformers_compat

    install_gemma4_transformers_compat()

    from vllm import LLM, SamplingParams

    model_id = config.model.model_id
    tp = config.model.tensor_parallel_size
    if tp is None:
        tp = torch.cuda.device_count()
    trust = config.model.trust_remote_code or False

    # Get number of layers from config. Some newly released vLLM-supported
    # models may not exist in the installed Transformers registry yet.
    text_cfg = _load_text_config_data(model_id, trust, config.model.revision)
    num_layers = text_cfg["num_hidden_layers"]
    layer_ids = aux_layer_ids(num_layers)

    print(f"* Loading model in vLLM with TP={tp} for hidden state extraction...")

    # The connector writes per-layer hidden states for every processed prompt
    # — for a 36-layer 400-prompt run that's ~16 GB.  Container /tmp is often
    # a 20 GB overlayfs that fills fast; route to /workspace (or configured
    # HF_HOME parent) which on RunPod is a 1 TB+ volume.
    hs_parent = os.environ.get("AX_HIDDEN_STATES_DIR")
    if not hs_parent:
        hf_home = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
        hs_parent = (os.path.dirname(hf_home) if hf_home else None) or (
            "/workspace" if os.path.isdir("/workspace") else None
        )
    if hs_parent and os.path.isdir(hs_parent):
        os.makedirs(hs_parent, exist_ok=True)
        tmpdir = tempfile.mkdtemp(prefix="abliterix_hs_", dir=hs_parent)
    else:
        tmpdir = tempfile.mkdtemp(prefix="abliterix_hs_")

    draft_hf_config = dict(text_cfg)
    draft_hf_config["eagle_aux_hidden_state_layer_ids"] = layer_ids

    kwargs: dict[str, Any] = dict(
        model=model_id,
        revision=config.model.revision,
        tensor_parallel_size=tp,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        trust_remote_code=trust,
        enforce_eager=True,  # Safer for extraction
        # Disable custom all-reduce on Blackwell PCIe (sm_120 RTX PRO 6000 etc.):
        # NCCL inits but engine deadlocks before weight loading (vllm issue
        # #33041). Same fix as vllm_backend.py.
        disable_custom_all_reduce=True,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": draft_hf_config,
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": tmpdir,
            },
        },
    )
    # Forward max_model_len so vLLM doesn't reserve KV for the model's full
    # native context (e.g. 196K for MiniMax-M2 would require 81GB/GPU for
    # a single request and OOMs the engine). Abliteration prompts + gen are
    # < 2K, so the config-level cap is the right ceiling.
    if config.model.max_model_len:
        kwargs["max_model_len"] = config.model.max_model_len
    if config.model.max_num_seqs:
        kwargs["max_num_seqs"] = config.model.max_num_seqs

    # Text prompts only, as the generation engine (vllm_backend) runs them: no
    # image/video/audio inputs, which also skips vLLM's multimodal profiling
    # (Gemma 4's processor fails on its own dummy video in some builds).
    kwargs["limit_mm_per_prompt"] = config.model.limit_mm_per_prompt or {
        "image": 0,
        "video": 0,
        "audio": 0,
    }

    # Extraction is one prefill per prompt, so the engine needs no warmup. Builds
    # with an extensible KV cache warm up anyway, with a decode step that carries
    # the extractor's draft token; on Inkling 2 that step fails in the NVFP4 MoE
    # GEMM. Without the extensible cache (and eager), vLLM skips the warmup and
    # sizes the KV cache by profiling, as older builds did.
    if _engine_accepts("enable_extensible_kv_cache"):
        kwargs["enable_extensible_kv_cache"] = False

    # The generation engine's attention backend: some builds pick one that
    # lacks kernels for the model's head size (Gemma 4's 512 on B300).
    from .vllm_backend import _resolve_attention_backend

    attention_backend = _resolve_attention_backend(
        config.model.attention_backend,
        (_config_json(model_id, config.model.revision).get("architectures") or [""])[0],
    )
    if attention_backend is not None:
        kwargs["attention_config"] = {"backend": attention_backend}

    # Model config overrides (e.g. MTP-3 → MTP-1 for Step-3.5-Flash).
    if config.model.hf_overrides:
        kwargs["hf_overrides"] = config.model.hf_overrides

    # FP8: let vLLM auto-detect from config.json.
    is_fp8 = config.model.quant_method and config.model.quant_method.value == "fp8"
    if is_fp8:
        kwargs["quantization"] = "fp8"

    # ``LLM(**kwargs)`` is the most likely failure point here (OOM, bad
    # max_model_len, unsupported arch). It has to be inside the try: the
    # finally block below exists precisely to guarantee teardown, and its
    # own comment notes that a partially-initialised engine pins the frame
    # and keeps its VRAM. It also guarantees the temp dir is removed.
    aux_capture = config.model.vllm_hidden_state_capture == "aux_layers"
    if aux_capture:
        # A plain engine: no speculator, no connector. Every token is
        # prefilled, so each prompt is one forward whose last row is read.
        del kwargs["speculative_config"], kwargs["kv_transfer_config"]
        kwargs["enable_prefix_caching"] = False

    try:
        with _without_expandable_segments():
            llm = LLM(**kwargs)
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    try:
        # Tokenize prompts.  Flatten every set into a single prompt list and
        # remember each set's slice so we can split hidden states back on return.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust,
                revision=config.model.revision,
            )
        except AttributeError as exc:
            if "'list' object has no attribute 'keys'" not in str(exc):
                raise
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust,
                revision=config.model.revision,
                extra_special_tokens={},
            )
        prompts: list[str] = []
        set_slices: dict[str, tuple[int, int]] = {}
        for set_name, messages in prompt_sets.items():
            start = len(prompts)
            for msg in messages:
                chat = []
                if msg.system:
                    chat.append({"role": "system", "content": msg.system})
                chat.append({"role": "user", "content": msg.user})
                try:
                    text = tokenizer.apply_chat_template(
                        chat,
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                        **chat_template_kwargs(config),
                    )
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        chat,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                prompts.append(text)
            set_slices[set_name] = (start, len(prompts))

        set_summary = ", ".join(
            f"{name}={end - start}" for name, (start, end) in set_slices.items()
        )
        print(
            f"* Extracting hidden states for {len(prompts)} prompts "
            f"({set_summary}; {num_layers} layers, TP={tp})..."
        )

        sampling_params = SamplingParams(max_tokens=1)
        if aux_capture:
            batch_residuals = _capture_aux_rows(
                llm, prompts, sampling_params, layer_ids, num_layers, token_offset
            )
            outputs = []
        else:
            outputs = llm.generate(prompts, sampling_params)
            batch_residuals = []

        # Collect hidden states from safetensors files.
        for out in outputs:
            hs_path = out.kv_transfer_params.get("hidden_states_path")
            if hs_path is None:
                raise RuntimeError(
                    f"No hidden_states_path in output for request {out.request_id}. "
                    f"kv_transfer_params={out.kv_transfer_params}"
                )

            with safe_open(hs_path, "pt") as f:
                hs = f.get_tensor("hidden_states")

            batch_residuals.append(layer_rows(hs, num_layers, token_offset))

        residuals = torch.stack(batch_residuals, dim=0).to(torch.float32)

        results: dict[str, Tensor] = {}
        for set_name, (start, end) in set_slices.items():
            results[set_name] = residuals[start:end].contiguous()
            print(f"  [green]Ok[/] — {set_name}: shape {list(results[set_name].shape)}")

    finally:
        # Guaranteed teardown: on the exception path the caller's handler
        # holds exc.__traceback__, which pins this frame and with it the
        # vLLM engine (the fall-through `del llm` never ran), so no amount
        # of gc in the handler could free the GPUs for the fallback
        # backend.  Deleting the frame's slot here lets flush_memory()
        # actually tear the engine down (issue #83 failure shape).
        del llm
        flush_memory()

        # Clean up temp files.
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results
