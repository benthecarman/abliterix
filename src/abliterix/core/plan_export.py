# Abliterix — steering plans without loading the model
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Search and export a checkpoint that transformers cannot load.

Inkling 2's NVFP4 release is searched in vLLM (``backend = "vllm"``,
``use_in_place_editing``, ``model.plans_without_hf_model``): vLLM extracts
the hidden states and profiles the safety experts, and each trial's steering
plan is recorded on a weightless skeleton of the model built from its config
(:func:`config_skeleton`) and applied by the vLLM workers. The export records
the same plan the same way, so it needs no model either:

* The search saves what the plan is computed from that only a model can
  give, the benign/target hidden states and the safety-expert ranking, in
  ``<checkpoint_dir>/<model slug>.search-state.pt``
  (:func:`save_search_state`). A resumed search reuses them, so every trial
  of a study is scored on the same vectors.
* :func:`export_plan_without_model` rebuilds a trial's vectors from them,
  records its plan on the skeleton, saves it, and replays it onto the
  checkpoint with :func:`~.fp4_repack.abliterate_fp4_to_disk`.

The skeleton's conv taps (the conv-span edit's only weight values) are read
from the checkpoint's safetensors, mapped to the in-memory names with
:func:`~.fp4_repack.checkpoint_name_map`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..settings import AbliterixConfig
from ..util import print

SEARCH_STATE_VERSION = 1


# ---------------------------------------------------------------------------
# The model's files
# ---------------------------------------------------------------------------


def model_dir(config: AbliterixConfig, *, weights: bool = True) -> Path:
    """The checkpoint as a local directory: ``model_id`` itself, or its Hub
    snapshot (only the JSON files unless ``weights``)."""
    path = Path(config.model.model_id)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    patterns = None if weights else ["*.json", "*.py"]
    return Path(
        snapshot_download(
            config.model.model_id,
            revision=config.model.revision,
            allow_patterns=patterns,
        )
    )


def _loadable_config_dir(src: Path, dst: Path) -> Path:
    """``src``'s config as transformers should read it for an unquantized
    skeleton: without ``quantization_config`` (the in-memory names and shapes
    are those of the dequantized model), and with ``transformers_version``
    set, or transformers gives Inkling 2's release config an FP8
    ``quantization_config`` of its own. Remote code files come along."""
    import transformers

    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    for sub in cfg.values():
        if isinstance(sub, dict):
            sub.pop("quantization_config", None)
    cfg.setdefault("transformers_version", transformers.__version__)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(cfg))
    for f in src.glob("*.py"):
        shutil.copy(f, dst / f.name)
    return dst


# ---------------------------------------------------------------------------
# A weightless engine
# ---------------------------------------------------------------------------


def _skeleton_dtype(config: AbliterixConfig, hf_config: Any) -> torch.dtype:
    first = (config.model.dtype_fallback_order or ["auto"])[0]
    if first == "auto":
        text = getattr(hf_config, "text_config", hf_config)
        dtype = getattr(text, "dtype", None) or getattr(hf_config, "dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        return dtype or torch.bfloat16
    return getattr(torch, first)


def _apply_dtype_plan(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Give the skeleton the dtypes ``from_pretrained`` loads it in:
    ``_from_config`` ignores the model's ``_keep_in_fp32_modules(_strict)``
    (Inkling 2 keeps its router and short convolutions in FP32)."""
    from torch.nn import Parameter

    get_plan = getattr(model, "_get_dtype_plan", None)
    plan = get_plan(dtype) if get_plan is not None else {}
    if not plan:
        return
    from transformers.core_model_loading import build_glob_alternation

    alt, group_of = build_glob_alternation(list(plan))[:2]
    for module_name, module in model.named_modules():
        for store in (module._parameters, module._buffers):
            for name, t in list(store.items()):
                if t is None:
                    continue
                full = f"{module_name}.{name}" if module_name else name
                match = alt.search(full)
                if match is None or t.dtype == plan[group_of[match.lastgroup]]:
                    continue
                empty = torch.empty(t.shape, dtype=plan[group_of[match.lastgroup]], device="meta")
                store[name] = Parameter(empty, requires_grad=False) if store is module._parameters else empty


def _conv_taps_from_checkpoint(engine, src: Path) -> dict[tuple[int, str], Tensor]:
    """Every ``(layer, component)``'s output conv taps, read from ``src``.

    The engine's :meth:`output_conv_weight` names the conv weight on the
    skeleton (shape only); the checkpoint key behind it comes from
    :func:`checkpoint_name_map`. The weight is cast to the skeleton's dtype,
    the one transformers loads it in, so the taps are the loaded model's."""
    from safetensors import safe_open

    from .fp4_repack import _iter_shards, checkpoint_name_map

    name_of = {id(p): n for n, p in engine.model.named_parameters()}
    wanted: dict[str, tuple[int, str, Tensor]] = {}
    for i in range(len(engine.transformer_layers)):
        for component in ("attn.o_proj", "mlp.down_proj"):
            weight = engine.output_conv_weight(i, component)
            if weight is not None:
                wanted[name_of[id(weight)]] = (i, component, weight)
    if not wanted:
        return {}
    shards = _iter_shards(src)
    shard_of = {k: f for f, keys in shards.items() for k in keys}
    names = checkpoint_name_map(src, list(shard_of), getattr(engine, "_skeleton_class", None))
    taps: dict[tuple[int, str], Tensor] = {}
    for name, (i, component, weight) in wanted.items():
        if name in shard_of:
            key = name
        else:
            keys = [k for k, expert in names.get(name, []) if expert is None]
            if len(keys) != 1:
                raise ValueError(f"{src}: no checkpoint tensor for '{name}' (conv taps)")
            key = keys[0]
        with safe_open(src / shard_of[key], framework="pt") as f:
            t = f.get_tensor(key)
        if t.numel() != weight.numel():
            raise ValueError(f"{src}: '{key}' has shape {list(t.shape)}, '{name}' {list(weight.shape)}")
        taps[(i, component)] = t.reshape(weight.shape).to(weight.dtype)[:, 0, :].contiguous()
    return taps


def config_skeleton(config: AbliterixConfig, src: Path | None = None):
    """A :class:`SteeringEngine` whose model is built from the checkpoint's
    config on the meta device: the structure a plan is recorded on, without
    loading (or being able to load) the weights. The counterpart of
    :meth:`SteeringEngine.plan_skeleton` for a model that was never loaded.

    The model class is the one the engine would load (``resolve_model_class``)
    in the dtype it would load it in (the first of ``dtype_fallback_order``),
    unquantized. Conv taps come from the checkpoint (``src``, by default the
    model's own files)."""
    from transformers import AutoConfig

    from .engine import SteeringEngine, resolve_model_class

    src = src or model_dir(config)
    trust = config.model.trust_remote_code or False
    with tempfile.TemporaryDirectory(prefix="abliterix_skeleton_") as tmp:
        hf_config = AutoConfig.from_pretrained(
            str(_loadable_config_dir(src, Path(tmp))), trust_remote_code=trust
        )
        cls = resolve_model_class(str(src), text_only=config.model.text_only)
        dtype = _skeleton_dtype(config, hf_config)
        with torch.device("meta"):
            model = cls.from_config(hf_config, dtype=dtype, trust_remote_code=trust)
    model.requires_grad_(False)
    _apply_dtype_plan(model, dtype)

    engine = SteeringEngine.without_model(config, tokenizer=False)
    engine.model = model
    engine._skeleton_class = type(model).__name__
    text = getattr(hf_config, "text_config", hf_config)
    engine._fused_down_proj_transposed = getattr(text, "model_type", "") in {"gpt_oss"}
    engine._lora_b_weights = []
    engine.peft_config = None
    engine._init_expert_routing()
    taps = _conv_taps_from_checkpoint(engine, src)
    engine.output_conv_taps = lambda layer_index, component: taps.get((layer_index, component))
    return engine


# ---------------------------------------------------------------------------
# What the search saves for the export
# ---------------------------------------------------------------------------


def search_state_path(config: AbliterixConfig) -> Path:
    from ..util import slugify_model_name

    return Path(config.optimization.checkpoint_dir) / (
        slugify_model_name(config.model.model_id) + ".search-state.pt"
    )


def _fingerprint(config: AbliterixConfig) -> dict:
    """What the saved states depend on: the model and the direction prompts.
    The export compares it with the trial's own direction inputs."""
    return {
        "model_id": config.model.model_id,
        "revision": config.model.revision,
        "system_prompt": config.system_prompt,
        "benign_prompts": config.benign_prompts.model_dump(mode="json"),
        "target_prompts": config.target_prompts.model_dump(mode="json"),
        "chat_template_kwargs": dict(config.model.chat_template_kwargs or {}),
    }


def save_search_state(
    config: AbliterixConfig,
    benign_states: Tensor,
    target_states: Tensor,
    safety_experts: dict[int, list[tuple[int, float]]] | None,
    *,
    safety_experts_method: str | None = None,
) -> Path:
    """Save the hidden states and safety-expert ranking a search computed
    its plans from (written to a temporary file and renamed into place)."""
    path = search_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SEARCH_STATE_VERSION,
        "fingerprint": _fingerprint(config),
        "benign_states": benign_states.detach().cpu(),
        "target_states": target_states.detach().cpu(),
        "safety_experts": safety_experts,
        "safety_experts_method": safety_experts_method,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)
    return path


def load_search_state(config: AbliterixConfig, *, required: bool = False) -> dict | None:
    """The saved search state if it was computed for this config's model and
    direction prompts; otherwise None, or with ``required`` an error."""
    path = search_state_path(config)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"no saved search state at {path}")
        return None
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if state.get("version") != SEARCH_STATE_VERSION or state.get("fingerprint") != _fingerprint(config):
        if required:
            raise ValueError(
                f"{path} was saved for another model or other direction prompts "
                f"({state.get('fingerprint')} != {_fingerprint(config)})"
            )
        return None
    return state


def adopt_search_revision(config: AbliterixConfig) -> None:
    """Give an unpinned config the model revision its search ran on.

    The search pins ``model.revision`` to the Hub commit it resolved
    (``pin_remote_sources``) and saves that in the search state; a config
    loaded again from the same TOML has none. Taking the saved one keeps
    the export on the search's commit, where resolving again could land on
    a newer one. Changes nothing unless the saved state differs from the
    config only in that revision."""
    if config.model.revision is not None:
        return
    path = search_state_path(config)
    if not path.exists():
        return
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    saved = state.get("fingerprint") or {}
    if state.get("version") == SEARCH_STATE_VERSION and {**saved, "revision": None} == _fingerprint(config):
        config.model.revision = saved.get("revision")


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------


@dataclass
class PlanExport:
    edits: list
    meta: dict
    plan_file: Path | None
    stats: Any | None  # RepackStats of the bake, None without one


def export_plan_without_model(
    config: AbliterixConfig,
    trial,
    out: str | Path | None,
    *,
    plan_file: str | Path | None = None,
    src: str | Path | None = None,
    use_cuda: bool = True,
) -> PlanExport:
    """Record a trial's steering plan without loading the model, save it,
    and bake it onto the checkpoint.

    Contract:

    * ``config``: the search's config, as loaded from its TOML (``model_id``
      is the checkpoint the search ran on; ``optimization.checkpoint_dir``
      holds the study and the saved search state). The trial's artifact is
      applied to it here (it is modified in place).
    * ``trial``: the Optuna ``FrozenTrial`` of the study.
    * Needs the search state the search saved (:func:`save_search_state`;
      a search with ``model.plans_without_hf_model`` saves it). Raises if it
      is missing or was computed for other prompts.
    * Records the plan on :func:`config_skeleton` with the trial's vectors
      (:func:`~abliterix.scriptlib.compute_trial_vectors` on the saved
      states), the saved safety experts and the trial's routing: the same
      inputs, code and skeleton the search recorded the trial's plan with.
      There is no model to check the plan against
      (``verify_plan_against_engine``).
    * ``plan_file``: where to save the plan (``save_plan``, with
      ``plan_meta``), if given.
    * ``out``: where to bake (``abliterate_fp4_to_disk`` on ``src``, by
      default the model's own files, after ``map_plan_to_checkpoint``;
      ``use_cuda`` as there); None
      records (and saves) the plan only. The bake writes the weights and the
      checkpoint's other files; the caller adds its own records
      (``trial.json``) and marks the export complete.
    """
    from ..scriptlib import apply_trial_artifact, compute_trial_vectors, extract_trial_artifact
    from .fp4_repack import (
        abliterate_fp4_to_disk,
        map_plan_to_checkpoint,
        plan_meta,
        record_steering_plan_from_trial,
        save_plan,
    )

    artifact = extract_trial_artifact(trial)
    apply_trial_artifact(config, artifact, require_hf_backend=False)
    adopt_search_revision(config)
    state = load_search_state(config, required=True)
    benign, target = state["benign_states"], state["target_states"]
    vectors = compute_trial_vectors(artifact, benign, target, config)

    src_path = Path(src) if src is not None else model_dir(config)
    engine = config_skeleton(config, src_path)
    edits = record_steering_plan_from_trial(
        engine,
        vectors,
        artifact.vector_index,
        artifact.profiles,
        config,
        benign_states=benign,
        target_states=target,
        safety_experts=state["safety_experts"],
        routing_config=artifact.routing,
    )
    meta = plan_meta(engine)
    print(f"* Recorded the plan of trial {trial.number} without the model: {len(edits)} edits")
    saved = None
    if plan_file is not None:
        saved = Path(plan_file)
        save_plan(edits, saved, meta=meta)
    stats = None
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        stats = abliterate_fp4_to_disk(
            src_path,
            out,
            map_plan_to_checkpoint(edits, src_path, model_class=meta["model_class"]),
            use_cuda=use_cuda,
        )
    return PlanExport(edits=edits, meta=meta, plan_file=saved, stats=stats)
