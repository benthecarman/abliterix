# Abliterix — small vLLM compatibility shims
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compatibility helpers for vLLM version gating, env-var setup, and model
families that vLLM supports before HF does."""

from __future__ import annotations

import os
import warnings
from typing import Any

# vLLM versions abliterix is tested against. Lower bound is v0.18 because
# `VLLM_ALLOW_INSECURE_SERIALIZATION` (PR #35928) is required for the
# `collective_rpc` path that VLLMMoEEditor depends on. Upper bound is the
# next minor we have not yet exercised; bump after a passing CI run on the
# new minor.
_MIN_VLLM = (0, 18, 0)
_MAX_VLLM_EXCLUSIVE = (0, 21, 0)

# Env vars VLLMGenerator sets before constructing LLM(). Each is a no-op when
# the user has already exported a value; we never override.
_REQUIRED_ENV_BASE: tuple[tuple[str, str], ...] = (
    # Skips a noisy version assertion in flashinfer when transformers gets
    # bumped underneath it; vLLM's own version pin is already authoritative.
    ("FLASHINFER_DISABLE_VERSION_CHECK", "1"),
)
_REQUIRED_ENV_RPC: tuple[tuple[str, str], ...] = (
    # Required for collective_rpc to pickle Python callables sent to TP
    # workers. Introduced in vLLM v0.18 (PR #35928). Without this, MoE
    # router suppression and in-place expert editing silently no-op.
    ("VLLM_ALLOW_INSECURE_SERIALIZATION", "1"),
)


def _parse_version(spec: str) -> tuple[int, ...]:
    """Parse a vLLM version string into a comparable tuple. Handles dev/rc
    suffixes by taking the leading numeric prefix only, and tolerates an
    optional ``v`` prefix (``v0.20.1`` → ``(0, 20, 1)``)."""
    head = spec.split("+", 1)[0].lstrip("v")
    parts: list[int] = []
    for chunk in head.split("."):
        digits = ""
        for c in chunk:
            if c.isdigit():
                digits += c
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def check_vllm_version(installed: str | None = None) -> tuple[int, ...]:
    """Validate that the installed vLLM version is in the supported window.

    Parameters
    ----------
    installed
        Version string. When None, queried from
        ``importlib.metadata.version("vllm")``.

    Returns
    -------
    tuple[int, ...]
        The parsed (major, minor, patch) version tuple.

    Raises
    ------
    ImportError
        If vLLM is not installed, or the installed version is below the
        supported floor (currently 0.18.0). The PRD bumped abliterix's
        floor to v0.18 because earlier versions lack the
        `VLLM_ALLOW_INSECURE_SERIALIZATION` flag that
        :class:`VLLMMoEEditor` requires.
    """
    if installed is None:
        try:
            from importlib.metadata import PackageNotFoundError, version

            installed = version("vllm")
        except PackageNotFoundError as exc:
            raise ImportError(
                'vLLM is not installed. Install with `pip install "vllm>=0.18,<0.21"`.'
            ) from exc

    parsed = _parse_version(installed)
    if parsed[:2] == (0, 1) and ".dev" in installed:
        # A source build without a release tag (vLLM's model-launch images report
        # "0.1.devN+g<commit>"): its version says nothing about its age.
        warnings.warn(
            f"vLLM {installed} is an untagged source build; abliterix cannot "
            "tell its version and assumes it is recent. Run smoke tests first.",
            stacklevel=2,
        )
        return parsed
    if parsed < _MIN_VLLM:
        raise ImportError(
            f"abliterix requires vllm>={'.'.join(str(p) for p in _MIN_VLLM)}, "
            f'got {installed}. Upgrade with `pip install -U "vllm>=0.18,<0.21"`.'
        )
    if parsed >= _MAX_VLLM_EXCLUSIVE:
        warnings.warn(
            f"vLLM {installed} is newer than the highest version abliterix has been "
            f"validated against ({'.'.join(str(p) for p in _MAX_VLLM_EXCLUSIVE)}). "
            "Run smoke tests before relying on this combination in production.",
            stacklevel=2,
        )
    return parsed


def ensure_vllm_env(
    *,
    needs_collective_rpc: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Set the small set of env vars vLLM needs before LLM() construction.

    Idempotent and non-destructive: never overwrites a value the user has
    already exported. Returns a dict of the names+values that this call
    actually wrote, suitable for logging.

    Parameters
    ----------
    needs_collective_rpc
        When True, also sets `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. This is
        required whenever VLLMMoEEditor / VLLMExpertEditor / use_in_place_editing
        is active so `collective_rpc` can pickle Python callables.
    env
        Mutable mapping treated as the environment. Defaults to `os.environ`.
    """
    target = env if env is not None else os.environ
    written: dict[str, str] = {}
    pairs = list(_REQUIRED_ENV_BASE)
    if needs_collective_rpc:
        pairs.extend(_REQUIRED_ENV_RPC)
    for name, value in pairs:
        if name not in target:
            target[name] = value
            written[name] = value
    return written


def _transformers_knows(model_type: str) -> bool:
    """Whether this Transformers has its own config class for ``model_type``."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except ImportError:
        return False
    return model_type in CONFIG_MAPPING_NAMES


def install_gemma4_transformers_compat() -> None:
    """Register minimal Gemma4 configs and patch a tokenizer_config quirk.

    vLLM 0.19 includes Gemma4 model executors, but the paired Transformers
    release may not yet know ``model_type="gemma4"``. Gemma4 tokenizer configs
    can also carry ``extra_special_tokens`` as a list, while Transformers 4.x
    expects a mapping. Both fixes are process-local and only affect this run.
    """

    from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

    class Gemma4TextConfig(PretrainedConfig):
        model_type = "gemma4_text"

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)

    class Gemma4Config(PretrainedConfig):
        model_type = "gemma4"

        def __init__(self, text_config: Any | None = None, **kwargs: Any):
            super().__init__(**kwargs)
            if isinstance(text_config, dict):
                self.text_config = Gemma4TextConfig(**text_config)
            elif text_config is None:
                self.text_config = Gemma4TextConfig()
            else:
                self.text_config = text_config

    # Only where Transformers lacks Gemma4: registering with exist_ok replaces
    # its real config classes, and vLLM then rejects the config it is handed
    # ("Invalid type of HuggingFace config").
    if not _transformers_knows("gemma4"):
        try:
            AutoConfig.register("gemma4", Gemma4Config, exist_ok=True)
            AutoConfig.register("gemma4_text", Gemma4TextConfig, exist_ok=True)
        except ValueError:
            pass

        try:
            from vllm.transformers_utils import config as vllm_config

            vllm_config._CONFIG_REGISTRY.setdefault("gemma4", Gemma4Config)
            vllm_config._CONFIG_REGISTRY.setdefault("gemma4_text", Gemma4TextConfig)
        except Exception:
            pass

    if getattr(AutoTokenizer.from_pretrained, "_abliterix_gemma4_patch", False):
        return

    original_from_pretrained = AutoTokenizer.from_pretrained

    def from_pretrained_with_gemma4_extra_tokens(*args: Any, **kwargs: Any):
        try:
            return original_from_pretrained(*args, **kwargs)
        except AttributeError as exc:
            if "'list' object has no attribute 'keys'" not in str(exc):
                raise
            kwargs = dict(kwargs)
            kwargs["extra_special_tokens"] = {}
            return original_from_pretrained(*args, **kwargs)

    from_pretrained_with_gemma4_extra_tokens._abliterix_gemma4_patch = True  # type: ignore[attr-defined]
    AutoTokenizer.from_pretrained = from_pretrained_with_gemma4_extra_tokens


def transformers_config(path, **kwargs):
    """``AutoConfig.from_pretrained``, but always transformers' own config class.

    Loading some models in vLLM (Inkling 2) registers vLLM's config class over
    transformers' for the same model type (``AutoConfig.register(...,
    exist_ok=True)``). After an in-process vLLM engine, ``AutoConfig`` then
    returns vLLM's class, which transformers' AutoModel mappings and model
    classes do not accept. Configs from remote code are returned unchanged."""
    import transformers
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    config = AutoConfig.from_pretrained(path, **kwargs)
    if not type(config).__module__.startswith("vllm."):
        return config
    name = CONFIG_MAPPING_NAMES.get(getattr(config, "model_type", None))
    native = getattr(transformers, name, None) if name else None
    if native is None:
        return config
    kwargs.pop("trust_remote_code", None)
    return native.from_pretrained(path, **kwargs)
