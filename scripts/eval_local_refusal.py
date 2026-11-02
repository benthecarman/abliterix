#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evaluate local refusal rate on Abliterix prompts.

This is intended for Mac/local workstation runs where the model may be served
by Transformers on MPS, MLX/llama.cpp/OpenAI-compatible servers, or another
local endpoint.  It keeps the evaluation dataset and refusal judge consistent
with the rest of Abliterix.

Examples:
    # Full HF checkpoint on Apple Silicon (128 GB unified memory recommended)
    python scripts/eval_local_refusal.py \
        --backend transformers \
        --model wangzhang/Qwen3.6-27B-abliterated-v2 \
        --config configs/qwen3.6_27b_v2.toml

    # Local OpenAI-compatible server (llama.cpp, mlx-lm, LM Studio, Ollama)
    python scripts/eval_local_refusal.py \
        --backend openai \
        --model qwen3.6-27b-abliterated-v2 \
        --base-url http://127.0.0.1:8080/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_env_file() -> None:
    """Load repo .env so HF/OpenRouter tokens work from local app shells."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "openai"],
        default="transformers",
        help=(
            "Generation backend. Use openai for llama.cpp/MLX/LM Studio/Ollama servers."
        ),
    )
    parser.add_argument(
        "--model",
        default="wangzhang/Qwen3.6-27B-abliterated-v2",
        help=(
            "HF model ID/local path, or model name exposed by the "
            "OpenAI-compatible server."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/qwen3.6_27b_v2.toml",
        help="Abliterix config to borrow detector/judge settings from.",
    )
    parser.add_argument(
        "--dataset",
        default="datasets/harmful_1000",
        help="Dataset path or HF dataset ID containing harmful prompts.",
    )
    parser.add_argument(
        "--split",
        default="train[800:1000]",
        help="Dataset split to evaluate. Default is the last 200 harmful_1000 prompts.",
    )
    parser.add_argument("--column", default="prompt", help="Prompt text column.")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override system prompt. Defaults to the config/global system prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Generation length. Defaults to config inference.max_gen_tokens.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=None,
        help="Optional minimum generation length. Defaults to config value.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. 0.0 gives deterministic decoding.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Transformers batch size. On MPS, start with 1 or 2.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cpu", "auto"],
        help="Transformers device. 'auto' uses mps when available, else cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Transformers load dtype. For M4 Max, float16 is the safest default.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom tokenizer/model code from Hub repos.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible base URL for --backend openai.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Optional API key env var for OpenAI-compatible generation server.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="OpenAI-compatible generation request concurrency.",
    )
    parser.add_argument(
        "--judge",
        dest="judge",
        action="store_true",
        default=True,
        help="Use the Abliterix LLM judge from config (default).",
    )
    parser.add_argument(
        "--no-judge",
        dest="judge",
        action="store_false",
        help="Use keyword/degeneracy detector only. Faster but less faithful.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override config.detection.llm_judge_model.",
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=None,
        help="Override config.detection.llm_judge_batch_size.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=None,
        help="Override config.detection.llm_judge_concurrency.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Defaults to artifacts/<model>_<split>_refusal_eval.json."
        ),
    )
    parser.add_argument(
        "--print-responses",
        action="store_true",
        help="Print every prompt/response/verdict as it is produced.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def load_config(args: argparse.Namespace):
    os.environ["AX_CONFIG"] = args.config
    sys.argv = [sys.argv[0]]

    from abliterix.settings import AbliterixConfig

    config = AbliterixConfig()
    config.detection.llm_judge = args.judge
    if args.judge_model:
        config.detection.llm_judge_model = args.judge_model
    if args.judge_batch_size is not None:
        config.detection.llm_judge_batch_size = args.judge_batch_size
    if args.judge_concurrency is not None:
        config.detection.llm_judge_concurrency = args.judge_concurrency
    if args.print_responses:
        config.display.print_responses = True
    return config


def load_messages(args: argparse.Namespace, config) -> list[Any]:
    from abliterix.data import load_prompt_dataset
    from abliterix.types import PromptSource

    source = PromptSource(
        dataset=args.dataset,
        split=args.split,
        column=args.column,
        system_prompt=args.system_prompt,
    )
    messages = load_prompt_dataset(config, source)
    if args.limit is not None:
        messages = messages[: args.limit]
    return messages


def render_chat(tokenizer, msg) -> str:
    messages = [
        {"role": "system", "content": msg.system},
        {"role": "user", "content": msg.user},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return f"{msg.system}\n\nUser: {msg.user}\nAssistant:"


def generate_transformers(
    args: argparse.Namespace, config, messages: list[Any]
) -> list[str]:
    if args.device == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = args.device

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading {args.model} with Transformers on {device_name} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        load_kwargs["dtype"] = dtype

    def load_with_optional_device_map(device_map):
        try:
            return AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map=device_map,
                **load_kwargs,
            )
        except TypeError:
            compat_kwargs = dict(load_kwargs)
            if "dtype" in compat_kwargs:
                compat_kwargs["torch_dtype"] = compat_kwargs.pop("dtype")
            return AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map=device_map,
                **compat_kwargs,
            )

    try:
        model = load_with_optional_device_map(
            {"": device_name} if device_name != "cpu" else None
        )
    except Exception as exc:
        if device_name == "cpu":
            raise
        print(
            f"Direct load to {device_name} failed ({exc}). "
            f"Retrying CPU load followed by model.to('{device_name}')."
        )
        model = load_with_optional_device_map(None)
        model.to(device_name)

    model.eval()
    prompts = [render_chat(tokenizer, msg) for msg in messages]
    max_new_tokens = args.max_new_tokens or config.inference.max_gen_tokens
    min_new_tokens = (
        args.min_new_tokens
        if args.min_new_tokens is not None
        else config.inference.min_gen_tokens
    )

    responses: list[str] = []
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start : start + args.batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device_name)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": args.temperature > 0,
            "temperature": args.temperature if args.temperature > 0 else None,
            "top_p": args.top_p,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.inference_mode():
            out = model.generate(**enc, **gen_kwargs)
        gen = out[:, enc["input_ids"].shape[1] :]
        responses.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        print(f"Generated {len(responses)}/{len(prompts)} responses")

    return responses


def post_chat_completion(
    *,
    base_url: str,
    api_key_env: str,
    model: str,
    system: str,
    user: str,
    max_new_tokens: int,
    min_new_tokens: int | None,
    temperature: float,
    top_p: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if min_new_tokens is not None:
        body["min_tokens"] = min_new_tokens

    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(api_key_env, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = json.dumps(body).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Generation request failed after 3 attempts: {last_exc}")


def generate_openai(args: argparse.Namespace, config, messages: list[Any]) -> list[str]:
    max_new_tokens = args.max_new_tokens or config.inference.max_gen_tokens
    min_new_tokens = (
        args.min_new_tokens
        if args.min_new_tokens is not None
        else config.inference.min_gen_tokens
    )

    responses: list[str | None] = [None] * len(messages)
    print(
        f"Generating via {args.base_url.rstrip('/')}/chat/completions "
        f"(concurrency={args.concurrency})..."
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                post_chat_completion,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                model=args.model,
                system=msg.system,
                user=msg.user,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            ): i
            for i, msg in enumerate(messages)
        }
        completed = 0
        for fut in as_completed(futures):
            i = futures[fut]
            responses[i] = fut.result()
            completed += 1
            print(f"Generated {completed}/{len(messages)} responses")

    return [r or "" for r in responses]


def classify(config, messages: list[Any], responses: list[str]) -> list[bool]:
    from abliterix.eval.detector import RefusalDetector

    detector = RefusalDetector(config)
    if config.detection.llm_judge:
        pairs = [(msg.user, resp) for msg, resp in zip(messages, responses)]
        return detector._batch_judge_classify(pairs)
    return [detector.detect_refusal(resp) for resp in responses]


def main() -> None:
    load_env_file()
    args = parse_args()
    config = load_config(args)
    messages = load_messages(args, config)
    if not messages:
        raise SystemExit("No prompts loaded. Check --dataset/--split/--column.")

    print(f"Loaded {len(messages)} prompts from {args.dataset} {args.split}")
    if args.backend == "transformers":
        responses = generate_transformers(args, config, messages)
    else:
        responses = generate_openai(args, config, messages)

    refusals = classify(config, messages, responses)
    refusal_count = sum(1 for x in refusals if x)
    rate = 100.0 * refusal_count / len(messages)

    rows = []
    for i, (msg, resp, is_refusal) in enumerate(zip(messages, responses, refusals), 1):
        label = "R" if is_refusal else "C"
        rows.append(
            {
                "index": i,
                "system": msg.system,
                "prompt": msg.user,
                "response": resp,
                "label": label,
                "is_refusal": is_refusal,
            }
        )
        if args.print_responses:
            print("\n" + "=" * 80)
            print(f"[{i}/{len(messages)}] {label}")
            print(f"Q: {msg.user}")
            print(f"A: {resp}")

    output = args.output
    if output is None:
        model_name = safe_name(args.model.split("/")[-1])
        split_name = safe_name(args.split)
        output = f"artifacts/{model_name}_{split_name}_refusal_eval.json"
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "backend": args.backend,
                "dataset": args.dataset,
                "split": args.split,
                "config": args.config,
                "judge_enabled": config.detection.llm_judge,
                "judge_model": (
                    config.detection.llm_judge_model
                    if config.detection.llm_judge
                    else None
                ),
                "total": len(messages),
                "refusals": refusal_count,
                "refusal_rate": refusal_count / len(messages),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("RESULTS")
    print(f"Model: {args.model}")
    print(f"Dataset split: {args.dataset} {args.split}")
    print(f"Refusals: {refusal_count}/{len(messages)}")
    print(f"Refusal rate: {rate:.2f}%")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
