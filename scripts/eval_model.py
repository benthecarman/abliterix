# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluate refusal rate for a base or abliterated model.

Example:
    python scripts/eval_model.py \
        --model wangzhang/Qwen3.5-35B-A3B-abliterated \
        --config configs/qwen3.5_35b.toml \
        --batch-size 8
"""

import argparse
import sys

from abliterix.scriptlib import setup_io

setup_io()

parser = argparse.ArgumentParser(description="Evaluate refusal rate for a model")
parser.add_argument("--model", required=True, help="Model ID or local path to evaluate")
parser.add_argument("--config", default=None, help="Abliterix config path")
parser.add_argument(
    "--batch-size", type=int, default=0, help="Override inference batch size"
)
parser.add_argument(
    "--judge",
    action="store_true",
    help="Enable OpenRouter LLM judge instead of keyword-only detection",
)
parser.add_argument(
    "--trust-remote-code",
    action="store_true",
    help="Allow custom tokenizer/model code from Hub repos",
)
parser.add_argument(
    "--eval-set",
    choices=["target", "benign", "both"],
    default="target",
    help=(
        "Which held-out set to judge: target refusal prompts, benign "
        "over-refusal prompts, or both"
    ),
)
args = parser.parse_args()

sys.argv = ["eval", "--model.model-id", args.model]
if args.config:
    sys.argv.extend(["--config", args.config])
if args.batch_size > 0:
    sys.argv.extend(["--inference.batch-size", str(args.batch_size)])
if args.judge:
    sys.argv.append("--detection.llm-judge")
else:
    sys.argv.append("--no-detection.llm-judge")
if args.trust_remote_code:
    sys.argv.extend(["--model.trust-remote-code", "true"])

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402

config = AbliterixConfig()

print(f"Loading {args.model}...")
engine = SteeringEngine(config)
detector = RefusalDetector(config)

print("Initializing scorer...")
scorer = TrialScorer(
    config,
    engine,
    detector,
    defer_baseline=args.eval_set == "benign",
)

objectives = None
kl_divergence = None
length_deviation = None
target_refusals = None
benign_refusals = None

if args.eval_set in {"target", "both"}:
    objectives, kl_divergence, target_refusals, length_deviation = scorer.score_trial(
        engine
    )

if args.eval_set in {"benign", "both"}:
    print("Counting benign over-refusals...")
    benign_refusals = detector.evaluate_compliance(engine, scorer.benign_msgs)

print()
print("RESULTS")
print(f"Model: {args.model}")
if target_refusals is not None:
    print(f"Target eval prompts: {len(scorer.target_msgs)}")
    print(f"Target refusals: {target_refusals}/{len(scorer.target_msgs)}")
    print(
        f"Target refusal rate: {100.0 * target_refusals / len(scorer.target_msgs):.2f}%"
    )
if benign_refusals is not None:
    print(f"Benign eval prompts: {len(scorer.benign_msgs)}")
    print(f"Benign over-refusals: {benign_refusals}/{len(scorer.benign_msgs)}")
    print(
        f"Benign over-refusal rate: "
        f"{100.0 * benign_refusals / len(scorer.benign_msgs):.2f}%"
    )
if kl_divergence is not None:
    print(f"KL divergence: {kl_divergence:.4f}")
if length_deviation is not None:
    print(f"Length deviation: {length_deviation:.2f}")
if objectives is not None:
    print(f"Objectives: {objectives}")
