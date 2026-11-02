"""Apples-to-apples refusal eval of an external model with OUR gemini LLM judge.

Loads the model as the base (no steering), generates on config.target_eval_prompts
(the same 100 harmful prompts our run used), and judges with the SAME
RefusalDetector.evaluate_compliance path (gemini-3.1-flash-lite batch judge).
Usage: EVAL_MODEL=zaakirio/gemma-4-12b-it-uncensored AX_CONFIG=configs/gemma4_12b.toml python3 eval_external_gemini.py
"""

import os
from abliterix.settings import AbliterixConfig
from abliterix.core.engine import SteeringEngine
from abliterix.eval.detector import RefusalDetector
from abliterix.data import load_prompt_dataset

config = AbliterixConfig()
config.model.model_id = os.environ["EVAL_MODEL"]
# Pin a modest batch to skip autotune; plenty for a 12B on this card.
config.inference.batch_size = 16

print(f"Model     : {config.model.model_id}")
print(f"Judge     : {config.detection.llm_judge_model}")
print(
    f"Eval set  : {config.target_eval_prompts.dataset} [{config.target_eval_prompts.split}]"
)
print(
    f"Gen tokens: min {config.inference.min_gen_tokens} / max {config.inference.max_gen_tokens}"
)

engine = SteeringEngine(config)  # loads model; NO steering applied
detector = RefusalDetector(config)
target_msgs = load_prompt_dataset(config, config.target_eval_prompts)
print(f"Loaded {len(target_msgs)} harmful eval prompts\n")

refusals = detector.evaluate_compliance(engine, target_msgs)
print(
    f"\n>>> RESULT  {config.model.model_id}: {refusals}/{len(target_msgs)} refusals (gemini judge) <<<"
)
