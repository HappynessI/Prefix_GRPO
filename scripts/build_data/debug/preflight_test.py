#!/usr/bin/env python3
"""Preflight smoke test for TextCraft prefix-GRPO."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoTokenizer


def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_debug_samples(data_path: str, max_samples: int = 4):
    df = pd.read_parquet(data_path)
    print(f"总样本数: {len(df)}")
    print(f"列名: {list(df.columns)}")

    samples = df.head(max_samples).to_dict("records")
    for idx, sample in enumerate(samples):
        extra_info = sample.get("extra_info", {})
        interaction_kwargs = extra_info.get("interaction_kwargs", {}) if isinstance(extra_info, dict) else {}
        prefix_actions = interaction_kwargs.get("prefix_actions", [])
        has_prefix = prefix_actions is not None and hasattr(prefix_actions, "__len__") and len(prefix_actions) > 0
        print(f"sample={idx}, has_prefix_actions={has_prefix}, prefix_actions_len={len(prefix_actions) if has_prefix else 0}")
        if not has_prefix:
            raise ValueError(f"sample {idx} 缺少 prefix_actions")
    return samples


def test_model_inference(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokens = tokenizer.encode("Hello, world!")
    print(f"Tokenizer 加载成功，测试 tokens={len(tokens)}")


async def test_interaction_replay(textcraft_server: str, samples: list):
    from verl.interactions.textcraft_interaction import TextCraftInteraction

    config = OmegaConf.create({"env_server_base": textcraft_server})
    interaction = TextCraftInteraction(config)

    for idx, sample in enumerate(samples):
        extra_info = sample.get("extra_info", {})
        interaction_kwargs = extra_info.get("interaction_kwargs", {}) if isinstance(extra_info, dict) else {}
        request_id = f"preflight_{idx}"
        await interaction.start_interaction(request_id, prompt=sample.get("prompt"), **interaction_kwargs)
        session = interaction.instance_sessions.get(request_id)
        if not session:
            raise RuntimeError(f"sample {idx}: 未创建 interaction session")
        if session["step_count"] == 0:
            raise RuntimeError(f"sample {idx}: replay 后 step_count 仍为 0")

        prompt = sample.get("prompt")
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        should_terminate, obs, reward, metrics = await interaction.generate_response(
            request_id,
            messages,
            **interaction_kwargs,
        )
        print(
            f"sample={idx}, env_id={session['env_id']}, step_count={session['step_count']}, "
            f"obs_len={len(str(obs))}, reward={reward}, should_terminate={should_terminate}"
        )
        if not obs:
            raise RuntimeError(f"sample {idx}: generate_response 未返回 observation")
        if hasattr(interaction, "end_interaction"):
            await interaction.end_interaction(request_id)


async def main():
    parser = argparse.ArgumentParser(description="Preflight smoke test for TextCraft prefix-GRPO")
    parser.add_argument("--project_root", type=str, default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--textcraft_server", type=str, default="http://127.0.0.1:36001")
    parser.add_argument("--max_samples", type=int, default=4)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root / "verl"))
    os.environ["VERL_DEBUG_MODE"] = "1"
    setup_logging()

    print(f"project_root: {project_root}")
    print(f"data_path: {args.data_path}")
    print(f"model_path: {args.model_path}")
    print(f"textcraft_server: {args.textcraft_server}")

    samples = load_debug_samples(args.data_path, args.max_samples)
    test_model_inference(args.model_path)
    await test_interaction_replay(args.textcraft_server, samples)
    print("Preflight smoke test PASSED")


if __name__ == "__main__":
    asyncio.run(main())
