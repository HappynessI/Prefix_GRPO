#!/usr/bin/env python3
"""Compute teacher-forced per-token entropy for full TextCraft teacher trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from textcraft_entropy_utils import (
    compute_message_spans,
    compute_token_entropy,
    load_jsonl,
    make_sample_uid,
    parse_task_id,
    select_role_spans,
    tokenize_conversations,
)


DEFAULT_INPUT_PATH = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage0_teacher/teacher_normalized.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage1_entropy/"
    "textcraft_teacher_entropy_step200.parquet"
)
DEFAULT_MANIFEST_PATH = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/manifests/stage1_entropy_manifest.json"
)
DEFAULT_MODEL_PATH = "/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def build_record(
    model,
    tokenizer,
    sample: Dict[str, Any],
    device: torch.device,
    enable_thinking: bool,
) -> Dict[str, Any]:
    item_id = sample["item_id"]
    sample_idx = int(sample["sample_idx"])
    sample_uid = sample.get("sample_uid") or make_sample_uid(item_id, sample_idx)
    task_id = int(sample.get("task_id", parse_task_id(item_id)))
    conversations = sample["conversations"]

    input_ids = tokenize_conversations(tokenizer, conversations, enable_thinking=enable_thinking).to(device)
    token_length = int(input_ids.shape[0])
    attention_mask = torch.ones_like(input_ids)

    message_spans = compute_message_spans(tokenizer, conversations, enable_thinking=enable_thinking)
    assistant_turn_spans = select_role_spans(message_spans, "assistant")
    user_message_spans = select_role_spans(message_spans, "user")

    sequence_entropies = compute_token_entropy(model, input_ids, attention_mask)
    entropy_length = int(sequence_entropies.shape[0])
    if entropy_length != max(0, token_length - 1):
        raise ValueError(
            f"Entropy length mismatch for {sample_uid}: {entropy_length} vs token_length-1={token_length - 1}"
        )

    for span in message_spans:
        if not (0 <= span["token_start"] < span["token_end"] <= token_length):
            raise ValueError(f"Token span out of range for {sample_uid}: {span}")
        if not (0 <= span["entropy_start"] <= span["entropy_end"] <= entropy_length):
            raise ValueError(f"Entropy span out of range for {sample_uid}: {span}")

    return {
        "sample_uid": sample_uid,
        "item_id": item_id,
        "sample_idx": sample_idx,
        "task_id": task_id,
        "goal": sample.get("goal"),
        "success": int(sample.get("success", 0)),
        "reward": sample.get("reward", 0),
        "num_messages": len(conversations),
        "num_assistant_messages": len(assistant_turn_spans),
        "num_user_messages": len(user_message_spans),
        "token_length": token_length,
        "entropy_length": entropy_length,
        "entropy_coordinate_note": "sequence_entropies[i] aligns to token_position=i+1 in the full trajectory",
        "sequence_entropies": sequence_entropies.cpu().tolist(),
        "message_spans": message_spans,
        "assistant_turn_spans": assistant_turn_spans,
        "user_message_spans": user_message_spans,
        "model_path": sample.get("model_path", ""),
        "entropy_model_path": tokenizer.name_or_path,
    }


def main() -> None:
    args = parse_args()

    samples = load_jsonl(args.input_path)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(f"No rows found in {args.input_path}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        pad_token="<|endoftext|>",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    records: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="entropy-forward"):
        records.append(
            build_record(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                device=device,
                enable_thinking=args.enable_thinking,
            )
        )

    df = pd.DataFrame(records)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)

    manifest = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "model_path": args.model_path,
        "rows": len(df),
        "unique_sample_uid": int(df["sample_uid"].nunique()),
        "max_samples": args.max_samples,
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
