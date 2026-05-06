#!/usr/bin/env python3
"""Append logged TextCraft teacher-demo rows with precomputed SFT old logprobs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts" / "build_data") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "build_data"))

import build_textcraft_main_prefix_datasets as main_builder  # noqa: E402


DEFAULT_BASE_PARQUET = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/"
    "replay_validated/main_change_top3_w11_fullflow.parquet"
)
DEFAULT_TEACHER_PATH = main_builder.DEFAULT_TEACHER_PATH
DEFAULT_MODEL_PATH = main_builder.DEFAULT_MODEL_PATH
DEFAULT_OUTPUT_PATH = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/"
    "replay_validated/main_change_top3_w11_fullflow_with_teacher_demo.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-parquet", type=Path, default=DEFAULT_BASE_PARQUET)
    parser.add_argument("--teacher-path", type=Path, default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-response-tokens", type=int, default=8192)
    parser.add_argument("--min-demo-reward", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    return main_builder.normalize_messages(messages)


def canonicalize_response_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if role == "assistant":
            if main_builder.is_warmup_assistant_message(content):
                continue
            output.append({"role": "assistant", "content": main_builder.canonicalize_assistant_content(content)})
        elif role == "user":
            if main_builder.is_warmup_user_message(content):
                continue
            output.append({"role": "user", "content": content})
    return output


def teacher_response_messages(conversations: Any) -> List[Dict[str, str]]:
    messages = normalize_messages(conversations)
    task_user_idx = None
    for idx, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", ""))
        if main_builder.is_warmup_user_message(content):
            continue
        task_user_idx = idx
        break
    if task_user_idx is None:
        raise ValueError("Could not find non-warmup task user message in teacher conversation.")
    return canonicalize_response_messages(messages[task_user_idx + 1 :])


def apply_chat_template(tokenizer, messages: Sequence[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_response_sidecars(tokenizer, prompt: Sequence[Dict[str, Any]], response_messages: Sequence[Dict[str, Any]]):
    prompt = list(prompt)
    response_messages = list(response_messages)
    full_messages = prompt + response_messages
    prompt_text = apply_chat_template(tokenizer, prompt, add_generation_prompt=True)
    full_text = apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("Full teacher-demo text does not share the raw prompt generation prefix.")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    tokenized = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(tokenized.input_ids)
    offsets = list(tokenized.offset_mapping)
    prompt_len = len(prompt_ids)
    response_ids = input_ids[prompt_len:]
    if not response_ids:
        raise ValueError("Teacher-demo response is empty after raw prompt.")

    loss_mask = np.zeros((len(input_ids),), dtype=np.int64)
    start_matches = list(main_builder.re.finditer(r"<\|im_start\|>(user|assistant|tool|system)\s*\n?", full_text))
    end_matches = list(main_builder.CHAT_TEMPLATE_END_RE.finditer(full_text))
    if len(start_matches) != len(full_messages) or len(end_matches) != len(full_messages):
        raise ValueError("Chat template tag count does not match teacher-demo message count.")

    for msg_idx, msg in enumerate(full_messages):
        if msg_idx < len(prompt) or msg.get("role") != "assistant":
            continue
        start_char = start_matches[msg_idx].end()
        end_char = end_matches[msg_idx].end()
        for token_idx, (char_start, char_end) in enumerate(offsets):
            if char_start is None:
                continue
            if char_start < end_char and char_end > start_char:
                loss_mask[token_idx] = 1

    response_loss_mask = loss_mask[prompt_len : prompt_len + len(response_ids)].astype(np.int64).tolist()
    if sum(response_loss_mask) <= 0:
        raise ValueError("Teacher-demo response has no assistant loss tokens.")
    return {
        "full_text": full_text,
        "prompt_token_len": int(prompt_len),
        "response_ids": [int(x) for x in response_ids],
        "response_attention_mask": [1] * len(response_ids),
        "response_loss_mask": response_loss_mask,
    }


def compute_old_logprobs(model, tokenizer, texts: Sequence[str], device: str) -> List[List[float]]:
    batch = tokenizer(list(texts), add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        targets = input_ids[:, 1:].unsqueeze(-1)
        scalar_logprobs = log_probs.gather(dim=-1, index=targets).squeeze(-1)

    lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
    result: List[List[float]] = []
    for row_idx, seq_len in enumerate(lengths):
        valid_len = int(seq_len) - 1
        result.append(scalar_logprobs[row_idx, :valid_len].detach().cpu().float().tolist())

    del outputs, logits, log_probs, targets, scalar_logprobs, input_ids, attention_mask, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def materialize_demo_old_logprobs(rows: List[Dict[str, Any]], *, model_path: str, device: str, batch_size: int) -> None:
    if not rows:
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch_dtype)
    model.to(device)
    model.eval()

    for start in range(0, len(rows), max(1, batch_size)):
        chunk = rows[start : start + max(1, batch_size)]
        old_logprobs = compute_old_logprobs(model, tokenizer, [row["_teacher_demo_full_text"] for row in chunk], device)
        for row, sequence_old_logprobs in zip(chunk, old_logprobs, strict=True):
            prompt_len = int(row["_teacher_demo_prompt_token_len"])
            response_len = len(row["teacher_demo_response_ids"])
            slice_start = prompt_len - 1
            slice_end = slice_start + response_len
            response_old_logprobs = sequence_old_logprobs[slice_start:slice_end]
            if len(response_old_logprobs) != response_len:
                raise ValueError(
                    f"{row['record_uid']}: old_logprob len={len(response_old_logprobs)} "
                    f"!= response_len={response_len}"
                )
            row["teacher_demo_old_log_probs"] = [float(x) for x in response_old_logprobs]
        print(f"[teacher-demo] old_logprobs {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
        gc.collect()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_demo_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_df = pd.read_parquet(args.base_parquet)
    teacher_df = pd.read_parquet(args.teacher_path)
    if args.max_samples is not None:
        teacher_df = teacher_df.head(args.max_samples).copy()

    demo_columns = [
        "teacher_demo_response_ids",
        "teacher_demo_response_attention_mask",
        "teacher_demo_response_loss_mask",
        "teacher_demo_old_log_probs",
        "teacher_demo_reward",
        "teacher_demo_old_logprob_source",
    ]
    base_records = base_df.to_dict(orient="records")
    for record in base_records:
        for column in demo_columns:
            record.setdefault(column, None)

    raw_rows = {
        str(row["sample_uid"]): dict(row)
        for row in base_records
        if str(row.get("variant_label", "")).lower() == "raw"
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    demo_rows: List[Dict[str, Any]] = []
    skipped_missing_raw = 0
    skipped_reward = 0
    skipped_overlong = 0
    skipped_invalid = 0
    for teacher_row in teacher_df.to_dict(orient="records"):
        sample_uid = str(teacher_row["sample_uid"])
        raw_row = raw_rows.get(sample_uid)
        if raw_row is None:
            skipped_missing_raw += 1
            continue
        reward = float(teacher_row.get("reward", teacher_row.get("success", 0.0)))
        if reward < float(args.min_demo_reward):
            skipped_reward += 1
            continue
        try:
            response_messages = teacher_response_messages(teacher_row["conversations"])
            sidecars = build_response_sidecars(tokenizer, raw_row["prompt"], response_messages)
        except Exception:
            skipped_invalid += 1
            continue
        if len(sidecars["response_ids"]) > int(args.max_response_tokens):
            skipped_overlong += 1
            continue

        row = deepcopy(raw_row)
        main_dataset = str(row.get("main_dataset", "textcraft_main"))
        row.update(
            {
                "record_uid": f"{main_dataset}::{sample_uid}::teacher_demo",
                "variant_label": "teacher_demo",
                "is_raw_variant": False,
                "strategy": "teacher_demo",
                "replay_category": "teacher_demo",
                "prefix_messages": [],
                "continuation_messages": response_messages,
                "prefix_actions": [],
                "assistant_prefix_old_log_probs": [],
                "prefix_mask": [],
                "prefix_token_count": 0,
                "assistant_prefix_span": {"start": 0, "end": 0},
                "teacher_demo_response_ids": sidecars["response_ids"],
                "teacher_demo_response_attention_mask": sidecars["response_attention_mask"],
                "teacher_demo_response_loss_mask": sidecars["response_loss_mask"],
                "teacher_demo_old_log_probs": None,
                "teacher_demo_reward": reward,
                "teacher_demo_old_logprob_source": "sft_fixed",
                "source_oldlogprob_model_path": args.model_path,
                "_teacher_demo_full_text": sidecars["full_text"],
                "_teacher_demo_prompt_token_len": sidecars["prompt_token_len"],
            }
        )
        extra_info = deepcopy(row.get("extra_info", {}) or {})
        interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})
        interaction_kwargs["prefix_actions"] = []
        extra_info["interaction_kwargs"] = interaction_kwargs
        extra_info["variant_label"] = "teacher_demo"
        extra_info["sample_uid"] = sample_uid
        row["extra_info"] = extra_info
        demo_rows.append(row)

    materialize_demo_old_logprobs(
        demo_rows,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    for row in demo_rows:
        row.pop("_teacher_demo_full_text", None)
        row.pop("_teacher_demo_prompt_token_len", None)

    output_df = pd.DataFrame(base_records + demo_rows)
    manifest = {
        "base_parquet": str(args.base_parquet),
        "teacher_path": str(args.teacher_path),
        "output": str(args.output),
        "base_rows": int(len(base_records)),
        "demo_rows": int(len(demo_rows)),
        "skipped_missing_raw": int(skipped_missing_raw),
        "skipped_low_reward": int(skipped_reward),
        "skipped_overlong": int(skipped_overlong),
        "skipped_invalid": int(skipped_invalid),
        "model_path": str(args.model_path),
        "old_logprob_source": "sft_fixed",
    }
    return output_df, manifest


def main() -> None:
    args = parse_args()
    output_df, manifest = build_demo_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
