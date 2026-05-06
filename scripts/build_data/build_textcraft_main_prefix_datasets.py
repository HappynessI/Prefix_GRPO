#!/usr/bin/env python3
"""Build TextCraft main-prefix training datasets with raw+prefix variants."""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TEACHER_PATH = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage0_teacher/teacher_normalized.parquet"
)
DEFAULT_ENTROPY_CANDIDATES_PATH = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage2_splits/prefix_candidates_entropy_topk.parquet"
)
DEFAULT_TRAIN_PARQUET_PATH = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/train.parquet"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/complete_split"
)
DEFAULT_MODEL_PATH = (
    "/Data/wyh/datasets/Verl-Data/outputs/textcraft_sft/qwen3-1.7b-sft/global_step_200/huggingface"
)

GOAL_RE = re.compile(r"Goal:\s*craft\s+(.+?)\.?\s*$", re.IGNORECASE | re.MULTILINE)
ACTION_LINE_RE = re.compile(r"Action:\s*(.*?)(?=\n|$)", re.DOTALL)
CHAT_TEMPLATE_ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\s*\n?", re.IGNORECASE)
CHAT_TEMPLATE_END_RE = re.compile(r"<\|im_end\|>")
PLACEHOLDER_ACTION_RE = re.compile(
    r"(?:^|\b)(?:my next action|your next action|next action)(?:\b|$)",
    re.IGNORECASE,
)
CONCAT_LIKE_ACTION_RE = re.compile(
    r"(?:\bget\b.*\bget\b|\bcraft\b.*\bcraft\b|\bget\b.*\bcraft\b|\bcraft\b.*\bget\b)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-path", type=Path, default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--entropy-candidates-path", type=Path, default=DEFAULT_ENTROPY_CANDIDATES_PATH)
    parser.add_argument("--train-parquet-path", type=Path, default=DEFAULT_TRAIN_PARQUET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-batch-prompt-tokens",
        type=int,
        default=1600,
        help="Upper bound on summed prompt lengths for each old-logprob forward batch.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--datasets",
        type=str,
        default="main_fixed_gp1,main_fixed_gp2,main_raw_top3,main_change_top3_w11",
        help="Comma-separated dataset names to build.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print one materialization progress line after this many unique prompts.",
    )
    return parser.parse_args()


def normalize_prompt_messages(prompt: Any) -> List[Dict[str, Any]]:
    if prompt is None:
        return []
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        raise ValueError(f"Prompt must be a list, got {type(prompt)}")
    normalized: List[Dict[str, Any]] = []
    for msg in prompt:
        if not isinstance(msg, dict):
            raise ValueError(f"Prompt message must be dict, got {type(msg)}")
        normalized.append(dict(msg))
    return normalized


def normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, list):
        raise ValueError(f"Messages must be a list, got {type(messages)}")
    return [dict(msg) for msg in messages]


def normalize_ws(text: str) -> str:
    return " ".join(str(text).split()).strip()


def strip_chat_template_markers(text: str) -> str:
    text = CHAT_TEMPLATE_ASSISTANT_RE.sub("", text)
    text = CHAT_TEMPLATE_END_RE.sub("", text)
    return text


def runtime_action_matches(text: str) -> List[str]:
    if not text:
        return []
    normalized_text = strip_chat_template_markers(str(text).strip())
    return ACTION_LINE_RE.findall(normalized_text)


def sanitize_runtime_action(action: str) -> str:
    action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
    return normalize_ws(action)


def extract_goal_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        match = GOAL_RE.search(content)
        if match:
            return match.group(1).strip()
    return None


def extract_action(text: str) -> Optional[str]:
    matches = runtime_action_matches(text)
    if len(matches) != 1:
        return None
    action = sanitize_runtime_action(matches[-1])
    if action:
        return action
    return None


def extract_actions_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    actions: List[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        action = extract_action(msg.get("content", ""))
        if action:
            actions.append(action)
    return actions


def prefix_row_has_multi_action_tag(messages: Sequence[Dict[str, Any]]) -> bool:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if len(runtime_action_matches(msg.get("content", ""))) > 1:
            return True
    return False


def inspect_runtime_assistant_message(
    content: str,
) -> Tuple[bool, Optional[str], Optional[str], int]:
    text = str(content or "")
    if is_warmup_assistant_message(text):
        return True, None, None, 0

    matches = runtime_action_matches(text)
    if len(matches) == 0:
        return False, "missing_action_tag", None, 0
    if len(matches) > 1:
        return False, "multiple_action_tags", None, len(matches)

    action = sanitize_runtime_action(matches[0])
    if not action:
        return False, "empty_sanitized_action", None, 1
    return True, None, action, 1


def inspect_prefix_runtime_validity(
    messages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    invalid_details: List[Dict[str, Any]] = []
    assistant_turn_idx = -1
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        assistant_turn_idx += 1
        is_valid, reason, _, action_tag_count = inspect_runtime_assistant_message(msg.get("content", ""))
        if is_valid:
            continue

        invalid_details.append(
            {
                "assistant_message_index": int(msg_idx),
                "assistant_turn_index": int(assistant_turn_idx),
                "runtime_invalid_reason": str(reason),
                "runtime_action_tag_count": int(action_tag_count),
            }
        )
    return invalid_details


def assistant_message_indices(messages: List[Dict[str, Any]]) -> List[int]:
    return [idx for idx, msg in enumerate(messages) if msg.get("role") == "assistant"]


def is_warmup_user_message(content: str) -> bool:
    return (
        "You are given few useful crafting recipes" in content
        or content.startswith("Every round I will give you")
    )


def is_warmup_assistant_message(content: str) -> bool:
    return "OK. I'll follow your instructions" in content


def extract_think_text(content: str) -> Optional[str]:
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", content, re.DOTALL)
    if think_match:
        value = think_match.group(1).strip()
        return value or None

    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", content, re.DOTALL)
    if thought_match:
        value = thought_match.group(1).strip()
        return value or None

    return None


def canonicalize_assistant_content(content: str) -> str:
    action = extract_action(content)
    think_text = extract_think_text(content)
    if action and think_text:
        return f"Think: {think_text}\nAction: [[ {action} ]]"
    if action:
        return f"Action: [[ {action} ]]"
    return content


def ratio_to_name(value: float) -> str:
    return str(value).replace(".", "p")


def choose_cut_turn_idx(num_assistant_messages: int, target_ratio: float) -> Tuple[int, float]:
    if num_assistant_messages <= 0:
        return 0, 0.0
    if num_assistant_messages == 1:
        return 0, 0.0

    for turn_idx in range(num_assistant_messages):
        q = turn_idx / (num_assistant_messages - 1)
        if q >= target_ratio:
            return turn_idx, q
    return num_assistant_messages - 1, 1.0


def split_messages(messages: List[Dict[str, Any]], cut_turn_idx: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    assistant_indices = assistant_message_indices(messages)
    if not assistant_indices:
        return [], list(messages)

    last_prefix_message_index = assistant_indices[min(cut_turn_idx, len(assistant_indices) - 1)]
    cut_position = last_prefix_message_index + 1
    return list(messages[:cut_position]), list(messages[cut_position:])


def load_reference_system_prompt(train_parquet_path: Path) -> str:
    df = pd.read_parquet(train_parquet_path)
    if df.empty:
        raise RuntimeError(f"Reference train parquet is empty: {train_parquet_path}")
    row = df.iloc[0]
    for column in ("messages", "prompt"):
        if column not in row:
            continue
        messages = normalize_prompt_messages(row[column])
        for msg in messages:
            if msg.get("role") == "system":
                return msg.get("content", "")
    raise RuntimeError(f"Could not find system prompt in {train_parquet_path}")


def resolve_task_id_from_extra_info(extra_info: Dict[str, Any]) -> Optional[int]:
    interaction_kwargs = extra_info.get("interaction_kwargs", {}) if isinstance(extra_info, dict) else {}
    for key in ("data_idx", "session_id", "task_id"):
        value = interaction_kwargs.get(key)
        if value is not None:
            return int(value)
    return None


def load_raw_base_rows(train_parquet_path: Path) -> Dict[int, Dict[str, Any]]:
    df = pd.read_parquet(train_parquet_path)
    base_rows: Dict[int, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        prompt = normalize_prompt_messages(row["prompt"])
        extra_info = deepcopy(row.get("extra_info", {}) or {})
        if not isinstance(extra_info, dict):
            raise ValueError(f"extra_info must be dict, got {type(extra_info)}")
        task_id = resolve_task_id_from_extra_info(extra_info)
        if task_id is None:
            raise ValueError(f"Could not resolve task_id from raw row extra_info: {extra_info}")
        if task_id in base_rows:
            raise ValueError(f"Duplicate raw base row for task_id={task_id}")
        base_rows[task_id] = {
            "task_id": task_id,
            "prompt": prompt,
            "data_source": row.get("data_source", "textcraft"),
            "ability": row.get("ability", "crafting"),
            "reward_model": deepcopy(row.get("reward_model", {"ground_truth": "", "style": "interaction"})),
            "extra_info": extra_info,
            "goal": extract_goal_from_messages(prompt),
        }
    return base_rows


def build_training_prompt(
    prefix_messages: List[Dict[str, Any]],
    continuation_messages: List[Dict[str, Any]],
    system_prompt: str,
) -> List[Dict[str, str]]:
    prompt: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for msg in prefix_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            if is_warmup_user_message(content):
                continue
            prompt.append({"role": "user", "content": content})
            continue
        if role == "assistant":
            if is_warmup_assistant_message(content):
                continue
            prompt.append({"role": "assistant", "content": canonicalize_assistant_content(content)})

    cut_observation: Optional[str] = None
    for msg in continuation_messages:
        if msg.get("role") == "user":
            cut_observation = msg.get("content", "")
            break
    if cut_observation:
        prompt.append({"role": "user", "content": cut_observation})

    return prompt


def build_interaction_kwargs(task_id: int, goal: Optional[str], prefix_actions: Sequence[str]) -> Dict[str, Any]:
    return {
        "name": "textcraft",
        "task_id": int(task_id),
        "data_idx": int(task_id),
        "session_id": int(task_id),
        "goal": goal,
        "eval_mode": False,
        "prefix_actions": list(prefix_actions),
    }


def build_prefix_record(
    base_row: Dict[str, Any],
    *,
    main_dataset: str,
    sample_uid: str,
    item_id: str,
    sample_idx: int,
    task_id: int,
    goal: Optional[str],
    system_prompt: str,
    prefix_messages: List[Dict[str, Any]],
    continuation_messages: List[Dict[str, Any]],
    strategy: str,
    variant_label: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prefix_actions = extract_actions_from_messages(prefix_messages)
    prompt = build_training_prompt(prefix_messages, continuation_messages, system_prompt)
    extra_info = deepcopy(base_row["extra_info"])
    interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})
    interaction_kwargs.update(build_interaction_kwargs(task_id=task_id, goal=goal, prefix_actions=prefix_actions))
    extra_info["interaction_kwargs"] = interaction_kwargs
    extra_info["sample_uid"] = sample_uid
    extra_info["main_dataset"] = main_dataset
    extra_info["variant_label"] = variant_label

    record: Dict[str, Any] = {
        "record_uid": f"{main_dataset}::{sample_uid}::{variant_label}",
        "main_dataset": main_dataset,
        "variant_label": variant_label,
        "is_raw_variant": False,
        "sample_uid": sample_uid,
        "item_id": item_id,
        "sample_idx": int(sample_idx),
        "task_id": int(task_id),
        "goal": goal,
        "strategy": strategy,
        "data_source": base_row["data_source"],
        "ability": base_row["ability"],
        "prompt": prompt,
        "reward_model": deepcopy(base_row["reward_model"]),
        "extra_info": extra_info,
        "prefix_messages": prefix_messages,
        "continuation_messages": continuation_messages,
        "prefix_actions": prefix_actions,
        "replay_category": "constructed_from_sampling",
        "assistant_prefix_old_log_probs": None,
        "prefix_mask": None,
        "prefix_token_count": None,
        "assistant_prefix_span": None,
        "source_oldlogprob_model_path": None,
        "prefix_coordinate_system": "canonicalized_prompt",
    }
    if metadata:
        record.update(metadata)
    return record


def build_raw_record(
    base_row: Dict[str, Any],
    *,
    main_dataset: str,
    sample_uid: str,
    item_id: str,
    sample_idx: int,
    task_id: int,
    goal: Optional[str],
    model_path: str,
) -> Dict[str, Any]:
    prompt = normalize_prompt_messages(base_row["prompt"])
    extra_info = deepcopy(base_row["extra_info"])
    interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})
    interaction_kwargs.update(build_interaction_kwargs(task_id=task_id, goal=goal, prefix_actions=[]))
    extra_info["interaction_kwargs"] = interaction_kwargs
    extra_info["sample_uid"] = sample_uid
    extra_info["main_dataset"] = main_dataset
    extra_info["variant_label"] = "raw"

    return {
        "record_uid": f"{main_dataset}::{sample_uid}::raw",
        "main_dataset": main_dataset,
        "variant_label": "raw",
        "is_raw_variant": True,
        "sample_uid": sample_uid,
        "item_id": item_id,
        "sample_idx": int(sample_idx),
        "task_id": int(task_id),
        "goal": goal,
        "strategy": "raw",
        "data_source": base_row["data_source"],
        "ability": base_row["ability"],
        "prompt": prompt,
        "reward_model": deepcopy(base_row["reward_model"]),
        "extra_info": extra_info,
        "prefix_messages": [],
        "continuation_messages": [],
        "prefix_actions": [],
        "replay_category": "raw",
        "assistant_prefix_old_log_probs": [],
        "prefix_mask": [],
        "prefix_token_count": 0,
        "assistant_prefix_span": {"start": 0, "end": 0},
        "source_oldlogprob_model_path": model_path,
        "prefix_coordinate_system": "canonicalized_prompt",
    }


def build_fixed_dataset_rows(
    teacher_rows: List[Dict[str, Any]],
    raw_base_rows: Dict[int, Dict[str, Any]],
    *,
    main_dataset: str,
    system_prompt: str,
    ratios: Sequence[float],
    model_path: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in teacher_rows:
        sample_uid = row["sample_uid"]
        item_id = row["item_id"]
        sample_idx = int(row["sample_idx"])
        task_id = int(row["task_id"])
        goal = row.get("goal")
        base_row = raw_base_rows[task_id]
        messages = normalize_messages(row["conversations"])
        num_assistant_messages = len(assistant_message_indices(messages))

        for ratio in ratios:
            cut_turn_idx, cut_q = choose_cut_turn_idx(num_assistant_messages, ratio)
            prefix_messages, continuation_messages = split_messages(messages, cut_turn_idx)
            strategy = f"fixed_ratio_{ratio_to_name(ratio)}"
            variant_label = strategy
            prefix_record = build_prefix_record(
                base_row,
                main_dataset=main_dataset,
                sample_uid=sample_uid,
                item_id=item_id,
                sample_idx=sample_idx,
                task_id=task_id,
                goal=goal,
                system_prompt=system_prompt,
                prefix_messages=prefix_messages,
                continuation_messages=continuation_messages,
                strategy=strategy,
                variant_label=variant_label,
                metadata={
                    "cut_turn_idx": int(cut_turn_idx),
                    "cut_relative_position_q": float(cut_q),
                    "num_assistant_messages_total": int(num_assistant_messages),
                    "source_dataset": "teacher_normalized",
                },
            )
            rows.append(prefix_record)

        rows.append(
            build_raw_record(
                base_row,
                main_dataset=main_dataset,
                sample_uid=sample_uid,
                item_id=item_id,
                sample_idx=sample_idx,
                task_id=task_id,
                goal=goal,
                model_path=model_path,
            )
        )
    return rows


def build_entropy_dataset_rows(
    teacher_rows: List[Dict[str, Any]],
    entropy_rows: List[Dict[str, Any]],
    raw_base_rows: Dict[int, Dict[str, Any]],
    *,
    main_dataset: str,
    system_prompt: str,
    model_path: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    teacher_map = {row["sample_uid"]: row for row in teacher_rows}

    for row in entropy_rows:
        sample_uid = row["sample_uid"]
        teacher_row = teacher_map[sample_uid]
        task_id = int(row["task_id"])
        base_row = raw_base_rows[task_id]
        prefix_record = build_prefix_record(
            base_row,
            main_dataset=main_dataset,
            sample_uid=sample_uid,
            item_id=row["item_id"],
            sample_idx=int(row["sample_idx"]),
            task_id=task_id,
            goal=row.get("goal"),
            system_prompt=system_prompt,
            prefix_messages=normalize_messages(row["prefix_messages"]),
            continuation_messages=normalize_messages(row["continuation_messages"]),
            strategy=row["strategy"],
            variant_label=f"rank{int(row['candidate_rank'])}",
            metadata={
                "candidate_uid": row.get("candidate_uid"),
                "candidate_rank": int(row["candidate_rank"]),
                "cut_turn_idx": int(row["cut_turn_idx"]),
                "cut_relative_position_q": float(row.get("cut_relative_position_q", 0.0)),
                "num_assistant_messages_total": int(teacher_row["num_assistant_messages"]),
                "selection_score": float(row.get("selection_score", 0.0)),
                "source_role": row.get("source_role"),
                "source_token_position": int(row.get("source_token_position", -1)),
                "source_dataset": "entropy_stage2_candidates",
            },
        )
        rows.append(prefix_record)

    for row in teacher_rows:
        task_id = int(row["task_id"])
        base_row = raw_base_rows[task_id]
        rows.append(
            build_raw_record(
                base_row,
                main_dataset=main_dataset,
                sample_uid=row["sample_uid"],
                item_id=row["item_id"],
                sample_idx=int(row["sample_idx"]),
                task_id=task_id,
                goal=row.get("goal"),
                model_path=model_path,
            )
        )
    return rows


def split_runtime_invalid_prefix_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept_rows: List[Dict[str, Any]] = []
    dropped_rows: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("is_raw_variant"):
            kept_rows.append(row)
            continue
        invalid_details = inspect_prefix_runtime_validity(row.get("prefix_messages", []))
        prefix_actions = list(row.get("prefix_actions", []))
        if invalid_details or not prefix_actions:
            dropped_row = dict(row)
            if invalid_details:
                dropped_row["runtime_invalid_details"] = invalid_details
                dropped_row["runtime_invalid_reasons"] = [
                    detail["runtime_invalid_reason"] for detail in invalid_details
                ]
            elif not prefix_actions:
                dropped_row["runtime_invalid_details"] = [
                    {
                        "assistant_message_index": -1,
                        "assistant_turn_index": -1,
                        "runtime_invalid_reason": "no_runtime_actions_after_filter",
                        "runtime_action_tag_count": 0,
                    }
                ]
                dropped_row["runtime_invalid_reasons"] = ["no_runtime_actions_after_filter"]
            dropped_rows.append(dropped_row)
            continue
        kept_rows.append(row)
    return kept_rows, dropped_rows


def prompt_key(prompt: List[Dict[str, Any]]) -> str:
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_prompt_text(prompt: List[Dict[str, Any]], tokenizer) -> str:
    return tokenizer.apply_chat_template(prompt, add_generation_prompt=False, tokenize=False)


def tokenize_conversations(tokenizer, conversations: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, int]:
    text = build_prompt_text(list(conversations), tokenizer)
    tokens = tokenizer(text, add_special_tokens=True, return_tensors="pt")
    input_ids = tokens.input_ids[0]
    return input_ids, len(input_ids)


def compute_prompt_assistant_token_spans(prompt: List[Dict[str, Any]], tokenizer) -> Tuple[str, List[Dict[str, Any]]]:
    full_text = build_prompt_text(prompt, tokenizer)
    result = tokenizer(
        full_text,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = result.offset_mapping[0]

    start_pattern = re.compile(r"<\|im_start\|>(user|assistant|tool|system)")
    end_pattern = re.compile(r"<\|im_end\|>")
    start_matches = list(start_pattern.finditer(full_text))
    end_matches = list(end_pattern.finditer(full_text))
    if len(start_matches) != len(prompt) or len(end_matches) != len(prompt):
        raise ValueError("Conversation tag count does not match prompt message count")

    token_spans: List[Dict[str, Any]] = []
    for idx, msg in enumerate(prompt):
        role = msg.get("role", "")
        if role != "assistant":
            continue

        action = extract_action(msg.get("content", ""))
        if action is None:
            continue

        start_char = start_matches[idx].end()
        end_char = end_matches[idx].end()
        start_token = None
        end_token = None
        for token_idx, (char_start, char_end) in enumerate(offset_mapping.tolist()):
            if char_start is None:
                continue
            if char_start < end_char and char_end > start_char:
                if start_token is None:
                    start_token = token_idx
                end_token = token_idx + 1

        if start_token is None or end_token is None:
            raise ValueError(f"Could not map assistant prompt message to token span: idx={idx}")

        token_spans.append(
            {
                "token_start": start_token,
                "token_end": end_token,
                "action": action,
            }
        )

    return full_text, token_spans


def compute_batch_old_logprobs(model, tokenizer, texts: List[str], device: str) -> List[List[float]]:
    if not texts:
        return []

    batch = None
    input_ids = None
    attention_mask = None
    outputs = None
    logits = None
    log_probs = None
    scalar_logprobs = None
    try:
        batch = tokenizer(
            texts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            target_tokens = input_ids[:, 1:].unsqueeze(-1)
            scalar_logprobs = log_probs.gather(dim=-1, index=target_tokens).squeeze(-1)

        lengths = attention_mask.sum(dim=1).tolist()
        result: List[List[float]] = []
        for row_idx, seq_len in enumerate(lengths):
            valid_len = int(seq_len) - 1
            result.append(scalar_logprobs[row_idx, :valid_len].detach().cpu().float().tolist())

        del outputs, logits, log_probs, scalar_logprobs, input_ids, attention_mask, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except torch.OutOfMemoryError:
        del outputs, logits, log_probs, scalar_logprobs, input_ids, attention_mask, batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(texts) == 1:
            raise
        mid = len(texts) // 2
        left = compute_batch_old_logprobs(model, tokenizer, texts[:mid], device)
        right = compute_batch_old_logprobs(model, tokenizer, texts[mid:], device)
        return left + right


def build_prefix_window_from_prompt_spans(
    prompt_spans: List[Dict[str, Any]],
    sequence_old_logprobs: List[float],
) -> Tuple[List[float], List[int], Dict[str, int], int]:
    if not prompt_spans:
        return [], [], {"start": 0, "end": 0}, 0

    first_token = prompt_spans[0]["token_start"]
    last_token = prompt_spans[-1]["token_end"]

    lp_start = max(0, first_token - 1)
    lp_end = min(len(sequence_old_logprobs), last_token - 1)
    prefix_old_logprobs = list(sequence_old_logprobs[lp_start:lp_end])

    prefix_mask = [0] * (last_token - first_token)
    for span in prompt_spans:
        rel_start = span["token_start"] - first_token
        rel_end = span["token_end"] - first_token
        for pos in range(rel_start, rel_end):
            prefix_mask[pos] = 1

    prefix_token_count = int(sum(prefix_mask))
    return prefix_old_logprobs, prefix_mask, {"start": first_token, "end": last_token}, prefix_token_count


def validate_rebuilt_sample(
    record_uid: str,
    prompt_token_length: int,
    prompt_spans: List[Dict[str, Any]],
    sequence_old_logprobs: List[float],
    prefix_old_logprobs: List[float],
    prefix_mask: List[int],
    prefix_span: Dict[str, int],
    prefix_token_count: int,
) -> None:
    if not prompt_spans:
        raise RuntimeError(f"{record_uid}: prompt contains no assistant action spans")

    if len(sequence_old_logprobs) != prompt_token_length - 1:
        raise RuntimeError(
            f"{record_uid}: len(sequence_old_logprobs)={len(sequence_old_logprobs)} "
            f"!= prompt_token_length-1={prompt_token_length - 1}"
        )

    span_len = int(prefix_span["end"]) - int(prefix_span["start"])
    if span_len <= 0:
        raise RuntimeError(f"{record_uid}: invalid prefix span {prefix_span}")
    if len(prefix_old_logprobs) != span_len:
        raise RuntimeError(f"{record_uid}: len(prefix_old_logprobs)={len(prefix_old_logprobs)} != span_len={span_len}")
    if len(prefix_mask) != span_len:
        raise RuntimeError(f"{record_uid}: len(prefix_mask)={len(prefix_mask)} != span_len={span_len}")
    if prefix_token_count != int(sum(prefix_mask)):
        raise RuntimeError(
            f"{record_uid}: prefix_token_count={prefix_token_count} != sum(prefix_mask)={int(sum(prefix_mask))}"
        )
    if prefix_token_count <= 0:
        raise RuntimeError(f"{record_uid}: prefix_token_count must be positive")


def materialize_prefix_sidecars(
    rows: List[Dict[str, Any]],
    *,
    model_path: str,
    device: str,
    batch_size: int,
    max_batch_prompt_tokens: int,
    progress_every: int,
) -> None:
    prefix_prompt_rows = [row for row in rows if not row["is_raw_variant"]]
    if not prefix_prompt_rows:
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    unique_prompts: Dict[str, Dict[str, Any]] = {}
    for row in prefix_prompt_rows:
        prompt = normalize_prompt_messages(row["prompt"])
        key = prompt_key(prompt)
        if key in unique_prompts:
            continue
        prompt_text, prompt_spans = compute_prompt_assistant_token_spans(prompt, tokenizer)
        _, prompt_token_length = tokenize_conversations(tokenizer, prompt)
        unique_prompts[key] = {
            "prompt_text": prompt_text,
            "prompt_spans": prompt_spans,
            "prompt_token_length": prompt_token_length,
        }

    prompt_items = sorted(
        unique_prompts.items(),
        key=lambda item: int(item[1]["prompt_token_length"]),
        reverse=True,
    )
    total_unique_prompts = len(prompt_items)
    total_prompt_tokens = sum(int(item[1]["prompt_token_length"]) for item in prompt_items)
    max_prompt_tokens = max(int(item[1]["prompt_token_length"]) for item in prompt_items)
    print(
        "[materialize] "
        f"device={device} unique_prefix_prompts={total_unique_prompts} "
        f"total_prompt_tokens={total_prompt_tokens} max_prompt_tokens={max_prompt_tokens} "
        f"batch_size={max(1, batch_size)} max_batch_prompt_tokens={max_batch_prompt_tokens}",
        flush=True,
    )

    processed_unique_prompts = 0
    start_time = time.time()
    start_idx = 0
    while start_idx < len(prompt_items):
        chunk: List[Tuple[str, Dict[str, Any]]] = []
        token_budget = 0
        while start_idx < len(prompt_items) and len(chunk) < max(1, batch_size):
            candidate = prompt_items[start_idx]
            candidate_tokens = int(candidate[1]["prompt_token_length"])
            if chunk and token_budget + candidate_tokens > max_batch_prompt_tokens:
                break
            chunk.append(candidate)
            token_budget += candidate_tokens
            start_idx += 1
        if not chunk:
            chunk = [prompt_items[start_idx]]
            start_idx += 1

        texts = [item[1]["prompt_text"] for item in chunk]
        batch_old_logprobs = compute_batch_old_logprobs(model, tokenizer, texts, device)
        for (key, prepared), sequence_old_logprobs in zip(chunk, batch_old_logprobs, strict=True):
            prefix_old_logprobs, prefix_mask, prefix_span, prefix_token_count = build_prefix_window_from_prompt_spans(
                prompt_spans=prepared["prompt_spans"],
                sequence_old_logprobs=sequence_old_logprobs,
            )
            validate_rebuilt_sample(
                record_uid=f"prompt_key={key[:64]}",
                prompt_token_length=prepared["prompt_token_length"],
                prompt_spans=prepared["prompt_spans"],
                sequence_old_logprobs=sequence_old_logprobs,
                prefix_old_logprobs=prefix_old_logprobs,
                prefix_mask=prefix_mask,
                prefix_span=prefix_span,
                prefix_token_count=prefix_token_count,
            )
            prepared["assistant_prefix_old_log_probs"] = prefix_old_logprobs
            prepared["prefix_mask"] = prefix_mask
            prepared["assistant_prefix_span"] = prefix_span
            prepared["prefix_token_count"] = prefix_token_count

        processed_unique_prompts += len(chunk)
        gc.collect()
        if (
            processed_unique_prompts == total_unique_prompts
            or processed_unique_prompts % max(1, progress_every) == 0
        ):
            elapsed = time.time() - start_time
            print(
                "[materialize] "
                f"processed={processed_unique_prompts}/{total_unique_prompts} "
                f"({processed_unique_prompts / total_unique_prompts:.1%}) "
                f"chunk_size={len(chunk)} chunk_prompt_tokens={token_budget} "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for row in prefix_prompt_rows:
        key = prompt_key(normalize_prompt_messages(row["prompt"]))
        prepared = unique_prompts[key]
        row["assistant_prefix_old_log_probs"] = prepared["assistant_prefix_old_log_probs"]
        row["prefix_mask"] = prepared["prefix_mask"]
        row["assistant_prefix_span"] = prepared["assistant_prefix_span"]
        row["prefix_token_count"] = prepared["prefix_token_count"]
        row["source_oldlogprob_model_path"] = model_path


def summarize_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    raw_mask = df["is_raw_variant"] == True
    prefix_mask = ~raw_mask
    prefix_action_lengths = df["prefix_actions"].apply(lambda value: len(list(value)) if value is not None else 0)
    summary = {
        "rows": len(df),
        "unique_record_uid": int(df["record_uid"].nunique()),
        "unique_sample_uid": int(df["sample_uid"].nunique()),
        "raw_rows": int(raw_mask.sum()),
        "prefix_rows": int(prefix_mask.sum()),
        "strategy_counts": df["strategy"].value_counts().sort_index().to_dict(),
        "variant_counts": df["variant_label"].value_counts().sort_index().to_dict(),
        "zero_prefix_rows": int((df["prefix_token_count"] == 0).sum()),
        "positive_prefix_rows": int((df["prefix_token_count"] > 0).sum()),
        "prefix_action_length_counts": prefix_action_lengths.value_counts().sort_index().to_dict(),
        "empty_prefix_action_rows": int((prefix_mask & (prefix_action_lengths == 0)).sum()),
        "placeholder_like_prefix_action_rows": int(
            df.apply(
                lambda row: (
                    (not bool(row["is_raw_variant"]))
                    and any(PLACEHOLDER_ACTION_RE.search(str(action)) for action in list(row["prefix_actions"]))
                ),
                axis=1,
            ).sum()
        ),
        "concat_like_prefix_action_rows": int(
            df.apply(
                lambda row: (
                    (not bool(row["is_raw_variant"]))
                    and any(CONCAT_LIKE_ACTION_RE.search(str(action)) for action in list(row["prefix_actions"]))
                ),
                axis=1,
            ).sum()
        ),
        "rows_with_multi_action_tag_in_prefix_messages": int(
            df.apply(
                lambda row: (
                    (not bool(row["is_raw_variant"]))
                    and prefix_row_has_multi_action_tag(list(row["prefix_messages"]))
                ),
                axis=1,
            ).sum()
        ),
    }
    if "candidate_rank" in df.columns:
        rank_series = df["candidate_rank"].dropna()
        if not rank_series.empty:
            summary["candidate_rank_counts"] = rank_series.astype(int).value_counts().sort_index().to_dict()
    return summary


def write_dataset(
    output_path: Path,
    rows: List[Dict[str, Any]],
    manifest_path: Path,
    *,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    summary = summarize_dataset(rows)
    if extra_summary:
        summary.update(extra_summary)
    summary["output_path"] = str(output_path)
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_report_markdown(
    summaries: Dict[str, Dict[str, Any]],
    *,
    teacher_path: Path,
    entropy_candidates_path: Path,
    train_parquet_path: Path,
    model_path: str,
) -> str:
    lines = [
        "# TextCraft Main Prefix Dataset Build",
        "",
        "## Inputs",
        f"- teacher_normalized: `{teacher_path}`",
        f"- entropy stage2 candidates: `{entropy_candidates_path}`",
        f"- official raw train parquet: `{train_parquet_path}`",
        f"- prompt-space old-logprob model: `{model_path}`",
        "",
        "## Output Datasets",
    ]
    for name, summary in summaries.items():
        lines.extend(
            [
                f"### {name}",
                f"- rows: `{summary['rows']}`",
                f"- unique_sample_uid: `{summary['unique_sample_uid']}`",
                f"- raw_rows: `{summary['raw_rows']}`",
                f"- prefix_rows: `{summary['prefix_rows']}`",
                f"- zero_prefix_rows: `{summary['zero_prefix_rows']}`",
                f"- positive_prefix_rows: `{summary['positive_prefix_rows']}`",
                f"- output_path: `{summary['output_path']}`",
                f"- strategy_counts: `{summary['strategy_counts']}`",
            ]
        )
        if "candidate_rank_counts" in summary:
            lines.append(f"- candidate_rank_counts: `{summary['candidate_rank_counts']}`")
        if "dropped_runtime_invalid_prefix_rows" in summary:
            lines.append(
                f"- dropped_runtime_invalid_prefix_rows: `{summary['dropped_runtime_invalid_prefix_rows']}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "- `main_fixed_gp1` and `main_fixed_gp2` each duplicate every sampled trajectory into `3 prefix variants + 1 raw variant`.",
            "- `main_raw_top3` and `main_change_top3_w11` use the full stage2 top-k split points to preserve all `1496` sampled trajectories.",
            "- `raw` variants keep the official train parquet prompt and attach an empty prefix sidecar (`prefix_token_count=0`).",
            "- Prefix sidecars are rebuilt in prompt-space on the canonicalized training prompt.",
            "- `extract_action` / `prefix_actions` / assistant canonicalization now follow the runtime TextCraft ReAct parser exactly: a message contributes an action only when it contains exactly one `Action:` tag.",
            "- Assistant messages with multiple `Action:` tags are treated as runtime-invalid and are no longer collapsed into synthetic fake actions during dataset rebuild.",
            "- The legacy `new_prefix_rl` fixed-ratio release sidecars are not reused here; `main_prefix` rebuilds all prefix sidecars under one prompt-space protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def load_existing_summaries(output_root: Path) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for manifest_path in sorted(output_root.glob("*.manifest.json")):
        try:
            summaries[manifest_path.stem.replace(".manifest", "")] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue
    return summaries


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_datasets = {
        name.strip() for name in args.datasets.split(",") if name.strip()
    }
    if not selected_datasets:
        raise RuntimeError("No datasets selected. Pass at least one dataset name via --datasets.")

    teacher_df = pd.read_parquet(args.teacher_path)
    if args.max_samples is not None:
        teacher_df = teacher_df.head(args.max_samples).copy()
    teacher_rows = teacher_df.to_dict(orient="records")
    if not teacher_rows:
        raise RuntimeError(f"Teacher parquet is empty: {args.teacher_path}")

    sample_uids = set(teacher_df["sample_uid"].tolist())
    entropy_df = pd.read_parquet(args.entropy_candidates_path)
    entropy_df = entropy_df[entropy_df["sample_uid"].isin(sample_uids)].copy()

    raw_base_rows = load_raw_base_rows(args.train_parquet_path)
    system_prompt = load_reference_system_prompt(args.train_parquet_path)

    raw_top3_rows = entropy_df[
        entropy_df["strategy"] == "entropy_raw_topk_interaction_assistant_k3"
    ].to_dict(orient="records")
    change_top3_rows = entropy_df[
        entropy_df["strategy"] == "entropy_change_topk_w11_interaction_assistant_k3"
    ].to_dict(orient="records")

    datasets_all: Dict[str, List[Dict[str, Any]]] = {
        "main_fixed_gp1": build_fixed_dataset_rows(
            teacher_rows,
            raw_base_rows,
            main_dataset="main_fixed_gp1",
            system_prompt=system_prompt,
            ratios=(0.1, 0.3, 0.5),
            model_path=args.model_path,
        ),
        "main_fixed_gp2": build_fixed_dataset_rows(
            teacher_rows,
            raw_base_rows,
            main_dataset="main_fixed_gp2",
            system_prompt=system_prompt,
            ratios=(0.25, 0.5, 0.7),
            model_path=args.model_path,
        ),
        "main_raw_top3": build_entropy_dataset_rows(
            teacher_rows,
            raw_top3_rows,
            raw_base_rows,
            main_dataset="main_raw_top3",
            system_prompt=system_prompt,
            model_path=args.model_path,
        ),
        "main_change_top3_w11": build_entropy_dataset_rows(
            teacher_rows,
            change_top3_rows,
            raw_base_rows,
            main_dataset="main_change_top3_w11",
            system_prompt=system_prompt,
            model_path=args.model_path,
        ),
    }
    datasets = {name: rows for name, rows in datasets_all.items() if name in selected_datasets}
    unknown = sorted(selected_datasets - set(datasets_all))
    if unknown:
        raise RuntimeError(f"Unknown dataset names: {unknown}")

    dataset_runtime_drop_summary: Dict[str, Dict[str, Any]] = {}
    for name, rows in list(datasets.items()):
        kept_rows, dropped_rows = split_runtime_invalid_prefix_rows(rows)
        datasets[name] = kept_rows
        dataset_runtime_drop_summary[name] = {
            "dropped_runtime_invalid_prefix_rows": int(len(dropped_rows)),
            "dropped_runtime_invalid_prefix_sample_uid": sorted({str(row["sample_uid"]) for row in dropped_rows})[:50],
        }

    all_rows: List[Dict[str, Any]] = []
    for rows in datasets.values():
        all_rows.extend(rows)
    materialize_prefix_sidecars(
        all_rows,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_batch_prompt_tokens=args.max_batch_prompt_tokens,
        progress_every=args.progress_every,
    )

    summaries: Dict[str, Dict[str, Any]] = {}
    for name, rows in datasets.items():
        output_path = args.output_root / f"{name}.parquet"
        manifest_path = args.output_root / f"{name}.manifest.json"
        summaries[name] = write_dataset(
            output_path,
            rows,
            manifest_path,
            extra_summary=dataset_runtime_drop_summary.get(name),
        )

    report_path = args.output_root / "README.md"
    combined_summaries = load_existing_summaries(args.output_root)
    report_path.write_text(
        build_report_markdown(
            combined_summaries if combined_summaries else summaries,
            teacher_path=args.teacher_path,
            entropy_candidates_path=args.entropy_candidates_path,
            train_parquet_path=args.train_parquet_path,
            model_path=args.model_path,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
