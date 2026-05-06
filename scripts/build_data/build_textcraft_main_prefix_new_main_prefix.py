#!/usr/bin/env python3
"""Build sample-faithful TextCraft main-prefix datasets with the online loose action parser."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "verl"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from verl.interactions.textcraft_action_parser import (  # noqa: E402
    extract_textcraft_action_loose,
    extract_textcraft_action_loose_with_mode,
)

import build_textcraft_main_prefix_datasets as main_builder  # noqa: E402
import build_textcraft_main_prefix_fullflow_datasets as fullflow_builder  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix")
DEFAULT_LEGACY_COMPLETE_SPLIT_ROOT = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/complete_split"
)
DEFAULT_LEGACY_REPLAY_VALIDATED_ROOT = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/replay_validated"
)

DATASET_NAMES = (
    "main_fixed_gp1",
    "main_fixed_gp2",
    "main_raw_top3",
    "main_change_top3_w11",
)
LEGACY_NAME_TO_FILENAME = {
    "main_fixed_gp1": "main_fixed_gp1.parquet",
    "main_fixed_gp2": "main_fixed_gp2.parquet",
    "main_raw_top3": "main_raw_top3.parquet",
    "main_change_top3_w11": "main_change_top3_w11.parquet",
    "main_fixed_gp1_fullflow": "main_fixed_gp1_fullflow.parquet",
    "main_fixed_gp2_fullflow": "main_fixed_gp2_fullflow.parquet",
    "main_raw_top3_fullflow": "main_raw_top3_fullflow.parquet",
    "main_change_top3_w11_fullflow": "main_change_top3_w11_fullflow.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-path", type=Path, default=main_builder.DEFAULT_TEACHER_PATH)
    parser.add_argument("--entropy-candidates-path", type=Path, default=main_builder.DEFAULT_ENTROPY_CANDIDATES_PATH)
    parser.add_argument("--train-parquet-path", type=Path, default=main_builder.DEFAULT_TRAIN_PARQUET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=str, default=main_builder.DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-prompt-tokens", type=int, default=2400)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--server", type=str, default=fullflow_builder.DEFAULT_SERVER)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--replay-concurrency", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--datasets",
        type=str,
        default="main_fixed_gp1,main_fixed_gp2,main_raw_top3,main_change_top3_w11",
        help="Comma-separated base dataset names. Matching *_fullflow datasets are rebuilt automatically.",
    )
    parser.add_argument("--legacy-complete-split-root", type=Path, default=DEFAULT_LEGACY_COMPLETE_SPLIT_ROOT)
    parser.add_argument("--legacy-replay-validated-root", type=Path, default=DEFAULT_LEGACY_REPLAY_VALIDATED_ROOT)
    return parser.parse_args()


def normalize_records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def sampling_action_tag_count(text: str) -> int:
    return len(main_builder.runtime_action_matches(text))


def collect_sampling_prefix_diagnostics(
    messages: Sequence[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
    actions: List[str] = []
    extraction_mode_counts: Dict[str, int] = {}
    details: List[Dict[str, Any]] = []

    assistant_turn_count = 0
    non_warmup_assistant_turn_count = 0
    missing_action_turn_count = 0
    multi_action_tag_turn_count = 0

    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        assistant_turn_count += 1
        content = str(msg.get("content", ""))
        is_warmup = main_builder.is_warmup_assistant_message(content)
        if is_warmup:
            details.append(
                {
                    "assistant_message_index": int(msg_idx),
                    "assistant_turn_index": int(assistant_turn_count - 1),
                    "is_warmup": True,
                    "sampling_action_tag_count": 0,
                    "sampling_has_multiple_action_tags": False,
                    "sampling_extracted_action": None,
                    "sampling_extraction_mode": None,
                }
            )
            continue

        non_warmup_assistant_turn_count += 1
        action, mode = extract_textcraft_action_loose_with_mode(content)
        tag_count = sampling_action_tag_count(content)
        if tag_count > 1:
            multi_action_tag_turn_count += 1
        if action is None:
            missing_action_turn_count += 1
        else:
            actions.append(action)
            if mode:
                extraction_mode_counts[mode] = extraction_mode_counts.get(mode, 0) + 1

        details.append(
            {
                "assistant_message_index": int(msg_idx),
                "assistant_turn_index": int(assistant_turn_count - 1),
                "is_warmup": False,
                "sampling_action_tag_count": int(tag_count),
                "sampling_has_multiple_action_tags": bool(tag_count > 1),
                "sampling_extracted_action": action,
                "sampling_extraction_mode": mode,
            }
        )

    summary = {
        "sampling_prefix_action_count": int(len(actions)),
        "sampling_assistant_turn_count": int(assistant_turn_count),
        "sampling_non_warmup_assistant_turn_count": int(non_warmup_assistant_turn_count),
        "sampling_missing_action_turn_count": int(missing_action_turn_count),
        "sampling_multi_action_tag_turn_count": int(multi_action_tag_turn_count),
        "sampling_extraction_mode_counts": dict(sorted(extraction_mode_counts.items())),
        "sampling_has_empty_prefix_actions": bool(len(actions) == 0),
    }
    return actions, summary, details


def canonicalize_assistant_content_sampling(content: str) -> str:
    action = extract_textcraft_action_loose(content)
    think_text = main_builder.extract_think_text(content)
    if action and think_text:
        return f"Think: {think_text}\nAction: [[ {action} ]]"
    if action:
        return f"Action: [[ {action} ]]"
    return content


def build_training_prompt_sampling(
    prefix_messages: List[Dict[str, Any]],
    continuation_messages: List[Dict[str, Any]],
    system_prompt: str,
) -> List[Dict[str, str]]:
    prompt: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for msg in prefix_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            if main_builder.is_warmup_user_message(content):
                continue
            prompt.append({"role": "user", "content": content})
            continue
        if role == "assistant":
            if main_builder.is_warmup_assistant_message(content):
                continue
            prompt.append({"role": "assistant", "content": canonicalize_assistant_content_sampling(content)})

    cut_observation: Optional[str] = None
    for msg in continuation_messages:
        if msg.get("role") == "user":
            cut_observation = msg.get("content", "")
            break
    if cut_observation:
        prompt.append({"role": "user", "content": cut_observation})

    return prompt


def build_prefix_record_sampling(
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
    prefix_actions, diagnostics, details = collect_sampling_prefix_diagnostics(prefix_messages)
    prompt = build_training_prompt_sampling(prefix_messages, continuation_messages, system_prompt)
    extra_info = deepcopy(base_row["extra_info"])
    interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})
    interaction_kwargs.update(
        main_builder.build_interaction_kwargs(task_id=task_id, goal=goal, prefix_actions=prefix_actions)
    )
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
        "sampling_parser": "online_loose",
        "sampling_prefix_diagnostics": details,
    }
    record.update(diagnostics)
    if metadata:
        record.update(metadata)
    return record


def build_fixed_complete_split_rows(
    teacher_rows: Sequence[Dict[str, Any]],
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
        messages = main_builder.normalize_messages(row["conversations"])
        num_assistant_messages = len(main_builder.assistant_message_indices(messages))

        for ratio in ratios:
            cut_turn_idx, cut_q = main_builder.choose_cut_turn_idx(num_assistant_messages, ratio)
            prefix_messages, continuation_messages = main_builder.split_messages(messages, cut_turn_idx)
            strategy = f"fixed_ratio_{main_builder.ratio_to_name(ratio)}"
            rows.append(
                build_prefix_record_sampling(
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
                    variant_label=strategy,
                    metadata={
                        "cut_turn_idx": int(cut_turn_idx),
                        "cut_relative_position_q": float(cut_q),
                        "num_assistant_messages_total": int(num_assistant_messages),
                        "source_dataset": "teacher_normalized",
                    },
                )
            )

        rows.append(
            main_builder.build_raw_record(
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


def build_entropy_complete_split_rows(
    teacher_rows: Sequence[Dict[str, Any]],
    entropy_rows: Sequence[Dict[str, Any]],
    raw_base_rows: Dict[int, Dict[str, Any]],
    *,
    main_dataset: str,
    system_prompt: str,
    model_path: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    teacher_map = {str(row["sample_uid"]): row for row in teacher_rows}

    for row in entropy_rows:
        sample_uid = str(row["sample_uid"])
        teacher_row = teacher_map[sample_uid]
        task_id = int(row["task_id"])
        base_row = raw_base_rows[task_id]
        rows.append(
            build_prefix_record_sampling(
                base_row,
                main_dataset=main_dataset,
                sample_uid=sample_uid,
                item_id=row["item_id"],
                sample_idx=int(row["sample_idx"]),
                task_id=task_id,
                goal=row.get("goal"),
                system_prompt=system_prompt,
                prefix_messages=main_builder.normalize_messages(row["prefix_messages"]),
                continuation_messages=main_builder.normalize_messages(row["continuation_messages"]),
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
        )

    for row in teacher_rows:
        task_id = int(row["task_id"])
        base_row = raw_base_rows[task_id]
        rows.append(
            main_builder.build_raw_record(
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


def build_fixed_replay_candidate_rows(
    teacher_rows: Sequence[Dict[str, Any]],
    *,
    ratio: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    strategy = f"fixed_ratio_{main_builder.ratio_to_name(ratio)}"
    for row in teacher_rows:
        messages = main_builder.normalize_messages(row["conversations"])
        num_assistant_messages = len(main_builder.assistant_message_indices(messages))
        cut_turn_idx, cut_q = main_builder.choose_cut_turn_idx(num_assistant_messages, ratio)
        prefix_messages, continuation_messages = main_builder.split_messages(messages, cut_turn_idx)
        prefix_actions, diagnostics, details = collect_sampling_prefix_diagnostics(prefix_messages)
        rows.append(
            {
                "sample_uid": row["sample_uid"],
                "item_id": row["item_id"],
                "sample_idx": int(row["sample_idx"]),
                "task_id": int(row["task_id"]),
                "goal": row.get("goal"),
                "strategy": strategy,
                "cut_turn_idx": int(cut_turn_idx),
                "cut_relative_position_q": float(cut_q),
                "num_assistant_messages_total": int(num_assistant_messages),
                "prefix_messages": prefix_messages,
                "continuation_messages": continuation_messages,
                "prefix_actions": prefix_actions,
                "sampling_parser": "online_loose",
                "sampling_prefix_diagnostics": details,
                **diagnostics,
            }
        )
    return rows


def rebuild_entropy_candidate_rows_sampling(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rebuilt_rows: List[Dict[str, Any]] = []
    for row in rows:
        prefix_messages = main_builder.normalize_messages(row["prefix_messages"])
        continuation_messages = main_builder.normalize_messages(row["continuation_messages"])
        prefix_actions, diagnostics, details = collect_sampling_prefix_diagnostics(prefix_messages)
        rebuilt = dict(row)
        rebuilt["prefix_messages"] = prefix_messages
        rebuilt["continuation_messages"] = continuation_messages
        rebuilt["prefix_actions"] = prefix_actions
        rebuilt["sampling_parser"] = "online_loose"
        rebuilt["sampling_prefix_diagnostics"] = details
        rebuilt.update(diagnostics)
        rebuilt_rows.append(rebuilt)
    return rebuilt_rows


def compute_prompt_assistant_token_spans_sampling(prompt: List[Dict[str, Any]], tokenizer) -> Tuple[str, List[Dict[str, Any]]]:
    full_text = main_builder.build_prompt_text(prompt, tokenizer)
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
        if msg.get("role") != "assistant":
            continue

        action = extract_textcraft_action_loose(msg.get("content", ""))
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


def materialize_prefix_sidecars_sampling(
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

    unique_prompts: Dict[str, Dict[str, Any]] = {}
    for row in prefix_prompt_rows:
        prompt = main_builder.normalize_prompt_messages(row["prompt"])
        key = main_builder.prompt_key(prompt)
        if key in unique_prompts:
            continue
        prompt_text, prompt_spans = compute_prompt_assistant_token_spans_sampling(prompt, tokenizer)
        _, prompt_token_length = main_builder.tokenize_conversations(tokenizer, prompt)
        unique_prompts[key] = {
            "prompt_text": prompt_text,
            "prompt_spans": prompt_spans,
            "prompt_token_length": prompt_token_length,
        }

    zero_prefix_prompt_count = 0
    prompt_items: List[Tuple[str, Dict[str, Any]]] = []
    for key, prepared in unique_prompts.items():
        if prepared["prompt_spans"]:
            prompt_items.append((key, prepared))
        else:
            prepared["assistant_prefix_old_log_probs"] = []
            prepared["prefix_mask"] = []
            prepared["assistant_prefix_span"] = {"start": 0, "end": 0}
            prepared["prefix_token_count"] = 0
            zero_prefix_prompt_count += 1

    if prompt_items:
        torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch_dtype,
        )
        model.to(device)
        model.eval()

        prompt_items = sorted(
            prompt_items,
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
            f"batch_size={max(1, batch_size)} max_batch_prompt_tokens={max_batch_prompt_tokens} "
            f"zero_prefix_prompts={zero_prefix_prompt_count}",
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
            batch_old_logprobs = main_builder.compute_batch_old_logprobs(model, tokenizer, texts, device)
            for (key, prepared), sequence_old_logprobs in zip(chunk, batch_old_logprobs, strict=True):
                prefix_old_logprobs, prefix_mask, prefix_span, prefix_token_count = (
                    main_builder.build_prefix_window_from_prompt_spans(
                        prompt_spans=prepared["prompt_spans"],
                        sequence_old_logprobs=sequence_old_logprobs,
                    )
                )
                main_builder.validate_rebuilt_sample(
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
            if processed_unique_prompts == total_unique_prompts or processed_unique_prompts % max(1, progress_every) == 0:
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
    else:
        print(f"[materialize] all prefix prompts have zero extracted actions; zero_prefix_prompts={zero_prefix_prompt_count}")

    for row in prefix_prompt_rows:
        key = main_builder.prompt_key(main_builder.normalize_prompt_messages(row["prompt"]))
        prepared = unique_prompts[key]
        row["assistant_prefix_old_log_probs"] = prepared["assistant_prefix_old_log_probs"]
        row["prefix_mask"] = prepared["prefix_mask"]
        row["assistant_prefix_span"] = prepared["assistant_prefix_span"]
        row["prefix_token_count"] = prepared["prefix_token_count"]
        row["source_oldlogprob_model_path"] = model_path


def dataset_flag_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sample_uids = sorted({str(row["sample_uid"]) for row in rows})
    extraction_mode_counts: Dict[str, int] = {}
    for row in rows:
        for mode, count in dict(row.get("sampling_extraction_mode_counts", {}) or {}).items():
            extraction_mode_counts[str(mode)] = extraction_mode_counts.get(str(mode), 0) + int(count)
    return {
        "flagged_rows": int(len(rows)),
        "flagged_sample_uid_count": int(len(sample_uids)),
        "flagged_sample_uid_examples": sample_uids[:20],
        "rows_with_empty_sampling_prefix_actions": int(
            sum(bool(row.get("sampling_has_empty_prefix_actions")) for row in rows)
        ),
        "rows_with_missing_action_turns": int(
            sum(int(row.get("sampling_missing_action_turn_count", 0)) > 0 for row in rows)
        ),
        "rows_with_multi_action_tag_turns": int(
            sum(int(row.get("sampling_multi_action_tag_turn_count", 0)) > 0 for row in rows)
        ),
        "sampling_extraction_mode_counts": dict(sorted(extraction_mode_counts.items())),
    }


def collect_flagged_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flagged: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("is_raw_variant"):
            continue
        if (
            bool(row.get("sampling_has_empty_prefix_actions"))
            or int(row.get("sampling_missing_action_turn_count", 0)) > 0
            or int(row.get("sampling_multi_action_tag_turn_count", 0)) > 0
        ):
            flagged.append(dict(row))
    return flagged


def save_flagged_artifacts(flag_root: Path, dataset_name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    flag_root.mkdir(parents=True, exist_ok=True)
    parquet_path = flag_root / f"{dataset_name}.parser_flagged.parquet"
    manifest_path = flag_root / f"{dataset_name}.parser_flagged.manifest.json"
    df = pd.DataFrame(normalize_records(rows))
    df.to_parquet(parquet_path, index=False)
    manifest = dataset_flag_summary(rows)
    manifest["output_path"] = str(parquet_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def build_complete_split_datasets(
    teacher_rows: Sequence[Dict[str, Any]],
    entropy_df: pd.DataFrame,
    raw_base_rows: Dict[int, Dict[str, Any]],
    *,
    system_prompt: str,
    model_path: str,
    selected_datasets: Sequence[str],
    audit_root: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    raw_top3_rows = entropy_df[
        entropy_df["strategy"] == "entropy_raw_topk_interaction_assistant_k3"
    ].to_dict(orient="records")
    change_top3_rows = entropy_df[
        entropy_df["strategy"] == "entropy_change_topk_w11_interaction_assistant_k3"
    ].to_dict(orient="records")

    datasets_all: Dict[str, List[Dict[str, Any]]] = {
        "main_fixed_gp1": build_fixed_complete_split_rows(
            teacher_rows,
            raw_base_rows,
            main_dataset="main_fixed_gp1",
            system_prompt=system_prompt,
            ratios=(0.1, 0.3, 0.5),
            model_path=model_path,
        ),
        "main_fixed_gp2": build_fixed_complete_split_rows(
            teacher_rows,
            raw_base_rows,
            main_dataset="main_fixed_gp2",
            system_prompt=system_prompt,
            ratios=(0.25, 0.5, 0.7),
            model_path=model_path,
        ),
        "main_raw_top3": build_entropy_complete_split_rows(
            teacher_rows,
            raw_top3_rows,
            raw_base_rows,
            main_dataset="main_raw_top3",
            system_prompt=system_prompt,
            model_path=model_path,
        ),
        "main_change_top3_w11": build_entropy_complete_split_rows(
            teacher_rows,
            change_top3_rows,
            raw_base_rows,
            main_dataset="main_change_top3_w11",
            system_prompt=system_prompt,
            model_path=model_path,
        ),
    }

    datasets: Dict[str, List[Dict[str, Any]]] = {}
    manifests: Dict[str, Dict[str, Any]] = {}
    for name in selected_datasets:
        rows = datasets_all[name]
        datasets[name] = rows
        manifests[name] = save_flagged_artifacts(audit_root, name, collect_flagged_rows(rows))
    return datasets, manifests


def build_fixed_replay_prefix_rows(
    teacher_rows: Sequence[Dict[str, Any]],
    raw_base_rows: Dict[int, Dict[str, Any]],
    *,
    main_dataset: str,
    ratios: Sequence[float],
    system_prompt: str,
    model_path: str,
    server: str,
    timeout: float,
    concurrency: int,
    audit_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        candidate_rows.extend(build_fixed_replay_candidate_rows(teacher_rows, ratio=ratio))

    flagged_candidates = collect_flagged_rows(candidate_rows)
    flagged_manifest = save_flagged_artifacts(audit_root, main_dataset, flagged_candidates)

    replay_df = fullflow_builder.replay_validate_rows(
        candidate_rows,
        server=server,
        timeout=timeout,
        concurrency=concurrency,
    )

    ratio_records: Dict[str, List[Dict[str, Any]]] = {}
    strategy_manifest: Dict[str, Any] = {}
    for strategy, strategy_df in replay_df.groupby("strategy", sort=True):
        validated_df = strategy_df[strategy_df["replay_category"] == "validated"].copy()
        unverifiable_df = strategy_df[strategy_df["replay_category"] == "unverifiable"].copy()

        usable_rows: List[Dict[str, Any]] = []
        for row in unverifiable_df.to_dict(orient="records"):
            bucket, subreason = fullflow_builder.classify_unverifiable_bucket(row)
            row["unverifiable_bucket"] = bucket
            row["unverifiable_subreason"] = subreason
            if bucket == "usable_state_feedback":
                usable_rows.append(row)

        trainable_df = pd.concat([validated_df, pd.DataFrame(usable_rows)], ignore_index=True)
        if not trainable_df.empty:
            trainable_df = trainable_df.sort_values(["task_id", "sample_idx"]).reset_index(drop=True)

        ratio_records[strategy] = []
        for row in trainable_df.to_dict(orient="records"):
            task_id = int(row["task_id"])
            base_row = raw_base_rows[task_id]
            prefix_messages = main_builder.normalize_messages(row["prefix_messages"])
            continuation_messages = main_builder.normalize_messages(row["continuation_messages"])
            prefix_actions = list(row.get("prefix_actions", []))
            prompt = build_training_prompt_sampling(
                prefix_messages=prefix_messages,
                continuation_messages=continuation_messages,
                system_prompt=system_prompt,
            )
            extra_info = deepcopy(base_row["extra_info"])
            interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})
            interaction_kwargs.update(
                main_builder.build_interaction_kwargs(
                    task_id=task_id,
                    goal=row.get("goal"),
                    prefix_actions=prefix_actions,
                )
            )
            extra_info["interaction_kwargs"] = interaction_kwargs
            extra_info["sample_uid"] = row["sample_uid"]
            extra_info["main_dataset"] = main_dataset
            extra_info["variant_label"] = str(row["strategy"])

            ratio_records[strategy].append(
                {
                    **row,
                    "data_source": "textcraft",
                    "ability": "crafting",
                    "prompt": prompt,
                    "reward_model": {"ground_truth": "", "style": "interaction"},
                    "extra_info": extra_info,
                    "source_dataset": "main_prefix_new_main_prefix_replay",
                }
            )

        strategy_manifest[strategy] = {
            "candidate_rows": int(len(strategy_df)),
            "validated_rows": int(len(validated_df)),
            "usable_unverifiable_rows": int(len(usable_rows)),
            "trainable_rows": int(len(trainable_df)),
            "unique_sample_uid": int(trainable_df["sample_uid"].nunique()) if not trainable_df.empty else 0,
            "replay_category_counts": strategy_df["replay_category"].value_counts().sort_index().to_dict(),
        }

    rows = fullflow_builder.build_fixed_fullflow_dataset(
        main_dataset=main_dataset,
        ratio_records=ratio_records,
        teacher_rows=teacher_rows,
        ratios=ratios,
        raw_base_rows=raw_base_rows,
        model_path=model_path,
    )
    manifest = dict(flagged_manifest)
    manifest["strategy_manifest"] = strategy_manifest
    return rows, manifest


def build_entropy_replay_prefix_rows(
    teacher_rows: Sequence[Dict[str, Any]],
    entropy_rows: Sequence[Dict[str, Any]],
    raw_base_rows: Dict[int, Dict[str, Any]],
    *,
    main_dataset: str,
    strategy: str,
    system_prompt: str,
    model_path: str,
    server: str,
    timeout: float,
    concurrency: int,
    audit_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rebuilt_candidates = rebuild_entropy_candidate_rows_sampling(entropy_rows)
    flagged_manifest = save_flagged_artifacts(audit_root, main_dataset, collect_flagged_rows(rebuilt_candidates))

    replay_df = fullflow_builder.replay_validate_rows(
        rebuilt_candidates,
        server=server,
        timeout=timeout,
        concurrency=concurrency,
    )
    validated_df = replay_df[replay_df["replay_category"] == "validated"].copy()

    teacher_map = {str(row["sample_uid"]): row for row in teacher_rows}
    prefix_rows: List[Dict[str, Any]] = []
    for row in validated_df.to_dict(orient="records"):
        sample_uid = str(row["sample_uid"])
        teacher_row = teacher_map[sample_uid]
        task_id = int(row["task_id"])
        base_row = raw_base_rows[task_id]
        prefix_messages = main_builder.normalize_messages(row["prefix_messages"])
        continuation_messages = main_builder.normalize_messages(row["continuation_messages"])
        prefix_actions = list(row.get("prefix_actions", []))
        prompt = build_training_prompt_sampling(
            prefix_messages=prefix_messages,
            continuation_messages=continuation_messages,
            system_prompt=system_prompt,
        )
        extra_info = deepcopy(base_row["extra_info"])
        interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})
        interaction_kwargs.update(
            main_builder.build_interaction_kwargs(
                task_id=task_id,
                goal=row.get("goal"),
                prefix_actions=prefix_actions,
            )
        )
        extra_info["interaction_kwargs"] = interaction_kwargs
        extra_info["sample_uid"] = sample_uid
        extra_info["main_dataset"] = main_dataset
        extra_info["variant_label"] = f"rank{int(row['candidate_rank'])}"

        prefix_rows.append(
            {
                "record_uid": f"{main_dataset}::{sample_uid}::rank{int(row['candidate_rank'])}",
                "main_dataset": main_dataset,
                "variant_label": f"rank{int(row['candidate_rank'])}",
                "is_raw_variant": False,
                "sample_uid": sample_uid,
                "item_id": row["item_id"],
                "sample_idx": int(row["sample_idx"]),
                "task_id": task_id,
                "goal": row.get("goal"),
                "strategy": strategy,
                "data_source": base_row["data_source"],
                "ability": base_row["ability"],
                "prompt": prompt,
                "reward_model": deepcopy(base_row["reward_model"]),
                "extra_info": extra_info,
                "prefix_messages": prefix_messages,
                "continuation_messages": continuation_messages,
                "prefix_actions": prefix_actions,
                "replay_category": row.get("replay_category", "validated"),
                "assistant_prefix_old_log_probs": None,
                "prefix_mask": None,
                "prefix_token_count": None,
                "assistant_prefix_span": None,
                "source_oldlogprob_model_path": None,
                "prefix_coordinate_system": "canonicalized_prompt",
                "candidate_uid": row.get("candidate_uid"),
                "candidate_rank": int(row["candidate_rank"]),
                "cut_turn_idx": int(row["cut_turn_idx"]),
                "cut_relative_position_q": float(row.get("cut_relative_position_q", 0.0)),
                "num_assistant_messages_total": int(teacher_row["num_assistant_messages"]),
                "selection_score": float(row.get("selection_score", 0.0)),
                "source_role": row.get("source_role"),
                "source_token_position": int(row.get("source_token_position", -1)),
                "source_dataset": "entropy_stage2_replay_rebuilt_sampling_faithful",
                "sampling_parser": row.get("sampling_parser", "online_loose"),
                "sampling_prefix_diagnostics": row.get("sampling_prefix_diagnostics", []),
                "sampling_prefix_action_count": int(row.get("sampling_prefix_action_count", len(prefix_actions))),
                "sampling_assistant_turn_count": int(row.get("sampling_assistant_turn_count", 0)),
                "sampling_non_warmup_assistant_turn_count": int(row.get("sampling_non_warmup_assistant_turn_count", 0)),
                "sampling_missing_action_turn_count": int(row.get("sampling_missing_action_turn_count", 0)),
                "sampling_multi_action_tag_turn_count": int(row.get("sampling_multi_action_tag_turn_count", 0)),
                "sampling_extraction_mode_counts": dict(row.get("sampling_extraction_mode_counts", {}) or {}),
                "sampling_has_empty_prefix_actions": bool(row.get("sampling_has_empty_prefix_actions", len(prefix_actions) == 0)),
            }
        )

    rows = prefix_rows[:]
    for teacher_row in teacher_rows:
        task_id = int(teacher_row["task_id"])
        rows.append(
            main_builder.build_raw_record(
                raw_base_rows[task_id],
                main_dataset=main_dataset,
                sample_uid=teacher_row["sample_uid"],
                item_id=teacher_row["item_id"],
                sample_idx=int(teacher_row["sample_idx"]),
                task_id=task_id,
                goal=teacher_row.get("goal"),
                model_path=model_path,
            )
        )

    manifest = dict(flagged_manifest)
    manifest.update(
        {
            "candidate_rows": int(len(replay_df)),
            "validated_rows": int(len(validated_df)),
            "replay_category_counts": replay_df["replay_category"].value_counts().sort_index().to_dict(),
        }
    )
    return rows, manifest


def build_replay_validated_datasets(
    teacher_rows: Sequence[Dict[str, Any]],
    entropy_df: pd.DataFrame,
    raw_base_rows: Dict[int, Dict[str, Any]],
    *,
    system_prompt: str,
    model_path: str,
    server: str,
    timeout: float,
    concurrency: int,
    audit_root: Path,
    selected_datasets: Sequence[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    manifests: Dict[str, Dict[str, Any]] = {}

    if "main_fixed_gp1" in selected_datasets:
        rows, manifest = build_fixed_replay_prefix_rows(
            teacher_rows,
            raw_base_rows,
            main_dataset="main_fixed_gp1_fullflow",
            ratios=(0.1, 0.3, 0.5),
            system_prompt=system_prompt,
            model_path=model_path,
            server=server,
            timeout=timeout,
            concurrency=concurrency,
            audit_root=audit_root,
        )
        datasets["main_fixed_gp1_fullflow"] = rows
        manifests["main_fixed_gp1_fullflow"] = manifest

    if "main_fixed_gp2" in selected_datasets:
        rows, manifest = build_fixed_replay_prefix_rows(
            teacher_rows,
            raw_base_rows,
            main_dataset="main_fixed_gp2_fullflow",
            ratios=(0.25, 0.5, 0.7),
            system_prompt=system_prompt,
            model_path=model_path,
            server=server,
            timeout=timeout,
            concurrency=concurrency,
            audit_root=audit_root,
        )
        datasets["main_fixed_gp2_fullflow"] = rows
        manifests["main_fixed_gp2_fullflow"] = manifest

    entropy_sample_uids = {str(row["sample_uid"]) for row in teacher_rows}
    entropy_df = entropy_df[entropy_df["sample_uid"].isin(entropy_sample_uids)].copy()

    if "main_raw_top3" in selected_datasets:
        rows, manifest = build_entropy_replay_prefix_rows(
            teacher_rows,
            entropy_df[entropy_df["strategy"] == "entropy_raw_topk_interaction_assistant_k3"].to_dict(orient="records"),
            raw_base_rows,
            main_dataset="main_raw_top3_fullflow",
            strategy="entropy_raw_topk_interaction_assistant_k3",
            system_prompt=system_prompt,
            model_path=model_path,
            server=server,
            timeout=timeout,
            concurrency=concurrency,
            audit_root=audit_root,
        )
        datasets["main_raw_top3_fullflow"] = rows
        manifests["main_raw_top3_fullflow"] = manifest

    if "main_change_top3_w11" in selected_datasets:
        rows, manifest = build_entropy_replay_prefix_rows(
            teacher_rows,
            entropy_df[
                entropy_df["strategy"] == "entropy_change_topk_w11_interaction_assistant_k3"
            ].to_dict(orient="records"),
            raw_base_rows,
            main_dataset="main_change_top3_w11_fullflow",
            strategy="entropy_change_topk_w11_interaction_assistant_k3",
            system_prompt=system_prompt,
            model_path=model_path,
            server=server,
            timeout=timeout,
            concurrency=concurrency,
            audit_root=audit_root,
        )
        datasets["main_change_top3_w11_fullflow"] = rows
        manifests["main_change_top3_w11_fullflow"] = manifest

    return datasets, manifests


def summarize_parquet(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return main_builder.summarize_dataset(df.to_dict(orient="records"))


def build_complete_split_readme(
    summaries: Dict[str, Dict[str, Any]],
    *,
    teacher_path: Path,
    entropy_candidates_path: Path,
    train_parquet_path: Path,
    model_path: str,
) -> str:
    lines = [
        "# TextCraft Main Prefix Complete Split (Sample Faithful)",
        "",
        "## Meaning",
        "- This directory keeps all sampled cutpoints and rebuilds them with the same loose action parser used by online TextCraft interaction.",
        "- Prefix rows are not pre-dropped for `multiple_action_tags` or `missing_action_tag`.",
        "- If the loose online parser still extracts zero prefix actions, the row is kept and receives a zero-length prefix sidecar.",
        "- Parser-risk rows are audited under `../audit/complete_split/` instead of being filtered out here.",
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
                f"- output_path: `{summary['output_path']}`",
            ]
        )
        if "rows_with_empty_sampling_prefix_actions" in summary:
            lines.append(
                f"- rows_with_empty_sampling_prefix_actions: `{summary['rows_with_empty_sampling_prefix_actions']}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "- This branch is sample-faithful rather than runtime-audit-strict.",
            "- `rows_with_multi_action_tag_in_prefix_messages` can remain non-zero here by design.",
            "- Zero-prefix rows are expected when the online parser would also fail to recover any prefix action from the stored assistant text.",
        ]
    )
    return "\n".join(lines)


def build_replay_validated_readme(
    summaries: Dict[str, Dict[str, Any]],
    *,
    teacher_path: Path,
    entropy_candidates_path: Path,
    train_parquet_path: Path,
    model_path: str,
    server: str,
) -> str:
    lines = [
        "# TextCraft Main Prefix Replay Validated (Sample Faithful)",
        "",
        "## Meaning",
        "- This directory reruns replay from sample-faithful candidate rows built with the same loose action parser used online.",
        "- Fixed-ratio datasets keep `validated + usable_state_feedback`; entropy datasets keep `validated` only, matching the legacy replay evidence policy.",
        "- Candidate rows are not dropped just because a prefix message contains multiple `Action:` tags.",
        "- Parser-risk candidates are audited under `../audit/replay_validated/` for later inspection.",
        "",
        "## Inputs",
        f"- teacher_normalized: `{teacher_path}`",
        f"- entropy stage2 candidates: `{entropy_candidates_path}`",
        f"- official raw train parquet: `{train_parquet_path}`",
        f"- prompt-space old-logprob model: `{model_path}`",
        f"- replay validation server: `{server}`",
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
                f"- output_path: `{summary['output_path']}`",
            ]
        )
        if "rows_with_empty_sampling_prefix_actions" in summary:
            lines.append(
                f"- rows_with_empty_sampling_prefix_actions: `{summary['rows_with_empty_sampling_prefix_actions']}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "- This branch keeps the legacy replay evidence policy but switches candidate reconstruction to the online loose parser.",
            "- Replay, not the parser prefilter, is the first hard filter here.",
        ]
    )
    return "\n".join(lines)


def build_audit_report(
    comparisons: Dict[str, Dict[str, Any]],
    *,
    output_root: Path,
) -> Tuple[str, Dict[str, Any]]:
    report = {
        "output_root": str(output_root),
        "comparisons": comparisons,
    }
    lines = [
        "# Sample-Faithful Main Prefix Audit",
        "",
        f"- Output root: `{output_root}`",
        "- Comparison baseline: existing `main_prefix/complete_split/` and `main_prefix/replay_validated/` datasets.",
        "- Parser rule: use the same loose extraction order as online TextCraft interaction (`[[...]]` last, then `Action:\\n`, then inline `Action:`).",
        "",
        "## Dataset Diffs",
    ]
    for dataset_name, comparison in comparisons.items():
        lines.extend(
            [
                f"### {dataset_name}",
                f"- legacy_rows: `{comparison.get('legacy_rows')}`",
                f"- rebuilt_rows: `{comparison.get('rebuilt_rows')}`",
                f"- row_delta: `{comparison.get('row_delta')}`",
                f"- legacy_empty_prefix_action_rows: `{comparison.get('legacy_empty_prefix_action_rows')}`",
                f"- rebuilt_empty_prefix_action_rows: `{comparison.get('rebuilt_empty_prefix_action_rows')}`",
                f"- legacy_rows_with_multi_action_tag_in_prefix_messages: `{comparison.get('legacy_rows_with_multi_action_tag_in_prefix_messages')}`",
                f"- rebuilt_rows_with_multi_action_tag_in_prefix_messages: `{comparison.get('rebuilt_rows_with_multi_action_tag_in_prefix_messages')}`",
                "",
            ]
        )
    return "\n".join(lines), report


def main() -> None:
    args = parse_args()
    selected_datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(selected_datasets) - set(DATASET_NAMES))
    if unknown:
        raise RuntimeError(f"Unknown base dataset names: {unknown}")

    if args.device == "auto":
        args.device = "cuda"

    teacher_df = pd.read_parquet(args.teacher_path)
    if args.max_samples is not None:
        teacher_df = teacher_df.head(args.max_samples).copy()
    teacher_rows = teacher_df.to_dict(orient="records")
    if not teacher_rows:
        raise RuntimeError(f"Teacher parquet is empty: {args.teacher_path}")

    sample_uids = set(teacher_df["sample_uid"].tolist())
    entropy_df = pd.read_parquet(args.entropy_candidates_path)
    entropy_df = entropy_df[entropy_df["sample_uid"].isin(sample_uids)].copy()

    raw_base_rows = main_builder.load_raw_base_rows(args.train_parquet_path)
    system_prompt = main_builder.load_reference_system_prompt(args.train_parquet_path)

    complete_split_root = args.output_root / "complete_split"
    replay_validated_root = args.output_root / "replay_validated"
    audit_complete_root = args.output_root / "audit" / "complete_split"
    audit_replay_root = args.output_root / "audit" / "replay_validated"

    complete_split_datasets, complete_split_manifests = build_complete_split_datasets(
        teacher_rows,
        entropy_df,
        raw_base_rows,
        system_prompt=system_prompt,
        model_path=args.model_path,
        selected_datasets=selected_datasets,
        audit_root=audit_complete_root,
    )
    replay_validated_datasets, replay_validated_manifests = build_replay_validated_datasets(
        teacher_rows,
        entropy_df,
        raw_base_rows,
        system_prompt=system_prompt,
        model_path=args.model_path,
        server=args.server,
        timeout=args.request_timeout,
        concurrency=args.replay_concurrency,
        audit_root=audit_replay_root,
        selected_datasets=selected_datasets,
    )

    all_rows: List[Dict[str, Any]] = []
    for rows in complete_split_datasets.values():
        all_rows.extend(rows)
    for rows in replay_validated_datasets.values():
        all_rows.extend(rows)
    materialize_prefix_sidecars_sampling(
        all_rows,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_batch_prompt_tokens=args.max_batch_prompt_tokens,
        progress_every=args.progress_every,
    )

    complete_split_summaries: Dict[str, Dict[str, Any]] = {}
    complete_split_root.mkdir(parents=True, exist_ok=True)
    for name, rows in complete_split_datasets.items():
        output_path = complete_split_root / f"{name}.parquet"
        manifest_path = complete_split_root / f"{name}.manifest.json"
        summary = main_builder.write_dataset(
            output_path,
            rows,
            manifest_path,
            extra_summary=complete_split_manifests.get(name),
        )
        complete_split_summaries[name] = summary

    replay_validated_summaries: Dict[str, Dict[str, Any]] = {}
    replay_validated_root.mkdir(parents=True, exist_ok=True)
    for name, rows in replay_validated_datasets.items():
        output_path = replay_validated_root / f"{name}.parquet"
        manifest_path = replay_validated_root / f"{name}.manifest.json"
        summary = main_builder.write_dataset(
            output_path,
            rows,
            manifest_path,
            extra_summary=replay_validated_manifests.get(name),
        )
        replay_validated_summaries[name] = summary

    (complete_split_root / "README.md").write_text(
        build_complete_split_readme(
            complete_split_summaries,
            teacher_path=args.teacher_path,
            entropy_candidates_path=args.entropy_candidates_path,
            train_parquet_path=args.train_parquet_path,
            model_path=args.model_path,
        )
        + "\n",
        encoding="utf-8",
    )
    (replay_validated_root / "README.md").write_text(
        build_replay_validated_readme(
            replay_validated_summaries,
            teacher_path=args.teacher_path,
            entropy_candidates_path=args.entropy_candidates_path,
            train_parquet_path=args.train_parquet_path,
            model_path=args.model_path,
            server=args.server,
        )
        + "\n",
        encoding="utf-8",
    )

    comparisons: Dict[str, Dict[str, Any]] = {}
    for dataset_name in selected_datasets:
        legacy_complete = summarize_parquet(args.legacy_complete_split_root / LEGACY_NAME_TO_FILENAME[dataset_name])
        rebuilt_complete = summarize_parquet(complete_split_root / LEGACY_NAME_TO_FILENAME[dataset_name])
        comparisons[dataset_name] = {
            "legacy_rows": legacy_complete.get("rows") if legacy_complete else None,
            "rebuilt_rows": rebuilt_complete.get("rows") if rebuilt_complete else None,
            "row_delta": (
                int(rebuilt_complete["rows"]) - int(legacy_complete["rows"])
                if legacy_complete and rebuilt_complete
                else None
            ),
            "legacy_empty_prefix_action_rows": (
                legacy_complete.get("empty_prefix_action_rows") if legacy_complete else None
            ),
            "rebuilt_empty_prefix_action_rows": (
                rebuilt_complete.get("empty_prefix_action_rows") if rebuilt_complete else None
            ),
            "legacy_rows_with_multi_action_tag_in_prefix_messages": (
                legacy_complete.get("rows_with_multi_action_tag_in_prefix_messages") if legacy_complete else None
            ),
            "rebuilt_rows_with_multi_action_tag_in_prefix_messages": (
                rebuilt_complete.get("rows_with_multi_action_tag_in_prefix_messages") if rebuilt_complete else None
            ),
        }

        fullflow_name = f"{dataset_name}_fullflow"
        legacy_fullflow = summarize_parquet(
            args.legacy_replay_validated_root / LEGACY_NAME_TO_FILENAME[fullflow_name]
        )
        rebuilt_fullflow = summarize_parquet(replay_validated_root / LEGACY_NAME_TO_FILENAME[fullflow_name])
        comparisons[fullflow_name] = {
            "legacy_rows": legacy_fullflow.get("rows") if legacy_fullflow else None,
            "rebuilt_rows": rebuilt_fullflow.get("rows") if rebuilt_fullflow else None,
            "row_delta": (
                int(rebuilt_fullflow["rows"]) - int(legacy_fullflow["rows"])
                if legacy_fullflow and rebuilt_fullflow
                else None
            ),
            "legacy_empty_prefix_action_rows": (
                legacy_fullflow.get("empty_prefix_action_rows") if legacy_fullflow else None
            ),
            "rebuilt_empty_prefix_action_rows": (
                rebuilt_fullflow.get("empty_prefix_action_rows") if rebuilt_fullflow else None
            ),
            "legacy_rows_with_multi_action_tag_in_prefix_messages": (
                legacy_fullflow.get("rows_with_multi_action_tag_in_prefix_messages") if legacy_fullflow else None
            ),
            "rebuilt_rows_with_multi_action_tag_in_prefix_messages": (
                rebuilt_fullflow.get("rows_with_multi_action_tag_in_prefix_messages") if rebuilt_fullflow else None
            ),
        }

    audit_report_md, audit_report_json = build_audit_report(comparisons, output_root=args.output_root)
    (args.output_root / "audit_report.md").write_text(audit_report_md + "\n", encoding="utf-8")
    (args.output_root / "audit_report.json").write_text(
        json.dumps(audit_report_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    root_readme = [
        "# TextCraft Main Prefix New Main Prefix",
        "",
        "## Contents",
        f"- complete_split: `{complete_split_root}`",
        f"- replay_validated: `{replay_validated_root}`",
        f"- audit: `{args.output_root / 'audit'}`",
        f"- audit_report.md: `{args.output_root / 'audit_report.md'}`",
        "",
        "## Meaning",
        "- This branch keeps sampled cutpoints first, then lets replay decide which prefixes remain usable.",
        "- Action extraction reuses the same loose parser as online TextCraft interaction instead of the runtime-audit strict single-`Action:` rule.",
        "- Parser-risk rows are audited instead of being pre-dropped from `complete_split/`.",
    ]
    (args.output_root / "README.md").write_text("\n".join(root_readme) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "complete_split": complete_split_summaries,
                "replay_validated": replay_validated_summaries,
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
