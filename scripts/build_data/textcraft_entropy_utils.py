#!/usr/bin/env python3
"""Shared helpers for offline TextCraft entropy analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch


START_TAG_RE = re.compile(r"<\|im_start\|>(user|assistant|tool|system)")
END_TAG_RE = re.compile(r"<\|im_end\|>")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def make_sample_uid(item_id: str, sample_idx: int) -> str:
    return f"{item_id}__{sample_idx}"


def parse_task_id(item_id: str) -> int:
    if not item_id.startswith("textcraft_"):
        raise ValueError(f"Unexpected item_id format: {item_id}")
    return int(item_id.split("_", 1)[1])


def render_conversations_text(tokenizer, conversations: List[Dict[str, Any]], enable_thinking: bool = False) -> str:
    if enable_thinking:
        text = ""
        for msg in conversations:
            role = msg["role"]
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return text

    return tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=False,
        tokenize=False,
    )


def tokenize_conversations(tokenizer, conversations: List[Dict[str, Any]], enable_thinking: bool = False) -> torch.Tensor:
    text = render_conversations_text(tokenizer, conversations, enable_thinking=enable_thinking)
    tokens = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
    )
    return tokens.input_ids[0]


def compute_message_spans(
    tokenizer,
    conversations: List[Dict[str, Any]],
    enable_thinking: bool = False,
) -> List[Dict[str, Any]]:
    if not conversations:
        return []

    full_text = render_conversations_text(tokenizer, conversations, enable_thinking=enable_thinking)
    result = tokenizer(
        full_text,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = result.offset_mapping[0]

    start_matches = list(START_TAG_RE.finditer(full_text))
    end_matches = list(END_TAG_RE.finditer(full_text))
    if len(start_matches) != len(conversations) or len(end_matches) != len(conversations):
        raise ValueError(
            "Conversation tag count does not match message count: "
            f"{len(start_matches)} starts, {len(end_matches)} ends, {len(conversations)} messages"
        )

    role_counters: Dict[str, int] = {}
    message_spans: List[Dict[str, Any]] = []
    for message_index, msg in enumerate(conversations):
        role = msg.get("role", "")
        role_counters[role] = role_counters.get(role, 0) + 1
        role_turn_idx = role_counters[role]

        start_char = start_matches[message_index].end()
        end_char = end_matches[message_index].end()

        token_start = None
        token_end = None
        for token_index, (char_start, char_end) in enumerate(offset_mapping):
            if char_start is None:
                continue
            if char_start < end_char and char_end > start_char:
                if token_start is None:
                    token_start = token_index
                token_end = token_index + 1

        if token_start is None or token_end is None:
            raise ValueError(f"Could not map message {message_index} ({role}) to token span")

        message_spans.append(
            {
                "message_index": message_index,
                "role": role,
                "role_turn_idx": role_turn_idx,
                "token_start": token_start,
                "token_end": token_end,
                "entropy_start": max(0, token_start - 1),
                "entropy_end": max(0, token_end - 1),
            }
        )

    return message_spans


def select_role_spans(message_spans: List[Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    return [span for span in message_spans if span.get("role") == role]


def compute_token_entropy(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            use_cache=False,
        )
        logits = outputs.logits[0]
        if logits.shape[0] <= 1:
            return logits.new_zeros((0,), dtype=torch.float32)
        logits = logits[:-1].to(torch.float32)
        probs = logits.softmax(dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
        return entropy


def _interval_distance(token_position: int, token_start: int, token_end: int) -> int:
    if token_start <= token_position < token_end:
        return 0
    if token_position < token_start:
        return token_start - token_position
    return token_position - token_end + 1


def map_token_position_to_assistant_turn(
    token_position: int,
    assistant_turn_spans: List[Dict[str, Any]],
    policy: str = "previous",
) -> Optional[Dict[str, Any]]:
    if not assistant_turn_spans:
        return None

    for span in assistant_turn_spans:
        if span["token_start"] <= token_position < span["token_end"]:
            result = dict(span)
            result["mapping_policy"] = "contained_assistant_turn"
            result["mapping_distance"] = 0
            return result

    if policy == "previous":
        previous = [span for span in assistant_turn_spans if span["token_end"] <= token_position]
        if previous:
            chosen = dict(previous[-1])
            chosen["mapping_policy"] = "previous_assistant_turn"
            chosen["mapping_distance"] = _interval_distance(
                token_position, chosen["token_start"], chosen["token_end"]
            )
            return chosen

        chosen = dict(assistant_turn_spans[0])
        chosen["mapping_policy"] = "first_assistant_fallback"
        chosen["mapping_distance"] = _interval_distance(
            token_position, chosen["token_start"], chosen["token_end"]
        )
        return chosen

    if policy == "nearest":
        chosen = min(
            assistant_turn_spans,
            key=lambda span: (
                _interval_distance(token_position, span["token_start"], span["token_end"]),
                span["turn_idx"],
            ),
        )
        result = dict(chosen)
        result["mapping_policy"] = "nearest_assistant_turn"
        result["mapping_distance"] = _interval_distance(
            token_position, result["token_start"], result["token_end"]
        )
        return result

    raise ValueError(f"Unsupported mapping policy: {policy}")
