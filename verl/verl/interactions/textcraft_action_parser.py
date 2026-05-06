"""Shared TextCraft action parsing helpers used by online interaction and offline builders."""

from __future__ import annotations

import re
from typing import Optional, Tuple


CHAT_TEMPLATE_ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\s*\n?", re.IGNORECASE)
CHAT_TEMPLATE_END_RE = re.compile(r"<\|im_end\|>")
BOXED_ACTION_RE = re.compile(r"\[\[\s*(.*?)\s*\]\]", re.DOTALL)
ACTION_NEWLINE_RE = re.compile(r"Action:\s*\n\s*(.+?)(?:\n|$)", re.DOTALL)
ACTION_INLINE_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.DOTALL)


def strip_textcraft_chat_template_markers(text: str) -> str:
    normalized = str(text or "").strip()
    normalized = CHAT_TEMPLATE_ASSISTANT_RE.sub("", normalized)
    normalized = CHAT_TEMPLATE_END_RE.sub("", normalized)
    return normalized


def _normalize_action_text(action: str) -> str:
    return " ".join(str(action or "").split()).strip()


def extract_textcraft_action_loose_with_mode(text: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = strip_textcraft_chat_template_markers(text)

    boxed_matches = BOXED_ACTION_RE.findall(normalized)
    if boxed_matches:
        action = _normalize_action_text(boxed_matches[-1])
        if action:
            return action, "boxed_last"

    newline_match = ACTION_NEWLINE_RE.search(normalized)
    if newline_match:
        action = _normalize_action_text(newline_match.group(1))
        if action:
            return action, "newline_first"

    inline_match = ACTION_INLINE_RE.search(normalized)
    if inline_match:
        action = _normalize_action_text(inline_match.group(1))
        if action:
            return action, "inline_first"

    return None, None


def extract_textcraft_action_loose(text: str) -> Optional[str]:
    action, _ = extract_textcraft_action_loose_with_mode(text)
    return action
