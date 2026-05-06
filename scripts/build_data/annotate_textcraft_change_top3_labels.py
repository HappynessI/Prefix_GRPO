#!/usr/bin/env python3
"""Add lightweight analysis/training labels to the current TextCraft change_top3 dataset.

This script does not rebuild the dataset. It reads the existing parquet,
derives row-level labels from the current prompt/prefix/continuation schema,
and writes a new labeled parquet sidecar plus a small JSON summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_PATH = Path(
    "/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/new_main_prefix/replay_validated/"
    "main_change_top3_w11_fullflow.parquet"
)
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_PATH.with_name("main_change_top3_w11_fullflow_labeled.parquet")
LABEL_VERSION = "20260425_change_top3_labels_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to <output>.label_summary.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output parquet/summary.",
    )
    return parser.parse_args()


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return list(value)
    return [value]


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def classify_cut_stage(is_raw_variant: bool, cut_progress_q: float | None) -> str:
    if is_raw_variant:
        return "raw"
    if cut_progress_q is None:
        return "unknown"
    if cut_progress_q <= (1.0 / 3.0):
        return "early"
    if cut_progress_q <= (2.0 / 3.0):
        return "mid"
    return "late"


def annotate_row(row: pd.Series) -> dict[str, Any]:
    is_raw_variant = bool(row.get("is_raw_variant", False))
    continuation_messages = to_list(row.get("continuation_messages"))
    continuation_roles = [msg.get("role") for msg in continuation_messages if isinstance(msg, dict)]

    continuation_message_count = len(continuation_messages)
    continuation_user_message_count = sum(role == "user" for role in continuation_roles)
    continuation_assistant_turn_count = sum(role == "assistant" for role in continuation_roles)

    cut_progress_q = maybe_float(row.get("cut_relative_position_q"))
    cut_stage = classify_cut_stage(is_raw_variant, cut_progress_q)

    if is_raw_variant:
        remaining_assistant_turns = None
        remaining_env_steps = None
        done_after_replay = False
        is_terminal_like = False
        training_row_type = "raw"
    else:
        remaining_assistant_turns = int(continuation_assistant_turn_count)
        # continuation_messages starts with the cut-state user observation, so the actual
        # future environment steps equal the remaining user messages after that observation.
        remaining_env_steps = max(int(continuation_user_message_count) - 1, 0)
        done_after_replay = remaining_assistant_turns == 0 and continuation_user_message_count == 1
        is_terminal_like = bool(done_after_replay)
        training_row_type = "prefix_terminal_like" if is_terminal_like else "prefix_non_terminal"

    return {
        "label_version": LABEL_VERSION,
        "training_row_type": training_row_type,
        "continuation_message_count": int(continuation_message_count),
        "continuation_user_message_count": int(continuation_user_message_count),
        "continuation_assistant_turn_count": int(continuation_assistant_turn_count),
        "remaining_assistant_turns": remaining_assistant_turns,
        "remaining_env_steps": remaining_env_steps,
        "cut_progress_q": cut_progress_q,
        "cut_stage": cut_stage,
        "done_after_replay": bool(done_after_replay),
        "is_terminal_like": bool(is_terminal_like),
        "terminal_anchor_candidate": bool(is_terminal_like),
    }


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    prefix_df = df[df["training_row_type"].ne("raw")].copy()
    terminal_df = df[df["is_terminal_like"].eq(True)].copy()
    return {
        "label_version": LABEL_VERSION,
        "rows": int(len(df)),
        "unique_sample_uid": int(df["sample_uid"].nunique()) if "sample_uid" in df.columns else None,
        "training_row_type_counts": df["training_row_type"].value_counts().sort_index().to_dict(),
        "cut_stage_counts": df["cut_stage"].value_counts().sort_index().to_dict(),
        "terminal_like_rows": int(len(terminal_df)),
        "terminal_like_unique_sample_uid": int(terminal_df["sample_uid"].nunique()) if len(terminal_df) else 0,
        "terminal_like_variant_counts": (
            terminal_df["variant_label"].value_counts().sort_index().to_dict() if "variant_label" in terminal_df else {}
        ),
        "prefix_remaining_assistant_turns": (
            prefix_df["remaining_assistant_turns"].dropna().astype(int).value_counts().sort_index().to_dict()
        ),
        "prefix_remaining_env_steps": (
            prefix_df["remaining_env_steps"].dropna().astype(int).value_counts().sort_index().to_dict()
        ),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()
    summary_path = (args.summary_path or output_path.with_suffix(".label_summary.json")).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    if not args.overwrite:
        for path in (output_path, summary_path):
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")

    df = pd.read_parquet(input_path)
    label_df = pd.DataFrame([annotate_row(row) for _, row in df.iterrows()])
    merged_df = pd.concat([df.reset_index(drop=True), label_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=False)

    summary = build_summary(merged_df)
    summary["input_path"] = str(input_path)
    summary["output_path"] = str(output_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
