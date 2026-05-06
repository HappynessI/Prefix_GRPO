#!/usr/bin/env python3
"""Build replay-filtered TextCraft main-prefix datasets aligned with the full legacy data flow."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

import build_textcraft_main_prefix_datasets as main_builder


DEFAULT_OUTPUT_ROOT = Path("/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/replay_validated")
DEFAULT_SERVER = "http://127.0.0.1:36001"

DEFAULT_EXISTING_FIXED_STAGE4 = {
    "fixed_ratio_0p1": Path(
        "/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage4_canonicalized/fixed_ratio_0p1_stage4_trainable_canonicalized.parquet"
    ),
    "fixed_ratio_0p3": Path(
        "/Data/wyh/datasets/Verl-Data/train/textcraft/new_prefix_rl/stage4_canonicalized/fixed_ratio_0p3_stage4_trainable_canonicalized.parquet"
    ),
}

DEFAULT_ENTROPY_AUDITED = {
    "main_raw_top3_fullflow": Path(
        "/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage7_audit_release/textcraft_prefix_entropy_raw_topk_interaction_assistant_k3_step200.audited.parquet"
    ),
    "main_change_top3_w11_fullflow": Path(
        "/Data/wyh/datasets/Verl-Data/train/textcraft/entropy_based_prefix/stage7_audit_release/textcraft_prefix_entropy_change_topk_w11_interaction_assistant_k3_step200.audited.parquet"
    ),
}

STATE_FEEDBACK_PREFIXES = (
    "Could not find",
    "Could not find enough items",
    "Could not find a valid recipe",
)

FORMAT_ERROR_PATTERNS = (
    "Only one 'Action' is allowed",
    "Error:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-path", type=Path, default=main_builder.DEFAULT_TEACHER_PATH)
    parser.add_argument("--train-parquet-path", type=Path, default=main_builder.DEFAULT_TRAIN_PARQUET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=str, default=main_builder.DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batch-prompt-tokens", type=int, default=3000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--replay-concurrency", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--datasets",
        type=str,
        default="main_fixed_gp1_fullflow,main_fixed_gp2_fullflow,main_raw_top3_fullflow,main_change_top3_w11_fullflow",
    )
    return parser.parse_args()


def normalize_text(text: Optional[str]) -> str:
    return " ".join((text or "").split()).strip()


def is_state_feedback(text: str) -> bool:
    return text.startswith(STATE_FEEDBACK_PREFIXES)


def is_format_error(text: str) -> bool:
    return any(pattern in text for pattern in FORMAT_ERROR_PATTERNS)


def first_user_message(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def first_assistant_action(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        action = main_builder.extract_action(msg.get("content", ""))
        if action:
            return action
    return None


def second_user_after_first_assistant(messages: List[Dict[str, Any]]) -> Optional[str]:
    saw_assistant = False
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            action = main_builder.extract_action(msg.get("content", ""))
            if action:
                saw_assistant = True
                continue
        if saw_assistant and role == "user":
            return msg.get("content", "")
    return None


def extract_observation_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    if not text:
        return fields

    for prefix, key in (
        ("Inventory:", "inventory"),
        ("Got ", "got"),
        ("Crafted ", "crafted"),
    ):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                fields[key] = stripped[len(prefix) :].strip()
                break
    return fields


def compare_observations(lhs: str, rhs: str) -> Dict[str, Any]:
    lhs_fields = extract_observation_fields(lhs)
    rhs_fields = extract_observation_fields(rhs)
    shared = sorted(set(lhs_fields) & set(rhs_fields))
    matched = [key for key in shared if lhs_fields[key] == rhs_fields[key]]
    mismatched = [key for key in shared if lhs_fields[key] != rhs_fields[key]]
    return {
        "shared_fields": shared,
        "matched_fields": matched,
        "mismatched_fields": mismatched,
        "strict_match": bool(shared) and not mismatched,
    }


def create_env(server: str, task_id: int, goal: Optional[str], timeout: float) -> Tuple[str, str]:
    payload: Dict[str, Any] = {"task_id": task_id, "eval_mode": False}
    if goal:
        payload["goal"] = goal
    response = requests.post(f"{server}/create", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["id"], data.get("observation", "")


def step_env(server: str, env_id: str, action: str, timeout: float) -> Dict[str, Any]:
    response = requests.post(
        f"{server}/step",
        json={"id": env_id, "action": action, "return_raw_obs": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def close_env(server: str, env_id: str, timeout: float) -> None:
    try:
        requests.post(f"{server}/close", json={"id": env_id}, timeout=timeout)
    except Exception:
        pass


def replay_validate_one(row: Dict[str, Any], server: str, timeout: float) -> Dict[str, Any]:
    task_id = int(row["task_id"])
    goal = row.get("goal")
    prefix_actions = [str(action) for action in list(row.get("prefix_actions", []))]
    continuation_messages = list(row["continuation_messages"])

    expected_cut_obs = first_user_message(continuation_messages)
    next_action = first_assistant_action(continuation_messages)
    expected_next_obs = second_user_after_first_assistant(continuation_messages)

    replay_cut_obs = ""
    replay_next_obs = None
    initial_obs = ""
    env_id = None
    error = None

    try:
        env_id, initial_obs = create_env(server, task_id=task_id, goal=goal, timeout=timeout)
        current_obs = initial_obs
        for action in prefix_actions:
            result = step_env(server, env_id, action, timeout=timeout)
            current_obs = result.get("observation", "")
        replay_cut_obs = current_obs

        if next_action:
            next_result = step_env(server, env_id, next_action, timeout=timeout)
            replay_next_obs = next_result.get("observation", "")
    except Exception as exc:
        error = str(exc)
    finally:
        if env_id is not None:
            close_env(server, env_id, timeout=timeout)

    cut_cmp = compare_observations(replay_cut_obs, expected_cut_obs or "")
    next_cmp = compare_observations(replay_next_obs or "", expected_next_obs or "")

    if error:
        replay_category = "error"
    elif not cut_cmp["shared_fields"]:
        replay_category = "unverifiable"
    elif not cut_cmp["strict_match"]:
        replay_category = "mismatch"
    elif next_action and expected_next_obs and next_cmp["shared_fields"] and not next_cmp["strict_match"]:
        replay_category = "mismatch"
    elif next_action and expected_next_obs and not next_cmp["shared_fields"]:
        replay_category = "unverifiable"
    else:
        replay_category = "validated"

    output = dict(row)
    output.update(
        {
            "replay_category": replay_category,
            "initial_observation": initial_obs,
            "replay_cut_observation": replay_cut_obs,
            "expected_cut_observation": expected_cut_obs,
            "continuation_first_action": next_action,
            "replay_next_observation": replay_next_obs,
            "expected_next_observation": expected_next_obs,
            "replay_error": error,
        }
    )
    return output


def classify_unverifiable_bucket(row: Dict[str, Any]) -> Tuple[str, str]:
    expected_cut = normalize_text(row.get("expected_cut_observation"))
    replay_cut = normalize_text(row.get("replay_cut_observation"))
    expected_next = normalize_text(row.get("expected_next_observation"))
    replay_next = normalize_text(row.get("replay_next_observation"))

    if (
        expected_cut
        and expected_cut == replay_cut
        and is_state_feedback(expected_cut)
        and not is_format_error(expected_cut)
    ):
        return "usable_state_feedback", "cut_state_feedback_exact_match"

    if any(is_format_error(text) for text in (expected_cut, replay_cut, expected_next, replay_next) if text):
        return "drop_nonusable", "format_error"
    return "drop_nonusable", "other_unverifiable"


def build_fixed_candidate_rows(teacher_rows: Sequence[Dict[str, Any]], ratio: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    strategy = f"fixed_ratio_{main_builder.ratio_to_name(ratio)}"
    for row in teacher_rows:
        messages = main_builder.normalize_messages(row["conversations"])
        num_assistant_messages = len(main_builder.assistant_message_indices(messages))
        cut_turn_idx, cut_q = main_builder.choose_cut_turn_idx(num_assistant_messages, ratio)
        prefix_messages, continuation_messages = main_builder.split_messages(messages, cut_turn_idx)
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
                "prefix_actions": main_builder.extract_actions_from_messages(prefix_messages),
            }
        )
    return rows


def replay_validate_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    server: str,
    timeout: float,
    concurrency: int,
) -> pd.DataFrame:
    rows_out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        iterator = executor.map(lambda row: replay_validate_one(row, server=server, timeout=timeout), rows)
        for index, row in enumerate(iterator, start=1):
            rows_out.append(row)
            if index % 200 == 0 or index == len(rows):
                print(f"[replay] validated {index}/{len(rows)} fixed-ratio candidates", flush=True)
    return pd.DataFrame(rows_out)


def canonicalize_fixed_trainable_rows(
    df: pd.DataFrame,
    *,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        prompt = main_builder.build_training_prompt(
            prefix_messages=main_builder.normalize_messages(row["prefix_messages"]),
            continuation_messages=main_builder.normalize_messages(row["continuation_messages"]),
            system_prompt=system_prompt,
        )
        output = dict(row)
        output.update(
            {
                "data_source": "textcraft",
                "ability": "crafting",
                "prompt": prompt,
                "reward_model": {"ground_truth": "", "style": "interaction"},
                "extra_info": {
                    "index": int(row["sample_idx"]),
                    "sample_uid": row["sample_uid"],
                    "strategy": row.get("strategy"),
                    "interaction_kwargs": {
                        "name": "textcraft",
                        "task_id": int(row["task_id"]),
                        "eval_mode": False,
                        "prefix_actions": list(row.get("prefix_actions", [])),
                        "goal": row.get("goal"),
                    },
                    "reward_model": {"ground_truth": "", "style": "interaction"},
                },
            }
        )
        records.append(output)
    return records


def wrap_fixed_prefix_row(
    row: Dict[str, Any],
    *,
    main_dataset: str,
    raw_base_rows: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    task_id = int(row["task_id"])
    base_row = raw_base_rows[task_id]
    extra_info = deepcopy(row.get("extra_info", {}) or {})
    interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})
    interaction_kwargs.update(
        main_builder.build_interaction_kwargs(
            task_id=task_id,
            goal=row.get("goal"),
            prefix_actions=list(row.get("prefix_actions", [])),
        )
    )
    extra_info["interaction_kwargs"] = interaction_kwargs
    extra_info["sample_uid"] = row["sample_uid"]
    extra_info["main_dataset"] = main_dataset
    extra_info["variant_label"] = str(row["strategy"])

    return {
        "record_uid": f"{main_dataset}::{row['sample_uid']}::{row['strategy']}",
        "main_dataset": main_dataset,
        "variant_label": str(row["strategy"]),
        "is_raw_variant": False,
        "sample_uid": row["sample_uid"],
        "item_id": row["item_id"],
        "sample_idx": int(row["sample_idx"]),
        "task_id": task_id,
        "goal": row.get("goal"),
        "strategy": row["strategy"],
        "data_source": base_row["data_source"],
        "ability": base_row["ability"],
        "prompt": main_builder.normalize_prompt_messages(row["prompt"]),
        "reward_model": deepcopy(base_row["reward_model"]),
        "extra_info": extra_info,
        "prefix_messages": main_builder.normalize_messages(row["prefix_messages"]),
        "continuation_messages": main_builder.normalize_messages(row["continuation_messages"]),
        "prefix_actions": list(row.get("prefix_actions", [])),
        "replay_category": row.get("replay_category", "validated"),
        "assistant_prefix_old_log_probs": None,
        "prefix_mask": None,
        "prefix_token_count": None,
        "assistant_prefix_span": None,
        "source_oldlogprob_model_path": None,
        "prefix_coordinate_system": "canonicalized_prompt",
        "cut_turn_idx": int(row.get("cut_turn_idx", 0)),
        "cut_relative_position_q": float(row.get("cut_relative_position_q", 0.0)),
        "num_assistant_messages_total": int(row.get("num_assistant_messages_total", 0)),
        "source_dataset": row.get("source_dataset", "full_flow_fixed"),
        "unverifiable_bucket": row.get("unverifiable_bucket"),
        "unverifiable_subreason": row.get("unverifiable_subreason"),
    }


def wrap_entropy_stage7_row(
    row: Dict[str, Any],
    *,
    main_dataset: str,
    raw_base_rows: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    task_id = int(row["task_id"])
    base_row = raw_base_rows[task_id]
    extra_info = deepcopy(row.get("extra_info", {}) or {})
    interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})
    interaction_kwargs.update(
        main_builder.build_interaction_kwargs(
            task_id=task_id,
            goal=row.get("goal"),
            prefix_actions=list(row.get("prefix_actions", [])),
        )
    )
    extra_info["interaction_kwargs"] = interaction_kwargs
    extra_info["sample_uid"] = row["sample_uid"]
    extra_info["main_dataset"] = main_dataset
    extra_info["variant_label"] = f"rank{int(row['candidate_rank'])}"

    output = dict(row)
    output.update(
        {
            "record_uid": f"{main_dataset}::{row['sample_uid']}::rank{int(row['candidate_rank'])}",
            "main_dataset": main_dataset,
            "variant_label": f"rank{int(row['candidate_rank'])}",
            "is_raw_variant": False,
            "data_source": base_row["data_source"],
            "ability": base_row["ability"],
            "reward_model": deepcopy(base_row["reward_model"]),
            "extra_info": extra_info,
            "source_dataset": "entropy_stage7_audited",
        }
    )
    return output


def load_or_build_fixed_trainable_rows(
    teacher_rows: Sequence[Dict[str, Any]],
    *,
    ratios: Sequence[float],
    system_prompt: str,
    server: str,
    timeout: float,
    concurrency: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    ratio_records: Dict[str, List[Dict[str, Any]]] = {}
    manifest: Dict[str, Any] = {"existing_stage4": {}, "replayed_new": {}}
    teacher_sample_uids = {str(row["sample_uid"]) for row in teacher_rows}

    missing_ratios: List[float] = []
    for ratio in ratios:
        strategy = f"fixed_ratio_{main_builder.ratio_to_name(ratio)}"
        existing_path = DEFAULT_EXISTING_FIXED_STAGE4.get(strategy)
        if existing_path and existing_path.exists():
            df = pd.read_parquet(existing_path)
            df = df[df["sample_uid"].isin(teacher_sample_uids)].copy()
            rows = df.to_dict(orient="records")
            for row in rows:
                row["source_dataset"] = "new_prefix_stage4_trainable_canonicalized"
            ratio_records[strategy] = rows
            manifest["existing_stage4"][strategy] = {
                "path": str(existing_path),
                "rows": int(len(df)),
                "unique_sample_uid": int(df["sample_uid"].nunique()),
                "replay_category_counts": df["replay_category"].value_counts().sort_index().to_dict(),
            }
        else:
            missing_ratios.append(ratio)

    if missing_ratios:
        candidate_rows: List[Dict[str, Any]] = []
        for ratio in missing_ratios:
            candidate_rows.extend(build_fixed_candidate_rows(teacher_rows, ratio))

        replay_df = replay_validate_rows(
            candidate_rows,
            server=server,
            timeout=timeout,
            concurrency=concurrency,
        )

        for strategy, strategy_df in replay_df.groupby("strategy", sort=True):
            validated_df = strategy_df[strategy_df["replay_category"] == "validated"].copy()
            unverifiable_df = strategy_df[strategy_df["replay_category"] == "unverifiable"].copy()

            usable_rows: List[Dict[str, Any]] = []
            for row in unverifiable_df.to_dict(orient="records"):
                bucket, subreason = classify_unverifiable_bucket(row)
                row["unverifiable_bucket"] = bucket
                row["unverifiable_subreason"] = subreason
                if bucket == "usable_state_feedback":
                    usable_rows.append(row)

            trainable_df = pd.concat([validated_df, pd.DataFrame(usable_rows)], ignore_index=True)
            trainable_df = trainable_df.sort_values(["task_id", "sample_idx"]).reset_index(drop=True)
            rows = canonicalize_fixed_trainable_rows(trainable_df, system_prompt=system_prompt)
            for row in rows:
                row["source_dataset"] = "main_prefix_full_flow_replay"
            ratio_records[strategy] = rows
            manifest["replayed_new"][strategy] = {
                "candidate_rows": int(len(strategy_df)),
                "validated_rows": int(len(validated_df)),
                "usable_unverifiable_rows": int(len(usable_rows)),
                "trainable_rows": int(len(trainable_df)),
                "unique_sample_uid": int(trainable_df["sample_uid"].nunique()),
                "replay_category_counts": strategy_df["replay_category"].value_counts().sort_index().to_dict(),
            }

    return ratio_records, manifest


def build_fixed_fullflow_dataset(
    main_dataset: str,
    ratio_records: Dict[str, List[Dict[str, Any]]],
    teacher_rows: Sequence[Dict[str, Any]],
    *,
    ratios: Sequence[float],
    raw_base_rows: Dict[int, Dict[str, Any]],
    model_path: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        strategy = f"fixed_ratio_{main_builder.ratio_to_name(ratio)}"
        for row in ratio_records[strategy]:
            rows.append(wrap_fixed_prefix_row(row, main_dataset=main_dataset, raw_base_rows=raw_base_rows))

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
    return rows


def build_entropy_fullflow_dataset(
    teacher_rows: Sequence[Dict[str, Any]],
    *,
    main_dataset: str,
    audited_path: Path,
    raw_base_rows: Dict[int, Dict[str, Any]],
    model_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    audited_df = pd.read_parquet(audited_path)
    rows = [
        wrap_entropy_stage7_row(row, main_dataset=main_dataset, raw_base_rows=raw_base_rows)
        for row in audited_df.to_dict(orient="records")
    ]
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
    manifest = {
        "audited_path": str(audited_path),
        "audited_rows": int(len(audited_df)),
        "audited_unique_sample_uid": int(audited_df["sample_uid"].nunique()),
        "candidate_rank_counts": audited_df["candidate_rank"].astype(int).value_counts().sort_index().to_dict(),
    }
    return rows, manifest


def build_report_markdown(
    summaries: Dict[str, Dict[str, Any]],
    *,
    teacher_path: Path,
    train_parquet_path: Path,
    model_path: str,
    server: str,
) -> str:
    lines = [
        "# TextCraft Main Prefix Full-Flow Supplement",
        "",
        "## Meaning",
        "- This directory contains replay-filtered supplemental datasets that explicitly follow the legacy `new_prefix_rl` / `entropy_based_prefix` full-flow semantics.",
        "- They do not replace the cutpoint-complete datasets in `main_prefix/complete_split/`.",
        "- The datasets in `complete_split/` are cutpoint-complete and prompt-space-correct.",
        "- The `full_flow` datasets below additionally inherit replay validation / refinement filtering semantics.",
        "",
        "## Inputs",
        f"- teacher_normalized: `{teacher_path}`",
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
                f"- positive_prefix_rows: `{summary['positive_prefix_rows']}`",
                f"- output_path: `{summary['output_path']}`",
                f"- strategy_counts: `{summary['strategy_counts']}`",
            ]
        )
        if "candidate_rank_counts" in summary:
            lines.append(f"- candidate_rank_counts: `{summary['candidate_rank_counts']}`")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "- Dataset-specific counts and protocol summaries are also stored beside each parquet as `*.manifest.json`.",
            "- `main_fixed_*_fullflow` keeps all `1496` raw rows, but prefix rows are replay-filtered (`validated + usable_state_feedback`).",
            "- `main_*_fullflow` is the pipeline-aligned replay-validated version; `complete_split/` remains the cutpoint-complete prompt-space version.",
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
        args.device = "cuda"

    selected_datasets = {name.strip() for name in args.datasets.split(",") if name.strip()}
    if not selected_datasets:
        raise RuntimeError("No datasets selected.")

    teacher_df = pd.read_parquet(args.teacher_path)
    if args.max_samples is not None:
        teacher_df = teacher_df.head(args.max_samples).copy()
    teacher_rows = teacher_df.to_dict(orient="records")
    raw_base_rows = main_builder.load_raw_base_rows(args.train_parquet_path)
    system_prompt = main_builder.load_reference_system_prompt(args.train_parquet_path)

    build_manifests: Dict[str, Any] = {}
    datasets: Dict[str, List[Dict[str, Any]]] = {}

    selected_fixed_specs = {
        "main_fixed_gp1_fullflow": (0.1, 0.3, 0.5),
        "main_fixed_gp2_fullflow": (0.25, 0.5, 0.7),
    }
    selected_fixed_ratios = sorted(
        {
            ratio
            for name, ratios in selected_fixed_specs.items()
            if name in selected_datasets
            for ratio in ratios
        }
    )
    fixed_ratio_records: Dict[str, List[Dict[str, Any]]] = {}
    fixed_manifest: Dict[str, Any] = {}
    if selected_fixed_ratios:
        fixed_ratio_records, fixed_manifest = load_or_build_fixed_trainable_rows(
            teacher_rows,
            ratios=selected_fixed_ratios,
            system_prompt=system_prompt,
            server=args.server,
            timeout=args.request_timeout,
            concurrency=args.replay_concurrency,
        )

    if "main_fixed_gp1_fullflow" in selected_datasets:
        rows = build_fixed_fullflow_dataset(
            main_dataset="main_fixed_gp1_fullflow",
            ratio_records=fixed_ratio_records,
            teacher_rows=teacher_rows,
            ratios=(0.1, 0.3, 0.5),
            raw_base_rows=raw_base_rows,
            model_path=args.model_path,
        )
        datasets["main_fixed_gp1_fullflow"] = rows
        build_manifests["main_fixed_gp1_fullflow"] = {
            "ratios": ["fixed_ratio_0p1", "fixed_ratio_0p3", "fixed_ratio_0p5"],
            "shared_fixed_manifest": fixed_manifest,
        }

    if "main_fixed_gp2_fullflow" in selected_datasets:
        rows = build_fixed_fullflow_dataset(
            main_dataset="main_fixed_gp2_fullflow",
            ratio_records=fixed_ratio_records,
            teacher_rows=teacher_rows,
            ratios=(0.25, 0.5, 0.7),
            raw_base_rows=raw_base_rows,
            model_path=args.model_path,
        )
        datasets["main_fixed_gp2_fullflow"] = rows
        build_manifests["main_fixed_gp2_fullflow"] = {
            "ratios": ["fixed_ratio_0p25", "fixed_ratio_0p5", "fixed_ratio_0p7"],
            "shared_fixed_manifest": fixed_manifest,
        }

    if "main_raw_top3_fullflow" in selected_datasets:
        rows, manifest = build_entropy_fullflow_dataset(
            teacher_rows,
            main_dataset="main_raw_top3_fullflow",
            audited_path=DEFAULT_ENTROPY_AUDITED["main_raw_top3_fullflow"],
            raw_base_rows=raw_base_rows,
            model_path=args.model_path,
        )
        datasets["main_raw_top3_fullflow"] = rows
        build_manifests["main_raw_top3_fullflow"] = manifest

    if "main_change_top3_w11_fullflow" in selected_datasets:
        rows, manifest = build_entropy_fullflow_dataset(
            teacher_rows,
            main_dataset="main_change_top3_w11_fullflow",
            audited_path=DEFAULT_ENTROPY_AUDITED["main_change_top3_w11_fullflow"],
            raw_base_rows=raw_base_rows,
            model_path=args.model_path,
        )
        datasets["main_change_top3_w11_fullflow"] = rows
        build_manifests["main_change_top3_w11_fullflow"] = manifest

    unknown = sorted(selected_datasets - set(datasets))
    if unknown:
        raise RuntimeError(f"Unknown dataset names: {unknown}")

    sidecar_rows: List[Dict[str, Any]] = []
    for name, rows in datasets.items():
        if name.startswith("main_fixed_"):
            sidecar_rows.extend(rows)
    if sidecar_rows:
        main_builder.materialize_prefix_sidecars(
            sidecar_rows,
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            max_batch_prompt_tokens=args.max_batch_prompt_tokens,
            progress_every=args.progress_every,
        )

    summaries: Dict[str, Dict[str, Any]] = {}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in datasets.items():
        output_path = args.output_root / f"{name}.parquet"
        manifest_path = args.output_root / f"{name}.manifest.json"
        summary = main_builder.write_dataset(output_path, rows, manifest_path)
        summary["output_path"] = str(output_path)
        summaries[name] = summary

    report_path = args.output_root / "README.md"
    combined_summaries = load_existing_summaries(args.output_root)
    report_path.write_text(
        build_report_markdown(
            combined_summaries if combined_summaries else summaries,
            teacher_path=args.teacher_path,
            train_parquet_path=args.train_parquet_path,
            model_path=args.model_path,
            server=args.server,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
