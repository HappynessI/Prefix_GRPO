#!/usr/bin/env python3
"""Build runtime-parser-aligned TextCraft main-prefix datasets and audit diffs."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

import build_textcraft_main_prefix_datasets as main_builder
import build_textcraft_main_prefix_fullflow_datasets as fullflow_builder


DEFAULT_OUTPUT_ROOT = Path("/Data/wyh/datasets/Verl-Data/train/textcraft/main_prefix/runtime_parser_aligned")
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
FULLFLOW_DATASET_NAMES = (
    "main_fixed_gp1_fullflow",
    "main_fixed_gp2_fullflow",
    "main_raw_top3_fullflow",
    "main_change_top3_w11_fullflow",
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
        help="Comma-separated base dataset names. The matching *_fullflow datasets are built automatically.",
    )
    parser.add_argument("--legacy-complete-split-root", type=Path, default=DEFAULT_LEGACY_COMPLETE_SPLIT_ROOT)
    parser.add_argument("--legacy-replay-validated-root", type=Path, default=DEFAULT_LEGACY_REPLAY_VALIDATED_ROOT)
    return parser.parse_args()


def normalize_records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def dataset_drop_summary(dropped_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sample_uids = sorted({str(row["sample_uid"]) for row in dropped_rows})
    strategies = sorted({str(row.get("strategy", "")) for row in dropped_rows})
    reason_counts: Dict[str, int] = {}
    for row in dropped_rows:
        for reason in list(row.get("runtime_invalid_reasons", [])):
            key = str(reason)
            reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "dropped_runtime_invalid_prefix_rows": int(len(dropped_rows)),
        "dropped_runtime_invalid_prefix_sample_uid_count": int(len(sample_uids)),
        "dropped_runtime_invalid_prefix_sample_uid_examples": sample_uids[:20],
        "dropped_runtime_invalid_strategies": strategies,
        "dropped_runtime_invalid_reason_counts": dict(sorted(reason_counts.items())),
    }


def save_drop_artifacts(drop_root: Path, dataset_name: str, dropped_rows: Sequence[Dict[str, Any]]) -> None:
    if not dropped_rows:
        return
    drop_root.mkdir(parents=True, exist_ok=True)
    parquet_path = drop_root / f"{dataset_name}.runtime_invalid_dropped.parquet"
    manifest_path = drop_root / f"{dataset_name}.runtime_invalid_dropped.manifest.json"
    df = pd.DataFrame(normalize_records(dropped_rows))
    df.to_parquet(parquet_path, index=False)
    manifest = dataset_drop_summary(dropped_rows)
    manifest["output_path"] = str(parquet_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_complete_split_datasets(
    teacher_rows: Sequence[Dict[str, Any]],
    entropy_df: pd.DataFrame,
    raw_base_rows: Dict[int, Dict[str, Any]],
    *,
    system_prompt: str,
    model_path: str,
    selected_datasets: Sequence[str],
    drop_root: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    raw_top3_rows = entropy_df[
        entropy_df["strategy"] == "entropy_raw_topk_interaction_assistant_k3"
    ].to_dict(orient="records")
    change_top3_rows = entropy_df[
        entropy_df["strategy"] == "entropy_change_topk_w11_interaction_assistant_k3"
    ].to_dict(orient="records")

    datasets_all: Dict[str, List[Dict[str, Any]]] = {
        "main_fixed_gp1": main_builder.build_fixed_dataset_rows(
            list(teacher_rows),
            raw_base_rows,
            main_dataset="main_fixed_gp1",
            system_prompt=system_prompt,
            ratios=(0.1, 0.3, 0.5),
            model_path=model_path,
        ),
        "main_fixed_gp2": main_builder.build_fixed_dataset_rows(
            list(teacher_rows),
            raw_base_rows,
            main_dataset="main_fixed_gp2",
            system_prompt=system_prompt,
            ratios=(0.25, 0.5, 0.7),
            model_path=model_path,
        ),
        "main_raw_top3": main_builder.build_entropy_dataset_rows(
            list(teacher_rows),
            raw_top3_rows,
            raw_base_rows,
            main_dataset="main_raw_top3",
            system_prompt=system_prompt,
            model_path=model_path,
        ),
        "main_change_top3_w11": main_builder.build_entropy_dataset_rows(
            list(teacher_rows),
            change_top3_rows,
            raw_base_rows,
            main_dataset="main_change_top3_w11",
            system_prompt=system_prompt,
            model_path=model_path,
        ),
    }

    datasets: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for name in selected_datasets:
        rows = datasets_all[name]
        kept_rows, dropped_rows = main_builder.split_runtime_invalid_prefix_rows(rows)
        datasets[name] = kept_rows
        summaries[name] = dataset_drop_summary(dropped_rows)
        save_drop_artifacts(drop_root, name, dropped_rows)
    return datasets, summaries


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
    drop_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_rows: List[Dict[str, Any]] = []
    dropped_rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        ratio_rows = fullflow_builder.build_fixed_candidate_rows(teacher_rows, ratio)
        kept_ratio_rows, dropped_ratio_rows = main_builder.split_runtime_invalid_prefix_rows(ratio_rows)
        candidate_rows.extend(kept_ratio_rows)
        dropped_rows.extend(dropped_ratio_rows)

    save_drop_artifacts(drop_root, main_dataset, dropped_rows)
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
        ratio_records[strategy] = fullflow_builder.canonicalize_fixed_trainable_rows(
            trainable_df,
            system_prompt=system_prompt,
        )
        strategy_manifest[strategy] = {
            "candidate_rows_after_runtime_drop": int(len(strategy_df)),
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
    manifest = dataset_drop_summary(dropped_rows)
    manifest["strategy_manifest"] = strategy_manifest
    return rows, manifest


def rebuild_entropy_candidate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rebuilt_rows: List[Dict[str, Any]] = []
    for row in rows:
        prefix_messages = main_builder.normalize_messages(row["prefix_messages"])
        continuation_messages = main_builder.normalize_messages(row["continuation_messages"])
        rebuilt = dict(row)
        rebuilt["prefix_messages"] = prefix_messages
        rebuilt["continuation_messages"] = continuation_messages
        rebuilt["prefix_actions"] = main_builder.extract_actions_from_messages(prefix_messages)
        rebuilt_rows.append(rebuilt)
    return rebuilt_rows


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
    drop_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rebuilt_candidates = rebuild_entropy_candidate_rows(entropy_rows)
    kept_candidates, dropped_candidates = main_builder.split_runtime_invalid_prefix_rows(rebuilt_candidates)
    save_drop_artifacts(drop_root, main_dataset, dropped_candidates)

    replay_df = fullflow_builder.replay_validate_rows(
        kept_candidates,
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
        prompt = main_builder.build_training_prompt(
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
                "source_dataset": "entropy_stage2_replay_rebuilt_runtime_parser_aligned",
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

    manifest = dataset_drop_summary(dropped_candidates)
    manifest.update(
        {
            "candidate_rows_after_runtime_drop": int(len(replay_df)),
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
    drop_root: Path,
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
            drop_root=drop_root,
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
            drop_root=drop_root,
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
            drop_root=drop_root,
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
            drop_root=drop_root,
        )
        datasets["main_change_top3_w11_fullflow"] = rows
        manifests["main_change_top3_w11_fullflow"] = manifest

    return datasets, manifests


def build_complete_split_readme(
    summaries: Dict[str, Dict[str, Any]],
    *,
    teacher_path: Path,
    entropy_candidates_path: Path,
    train_parquet_path: Path,
    model_path: str,
) -> str:
    lines = [
        "# TextCraft Main Prefix Complete Split (Runtime Parser Aligned)",
        "",
        "## Meaning",
        "- This directory rebuilds the complete-split datasets under a parser that matches the runtime TextCraft ReAct rule exactly.",
        "- A prefix assistant message contributes one action iff it contains exactly one `Action:` tag after chat-template marker stripping.",
        "- Prefix rows are dropped if any non-warmup assistant turn violates the runtime single-`Action:` protocol, or if no runtime-valid action remains after filtering.",
        "- Dropped rows are recorded under `../audit/complete_split/` with per-reason counts.",
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
                f"- dropped_runtime_invalid_prefix_rows: `{summary['dropped_runtime_invalid_prefix_rows']}`",
                f"- output_path: `{summary['output_path']}`",
            ]
        )
        if "candidate_rank_counts" in summary:
            lines.append(f"- candidate_rank_counts: `{summary['candidate_rank_counts']}`")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "- This is still the cutpoint-complete family: it preserves all remaining parser-valid cutpoints and then rebuilds prompt-space sidecars.",
            "- `concat_like_prefix_action_rows` may remain because the runtime env accepts a single `Action:` line whose payload still contains multiple commands.",
            "- `placeholder_like_prefix_action_rows` and `rows_with_multi_action_tag_in_prefix_messages` are the main parser-alignment audit fields to watch.",
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
        "# TextCraft Main Prefix Replay Validated (Runtime Parser Aligned)",
        "",
        "## Meaning",
        "- This directory rebuilds the replay-filtered datasets under a parser that matches the runtime TextCraft ReAct rule exactly.",
        "- Fixed-ratio datasets keep `validated + usable_state_feedback`; entropy datasets keep `validated` only.",
        "- Replay validation is rerun from parser-aligned candidate rows instead of reusing legacy stage4/stage7 outputs.",
        "- Candidate rows are dropped if any non-warmup assistant turn violates the runtime single-`Action:` protocol, or if no runtime-valid action remains after filtering.",
        "- Dropped rows are recorded under `../audit/replay_validated/` with per-reason counts.",
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
                f"- dropped_runtime_invalid_prefix_rows: `{summary['dropped_runtime_invalid_prefix_rows']}`",
                f"- output_path: `{summary['output_path']}`",
            ]
        )
        if "candidate_rank_counts" in summary:
            lines.append(f"- candidate_rank_counts: `{summary['candidate_rank_counts']}`")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "- This family is replay-filtered and parser-aligned, but it still uses the legacy replay evidence policy (`validated` / `usable_state_feedback`).",
            "- It should be interpreted as a new audit branch; it does not overwrite the current `main_prefix/replay_validated/` release.",
        ]
    )
    return "\n".join(lines)


def summarize_parquet(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return main_builder.summarize_dataset(df.to_dict(orient="records"))


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
        "# Runtime Parser Alignment Audit",
        "",
        f"- Output root: `{output_root}`",
        "- Comparison baseline: existing `main_prefix/complete_split/` and `main_prefix/replay_validated/` datasets.",
        "- Parser rule: one assistant message contributes an action iff it contains exactly one `Action:` tag after runtime-style chat-marker stripping.",
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
                f"- legacy_placeholder_like_prefix_action_rows: `{comparison.get('legacy_placeholder_like_prefix_action_rows')}`",
                f"- rebuilt_placeholder_like_prefix_action_rows: `{comparison.get('rebuilt_placeholder_like_prefix_action_rows')}`",
                f"- legacy_rows_with_multi_action_tag_in_prefix_messages: `{comparison.get('legacy_rows_with_multi_action_tag_in_prefix_messages')}`",
                f"- rebuilt_rows_with_multi_action_tag_in_prefix_messages: `{comparison.get('rebuilt_rows_with_multi_action_tag_in_prefix_messages')}`",
                f"- legacy_concat_like_prefix_action_rows: `{comparison.get('legacy_concat_like_prefix_action_rows')}`",
                f"- rebuilt_concat_like_prefix_action_rows: `{comparison.get('rebuilt_concat_like_prefix_action_rows')}`",
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
    audit_complete_drop_root = args.output_root / "audit" / "complete_split"
    audit_replay_drop_root = args.output_root / "audit" / "replay_validated"

    complete_split_datasets, complete_split_manifests = build_complete_split_datasets(
        teacher_rows,
        entropy_df,
        raw_base_rows,
        system_prompt=system_prompt,
        model_path=args.model_path,
        selected_datasets=selected_datasets,
        drop_root=audit_complete_drop_root,
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
        drop_root=audit_replay_drop_root,
        selected_datasets=selected_datasets,
    )

    all_rows: List[Dict[str, Any]] = []
    for rows in complete_split_datasets.values():
        all_rows.extend(rows)
    for rows in replay_validated_datasets.values():
        all_rows.extend(rows)
    main_builder.materialize_prefix_sidecars(
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
        complete_split_summaries[name] = main_builder.write_dataset(
            output_path,
            rows,
            manifest_path,
            extra_summary=complete_split_manifests.get(name),
        )

    replay_validated_summaries: Dict[str, Dict[str, Any]] = {}
    replay_validated_root.mkdir(parents=True, exist_ok=True)
    for name, rows in replay_validated_datasets.items():
        output_path = replay_validated_root / f"{name}.parquet"
        manifest_path = replay_validated_root / f"{name}.manifest.json"
        replay_validated_summaries[name] = main_builder.write_dataset(
            output_path,
            rows,
            manifest_path,
            extra_summary=replay_validated_manifests.get(name),
        )

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
            "legacy_placeholder_like_prefix_action_rows": (
                legacy_complete.get("placeholder_like_prefix_action_rows") if legacy_complete else None
            ),
            "rebuilt_placeholder_like_prefix_action_rows": (
                rebuilt_complete.get("placeholder_like_prefix_action_rows") if rebuilt_complete else None
            ),
            "legacy_rows_with_multi_action_tag_in_prefix_messages": (
                legacy_complete.get("rows_with_multi_action_tag_in_prefix_messages") if legacy_complete else None
            ),
            "rebuilt_rows_with_multi_action_tag_in_prefix_messages": (
                rebuilt_complete.get("rows_with_multi_action_tag_in_prefix_messages") if rebuilt_complete else None
            ),
            "legacy_concat_like_prefix_action_rows": (
                legacy_complete.get("concat_like_prefix_action_rows") if legacy_complete else None
            ),
            "rebuilt_concat_like_prefix_action_rows": (
                rebuilt_complete.get("concat_like_prefix_action_rows") if rebuilt_complete else None
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
            "legacy_placeholder_like_prefix_action_rows": (
                legacy_fullflow.get("placeholder_like_prefix_action_rows") if legacy_fullflow else None
            ),
            "rebuilt_placeholder_like_prefix_action_rows": (
                rebuilt_fullflow.get("placeholder_like_prefix_action_rows") if rebuilt_fullflow else None
            ),
            "legacy_rows_with_multi_action_tag_in_prefix_messages": (
                legacy_fullflow.get("rows_with_multi_action_tag_in_prefix_messages") if legacy_fullflow else None
            ),
            "rebuilt_rows_with_multi_action_tag_in_prefix_messages": (
                rebuilt_fullflow.get("rows_with_multi_action_tag_in_prefix_messages") if rebuilt_fullflow else None
            ),
            "legacy_concat_like_prefix_action_rows": (
                legacy_fullflow.get("concat_like_prefix_action_rows") if legacy_fullflow else None
            ),
            "rebuilt_concat_like_prefix_action_rows": (
                rebuilt_fullflow.get("concat_like_prefix_action_rows") if rebuilt_fullflow else None
            ),
        }

    audit_report_md, audit_report_json = build_audit_report(comparisons, output_root=args.output_root)
    (args.output_root / "audit_report.md").write_text(audit_report_md + "\n", encoding="utf-8")
    (args.output_root / "audit_report.json").write_text(
        json.dumps(audit_report_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    root_readme = [
        "# TextCraft Main Prefix Runtime Parser Aligned",
        "",
        "## Contents",
        f"- complete_split: `{complete_split_root}`",
        f"- replay_validated: `{replay_validated_root}`",
        f"- audit_report.md: `{args.output_root / 'audit_report.md'}`",
        "",
        "## Meaning",
        "- This branch rebuilds main-prefix datasets under a parser that matches the runtime TextCraft ReAct rule exactly.",
        "- Training-side interaction logic is intentionally left unchanged; this branch is for offline data audit and alternate experiments only.",
        "- Prefix rows are dropped if any non-warmup assistant turn violates the runtime single-`Action:` protocol, or if no runtime-valid action remains after filtering.",
        "- Dropped rows are stored as audit artifacts with per-reason counts.",
    ]
    (args.output_root / "README.md").write_text("\n".join(root_readme) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "complete_split": complete_split_summaries,
                "replay_validated": replay_validated_summaries,
                "audit_report_path": str(args.output_root / "audit_report.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
