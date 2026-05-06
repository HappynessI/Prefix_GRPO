#!/usr/bin/env python3
"""Repair AgentGym remaining environment parquet metadata in a reproducible way.

This script fixes the four non-TextCraft environments used in the current
AgentGym setup:

- babyai
- sciworld
- alfworld
- webshop

The repair has two goals:
1. Make the environment reset key explicit in parquet metadata so train/eval do
   not infer it from row order.
2. Add stable task category annotations for auditing and split analysis.

By default the script rewrites both:
- repo-local copies under `h200_grpo_from_oss/data`
- runtime copies under `/Data/wyh/datasets/Verl-Data`
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = PROJECT_ROOT.parent / "datasets" / "AgentGym-RL-Data"
REPO_DATA_ROOT = PROJECT_ROOT / "data"
VERL_DATA_ROOT = PROJECT_ROOT.parent / "datasets" / "Verl-Data"
ALFWORLD_CONFIG_ROOT = PROJECT_ROOT / "envs" / "AgentGym" / "agentenv-alfworld" / "configs"
WEBSHOP_DATA_ROOT = (
    PROJECT_ROOT / "envs" / "AgentGym" / "agentenv-webshop" / "webshop" / "data"
)

TARGET_CHOICES = ("repo", "verl-data")
SPLITS = ("train", "test")


BABYAI_LEVEL_TO_CATEGORY: dict[int, str] = {
    **{level: "GoTo" for level in range(1, 12)},
    **{level: "Pickup" for level in (19, 20, 21)},
    30: "AOD",
    31: "Find Room",
    33: "Find Room",
    36: "SLoc",
}

# Counts are for the released AgentGym RL split after excluding the omitted task
# ids in the environment wrapper. We assign labels by walking the full sorted
# session_id set across train+test, which preserves the simulator task order.
SCIWORLD_TASK_LAYOUT: list[tuple[str, int, str, str]] = [
    ("1-1", 11, "Other-Matter", "Changes of State (Boiling)"),
    ("1-2", 14, "Other-Matter", "Changes of State (Melting)"),
    ("1-3", 13, "Other-Matter", "Changes of State (Freezing)"),
    ("1-4", 14, "Other-Matter", "Changes of State (Any)"),
    ("2-1", 295, "Measure", "Use Thermometer"),
    ("2-2", 206, "Measure", "Measuring Boiling Point (known)"),
    ("2-3", 5, "Measure", "Measuring Boiling Point (unknown)"),
    ("3-1", 10, "Other-Electricity", "Create a circuit"),
    ("3-2", 10, "Other-Electricity", "Renewable vs Non-renewable Energy"),
    ("3-3", 481, "Test-Cond.", "Test Conductivity (known)"),
    ("3-4", 352, "Test-Cond.", "Test Conductivity (unknown)"),
    ("4-1", 161, "Find", "Find a living thing"),
    ("4-2", 167, "Find", "Find a non-living thing"),
    ("4-3", 159, "Find", "Find a plant"),
    ("4-4", 157, "Find", "Find an animal"),
    ("6-1", 14, "Chem-Mix", "Mixing (generic)"),
    ("6-2", 22, "Chem-Mix", "Mixing paints (secondary colours)"),
    ("6-3", 18, "Chem-Mix", "Mixing paints (tertiary colours)"),
    ("7-1", 77, "Lifespan", "Identify longest-lived animal"),
    ("7-2", 62, "Lifespan", "Identify shortest-lived animal"),
    ("7-3", 62, "Lifespan", "Identify longest-then-shortest-lived animal"),
    ("8-1", 6, "Other-Biology", "Identify life stages (plant)"),
    ("8-2", 4, "Other-Biology", "Identify life stages (animal)"),
]

ALFWORLD_TASK_FAMILIES = (
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGET_CHOICES,
        default=list(TARGET_CHOICES),
        help="Parquet roots to rewrite. Default: repo and verl-data.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_numeric_suffix(item_id: str) -> int:
    return int(item_id.rsplit("_", 1)[-1])


def target_path(target: str, env_name: str, split: str) -> Path:
    if target == "repo":
        return REPO_DATA_ROOT / env_name / f"{split}.parquet"
    if target == "verl-data":
        split_dir = "train" if split == "train" else "eval"
        return VERL_DATA_ROOT / split_dir / env_name / f"{split}.parquet"
    raise ValueError(f"Unsupported target: {target}")


def official_json_path(env_name: str, split: str) -> Path:
    split_dir = "train" if split == "train" else "eval"
    suffix = "train" if split == "train" else "test"
    return OFFICIAL_ROOT / split_dir / f"{env_name}_{suffix}.json"


def clone_extra_info(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    extra_info = deepcopy(row.get("extra_info") or {})
    interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs") or {})
    return extra_info, interaction_kwargs


def apply_common_metadata(
    row: dict[str, Any],
    *,
    row_index: int,
    split: str,
    item_id: str,
    session_id: int,
    task_category: str,
    task_subcategory: str,
    interaction_updates: dict[str, Any],
    top_level_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repaired = deepcopy(row)
    extra_info, interaction_kwargs = clone_extra_info(repaired)

    interaction_kwargs.update(
        {
            "official_item_id": item_id,
            "session_id": int(session_id),
            "task_category": task_category,
            "task_subcategory": task_subcategory,
        }
    )
    interaction_kwargs.update(interaction_updates)

    extra_info["index"] = row_index
    extra_info["official_item_id"] = item_id
    extra_info["interaction_kwargs"] = interaction_kwargs

    repaired["extra_info"] = extra_info
    repaired["item_id"] = item_id
    repaired["session_id"] = int(session_id)
    repaired["env_reset_key"] = int(session_id)
    repaired["data_split"] = split
    repaired["task_category"] = task_category
    repaired["task_subcategory"] = task_subcategory

    if top_level_updates:
        repaired.update(top_level_updates)

    return repaired


def babyai_metadata(session_id: int) -> tuple[str, str, int]:
    level = session_id % 40 + 1
    category = BABYAI_LEVEL_TO_CATEGORY.get(level)
    if category is None:
        raise ValueError(f"Unexpected BabyAI level {level} for session_id={session_id}")
    return category, f"level_{level:02d}", level


def build_sciworld_session_map() -> dict[int, dict[str, str]]:
    all_session_ids = []
    for split in SPLITS:
        official_rows = load_json(official_json_path("sciworld", split))
        all_session_ids.extend(extract_numeric_suffix(row["item_id"]) for row in official_rows)

    sorted_ids = sorted(all_session_ids)
    expected_total = sum(count for _task_id, count, _cat, _sub in SCIWORLD_TASK_LAYOUT)
    if len(sorted_ids) != expected_total:
        raise ValueError(
            f"SciWorld session_id count mismatch: got {len(sorted_ids)}, expected {expected_total}."
        )

    session_map: dict[int, dict[str, str]] = {}
    cursor = 0
    for task_id, count, category, task_name in SCIWORLD_TASK_LAYOUT:
        task_session_ids = sorted_ids[cursor : cursor + count]
        if len(task_session_ids) != count:
            raise ValueError(f"Failed to assign SciWorld task {task_id}: expected {count} rows.")
        for session_id in task_session_ids:
            assigned = {
                "task_id": task_id,
                "task_category": category,
                "task_name": task_name,
            }
            if session_id in session_map and session_map[session_id] != assigned:
                raise ValueError(
                    f"SciWorld session_id {session_id} maps to multiple tasks: "
                    f"{session_map[session_id]} vs {assigned}"
                )
            session_map[session_id] = assigned
        cursor += count

    if cursor != len(sorted_ids):
        raise ValueError("SciWorld session_id assignment did not consume all rows.")
    return session_map


def infer_alfworld_family(task_type: str) -> str:
    for family in ALFWORLD_TASK_FAMILIES:
        if task_type.startswith(family + "-"):
            return family
    raise ValueError(f"Could not infer ALFWorld family from task_type={task_type}")


def load_alfworld_mappings(split: str) -> list[dict[str, Any]]:
    filename = "mappings_train.json" if split == "train" else "mappings_test.json"
    return load_json(ALFWORLD_CONFIG_ROOT / filename)


def build_webshop_goal_metadata() -> list[dict[str, str]]:
    products = load_json(WEBSHOP_DATA_ROOT / "items_shuffle_1000.json")
    attributes = load_json(WEBSHOP_DATA_ROOT / "items_ins_v2_1000.json")

    seen_asins = set()
    synthetic_goals: list[dict[str, str]] = []

    for product in products:
        asin = product["asin"]
        if asin == "nan" or len(asin) > 10 or asin in seen_asins:
            continue
        seen_asins.add(asin)

        attr_payload = attributes.get(asin) or {}
        instruction_text = attr_payload.get("instruction")
        instruction_attributes = attr_payload.get("instruction_attributes") or []
        if not instruction_text or not instruction_attributes:
            continue

        raw_options = product.get("customization_options") or {}
        normalized_options: dict[str, list[str]] = {}
        for option_name, option_contents in raw_options.items():
            if option_contents is None:
                continue
            option_values = []
            for option_content in option_contents:
                option_values.append(option_content["value"].strip().replace("/", " | ").lower())
            normalized_options[option_name.lower()] = option_values

        option_names = sorted(normalized_options)
        combinations = list(
            itertools.product(*(normalized_options[name] for name in option_names))
        )
        if not combinations:
            combinations = [tuple()]

        for _combination in combinations:
            full_category = (product.get("product_category") or "").strip()
            top_category = full_category.split("›", 1)[0].strip() if full_category else ""
            synthetic_goals.append(
                {
                    "task_category": top_category,
                    "task_subcategory": full_category,
                }
            )

    random.seed(233)
    random.shuffle(synthetic_goals)
    return synthetic_goals


def repair_babyai_rows(
    current_rows: list[dict[str, Any]], official_rows: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    repaired = []
    for row_index, (row, official_row) in enumerate(zip(current_rows, official_rows)):
        item_id = official_row["item_id"]
        session_id = extract_numeric_suffix(item_id)
        category, subcategory, level = babyai_metadata(session_id)
        repaired.append(
            apply_common_metadata(
                row,
                row_index=row_index,
                split=split,
                item_id=item_id,
                session_id=session_id,
                task_category=category,
                task_subcategory=subcategory,
                interaction_updates={
                    "name": "babyai",
                    "data_idx": session_id,
                    "level": level,
                },
                top_level_updates={"babyai_level": level},
            )
        )
    return repaired


def repair_sciworld_rows(
    current_rows: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
    split: str,
    session_map: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    repaired = []
    for row_index, (row, official_row) in enumerate(zip(current_rows, official_rows)):
        item_id = official_row["item_id"]
        session_id = extract_numeric_suffix(item_id)
        meta = session_map.get(session_id)
        if meta is None:
            raise KeyError(f"Missing SciWorld session metadata for session_id={session_id}")
        repaired.append(
            apply_common_metadata(
                row,
                row_index=row_index,
                split=split,
                item_id=item_id,
                session_id=session_id,
                task_category=meta["task_category"],
                task_subcategory=meta["task_id"],
                interaction_updates={
                    "name": "sciworld",
                    "data_idx": session_id,
                    "task_id": meta["task_id"],
                    "task_name": meta["task_name"],
                },
                top_level_updates={"task_name": meta["task_name"]},
            )
        )
    return repaired


def repair_alfworld_rows(
    current_rows: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
    split: str,
    mappings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    repaired = []
    offset = 0 if split == "train" else 2420

    if len(official_rows) != len(mappings):
        raise ValueError(
            f"ALFWorld {split}: official row count {len(official_rows)} != mapping count {len(mappings)}"
        )

    for row_index, (row, official_row, mapping) in enumerate(zip(current_rows, official_rows, mappings)):
        expected_item_id = (
            f"{mapping['task_type']}_{mapping['task_id']}"
            if split == "train"
            else f"alfworld_{offset + row_index}"
        )
        item_id = official_row["item_id"]
        if item_id != expected_item_id:
            raise ValueError(
                f"ALFWorld {split} row {row_index}: item_id mismatch: {item_id} != {expected_item_id}"
            )

        session_id = offset + row_index
        task_type = mapping["task_type"]
        family = infer_alfworld_family(task_type)

        repaired.append(
            apply_common_metadata(
                row,
                row_index=row_index,
                split=split,
                item_id=item_id,
                session_id=session_id,
                task_category=family,
                task_subcategory=task_type,
                interaction_updates={
                    "name": "alfworld",
                    "game": session_id,
                    "world_type": "Text",
                    "game_id": item_id,
                    "task_type": task_type,
                },
                top_level_updates={
                    "game": session_id,
                    "world_type": "Text",
                    "task_type": task_type,
                },
            )
        )

    session_ids = [row["session_id"] for row in repaired]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError(f"ALFWorld {split}: repaired session_id still contains duplicates.")
    return repaired


def repair_webshop_rows(
    current_rows: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
    split: str,
    goal_metadata: list[dict[str, str]],
) -> list[dict[str, Any]]:
    repaired = []
    for row_index, (row, official_row) in enumerate(zip(current_rows, official_rows)):
        item_id = official_row["item_id"]
        session_id = extract_numeric_suffix(item_id)
        if session_id >= len(goal_metadata):
            raise ValueError(
                f"WebShop session_id {session_id} is out of range for {len(goal_metadata)} reconstructed goals."
            )
        meta = goal_metadata[session_id]
        repaired.append(
            apply_common_metadata(
                row,
                row_index=row_index,
                split=split,
                item_id=item_id,
                session_id=session_id,
                task_category=meta["task_category"],
                task_subcategory=meta["task_subcategory"],
                interaction_updates={
                    "name": "webshop",
                    "task_id": session_id,
                },
                top_level_updates={"product_category": meta["task_subcategory"]},
            )
        )
    return repaired


def assert_lengths(env_name: str, split: str, current_rows: list[dict[str, Any]], official_rows: list[dict[str, Any]]) -> None:
    if len(current_rows) != len(official_rows):
        raise ValueError(
            f"{env_name} {split}: parquet rows {len(current_rows)} != official rows {len(official_rows)}"
        )


def summarize_rows(env_name: str, split: str, rows: list[dict[str, Any]]) -> None:
    category_counts = Counter(row["task_category"] for row in rows)
    print(f"[{env_name}][{split}] rows={len(rows)} categories={dict(category_counts)}")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def repair_target(target: str, sciworld_session_map: dict[int, dict[str, str]], webshop_goal_metadata: list[dict[str, str]]) -> None:
    print(f"=== Repairing target: {target} ===")

    for env_name in ("babyai", "sciworld", "alfworld", "webshop"):
        for split in SPLITS:
            parquet_path = target_path(target, env_name, split)
            if not parquet_path.exists():
                raise FileNotFoundError(f"Missing parquet: {parquet_path}")

            official_rows = load_json(official_json_path(env_name, split))
            current_df = pd.read_parquet(parquet_path)
            current_rows = current_df.to_dict(orient="records")
            assert_lengths(env_name, split, current_rows, official_rows)

            if env_name == "babyai":
                repaired_rows = repair_babyai_rows(current_rows, official_rows, split)
            elif env_name == "sciworld":
                repaired_rows = repair_sciworld_rows(
                    current_rows,
                    official_rows,
                    split,
                    sciworld_session_map,
                )
            elif env_name == "alfworld":
                repaired_rows = repair_alfworld_rows(
                    current_rows,
                    official_rows,
                    split,
                    load_alfworld_mappings(split),
                )
            elif env_name == "webshop":
                repaired_rows = repair_webshop_rows(
                    current_rows,
                    official_rows,
                    split,
                    webshop_goal_metadata,
                )
            else:
                raise ValueError(f"Unsupported env: {env_name}")

            write_rows(parquet_path, repaired_rows)
            summarize_rows(env_name, split, repaired_rows)


def main() -> None:
    args = parse_args()
    sciworld_session_map = build_sciworld_session_map()
    webshop_goal_metadata = build_webshop_goal_metadata()

    for target in args.targets:
        repair_target(target, sciworld_session_map, webshop_goal_metadata)

    print("AgentGym remaining parquet repair completed.")


if __name__ == "__main__":
    main()
