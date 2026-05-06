#!/usr/bin/env python3
"""Align TextCraft parquet metadata with stable session_id/data_idx/goal binding."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "textcraft"
ENV_DIR = PROJECT_ROOT / "envs" / "AgentGym" / "agentenv-textcraft" / "agentenv_textcraft"


def item_id_to_text(item_id: str) -> str:
    return item_id.replace("minecraft:", "").replace("_", " ")


def load_sorted_item_depth_list():
    import sys
    import types
    import importlib.util

    base = str(ENV_DIR)
    pkg = types.ModuleType("agentenv_textcraft")
    pkg.__path__ = [base]
    sys.modules["agentenv_textcraft"] = pkg

    for name in ("utils", "crafting_tree"):
        path = f"{base}/{name}.py"
        spec = importlib.util.spec_from_file_location(f"agentenv_textcraft.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"agentenv_textcraft.{name}"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    crafting_tree_mod = sys.modules["agentenv_textcraft.crafting_tree"]
    tree = crafting_tree_mod.CraftingTree(minecraft_dir=base)
    item_depth_list = list(tree.item_recipes_min_depth(1))
    return sorted(item_depth_list, key=lambda x: x[1])


def extract_session_id(row: dict) -> int:
    item_id = row.get("item_id")
    if isinstance(item_id, str) and item_id.startswith("textcraft_"):
        return int(item_id.split("_")[-1])

    extra_info = row.get("extra_info", {}) or {}
    interaction_kwargs = extra_info.get("interaction_kwargs", {}) or {}

    session_id = interaction_kwargs.get("session_id")
    if session_id is not None:
        return int(session_id)

    original_index = row.get("original_index")
    if original_index is not None:
        return int(original_index)

    raise ValueError(f"Could not resolve session_id for row: {row}")


def align_parquet(path: Path, sorted_item_depth_list) -> None:
    df = pd.read_parquet(path)
    rows = df.to_dict(orient="records")
    aligned_rows = []

    for row in rows:
        session_id = extract_session_id(row)
        goal_item_id, _depth = sorted_item_depth_list[session_id]
        goal_text = item_id_to_text(goal_item_id)
        item_id = f"textcraft_{session_id}"

        aligned = deepcopy(row)
        extra_info = deepcopy(aligned.get("extra_info", {}) or {})
        interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})

        interaction_kwargs["name"] = "textcraft"
        interaction_kwargs["session_id"] = session_id
        interaction_kwargs["data_idx"] = session_id
        interaction_kwargs["item_id"] = item_id
        interaction_kwargs["goal"] = goal_text

        extra_info["interaction_kwargs"] = interaction_kwargs
        aligned["extra_info"] = extra_info
        aligned["item_id"] = item_id
        aligned["original_index"] = session_id

        aligned_rows.append(aligned)

    pd.DataFrame(aligned_rows).to_parquet(path, index=False)


def main() -> None:
    sorted_item_depth_list = load_sorted_item_depth_list()
    for name in ("train.parquet", "test.parquet"):
        align_parquet(DATA_DIR / name, sorted_item_depth_list)
        print(f"Aligned {DATA_DIR / name}")


if __name__ == "__main__":
    main()
