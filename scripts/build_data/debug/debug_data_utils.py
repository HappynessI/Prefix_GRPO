#!/usr/bin/env python3
"""Debug helpers for TextCraft prefix-GRPO datasets."""

import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _is_non_empty_sequence(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, (list, np.ndarray)):
        return len(obj) > 0
    return False


def get_prefix_actions(row: dict) -> list:
    try:
        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict):
            interaction_kwargs = extra_info.get("interaction_kwargs")
            if isinstance(interaction_kwargs, dict):
                prefix_actions = interaction_kwargs.get("prefix_actions")
                if _is_non_empty_sequence(prefix_actions):
                    return list(prefix_actions)
    except Exception:
        pass

    for key in ("extra_info.interaction_kwargs.prefix_actions", "prefix_actions"):
        try:
            prefix_actions = row.get(key)
            if _is_non_empty_sequence(prefix_actions):
                return list(prefix_actions)
        except Exception:
            pass
    return []


def create_debug_subset(input_path: str, output_path: str, max_samples: int = 16) -> dict:
    df = pd.read_parquet(input_path)
    rows = df.to_dict("records")

    with_prefix = [idx for idx, row in enumerate(rows) if len(get_prefix_actions(row)) > 0]
    without_prefix = [idx for idx, row in enumerate(rows) if len(get_prefix_actions(row)) == 0]

    selected_indices = with_prefix[:max_samples]
    if len(selected_indices) < max_samples:
        selected_indices.extend(without_prefix[: max_samples - len(selected_indices)])

    debug_df = df.iloc[selected_indices].reset_index(drop=True)
    debug_df.to_parquet(output_path, index=False)

    stats = {
        "total_samples": len(df),
        "debug_samples": len(debug_df),
        "samples_with_prefix_actions": sum(len(get_prefix_actions(row)) > 0 for row in debug_df.to_dict("records")),
    }
    logger.info("Debug subset stats: %s", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Create debug subsets for TextCraft prefix-GRPO")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--max-samples", type=int, default=16)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.command == "train":
        create_debug_subset(args.input, args.output, args.max_samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
