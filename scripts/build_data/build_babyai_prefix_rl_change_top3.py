#!/usr/bin/env python3
"""Build BabyAI prefix-RL change_top3 data from sampled trajectories."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SAMPLING_PATH = Path(
    "/Data/wyh/datasets/Sampling-Data/babyai_MiniMax-M2.1_20260307_150356/babyai_trajectories.jsonl"
)
DEFAULT_TRAIN_PARQUET = Path("/Data/wyh/datasets/Verl-Data/train/babyai/train.parquet")
DEFAULT_SFT_PARQUET = Path("/Data/wyh/datasets/SFT-Data/babyai/babyai_all_valid.parquet")
DEFAULT_MODEL_PATH = "/Data/wyh/datasets/Verl-Data/outputs/babyai_sft/global_step_300/huggingface"
DEFAULT_OUTPUT_ROOT = Path("/Data/wyh/datasets/Verl-Data/train/babyai/prefix-rl")
DEFAULT_SERVER = "http://127.0.0.1:36005"
DATASET_COMPLETE_NAMES = {
    "main_fixed_gp1_fullflow": "main_fixed_gp1",
    "main_fixed_gp2_fullflow": "main_fixed_gp2",
    "main_raw_top3_fullflow": "main_raw_top3",
    "main_change_top3_w11_fullflow": "main_change_top3_w11",
}
DATASET_STAGE2_FILENAMES = {
    "main_fixed_gp1_fullflow": "prefix_candidates_fixed_gp1.parquet",
    "main_fixed_gp2_fullflow": "prefix_candidates_fixed_gp2.parquet",
    "main_raw_top3_fullflow": "prefix_candidates_raw_top3.parquet",
    "main_change_top3_w11_fullflow": "prefix_candidates_change_top3_w11.parquet",
}
FIXED_DATASET_RATIOS = {
    "main_fixed_gp1_fullflow": (0.1, 0.3, 0.5),
    "main_fixed_gp2_fullflow": (0.25, 0.5, 0.7),
}

START_TAG_RE = re.compile(r"<\|im_start\|>(user|assistant|tool|system)")
END_TAG_RE = re.compile(r"<\|im_end\|>")
ACTION_LINE_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.IGNORECASE)
BRACKET_ACTION_RE = re.compile(r"\[\[\s*(.*?)\s*\]\]", re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=Action:|$)", re.DOTALL | re.IGNORECASE)
GOAL_RE = re.compile(r"Your goal:\s*(.+?)(?:\n|$)", re.IGNORECASE)
MAX_CANONICAL_THOUGHT_CHARS = 512
CANONICALIZATION_VERSION = "babyai_sft_system_skip_sampling_warmup_strict_action_brackets_thought512_cleanthink"

SYSTEM_PROMPT = (
    "You are an exploration master in the BabyAI environment. Every round I will give you an observation "
    "and the currently available actions. Your goal is to complete the given navigation or object-manipulation task.\n\n"
    "Common actions include:\n"
    "- turn left\n"
    "- turn right\n"
    "- move forward\n"
    "- go to <object> <id>\n"
    "- pickup <object> <id>\n"
    "- drop\n"
    "- toggle\n"
    "- go through <door> <id>\n"
    "- toggle and go through <door> <id>\n"
    "- check available actions\n\n"
    "Your response should use exactly this format:\n"
    "Thought:\n"
    "your thoughts.\n\n"
    "Action:\n"
    "[[ your next action ]]\n\n"
    "IMPORTANT: Output exactly one executable action and always wrap it in [[ ]] brackets."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "stage1", "build"), default="all")
    parser.add_argument("--sampling-path", type=Path, default=DEFAULT_SAMPLING_PATH)
    parser.add_argument("--train-parquet-path", type=Path, default=DEFAULT_TRAIN_PARQUET)
    parser.add_argument("--sft-parquet-path", type=Path, default=DEFAULT_SFT_PARQUET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--stage1-output-path", type=Path, default=None)
    parser.add_argument("--stage1-input-glob", type=str, default=None)
    parser.add_argument("--stage1-write-every", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--change-window", type=int, default=11)
    parser.add_argument(
        "--datasets",
        type=str,
        default="main_change_top3_w11_fullflow",
        help=(
            "Comma-separated datasets to build. Supported: "
            "main_fixed_gp1_fullflow,main_fixed_gp2_fullflow,"
            "main_raw_top3_fullflow,main_change_top3_w11_fullflow"
        ),
    )
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--replay-concurrency", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-prompt-tokens", type=int, default=2400)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--skip-replay", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [dict(msg) for msg in list(messages)]


def make_sample_uid(item_id: str, sample_idx: int) -> str:
    return f"{item_id}__{int(sample_idx)}"


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
            return turn_idx, float(q)
    return num_assistant_messages - 1, 1.0


def parse_task_id(item_id: str) -> int:
    if not str(item_id).startswith("babyai_"):
        raise ValueError(f"Unexpected BabyAI item_id: {item_id}")
    return int(str(item_id).split("_", 1)[1])


def load_system_prompt(sft_parquet_path: Path) -> str:
    if sft_parquet_path.exists():
        df = pd.read_parquet(sft_parquet_path, columns=["messages"])
        if not df.empty:
            messages = normalize_messages(df.iloc[0]["messages"])
            for msg in messages:
                if msg.get("role") == "system":
                    return str(msg.get("content", "")).strip() or SYSTEM_PROMPT
    return SYSTEM_PROMPT


def load_raw_base_rows(train_parquet_path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_parquet(train_parquet_path)
    rows: Dict[str, Dict[str, Any]] = {}
    for row_index, row in enumerate(df.to_dict(orient="records")):
        row = dict(row)
        extra_info = row.get("extra_info", {}) or {}
        # Sampling was run over the filtered train parquet row order, while the
        # official BabyAI data_idx may skip removed levels. Replay/training must
        # use the same reset key as sampling, otherwise the prompt and env state
        # refer to different BabyAI instances.
        row["sampling_env_reset_key"] = int(extra_info.get("index", row_index))
        rows[str(row["item_id"])] = dict(row)
    return rows


def is_sampling_instruction(content: str) -> bool:
    return "You are an exploration master that wants to finish every goal" in str(content)


def is_warmup_assistant(content: str) -> bool:
    return "OK. I'll follow your instructions" in str(content)


def sanitize_action(action: str) -> str:
    action = re.sub(r"^\[\[\s*|\s*\]\]$", "", str(action).strip())
    action = re.sub(r"^[`'\"]+|[`'\"]+$", "", action.strip())
    return " ".join(action.split()).strip().lower()


def action_marker_count(text: str) -> int:
    text = str(text or "")
    bracket_matches = BRACKET_ACTION_RE.findall(text)
    if bracket_matches:
        return len(bracket_matches)
    return len(ACTION_LINE_RE.findall(text))


def extract_action(text: str) -> Optional[str]:
    text = str(text or "").strip()
    bracket_matches = BRACKET_ACTION_RE.findall(text)
    if bracket_matches:
        action = sanitize_action(bracket_matches[-1])
        return action or None

    action_matches = ACTION_LINE_RE.findall(text)
    if action_matches:
        action = sanitize_action(action_matches[-1])
        return action or None

    cleaned = THINK_RE.sub("", text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = sanitize_action(line)
        if candidate and len(candidate) < 120 and any(
            key in candidate
            for key in ("turn", "move", "pickup", "pick up", "drop", "toggle", "go to", "go through", "check")
        ):
            return candidate
    return None


def extract_thought(text: str) -> str:
    text = str(text or "")
    thought_match = THOUGHT_RE.search(text)
    if thought_match:
        value = thought_match.group(1).strip()
        if value:
            return value

    think_match = THINK_RE.search(text)
    if think_match:
        value = think_match.group(1).strip()
        if value:
            return value

    before_action = ACTION_LINE_RE.split(text)[0].strip()
    before_action = THINK_RE.sub("", before_action).strip()
    return before_action or "I will take the next valid action."


def canonicalize_thought_text(text: str) -> str:
    thought = str(text or "")
    thought = THINK_RE.sub(lambda match: f" {match.group(1)} ", thought)
    thought = THINK_TAG_RE.sub(" ", thought)
    thought = re.sub(r"\bThought:\s*", " ", thought, flags=re.IGNORECASE)
    thought = " ".join(thought.split()).strip()
    if not thought:
        thought = "I will take the next valid action."
    if len(thought) > MAX_CANONICAL_THOUGHT_CHARS:
        thought = thought[:MAX_CANONICAL_THOUGHT_CHARS].rstrip() + "..."
    return thought


def canonicalize_assistant(content: str) -> str:
    action = extract_action(content)
    thought = canonicalize_thought_text(extract_thought(content))
    if action:
        return f"Thought:\n{thought}\n\nAction:\n[[ {action} ]]"
    return f"Thought:\n{thought}"


def canonicalize_sample(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    conversations = normalize_messages(sample["conversations"])
    episode_messages: List[Dict[str, Any]] = []
    for msg in conversations:
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if role == "user" and is_sampling_instruction(content):
            continue
        if role == "assistant" and is_warmup_assistant(content):
            continue
        if role == "assistant":
            episode_messages.append({"role": "assistant", "content": canonicalize_assistant(content)})
        elif role == "user":
            episode_messages.append({"role": "user", "content": content})

    if not episode_messages or episode_messages[0].get("role") != "user":
        raise ValueError(f"Could not find initial BabyAI user observation for {sample.get('item_id')}")

    item_id = str(sample["item_id"])
    task_id = parse_task_id(item_id)
    sample_idx = int(sample["sample_idx"])
    sample_uid = make_sample_uid(item_id, sample_idx)
    goal_match = GOAL_RE.search(str(episode_messages[0].get("content", "")))
    goal = goal_match.group(1).strip() if goal_match else None

    return {
        "sample_uid": sample_uid,
        "item_id": item_id,
        "sample_idx": sample_idx,
        "task_id": task_id,
        "goal": goal,
        "success": int(bool(sample.get("success", 0))),
        "reward": float(sample.get("reward", 0.0)),
        "model": sample.get("model", ""),
        "task_name": sample.get("task_name", "babyai"),
        "episode_messages": episode_messages,
        "canonical_messages": [{"role": "system", "content": system_prompt}] + episode_messages,
    }


def assistant_indices(messages: Sequence[Dict[str, Any]]) -> List[int]:
    return [idx for idx, msg in enumerate(messages) if msg.get("role") == "assistant"]


def split_episode_messages(
    episode_messages: List[Dict[str, Any]], cut_turn_idx: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    indices = assistant_indices(episode_messages)
    if not indices:
        return [], list(episode_messages)
    last_prefix_idx = indices[min(cut_turn_idx, len(indices) - 1)]
    return list(episode_messages[: last_prefix_idx + 1]), list(episode_messages[last_prefix_idx + 1 :])


def extract_actions_from_messages(messages: Sequence[Dict[str, Any]]) -> List[str]:
    actions: List[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        action = extract_action(str(msg.get("content", "")))
        if action:
            actions.append(action)
    return actions


def extract_replay_actions_from_messages(messages: Sequence[Dict[str, Any]]) -> List[str]:
    actions: List[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", ""))
        if action_marker_count(content) != 1:
            continue
        action = extract_action(content)
        if action:
            actions.append(action)
    return actions


def last_assistant_replay_action_taken(messages: Sequence[Dict[str, Any]]) -> bool:
    for msg in reversed(list(messages)):
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", ""))
        return action_marker_count(content) == 1 and extract_action(content) is not None
    return False


def is_parser_feedback(text: Optional[str]) -> bool:
    normalized = normalize_text(text)
    return normalized.startswith("Error: Only one 'Action' is allowed") or normalized.startswith(
        "Please provide a valid action"
    )


def render_chat(tokenizer, messages: List[Dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)


def tokenize_chat(tokenizer, messages: List[Dict[str, Any]]) -> torch.Tensor:
    text = render_chat(tokenizer, messages)
    return tokenizer(text, add_special_tokens=True, return_tensors="pt").input_ids[0]


def compute_message_spans(tokenizer, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text = render_chat(tokenizer, messages)
    result = tokenizer(text, add_special_tokens=True, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = result.offset_mapping[0].tolist()
    start_matches = list(START_TAG_RE.finditer(text))
    end_matches = list(END_TAG_RE.finditer(text))
    if len(start_matches) != len(messages) or len(end_matches) != len(messages):
        raise ValueError(
            "Conversation tag count does not match message count: "
            f"{len(start_matches)} starts, {len(end_matches)} ends, {len(messages)} messages"
        )

    role_counts: Dict[str, int] = {}
    spans: List[Dict[str, Any]] = []
    for message_index, msg in enumerate(messages):
        role = str(msg.get("role", ""))
        role_counts[role] = role_counts.get(role, 0) + 1
        start_char = start_matches[message_index].end()
        end_char = end_matches[message_index].end()
        token_start = None
        token_end = None
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start is None:
                continue
            if char_start < end_char and char_end > start_char:
                if token_start is None:
                    token_start = token_idx
                token_end = token_idx + 1
        if token_start is None or token_end is None:
            raise ValueError(f"Could not map message {message_index} ({role}) to token span")
        spans.append(
            {
                "message_index": int(message_index),
                "role": role,
                "role_turn_idx": int(role_counts[role]),
                "token_start": int(token_start),
                "token_end": int(token_end),
                "entropy_start": max(0, int(token_start) - 1),
                "entropy_end": max(0, int(token_end) - 1),
            }
        )
    return spans


def compute_sequence_forward(model, tokenizer, messages: List[Dict[str, Any]], device: str) -> Dict[str, Any]:
    text = render_chat(tokenizer, messages)
    batch = tokenizer(text, add_special_tokens=True, return_tensors="pt")
    input_ids = batch.input_ids[0].to(device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), use_cache=False)
        logits = outputs.logits[0]
        if logits.shape[0] <= 1:
            return {"token_length": int(logits.shape[0]), "sequence_old_log_probs": [], "sequence_entropies": []}
        next_logits = logits[:-1].float()
        target_tokens = input_ids[1:].unsqueeze(-1)
        log_probs = F.log_softmax(next_logits, dim=-1)
        old_log_probs = log_probs.gather(dim=-1, index=target_tokens).squeeze(-1)
        probs = torch.softmax(next_logits, dim=-1)
        entropies = torch.logsumexp(next_logits, dim=-1) - torch.sum(probs * next_logits, dim=-1)

    result = {
        "token_length": int(input_ids.shape[0]),
        "sequence_old_log_probs": old_log_probs.detach().cpu().float().tolist(),
        "sequence_entropies": entropies.detach().cpu().float().tolist(),
    }
    del outputs, logits, next_logits, target_tokens, log_probs, old_log_probs, probs, entropies, input_ids, attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def iter_selected_samples(samples: Sequence[Dict[str, Any]], max_samples: Optional[int], shard_index: int, num_shards: int):
    selected = list(samples)
    if max_samples is not None:
        selected = selected[:max_samples]
    for idx, sample in enumerate(selected):
        if idx % max(1, num_shards) == shard_index:
            yield idx, sample


def stage1_forward(args: argparse.Namespace, system_prompt: str) -> Path:
    samples = load_jsonl(args.sampling_path)
    output_path = args.stage1_output_path
    if output_path is None:
        suffix = "" if args.num_shards == 1 else f".shard{args.shard_index:02d}-of-{args.num_shards:02d}"
        output_path = args.output_root / "stage1_forward" / f"babyai_sft_step300_oldlogprob_entropy{suffix}.parquet"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(args.device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    part_paths: List[Path] = []
    parts_dir = output_path.parent / f"{output_path.stem}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    existing_sample_uids: set[str] = set()
    for existing_part in sorted(parts_dir.glob("part_*.parquet")):
        try:
            part_df = pd.read_parquet(existing_part, columns=["sample_uid", "canonicalization"])
        except Exception:
            continue
        if set(part_df["canonicalization"].dropna().astype(str).unique()) != {CANONICALIZATION_VERSION}:
            print(
                f"[stage1] ignoring stale canonicalization part: {existing_part}",
                flush=True,
            )
            continue
        existing_sample_uids.update(str(value) for value in part_df["sample_uid"].tolist())
        part_paths.append(existing_part)

    def flush_rows() -> None:
        nonlocal rows
        if not rows:
            return
        part_index = len(part_paths)
        part_path = parts_dir / f"part_{part_index:05d}.parquet"
        pd.DataFrame(rows).sort_values(["task_id", "sample_idx"]).to_parquet(part_path, index=False)
        part_paths.append(part_path)
        rows = []
        gc.collect()

    selected = list(iter_selected_samples(samples, args.max_samples, args.shard_index, args.num_shards))
    start = time.time()
    for local_idx, (global_idx, sample) in enumerate(selected, start=1):
        canonical = canonicalize_sample(sample, system_prompt)
        if canonical["sample_uid"] in existing_sample_uids:
            continue
        spans = compute_message_spans(tokenizer, canonical["canonical_messages"])
        forward = compute_sequence_forward(model, tokenizer, canonical["canonical_messages"], args.device)
        if len(forward["sequence_entropies"]) != forward["token_length"] - 1:
            raise RuntimeError(f"Entropy length mismatch for {canonical['sample_uid']}")
        if len(forward["sequence_old_log_probs"]) != forward["token_length"] - 1:
            raise RuntimeError(f"Old-logprob length mismatch for {canonical['sample_uid']}")

        assistant_spans = [span for span in spans if span["role"] == "assistant"]
        rows.append(
            {
                "global_index": int(global_idx),
                "sample_uid": canonical["sample_uid"],
                "item_id": canonical["item_id"],
                "sample_idx": canonical["sample_idx"],
                "task_id": canonical["task_id"],
                "goal": canonical["goal"],
                "success": canonical["success"],
                "reward": canonical["reward"],
                "num_messages": len(canonical["canonical_messages"]),
                "num_episode_messages": len(canonical["episode_messages"]),
                "num_assistant_messages": len(assistant_spans),
                "token_length": forward["token_length"],
                "entropy_length": len(forward["sequence_entropies"]),
                "old_logprob_length": len(forward["sequence_old_log_probs"]),
                "entropy_coordinate_note": "sequence_*[i] aligns to token_position=i+1 in the canonicalized trajectory",
                "sequence_old_log_probs": forward["sequence_old_log_probs"],
                "sequence_entropies": forward["sequence_entropies"],
                "message_spans": spans,
                "assistant_turn_spans": assistant_spans,
                "model_path": args.model_path,
                "source_sampling_path": str(args.sampling_path),
                "canonicalization": CANONICALIZATION_VERSION,
            }
        )
        existing_sample_uids.add(canonical["sample_uid"])
        if len(rows) >= max(1, args.stage1_write_every):
            flush_rows()
        if local_idx == len(selected) or local_idx % max(1, args.progress_every) == 0:
            elapsed = time.time() - start
            print(
                f"[stage1] shard={args.shard_index}/{args.num_shards} "
                f"processed={local_idx}/{len(selected)} elapsed_sec={elapsed:.1f}",
                flush=True,
            )
        gc.collect()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    flush_rows()

    if not part_paths:
        df = pd.DataFrame(rows)
    else:
        df = pd.concat([pd.read_parquet(path) for path in sorted(part_paths)], ignore_index=True)
    df = df.drop_duplicates(subset=["sample_uid"]).sort_values(["task_id", "sample_idx"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    manifest = {
        "stage": "stage1_forward",
        "output_path": str(output_path),
        "sampling_path": str(args.sampling_path),
        "model_path": args.model_path,
        "canonicalization": CANONICALIZATION_VERSION,
        "rows": int(len(df)),
        "unique_sample_uid": int(df["sample_uid"].nunique()) if not df.empty else 0,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return output_path


def centered_smooth(values: Sequence[float], window: int) -> List[float]:
    if not values:
        return []
    radius = max(0, int(window) // 2)
    output: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        output.append(float(sum(values[start:end]) / max(1, end - start)))
    return output


def select_change_topk_candidates(
    canonical_rows: Dict[str, Dict[str, Any]],
    forward_df: pd.DataFrame,
    *,
    top_k: int,
    change_window: int,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in forward_df.to_dict(orient="records"):
        sample_uid = str(row["sample_uid"])
        canonical = canonical_rows[sample_uid]
        episode_messages = canonical["episode_messages"]
        num_assistant = int(row["num_assistant_messages"])
        if num_assistant <= 0:
            continue

        entropies = [float(x) for x in list(row["sequence_entropies"])]
        assistant_spans = normalize_messages(row["assistant_turn_spans"])
        domain_tokens: List[Dict[str, Any]] = []
        for span in assistant_spans:
            message_index = int(span["message_index"])
            if message_index <= 0:
                continue
            source_content = str(canonical["canonical_messages"][message_index].get("content", ""))
            for entropy_idx in range(int(span["entropy_start"]), int(span["entropy_end"])):
                if not (0 <= entropy_idx < len(entropies)):
                    continue
                token_position = entropy_idx + 1
                domain_tokens.append(
                    {
                        "source_message_index": message_index - 1,
                        "source_role_turn_idx": int(span["role_turn_idx"]),
                        "source_token_position": int(token_position),
                        "source_message_content_preview": source_content[:240],
                        "entropy": float(entropies[entropy_idx]),
                    }
                )

        if not domain_tokens:
            continue

        smoothed = centered_smooth([tok["entropy"] for tok in domain_tokens], change_window)
        scored: List[Dict[str, Any]] = []
        for domain_idx, tok in enumerate(domain_tokens):
            change_score = 0.0 if domain_idx == 0 else abs(float(smoothed[domain_idx]) - float(smoothed[domain_idx - 1]))
            candidate = dict(tok)
            candidate.update(
                {
                    "source_domain_rank": int(domain_idx + 1),
                    "num_domain_tokens": int(len(domain_tokens)),
                    "source_token_entropy": float(tok["entropy"]),
                    "smoothed_entropy": float(smoothed[domain_idx]),
                    "change_score": float(change_score),
                    "selection_score": float(change_score),
                }
            )
            scored.append(candidate)

        chosen_by_cut: Dict[int, Dict[str, Any]] = {}
        for candidate in sorted(scored, key=lambda item: (-item["selection_score"], item["source_domain_rank"])):
            cut_turn_idx = int(candidate["source_role_turn_idx"]) - 1
            if cut_turn_idx < 0:
                continue
            if cut_turn_idx not in chosen_by_cut:
                chosen_by_cut[cut_turn_idx] = candidate
            if len(chosen_by_cut) >= top_k:
                break

        for rank, (cut_turn_idx, candidate) in enumerate(
            sorted(chosen_by_cut.items(), key=lambda item: (-item[1]["selection_score"], item[1]["source_domain_rank"])),
            start=1,
        ):
            prefix_messages, continuation_messages = split_episode_messages(episode_messages, cut_turn_idx)
            if not continuation_messages or continuation_messages[0].get("role") != "user":
                continue
            prefix_actions = extract_replay_actions_from_messages(prefix_messages)
            q = float(cut_turn_idx / (num_assistant - 1)) if num_assistant > 1 else 0.0
            records.append(
                {
                    "candidate_uid": f"{sample_uid}__entropy_change_topk_w11_interaction_assistant_k{top_k}__r{rank}",
                    "sample_uid": sample_uid,
                    "item_id": canonical["item_id"],
                    "sample_idx": int(canonical["sample_idx"]),
                    "task_id": int(canonical["task_id"]),
                    "goal": canonical.get("goal"),
                    "success": int(canonical.get("success", 0)),
                    "reward": float(canonical.get("reward", 0.0)),
                    "strategy": f"entropy_change_topk_w{change_window}_interaction_assistant_k{top_k}",
                    "strategy_family": "entropy_topk",
                    "scorer": "change_topk",
                    "domain": "interaction_assistant",
                    "candidate_rank": int(rank),
                    "top_k": int(top_k),
                    "change_window": int(change_window),
                    "min_domain_gap": 0,
                    "mapping_mode": "contained_assistant",
                    "cut_turn_idx": int(cut_turn_idx),
                    "cut_assistant_turn_idx_one_based": int(cut_turn_idx + 1),
                    "cut_relative_position_q": q,
                    "num_assistant_messages_total": int(num_assistant),
                    "source_role": "assistant",
                    "source_message_index": int(candidate["source_message_index"]),
                    "source_role_turn_idx": int(candidate["source_role_turn_idx"]),
                    "source_message_is_warmup": False,
                    "source_message_content_preview": candidate["source_message_content_preview"],
                    "source_token_position": int(candidate["source_token_position"]),
                    "source_domain_rank": int(candidate["source_domain_rank"]),
                    "num_domain_tokens": int(candidate["num_domain_tokens"]),
                    "mapped_assistant_message_index": int(candidate["source_message_index"]),
                    "source_token_entropy": float(candidate["source_token_entropy"]),
                    "smoothed_entropy": float(candidate["smoothed_entropy"]),
                    "change_score": float(candidate["change_score"]),
                    "selection_score": float(candidate["selection_score"]),
                    "num_prefix_messages": int(len(prefix_messages)),
                    "num_continuation_messages": int(len(continuation_messages)),
                    "num_prefix_assistant_messages": int(len(assistant_indices(prefix_messages))),
                    "num_continuation_assistant_messages": int(len(assistant_indices(continuation_messages))),
                    "prefix_messages": prefix_messages,
                    "continuation_messages": continuation_messages,
                    "prefix_actions": prefix_actions,
                    "canonicalization": CANONICALIZATION_VERSION,
                    "prefix_last_assistant_replay_action_taken": bool(
                        last_assistant_replay_action_taken(prefix_messages)
                    ),
                }
            )

    return pd.DataFrame(records).sort_values(["task_id", "sample_idx", "candidate_rank"]).reset_index(drop=True)


def select_raw_topk_candidates(
    canonical_rows: Dict[str, Dict[str, Any]],
    forward_df: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in forward_df.to_dict(orient="records"):
        sample_uid = str(row["sample_uid"])
        canonical = canonical_rows[sample_uid]
        episode_messages = canonical["episode_messages"]
        num_assistant = int(row["num_assistant_messages"])
        if num_assistant <= 0:
            continue

        entropies = [float(x) for x in list(row["sequence_entropies"])]
        assistant_spans = normalize_messages(row["assistant_turn_spans"])
        scored: List[Dict[str, Any]] = []
        domain_rank = 0
        for span in assistant_spans:
            message_index = int(span["message_index"])
            if message_index <= 0:
                continue
            source_content = str(canonical["canonical_messages"][message_index].get("content", ""))
            for entropy_idx in range(int(span["entropy_start"]), int(span["entropy_end"])):
                if not (0 <= entropy_idx < len(entropies)):
                    continue
                domain_rank += 1
                entropy = float(entropies[entropy_idx])
                scored.append(
                    {
                        "source_message_index": message_index - 1,
                        "source_role_turn_idx": int(span["role_turn_idx"]),
                        "source_token_position": int(entropy_idx + 1),
                        "source_message_content_preview": source_content[:240],
                        "source_domain_rank": int(domain_rank),
                        "num_domain_tokens": 0,
                        "source_token_entropy": entropy,
                        "smoothed_entropy": entropy,
                        "change_score": 0.0,
                        "selection_score": entropy,
                    }
                )

        if not scored:
            continue
        for candidate in scored:
            candidate["num_domain_tokens"] = len(scored)

        chosen_by_cut: Dict[int, Dict[str, Any]] = {}
        for candidate in sorted(scored, key=lambda item: (-item["selection_score"], item["source_domain_rank"])):
            cut_turn_idx = int(candidate["source_role_turn_idx"]) - 1
            if cut_turn_idx < 0:
                continue
            if cut_turn_idx not in chosen_by_cut:
                chosen_by_cut[cut_turn_idx] = candidate
            if len(chosen_by_cut) >= top_k:
                break

        for rank, (cut_turn_idx, candidate) in enumerate(
            sorted(chosen_by_cut.items(), key=lambda item: (-item[1]["selection_score"], item[1]["source_domain_rank"])),
            start=1,
        ):
            prefix_messages, continuation_messages = split_episode_messages(episode_messages, cut_turn_idx)
            if not continuation_messages or continuation_messages[0].get("role") != "user":
                continue
            prefix_actions = extract_replay_actions_from_messages(prefix_messages)
            q = float(cut_turn_idx / (num_assistant - 1)) if num_assistant > 1 else 0.0
            records.append(
                {
                    "candidate_uid": f"{sample_uid}__entropy_raw_topk_interaction_assistant_k{top_k}__r{rank}",
                    "sample_uid": sample_uid,
                    "item_id": canonical["item_id"],
                    "sample_idx": int(canonical["sample_idx"]),
                    "task_id": int(canonical["task_id"]),
                    "goal": canonical.get("goal"),
                    "success": int(canonical.get("success", 0)),
                    "reward": float(canonical.get("reward", 0.0)),
                    "strategy": f"entropy_raw_topk_interaction_assistant_k{top_k}",
                    "strategy_family": "entropy_topk",
                    "scorer": "raw_topk",
                    "domain": "interaction_assistant",
                    "candidate_rank": int(rank),
                    "variant_label": f"rank{int(rank)}",
                    "top_k": int(top_k),
                    "change_window": 0,
                    "min_domain_gap": 0,
                    "mapping_mode": "contained_assistant",
                    "cut_turn_idx": int(cut_turn_idx),
                    "cut_assistant_turn_idx_one_based": int(cut_turn_idx + 1),
                    "cut_relative_position_q": q,
                    "num_assistant_messages_total": int(num_assistant),
                    "source_role": "assistant",
                    "source_message_index": int(candidate["source_message_index"]),
                    "source_role_turn_idx": int(candidate["source_role_turn_idx"]),
                    "source_message_is_warmup": False,
                    "source_message_content_preview": candidate["source_message_content_preview"],
                    "source_token_position": int(candidate["source_token_position"]),
                    "source_domain_rank": int(candidate["source_domain_rank"]),
                    "num_domain_tokens": int(candidate["num_domain_tokens"]),
                    "mapped_assistant_message_index": int(candidate["source_message_index"]),
                    "source_token_entropy": float(candidate["source_token_entropy"]),
                    "smoothed_entropy": float(candidate["smoothed_entropy"]),
                    "change_score": 0.0,
                    "selection_score": float(candidate["selection_score"]),
                    "num_prefix_messages": int(len(prefix_messages)),
                    "num_continuation_messages": int(len(continuation_messages)),
                    "num_prefix_assistant_messages": int(len(assistant_indices(prefix_messages))),
                    "num_continuation_assistant_messages": int(len(assistant_indices(continuation_messages))),
                    "prefix_messages": prefix_messages,
                    "continuation_messages": continuation_messages,
                    "prefix_actions": prefix_actions,
                    "canonicalization": CANONICALIZATION_VERSION,
                    "prefix_last_assistant_replay_action_taken": bool(
                        last_assistant_replay_action_taken(prefix_messages)
                    ),
                }
            )

    return pd.DataFrame(records).sort_values(["task_id", "sample_idx", "candidate_rank"]).reset_index(drop=True)


def select_fixed_ratio_candidates(
    canonical_rows: Dict[str, Dict[str, Any]],
    *,
    ratios: Sequence[float],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    ratio_order = {float(ratio): idx + 1 for idx, ratio in enumerate(ratios)}
    for canonical in canonical_rows.values():
        episode_messages = canonical["episode_messages"]
        num_assistant = len(assistant_indices(episode_messages))
        if num_assistant <= 0:
            continue
        for ratio in ratios:
            ratio = float(ratio)
            strategy = f"fixed_ratio_{ratio_to_name(ratio)}"
            cut_turn_idx, q = choose_cut_turn_idx(num_assistant, ratio)
            prefix_messages, continuation_messages = split_episode_messages(episode_messages, cut_turn_idx)
            if not continuation_messages or continuation_messages[0].get("role") != "user":
                continue
            prefix_actions = extract_replay_actions_from_messages(prefix_messages)
            records.append(
                {
                    "candidate_uid": f"{canonical['sample_uid']}__{strategy}",
                    "sample_uid": canonical["sample_uid"],
                    "item_id": canonical["item_id"],
                    "sample_idx": int(canonical["sample_idx"]),
                    "task_id": int(canonical["task_id"]),
                    "goal": canonical.get("goal"),
                    "success": int(canonical.get("success", 0)),
                    "reward": float(canonical.get("reward", 0.0)),
                    "strategy": strategy,
                    "strategy_family": "fixed_ratio",
                    "scorer": "fixed_ratio",
                    "domain": "interaction_assistant",
                    "candidate_rank": int(ratio_order[ratio]),
                    "variant_label": strategy,
                    "top_k": 0,
                    "change_window": 0,
                    "min_domain_gap": 0,
                    "mapping_mode": "assistant_turn_ratio",
                    "cut_turn_idx": int(cut_turn_idx),
                    "cut_assistant_turn_idx_one_based": int(cut_turn_idx + 1),
                    "cut_relative_position_q": float(q),
                    "num_assistant_messages_total": int(num_assistant),
                    "source_role": "assistant",
                    "source_message_index": -1,
                    "source_role_turn_idx": int(cut_turn_idx + 1),
                    "source_message_is_warmup": False,
                    "source_message_content_preview": "",
                    "source_token_position": -1,
                    "source_domain_rank": -1,
                    "num_domain_tokens": 0,
                    "mapped_assistant_message_index": -1,
                    "source_token_entropy": 0.0,
                    "smoothed_entropy": 0.0,
                    "change_score": 0.0,
                    "selection_score": float(q),
                    "num_prefix_messages": int(len(prefix_messages)),
                    "num_continuation_messages": int(len(continuation_messages)),
                    "num_prefix_assistant_messages": int(len(assistant_indices(prefix_messages))),
                    "num_continuation_assistant_messages": int(len(assistant_indices(continuation_messages))),
                    "prefix_messages": prefix_messages,
                    "continuation_messages": continuation_messages,
                    "prefix_actions": prefix_actions,
                    "canonicalization": CANONICALIZATION_VERSION,
                    "prefix_last_assistant_replay_action_taken": bool(
                        last_assistant_replay_action_taken(prefix_messages)
                    ),
                }
            )
    return pd.DataFrame(records).sort_values(["task_id", "sample_idx", "candidate_rank"]).reset_index(drop=True)


def build_prompt(system_prompt: str, prefix_messages: List[Dict[str, Any]], continuation_messages: List[Dict[str, Any]]):
    prompt = [{"role": "system", "content": system_prompt}]
    prompt.extend(deepcopy(prefix_messages))
    for msg in continuation_messages:
        if msg.get("role") == "user":
            prompt.append({"role": "user", "content": str(msg.get("content", ""))})
            break
    return prompt


def sampling_env_reset_key(base_row: Dict[str, Any]) -> int:
    if "sampling_env_reset_key" in base_row:
        return int(base_row["sampling_env_reset_key"])
    extra_info = base_row.get("extra_info", {}) or {}
    if "index" in extra_info:
        return int(extra_info["index"])
    interaction_kwargs = extra_info.get("interaction_kwargs", {}) or {}
    return int(interaction_kwargs.get("data_idx", interaction_kwargs.get("session_id", 0)))


def update_babyai_interaction_kwargs(base_row: Dict[str, Any], prefix_actions: Sequence[str]) -> Dict[str, Any]:
    base_extra_info = base_row["extra_info"]
    extra_info = deepcopy(base_extra_info)
    interaction_kwargs = deepcopy(extra_info.get("interaction_kwargs", {}) or {})
    reset_key = sampling_env_reset_key(base_row)
    interaction_kwargs["name"] = "babyai"
    interaction_kwargs["data_idx"] = reset_key
    interaction_kwargs["session_id"] = reset_key
    interaction_kwargs["prefix_actions"] = list(prefix_actions)
    interaction_kwargs["official_data_idx"] = base_extra_info.get("interaction_kwargs", {}).get("data_idx")
    interaction_kwargs["official_session_id"] = base_extra_info.get("interaction_kwargs", {}).get("session_id")
    extra_info["sampling_env_reset_key"] = reset_key
    extra_info["interaction_kwargs"] = interaction_kwargs
    return extra_info


def build_prefix_record(
    row: Dict[str, Any],
    base_row: Dict[str, Any],
    *,
    main_dataset: str,
    system_prompt: str,
) -> Dict[str, Any]:
    prefix_messages = normalize_messages(row["prefix_messages"])
    continuation_messages = normalize_messages(row["continuation_messages"])
    prefix_actions = [str(x) for x in list(row.get("prefix_actions", []))]
    candidate_rank = int(row.get("candidate_rank", 1))
    variant_label = str(row.get("variant_label") or f"rank{candidate_rank}")
    extra_info = update_babyai_interaction_kwargs(base_row, prefix_actions)
    extra_info["sample_uid"] = row["sample_uid"]
    extra_info["main_dataset"] = main_dataset
    extra_info["variant_label"] = variant_label
    return {
        "record_uid": f"{main_dataset}::{row['sample_uid']}::{variant_label}",
        "main_dataset": main_dataset,
        "variant_label": variant_label,
        "is_raw_variant": False,
        "sample_uid": row["sample_uid"],
        "item_id": row["item_id"],
        "sample_idx": int(row["sample_idx"]),
        "task_id": int(row["task_id"]),
        "goal": row.get("goal"),
        "strategy": row["strategy"],
        "data_source": base_row["data_source"],
        "ability": base_row["ability"],
        "prompt": build_prompt(system_prompt, prefix_messages, continuation_messages),
        "reward_model": deepcopy(base_row["reward_model"]),
        "extra_info": extra_info,
        "prefix_messages": prefix_messages,
        "continuation_messages": continuation_messages,
        "prefix_actions": prefix_actions,
        "replay_category": row.get("replay_category", "constructed_from_sampling"),
        "assistant_prefix_old_log_probs": None,
        "prefix_mask": None,
        "prefix_token_count": None,
        "assistant_prefix_span": None,
        "source_oldlogprob_model_path": None,
        "prefix_coordinate_system": "canonicalized_prompt",
        "canonicalization": str(row.get("canonicalization", CANONICALIZATION_VERSION)),
        "candidate_uid": row.get("candidate_uid"),
        "candidate_rank": candidate_rank,
        "cut_turn_idx": int(row["cut_turn_idx"]),
        "cut_relative_position_q": float(row.get("cut_relative_position_q", 0.0)),
        "num_assistant_messages_total": int(row.get("num_assistant_messages_total", 0)),
        "selection_score": float(row.get("selection_score", 0.0)),
        "source_role": row.get("source_role"),
        "source_token_position": int(row.get("source_token_position", -1)),
        "source_dataset": row.get("source_dataset", "babyai_entropy_stage2_candidates"),
        "source_token_entropy": float(row.get("source_token_entropy", 0.0)),
        "smoothed_entropy": float(row.get("smoothed_entropy", 0.0)),
        "change_score": float(row.get("change_score", 0.0)),
        "prefix_last_assistant_replay_action_taken": bool(
            row.get("prefix_last_assistant_replay_action_taken", True)
        ),
    }


def build_raw_record(
    canonical: Dict[str, Any],
    base_row: Dict[str, Any],
    *,
    main_dataset: str,
    system_prompt: str,
    model_path: str,
) -> Dict[str, Any]:
    extra_info = update_babyai_interaction_kwargs(base_row, [])
    extra_info["sample_uid"] = canonical["sample_uid"]
    extra_info["main_dataset"] = main_dataset
    extra_info["variant_label"] = "raw"
    initial_user = deepcopy(canonical["episode_messages"][0])
    return {
        "record_uid": f"{main_dataset}::{canonical['sample_uid']}::raw",
        "main_dataset": main_dataset,
        "variant_label": "raw",
        "is_raw_variant": True,
        "sample_uid": canonical["sample_uid"],
        "item_id": canonical["item_id"],
        "sample_idx": int(canonical["sample_idx"]),
        "task_id": int(canonical["task_id"]),
        "goal": canonical.get("goal"),
        "strategy": "raw",
        "data_source": base_row["data_source"],
        "ability": base_row["ability"],
        "prompt": [{"role": "system", "content": system_prompt}, initial_user],
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
        "canonicalization": CANONICALIZATION_VERSION,
    }


def normalize_text(text: Optional[str]) -> str:
    return " ".join(str(text or "").split()).strip()


def create_env(server: str, timeout: float) -> int:
    response = requests.post(f"{server}/create", json={}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return int(data["id"])


def reset_env(server: str, env_id: int, data_idx: int, timeout: float) -> Dict[str, Any]:
    response = requests.post(f"{server}/reset", json={"id": env_id, "data_idx": int(data_idx)}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return data


def step_env(server: str, env_id: int, action: str, timeout: float) -> Dict[str, Any]:
    response = requests.post(f"{server}/step", json={"id": env_id, "action": str(action)}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return data


def close_env(server: str, env_id: int, timeout: float) -> None:
    try:
        requests.post(f"{server}/close", json={"id": env_id}, timeout=timeout)
    except Exception:
        pass


def first_user(messages: Sequence[Dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return None


def replay_validate_one(row: Dict[str, Any], raw_base_rows: Dict[str, Dict[str, Any]], server: str, timeout: float):
    item_id = str(row["item_id"])
    base_row = raw_base_rows[item_id]
    data_idx = sampling_env_reset_key(base_row)
    prefix_actions = [str(action) for action in list(row.get("prefix_actions", []))]
    expected_cut = first_user(normalize_messages(row["continuation_messages"]))
    env_id: Optional[int] = None
    replay_cut = ""
    replay_done = False
    initial_observation = ""
    error = None
    try:
        env_id = create_env(server, timeout)
        reset_data = reset_env(server, env_id, data_idx, timeout)
        initial_observation = str(reset_data.get("observation", ""))
        current = reset_data
        for action in prefix_actions:
            current = step_env(server, env_id, action, timeout)
        replay_cut = str(current.get("observation", ""))
        replay_done = bool(current.get("done", False))
    except Exception as exc:
        error = str(exc)
    finally:
        if env_id is not None:
            close_env(server, env_id, timeout)

    strict_match = normalize_text(replay_cut) == normalize_text(expected_cut)
    parser_feedback_match = (
        error is None
        and is_parser_feedback(expected_cut)
        and not bool(row.get("prefix_last_assistant_replay_action_taken", True))
    )
    output = dict(row)
    output.update(
        {
            "replay_category": (
                "error"
                if error
                else ("validated" if strict_match else ("validated_parser_feedback" if parser_feedback_match else "mismatch"))
            ),
            "initial_observation": initial_observation,
            "replay_cut_observation": replay_cut,
            "expected_cut_observation": expected_cut,
            "replay_cut_done": bool(replay_done),
            "replay_error": error,
            "replay_reset_key_mode": "train_row_index",
            "replay_data_idx": int(data_idx),
            "official_data_idx": base_row.get("extra_info", {})
            .get("interaction_kwargs", {})
            .get("data_idx", row["task_id"]),
        }
    )
    return output


def replay_validate_rows(
    candidate_df: pd.DataFrame,
    raw_base_rows: Dict[str, Dict[str, Any]],
    *,
    server: str,
    timeout: float,
    concurrency: int,
) -> pd.DataFrame:
    rows = candidate_df.to_dict(orient="records")
    out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        iterator = executor.map(lambda row: replay_validate_one(row, raw_base_rows, server, timeout), rows)
        for idx, row in enumerate(iterator, start=1):
            out.append(row)
            if idx == len(rows) or idx % 200 == 0:
                print(f"[replay] processed={idx}/{len(rows)}", flush=True)
    return pd.DataFrame(out)


def prompt_key(prompt: List[Dict[str, Any]]) -> str:
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_prompt_assistant_spans(tokenizer, prompt: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], int]:
    text = render_chat(tokenizer, prompt)
    result = tokenizer(text, add_special_tokens=True, return_tensors="pt", return_offsets_mapping=True)
    input_ids = result.input_ids[0]
    offset_mapping = result.offset_mapping[0].tolist()
    start_matches = list(START_TAG_RE.finditer(text))
    end_matches = list(END_TAG_RE.finditer(text))
    if len(start_matches) != len(prompt) or len(end_matches) != len(prompt):
        raise ValueError("Conversation tag count does not match prompt message count")
    spans: List[Dict[str, Any]] = []
    for idx, msg in enumerate(prompt):
        if msg.get("role") != "assistant":
            continue
        action = extract_action(str(msg.get("content", "")))
        if not action:
            continue
        start_char = start_matches[idx].end()
        end_char = end_matches[idx].end()
        token_start = None
        token_end = None
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start is None:
                continue
            if char_start < end_char and char_end > start_char:
                if token_start is None:
                    token_start = token_idx
                token_end = token_idx + 1
        if token_start is None or token_end is None:
            raise ValueError(f"Could not map assistant prompt message to tokens: idx={idx}")
        spans.append({"token_start": int(token_start), "token_end": int(token_end), "action": action})
    return text, spans, int(len(input_ids))


def compute_batch_old_logprobs(model, tokenizer, texts: List[str], device: str) -> List[List[float]]:
    if not texts:
        return []
    batch = None
    try:
        batch = tokenizer(texts, add_special_tokens=True, return_tensors="pt", padding=True)
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, :-1, :].float()
            log_probs = F.log_softmax(logits, dim=-1)
            target_tokens = input_ids[:, 1:].unsqueeze(-1)
            scalar = log_probs.gather(dim=-1, index=target_tokens).squeeze(-1)
        lengths = attention_mask.sum(dim=1).tolist()
        result: List[List[float]] = []
        for row_idx, seq_len in enumerate(lengths):
            result.append(scalar[row_idx, : int(seq_len) - 1].detach().cpu().float().tolist())
        del outputs, logits, log_probs, scalar, target_tokens, input_ids, attention_mask, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except torch.OutOfMemoryError:
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(texts) == 1:
            raise
        mid = len(texts) // 2
        return compute_batch_old_logprobs(model, tokenizer, texts[:mid], device) + compute_batch_old_logprobs(
            model, tokenizer, texts[mid:], device
        )


def build_prefix_window(prompt_spans: List[Dict[str, Any]], sequence_old_logprobs: List[float]):
    if not prompt_spans:
        return [], [], {"start": 0, "end": 0}, 0
    first_token = int(prompt_spans[0]["token_start"])
    last_token = int(prompt_spans[-1]["token_end"])
    lp_start = max(0, first_token - 1)
    lp_end = min(len(sequence_old_logprobs), last_token - 1)
    old_logprobs = list(sequence_old_logprobs[lp_start:lp_end])
    mask = [0] * (last_token - first_token)
    for span in prompt_spans:
        for pos in range(int(span["token_start"]) - first_token, int(span["token_end"]) - first_token):
            mask[pos] = 1
    return old_logprobs, mask, {"start": first_token, "end": last_token}, int(sum(mask))


def materialize_prefix_sidecars(
    rows: List[Dict[str, Any]],
    *,
    model_path: str,
    device: str,
    batch_size: int,
    max_batch_prompt_tokens: int,
    progress_every: int,
) -> None:
    prefix_rows = [row for row in rows if not bool(row["is_raw_variant"])]
    if not prefix_rows:
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    unique_prompts: Dict[str, Dict[str, Any]] = {}
    for row in prefix_rows:
        prompt = normalize_messages(row["prompt"])
        key = prompt_key(prompt)
        if key in unique_prompts:
            continue
        text, spans, token_len = compute_prompt_assistant_spans(tokenizer, prompt)
        unique_prompts[key] = {"prompt_text": text, "prompt_spans": spans, "prompt_token_length": token_len}

    prompt_items = sorted(unique_prompts.items(), key=lambda item: int(item[1]["prompt_token_length"]), reverse=True)
    torch_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    print(
        f"[materialize] unique_prefix_prompts={len(prompt_items)} "
        f"max_prompt_tokens={max(int(item[1]['prompt_token_length']) for item in prompt_items)}",
        flush=True,
    )
    processed = 0
    start_idx = 0
    start_time = time.time()
    while start_idx < len(prompt_items):
        chunk = []
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
        old_logprobs_batch = compute_batch_old_logprobs(model, tokenizer, [item[1]["prompt_text"] for item in chunk], device)
        for (key, prepared), sequence_old_logprobs in zip(chunk, old_logprobs_batch, strict=True):
            old_logprobs, mask, span, token_count = build_prefix_window(prepared["prompt_spans"], sequence_old_logprobs)
            span_len = int(span["end"]) - int(span["start"])
            if span_len > 0 and (len(old_logprobs) != span_len or len(mask) != span_len or token_count != int(sum(mask))):
                raise RuntimeError(f"Invalid prefix sidecar lengths for prompt {key[:64]}")
            prepared["assistant_prefix_old_log_probs"] = old_logprobs
            prepared["prefix_mask"] = mask
            prepared["assistant_prefix_span"] = span
            prepared["prefix_token_count"] = token_count
        processed += len(chunk)
        if processed == len(prompt_items) or processed % max(1, progress_every) == 0:
            print(
                f"[materialize] processed={processed}/{len(prompt_items)} "
                f"elapsed_sec={time.time() - start_time:.1f}",
                flush=True,
            )
        gc.collect()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for row in prefix_rows:
        prepared = unique_prompts[prompt_key(normalize_messages(row["prompt"]))]
        row["assistant_prefix_old_log_probs"] = prepared["assistant_prefix_old_log_probs"]
        row["prefix_mask"] = prepared["prefix_mask"]
        row["assistant_prefix_span"] = prepared["assistant_prefix_span"]
        row["prefix_token_count"] = prepared["prefix_token_count"]
        row["source_oldlogprob_model_path"] = model_path


def materialize_prefix_sidecars_from_stage1(rows: List[Dict[str, Any]], stage1_df: pd.DataFrame) -> None:
    """Fill prefix sidecars by slicing full-trajectory old logprobs from stage1.

    The BabyAI prefix prompt is a strict chat-template prefix of the canonicalized
    full trajectory used in stage1, so token positions before the cut observation
    are identical and do not require a second model forward.
    """

    stage1_map = {str(row["sample_uid"]): row for row in stage1_df.to_dict(orient="records")}
    filled = 0
    for row in rows:
        if bool(row.get("is_raw_variant")):
            continue
        sample_uid = str(row["sample_uid"])
        stage1_row = stage1_map.get(sample_uid)
        if stage1_row is None:
            raise RuntimeError(f"Missing stage1 forward row for {sample_uid}")

        cut_turn_idx = int(row["cut_turn_idx"])
        assistant_spans = normalize_messages(stage1_row["assistant_turn_spans"])
        selected_spans = [
            span for span in assistant_spans if int(span["role_turn_idx"]) <= cut_turn_idx + 1
        ]
        if not selected_spans:
            raise RuntimeError(f"{row['record_uid']}: no assistant spans selected for cut_turn_idx={cut_turn_idx}")

        sequence_old_logprobs = [float(value) for value in list(stage1_row["sequence_old_log_probs"])]
        prefix_old_logprobs, prefix_mask, prefix_span, prefix_token_count = build_prefix_window(
            selected_spans,
            sequence_old_logprobs,
        )
        span_len = int(prefix_span["end"]) - int(prefix_span["start"])
        if span_len <= 0:
            raise RuntimeError(f"{row['record_uid']}: invalid prefix span {prefix_span}")
        if len(prefix_old_logprobs) != span_len or len(prefix_mask) != span_len:
            raise RuntimeError(
                f"{row['record_uid']}: sidecar length mismatch "
                f"old={len(prefix_old_logprobs)} mask={len(prefix_mask)} span_len={span_len}"
            )
        if int(sum(prefix_mask)) != prefix_token_count or prefix_token_count <= 0:
            raise RuntimeError(f"{row['record_uid']}: invalid prefix mask/token count")

        row["assistant_prefix_old_log_probs"] = prefix_old_logprobs
        row["prefix_mask"] = prefix_mask
        row["assistant_prefix_span"] = prefix_span
        row["prefix_token_count"] = prefix_token_count
        row["source_oldlogprob_model_path"] = str(stage1_row.get("model_path", ""))
        row["prefix_sidecar_source"] = "stage1_full_trajectory_slice"
        filled += 1
    print(f"[materialize-stage1] filled prefix sidecars for {filled} rows", flush=True)


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    raw_mask = df["is_raw_variant"] == True
    prefix_mask = ~raw_mask
    prefix_action_lengths = df["prefix_actions"].apply(lambda value: len(list(value)) if value is not None else 0)
    summary = {
        "rows": int(len(df)),
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
    }
    if "candidate_rank" in df.columns:
        ranks = df["candidate_rank"].dropna()
        if not ranks.empty:
            summary["candidate_rank_counts"] = ranks.astype(int).value_counts().sort_index().to_dict()
    if "replay_category" in df.columns:
        summary["replay_category_counts"] = df["replay_category"].value_counts().sort_index().to_dict()
    return summary


def write_dataset(path: Path, rows: List[Dict[str, Any]], extra_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    summary = summarize_rows(rows)
    if extra_summary:
        summary.update(extra_summary)
    summary["output_path"] = str(path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def load_stage1_df(args: argparse.Namespace) -> pd.DataFrame:
    if args.stage1_input_glob:
        paths = sorted(Path().glob(args.stage1_input_glob) if not args.stage1_input_glob.startswith("/") else Path("/").glob(args.stage1_input_glob[1:]))
    else:
        stage1_dir = args.output_root / "stage1_forward"
        shard_paths = sorted(stage1_dir.glob("babyai_sft_step300_oldlogprob_entropy.shard*.parquet"))
        if shard_paths:
            paths = shard_paths
        else:
            paths = [stage1_dir / "babyai_sft_step300_oldlogprob_entropy.parquet"]
        paths = [path for path in paths if path.exists() and ".manifest" not in path.name]
    if not paths:
        raise RuntimeError("No stage1 parquet files found")
    dfs = [pd.read_parquet(path) for path in paths]
    df = pd.concat(dfs, ignore_index=True)
    if "canonicalization" not in df.columns:
        raise RuntimeError(
            "Stage1 parquet is missing canonicalization metadata; rerun stage1 with the current builder."
        )
    canonicalization_values = set(df["canonicalization"].dropna().astype(str).unique())
    if canonicalization_values != {CANONICALIZATION_VERSION}:
        raise RuntimeError(
            "Stage1 parquet canonicalization mismatch: "
            f"found={sorted(canonicalization_values)}, expected={CANONICALIZATION_VERSION}. "
            "Rerun stage1 or use a fresh OUTPUT_ROOT."
        )
    df = df.drop_duplicates(subset=["sample_uid"]).sort_values(["task_id", "sample_idx"]).reset_index(drop=True)
    merged_path = args.output_root / "stage1_forward" / "babyai_sft_step300_oldlogprob_entropy.parquet"
    if len(paths) > 1 or not merged_path.exists():
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(merged_path, index=False)
    return df


def write_candidate_table(path: Path, candidates_df: pd.DataFrame) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_parquet(path, index=False)
    manifest = {
        "rows": int(len(candidates_df)),
        "unique_sample_uid": int(candidates_df["sample_uid"].nunique()) if not candidates_df.empty else 0,
        "candidate_rank_counts": candidates_df["candidate_rank"].astype(int).value_counts().sort_index().to_dict()
        if not candidates_df.empty and "candidate_rank" in candidates_df.columns
        else {},
        "strategy_counts": candidates_df["strategy"].value_counts().sort_index().to_dict()
        if not candidates_df.empty and "strategy" in candidates_df.columns
        else {},
        "output_path": str(path),
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_or_replay_candidates(
    candidates_df: pd.DataFrame,
    raw_base_rows: Dict[str, Dict[str, Any]],
    *,
    replay_audit_path: Path,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    replay_manifest: Dict[str, Any] = {"skipped": bool(args.skip_replay)}
    if args.skip_replay:
        replay_df = candidates_df.copy()
        replay_df["replay_category"] = "not_replayed"
    elif replay_audit_path.exists():
        cached_replay_df = pd.read_parquet(replay_audit_path)
        cached_uids = set(str(value) for value in cached_replay_df["candidate_uid"].tolist())
        candidate_uids = set(str(value) for value in candidates_df["candidate_uid"].tolist())
        cache_uses_current_reset_key = (
            "replay_reset_key_mode" in cached_replay_df.columns
            and set(cached_replay_df["replay_reset_key_mode"].dropna().astype(str).unique()) == {"train_row_index"}
        )
        cache_uses_current_canonicalization = (
            "canonicalization" in cached_replay_df.columns
            and set(cached_replay_df["canonicalization"].dropna().astype(str).unique()) == {CANONICALIZATION_VERSION}
        )
        cache_has_non_error = (
            "replay_category" in cached_replay_df.columns
            and not cached_replay_df.empty
            and set(cached_replay_df["replay_category"].dropna().astype(str).unique()) != {"error"}
        )
        if (
            cached_uids == candidate_uids
            and cache_uses_current_reset_key
            and cache_uses_current_canonicalization
            and cache_has_non_error
        ):
            print(f"[replay] reusing cached replay audit: {replay_audit_path}", flush=True)
            replay_df = cached_replay_df
        else:
            if cached_uids == candidate_uids and not cache_uses_current_reset_key:
                print(
                    f"[replay] ignoring cached replay audit with stale reset-key mode: {replay_audit_path}",
                    flush=True,
                )
            if cached_uids == candidate_uids and not cache_uses_current_canonicalization:
                print(
                    f"[replay] ignoring cached replay audit with stale canonicalization: {replay_audit_path}",
                    flush=True,
                )
            if cached_uids == candidate_uids and not cache_has_non_error:
                print(
                    f"[replay] ignoring cached replay audit with only error rows: {replay_audit_path}",
                    flush=True,
                )
            replay_df = replay_validate_rows(
                candidates_df,
                raw_base_rows,
                server=args.server,
                timeout=args.request_timeout,
                concurrency=args.replay_concurrency,
            )
    else:
        replay_df = replay_validate_rows(
            candidates_df,
            raw_base_rows,
            server=args.server,
            timeout=args.request_timeout,
            concurrency=args.replay_concurrency,
        )

    replay_audit_path.parent.mkdir(parents=True, exist_ok=True)
    replay_df.to_parquet(replay_audit_path, index=False)
    replay_manifest.update(
        {
            "candidate_rows": int(len(replay_df)),
            "validated_rows": int(
                replay_df["replay_category"].isin(["validated", "validated_parser_feedback"]).sum()
            ),
            "replay_category_counts": replay_df["replay_category"].value_counts().sort_index().to_dict(),
            "replay_audit_path": str(replay_audit_path),
        }
    )
    return replay_df, replay_manifest


def build_datasets(args: argparse.Namespace, system_prompt: str) -> Dict[str, Any]:
    selected_datasets = {name.strip() for name in args.datasets.split(",") if name.strip()}
    unknown = sorted(selected_datasets - set(DATASET_COMPLETE_NAMES))
    if unknown:
        raise RuntimeError(f"Unknown dataset names: {unknown}")
    if not selected_datasets:
        raise RuntimeError("No datasets selected")

    samples = load_jsonl(args.sampling_path)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    canonical_rows = {
        canonical["sample_uid"]: canonical
        for canonical in (canonicalize_sample(sample, system_prompt) for sample in samples)
    }
    raw_base_rows = load_raw_base_rows(args.train_parquet_path)
    stage1_df = load_stage1_df(args)
    stage1_df = stage1_df[stage1_df["sample_uid"].isin(canonical_rows)].copy()

    complete_summaries: Dict[str, Any] = {}
    replay_summaries: Dict[str, Any] = {}
    stage2_manifests: Dict[str, Any] = {}
    all_rows_for_sidecars: List[Dict[str, Any]] = []

    for dataset_name in sorted(selected_datasets):
        complete_name = DATASET_COMPLETE_NAMES[dataset_name]
        if dataset_name in FIXED_DATASET_RATIOS:
            candidates_df = select_fixed_ratio_candidates(
                canonical_rows,
                ratios=FIXED_DATASET_RATIOS[dataset_name],
            )
        elif dataset_name == "main_raw_top3_fullflow":
            candidates_df = select_raw_topk_candidates(canonical_rows, stage1_df, top_k=args.top_k)
        elif dataset_name == "main_change_top3_w11_fullflow":
            candidates_df = select_change_topk_candidates(
                canonical_rows,
                stage1_df,
                top_k=args.top_k,
                change_window=args.change_window,
            )
        else:
            raise RuntimeError(f"Unsupported dataset: {dataset_name}")

        stage2_path = args.output_root / "stage2_splits" / DATASET_STAGE2_FILENAMES[dataset_name]
        stage2_manifest = write_candidate_table(stage2_path, candidates_df)
        stage2_manifests[dataset_name] = stage2_manifest

        complete_rows: List[Dict[str, Any]] = []
        for row in candidates_df.to_dict(orient="records"):
            row["source_dataset"] = f"babyai_{complete_name}_stage2_candidates"
            complete_rows.append(
                build_prefix_record(
                    row,
                    raw_base_rows[str(row["item_id"])],
                    main_dataset=complete_name,
                    system_prompt=system_prompt,
                )
            )
        for canonical in canonical_rows.values():
            complete_rows.append(
                build_raw_record(
                    canonical,
                    raw_base_rows[canonical["item_id"]],
                    main_dataset=complete_name,
                    system_prompt=system_prompt,
                    model_path=args.model_path,
                )
            )

        replay_audit_path = args.output_root / "audit" / "replay_validated" / f"{dataset_name}.replay_audit.parquet"
        replay_df, replay_manifest = load_or_replay_candidates(
            candidates_df,
            raw_base_rows,
            replay_audit_path=replay_audit_path,
            args=args,
        )

        replay_rows: List[Dict[str, Any]] = []
        trainable_df = replay_df[
            replay_df["replay_category"].isin(["validated", "validated_parser_feedback"])
        ].copy()
        for row in trainable_df.to_dict(orient="records"):
            row["source_dataset"] = f"babyai_{dataset_name}_stage2_replay_validated"
            replay_rows.append(
                build_prefix_record(
                    row,
                    raw_base_rows[str(row["item_id"])],
                    main_dataset=dataset_name,
                    system_prompt=system_prompt,
                )
            )
        for canonical in canonical_rows.values():
            replay_rows.append(
                build_raw_record(
                    canonical,
                    raw_base_rows[canonical["item_id"]],
                    main_dataset=dataset_name,
                    system_prompt=system_prompt,
                    model_path=args.model_path,
                )
            )

        all_rows_for_sidecars.extend(complete_rows)
        all_rows_for_sidecars.extend(replay_rows)

        complete_summaries[complete_name] = write_dataset(
            args.output_root / "complete_split" / f"{complete_name}.parquet",
            complete_rows,
            extra_summary={"stage2_manifest": stage2_manifest},
        )
        replay_summaries[dataset_name] = write_dataset(
            args.output_root / "replay_validated" / f"{dataset_name}.parquet",
            replay_rows,
            extra_summary=replay_manifest,
        )

    materialize_prefix_sidecars_from_stage1(all_rows_for_sidecars, stage1_df)
    for complete_name, summary in list(complete_summaries.items()):
        rows = [row for row in all_rows_for_sidecars if row["main_dataset"] == complete_name]
        complete_summaries[complete_name] = write_dataset(
            args.output_root / "complete_split" / f"{complete_name}.parquet",
            rows,
            extra_summary={"stage2_manifest": summary["stage2_manifest"]},
        )
    for dataset_name, summary in list(replay_summaries.items()):
        rows = [row for row in all_rows_for_sidecars if row["main_dataset"] == dataset_name]
        replay_summaries[dataset_name] = write_dataset(
            args.output_root / "replay_validated" / f"{dataset_name}.parquet",
            rows,
            extra_summary={
                "skipped": summary.get("skipped", False),
                "candidate_rows": summary.get("candidate_rows", 0),
                "validated_rows": summary.get("validated_rows", 0),
                "replay_category_counts": summary.get("replay_category_counts", {}),
                "replay_audit_path": summary.get("replay_audit_path"),
            },
        )

    report = {
        "output_root": str(args.output_root),
        "sampling_path": str(args.sampling_path),
        "train_parquet_path": str(args.train_parquet_path),
        "sft_parquet_path": str(args.sft_parquet_path),
        "model_path": args.model_path,
        "server": args.server,
        "stage1_rows": int(len(stage1_df)),
        "selected_datasets": sorted(selected_datasets),
        "stage2": stage2_manifests,
        "complete_split": complete_summaries,
        "replay_validated": replay_summaries,
    }
    write_readmes(args.output_root, report)
    (args.output_root / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def write_readmes(output_root: Path, report: Dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    root_lines = [
        "# BabyAI Prefix-RL Dataset Construction",
        "",
        "## Inputs",
        f"- sampling trajectories: `{report['sampling_path']}`",
        f"- raw BabyAI train parquet: `{report['train_parquet_path']}`",
        f"- SFT chat-format reference parquet: `{report['sft_parquet_path']}`",
        f"- old-logprob/entropy model: `{report['model_path']}`",
        "",
        "## Outputs",
        f"- stage1 token forward: `{output_root / 'stage1_forward'}`",
        f"- stage2 cut candidates: `{output_root / 'stage2_splits'}`",
        f"- complete split datasets: `{output_root / 'complete_split'}`",
        f"- replay validated datasets: `{output_root / 'replay_validated'}`",
        f"- replay audits: `{output_root / 'audit' / 'replay_validated'}`",
        f"- selected_datasets: `{report.get('selected_datasets', [])}`",
        "",
        "## Protocol",
        "- Sampling warmup instruction/ack messages are removed.",
        "- Assistant turns with any extractable action are strict-canonicalized to BabyAI SFT format with one `Action: [[ ... ]]`; no-action assistant turns are kept as short `Thought:` notes.",
        "- Assistant thought text strips `<think>` tags, normalizes whitespace, and is capped at 512 chars.",
        "- Stage1 stores per-token old log prob and entropy for the canonicalized full trajectory.",
        "- Fixed-ratio cutpoints use assistant-turn relative positions.",
        "- Raw-entropy Top3 chooses distinct assistant turns by maximum assistant-token entropy.",
        "- Change Top3 chooses distinct assistant turns from assistant-token entropy first differences after centered w11 smoothing.",
        "- `complete_split` keeps all selected cutpoints; `replay_validated` keeps only replay-exact/parser-feedback prefix rows plus raw rows.",
    ]
    (output_root / "README.md").write_text("\n".join(root_lines) + "\n", encoding="utf-8")

    for subdir, key, title in (
        ("complete_split", "complete_split", "BabyAI Complete Split Datasets"),
        ("replay_validated", "replay_validated", "BabyAI Replay Validated Datasets"),
    ):
        summaries = report[key]
        dir_lines = [f"# {title}", ""]
        for name, summary in sorted(summaries.items()):
            dir_lines.extend(
                [
                    f"## {name}",
                    f"- rows: `{summary['rows']}`",
                    f"- unique_sample_uid: `{summary['unique_sample_uid']}`",
                    f"- raw_rows: `{summary['raw_rows']}`",
                    f"- prefix_rows: `{summary['prefix_rows']}`",
                    f"- positive_prefix_rows: `{summary['positive_prefix_rows']}`",
                    f"- strategy_counts: `{summary.get('strategy_counts', {})}`",
                    f"- candidate_rank_counts: `{summary.get('candidate_rank_counts', {})}`",
                    f"- replay_category_counts: `{summary.get('replay_category_counts', {})}`",
                    f"- output_path: `{summary['output_path']}`",
                    "",
                ]
            )
        (output_root / subdir / "README.md").write_text("\n".join(dir_lines).rstrip() + "\n", encoding="utf-8")

    for name, summary in sorted(report["complete_split"].items()):
        lines = [
            f"# {name}",
            "",
            f"- rows: `{summary['rows']}`",
            f"- unique_sample_uid: `{summary['unique_sample_uid']}`",
            f"- raw_rows: `{summary['raw_rows']}`",
            f"- prefix_rows: `{summary['prefix_rows']}`",
            f"- positive_prefix_rows: `{summary['positive_prefix_rows']}`",
            f"- candidate_rank_counts: `{summary.get('candidate_rank_counts', {})}`",
            f"- replay_category_counts: `{summary.get('replay_category_counts', {})}`",
            f"- output_path: `{summary['output_path']}`",
        ]
        (output_root / "complete_split" / f"{name}.README.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    for name, summary in sorted(report["replay_validated"].items()):
        lines = [
            f"# {name}",
            "",
            f"- rows: `{summary['rows']}`",
            f"- unique_sample_uid: `{summary['unique_sample_uid']}`",
            f"- raw_rows: `{summary['raw_rows']}`",
            f"- prefix_rows: `{summary['prefix_rows']}`",
            f"- positive_prefix_rows: `{summary['positive_prefix_rows']}`",
            f"- strategy_counts: `{summary.get('strategy_counts', {})}`",
            f"- candidate_rank_counts: `{summary.get('candidate_rank_counts', {})}`",
            f"- replay_category_counts: `{summary.get('replay_category_counts', {})}`",
            f"- output_path: `{summary['output_path']}`",
        ]
        (output_root / "replay_validated" / f"{name}.README.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    audit_lines = [
        "# BabyAI Prefix-RL Data Construction Audit",
        "",
        f"- stage1_rows: `{report['stage1_rows']}`",
        f"- selected_datasets: `{report.get('selected_datasets', [])}`",
    ]
    for name, summary in sorted(report["stage2"].items()):
        audit_lines.append(f"- stage2[{name}]: rows=`{summary['rows']}`, ranks=`{summary['candidate_rank_counts']}`")
    for name, summary in sorted(report["complete_split"].items()):
        audit_lines.append(f"- complete[{name}]: rows=`{summary['rows']}`, prefix_rows=`{summary['prefix_rows']}`")
    for name, summary in sorted(report["replay_validated"].items()):
        audit_lines.append(
            f"- replay[{name}]: rows=`{summary['rows']}`, prefix_rows=`{summary['prefix_rows']}`, "
            f"replay_category_counts=`{summary.get('replay_category_counts', {})}`"
        )
    (output_root / "audit_report.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda"
    system_prompt = load_system_prompt(args.sft_parquet_path)
    if args.stage in ("all", "stage1"):
        stage1_forward(args, system_prompt)
    if args.stage in ("all", "build"):
        report = build_datasets(args, system_prompt)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
