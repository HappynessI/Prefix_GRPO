#!/usr/bin/env python3
"""
SciWorld Evaluation Client - vLLM Service Mode
科学实验环境评估脚本
"""

import os
import sys
import json
import logging
import argparse
import uuid
import asyncio
import math
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pyarrow.parquet as pq
from tqdm import tqdm
import aiohttp
import fcntl

def discover_project_root(script_path: Path) -> Path:
    """Find the nearest repo root that contains `verl/`."""
    resolved = script_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "verl").exists():
            return candidate
    raise RuntimeError(f"verl not found from {resolved}")


def default_data_path(root: Path) -> Path:
    local_data = root / "data" / "sciworld" / "test.parquet"
    shared_data = Path("/Data/wyh/datasets/Verl-Data/eval/sciworld/test.parquet")
    return local_data if local_data.exists() else shared_data


def default_output_dir(root: Path) -> Path:
    shared_output_root = Path("/Data/wyh/datasets/Verl-Data/outputs")
    if shared_output_root.exists():
        return shared_output_root / "sciworld_eval"
    return root / "outputs" / "sciworld_eval"


def bootstrap_import_paths(root: Path) -> None:
    """Prefer repo-local packages over any editable/installed fallback."""
    candidate_paths = [
        root / "envs" / "AgentGym" / "agentenv-sciworld",
        root / "verl",
        root,
    ]
    for candidate in reversed(candidate_paths):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def configure_logging(output_dir: str) -> Path:
    log_dir = Path(output_dir).expanduser().resolve() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if logger.handlers:
        return log_dir

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / f"eval_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return log_dir


# Bootstrap repo-local imports first.
project_root = discover_project_root(Path(__file__))
if not (project_root / "verl").exists():
    raise RuntimeError(f"verl not found in {project_root}")
bootstrap_import_paths(project_root)

DEFAULT_DATA_PATH = default_data_path(project_root)
DEFAULT_OUTPUT_DIR = default_output_dir(project_root)

from verl.interactions.sciworld_interaction import SciWorldInteraction
from agentgym_eval_utils import resolve_eval_row

logger = logging.getLogger("SciWorldEval")


SCIWORLD_TASK_LAYOUT = [
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

SCIWORLD_PAPER_CATEGORIES = (
    "Measure",
    "Test-Cond.",
    "Find",
    "Chem-Mix",
    "Lifespan",
)

SCIWORLD_CATEGORY_ORDER = (
    "Measure",
    "Test-Cond.",
    "Find",
    "Chem-Mix",
    "Lifespan",
    "Other-Matter",
    "Other-Electricity",
    "Other-Biology",
)

SCIWORLD_LAYOUT_BY_TASK_ID = {}
SCIWORLD_OFFICIAL_TASK_IDS_BY_CATEGORY = defaultdict(list)
for _task_id, _official_count, _task_category, _task_name in SCIWORLD_TASK_LAYOUT:
    SCIWORLD_LAYOUT_BY_TASK_ID[_task_id] = {
        "task_id": _task_id,
        "official_count": _official_count,
        "task_category": _task_category,
        "task_name": _task_name,
    }
    SCIWORLD_OFFICIAL_TASK_IDS_BY_CATEGORY[_task_category].append(_task_id)


def sciworld_category_sort_key(category: Optional[str]) -> tuple[int, str]:
    order = {name: idx for idx, name in enumerate(SCIWORLD_CATEGORY_ORDER)}
    category = category or ""
    return (order.get(category, len(order)), category)


def sciworld_task_id_sort_key(task_id: Optional[str]) -> tuple[int, int, str]:
    if not task_id:
        return (10**9, 10**9, "")
    try:
        major, minor = str(task_id).split("-", 1)
        return (int(major), int(minor), str(task_id))
    except Exception:
        return (10**9, 10**9, str(task_id))


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return f"{numerator}/{denominator}"
    return f"{numerator}/{denominator} ({(numerator / denominator) * 100:.1f}%)"


def build_sciworld_dataset_metadata(df) -> Dict[str, Any]:
    session_meta = {}
    split_episode_counts_by_category = defaultdict(int)
    split_episode_counts_by_task_id = defaultdict(int)
    split_task_ids_by_category = defaultdict(set)

    for _, row in df.iterrows():
        extra_info = row.get("extra_info") or {}
        interaction_kwargs = dict(extra_info.get("interaction_kwargs") or {})

        session_id = interaction_kwargs.get("session_id", row.get("session_id"))
        if session_id is None:
            continue
        session_id = int(session_id)

        task_id = interaction_kwargs.get("task_id")
        task_id = str(task_id) if task_id is not None else None
        official_meta = SCIWORLD_LAYOUT_BY_TASK_ID.get(task_id, {})

        task_category = row.get("task_category") or interaction_kwargs.get("task_category") or official_meta.get("task_category")
        task_name = row.get("task_name") or interaction_kwargs.get("task_name") or official_meta.get("task_name")
        task_subcategory = row.get("task_subcategory") or interaction_kwargs.get("task_subcategory")
        item_id = row.get("item_id") or interaction_kwargs.get("official_item_id") or f"sciworld_{session_id}"

        meta = {
            "item_id": str(item_id),
            "session_id": session_id,
            "task_id": task_id,
            "task_category": task_category,
            "task_name": task_name,
            "task_subcategory": task_subcategory,
        }
        session_meta[session_id] = meta

        if task_category:
            split_episode_counts_by_category[task_category] += 1
        if task_id:
            split_episode_counts_by_task_id[task_id] += 1
            if task_category:
                split_task_ids_by_category[task_category].add(task_id)

    official_categories = {task_category for _, _, task_category, _ in SCIWORLD_TASK_LAYOUT}
    official_task_ids = {task_id for task_id, _, _, _ in SCIWORLD_TASK_LAYOUT}
    paper_categories = set(SCIWORLD_PAPER_CATEGORIES)
    paper_task_ids = {task_id for task_id, _, task_category, _ in SCIWORLD_TASK_LAYOUT if task_category in paper_categories}
    split_categories = set(split_episode_counts_by_category)
    split_task_ids = set(split_episode_counts_by_task_id)

    return {
        "session_meta": session_meta,
        "split_episode_counts_by_category": dict(split_episode_counts_by_category),
        "split_episode_counts_by_task_id": dict(split_episode_counts_by_task_id),
        "split_task_ids_by_category": {k: set(v) for k, v in split_task_ids_by_category.items()},
        "official_categories": official_categories,
        "official_task_ids": official_task_ids,
        "paper_categories": paper_categories,
        "paper_task_ids": paper_task_ids,
        "split_categories": split_categories,
        "split_task_ids": split_task_ids,
    }


def merge_task_metadata(record: Dict[str, Any], task_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not task_meta:
        return record
    for key in ("task_id", "task_category", "task_name", "task_subcategory"):
        value = task_meta.get(key)
        if value is not None:
            record[key] = value
    return record


def compute_group_metrics(grouped_runs: Dict[Any, List[Dict[str, Any]]], k_values: List[int]) -> Dict[str, Any]:
    sum_avg_reward = 0.0
    sum_avg_success = 0.0
    pass_at_k_values = {k: [] for k in k_values}

    for task_runs in grouped_runs.values():
        n = len(task_runs)
        c = sum(1 for r in task_runs if r.get("success", False))
        total_r = sum(r.get("reward", 0.0) for r in task_runs)
        if n > 0:
            sum_avg_reward += total_r / n
            sum_avg_success += c / n
        for k in k_values:
            if n >= k:
                pass_at_k_values[k].append(estimate_pass_at_k(n, c, k))

    num_tasks = len(grouped_runs)
    return {
        "num_tasks": num_tasks,
        "avg_reward": sum_avg_reward / num_tasks if num_tasks else 0.0,
        "avg_success": sum_avg_success / num_tasks if num_tasks else 0.0,
        "pass_at_k_values": pass_at_k_values,
    }


def format_pass_metrics(metric_dict: Dict[str, Any], k_values: List[int]) -> str:
    chunks = []
    num_tasks = metric_dict.get("num_tasks", 0)
    pass_at_k_values = metric_dict.get("pass_at_k_values", {})
    for k in k_values:
        values = pass_at_k_values.get(k, [])
        if values:
            chunks.append(f"Pass@{k}: {sum(values) / len(values):.4f}")
        else:
            chunks.append(f"Pass@{k}: N/A")
    chunks.append(f"tasks: {num_tasks}")
    return " | ".join(chunks)


class AsyncSciWorldAgent:
    """SciWorld Agent - Uses HTTP calls to vLLM service"""
    
    def __init__(
        self,
        server_url: str,
        model_name: str = "qwen3",
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_context_tokens: int = 12288,
        context_safety_margin: int = 256,
        preserve_recent_messages: int = 8,
        history_sanitization: str = "raw",
        enable_thinking: bool = False,
    ):
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_context_tokens = max_context_tokens
        self.context_safety_margin = context_safety_margin
        self.preserve_recent_messages = preserve_recent_messages
        self.history_sanitization = history_sanitization
        self.enable_thinking = enable_thinking
        self.system_prompt = self._build_prompt()
        self.trim_event_count = 0
        self.trimmed_messages_total = 0
        self.truncated_messages_total = 0
        self.max_removed_messages = 0
        self.max_truncated_messages = 0
        self.max_estimated_input_tokens = 0
        self.sanitized_messages_total = 0
    
    def _build_prompt(self) -> str:
        return '''You are an agent for ScienceWorld. Every round the environment gives you one observation. Your job is to choose exactly one valid action to finish the task.

CORE PROTOCOL (strict):
1. Output exactly one action per turn.
2. Always use this format:
Thought:
brief reason for the next step.

Action:
[[ your next action ]]
3. The action must be wrapped in [[ ]] brackets.
4. Stop immediately after the Action line. Do not predict or simulate the environment response.
5. Do not output XML, HTML, chat-template, or thinking tags such as <think>, </think>, <|im_start|>, or <|im_end|>.
6. If you are unsure, inspect the world with a valid action such as [[ look around ]], [[ inventory ]], [[ look at OBJ ]], or [[ look in OBJ ]].

VALID ACTIONS:
- open/close OBJ
- activate/deactivate OBJ
- connect OBJ to OBJ
- disconnect OBJ
- use OBJ [on OBJ]
- look around
- look at OBJ
- look in OBJ
- read OBJ
- move OBJ to OBJ
- pick up OBJ
- put down OBJ
- pour OBJ into OBJ
- dunk OBJ into OBJ
- mix OBJ
- go to LOC
- eat OBJ
- flush OBJ
- focus on OBJ
- wait
- examine OBJ
- inventory

SCIENCEWORLD STRATEGY:
1. Read the task carefully and identify the target object, measurement, condition, or box.
2. Use navigation actions to reach relevant rooms.
3. Use look/open/read/inventory actions to gather missing information.
4. For find tasks, focus on the target object before moving it to the requested box.
5. For measure or test tasks, focus on required objects and use the appropriate device before choosing a box.
6. For chemistry tasks, put the required materials into the same container and use mix when needed.

EXAMPLES:
[Observation]
Task: Your task is to find a non-living thing and move it to the orange box.
This room is called the bedroom. In it, you see a bed, a table, and a book shelf.

[Assistant]
Thought:
I need to identify a non-living object in the room. A book is non-living, so I should inspect the shelf or take the book if accessible.

Action:
[[ look in book shelf ]]

[Observation]
The book shelf contains A book titled Beowulf.

[Assistant]
Thought:
The book is a non-living thing. I should focus on it before moving it.

Action:
[[ focus on book ]]

Remember: return only Thought and Action, with one bracketed action.'''

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """Conservative token estimate for context trimming without tokenizer dependency."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 3.0))

    def _estimate_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = self._estimate_text_tokens(self.system_prompt) + 16
        for message in messages:
            total += self._estimate_text_tokens(message.get("content", "")) + 12
        return total

    def _truncate_text_middle(self, text: str, max_tokens: int) -> str:
        if self._estimate_text_tokens(text) <= max_tokens:
            return text
        max_chars = max(64, max_tokens * 3)
        if len(text) <= max_chars:
            return text
        marker = "\n...[TRUNCATED HISTORY]...\n"
        keep = max_chars - len(marker)
        head = max(16, keep // 2)
        tail = max(16, keep - head)
        return text[:head] + marker + text[-tail:]

    def _trim_messages_for_context(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Keep the initial task prompt plus recent turns within the model context budget."""
        input_budget = self.max_context_tokens - self.max_new_tokens - self.context_safety_margin
        if input_budget <= 0:
            return messages

        trimmed = [{"role": m["role"], "content": m["content"]} for m in messages]
        removed_messages = 0
        truncated_messages = 0

        while (
            self._estimate_messages_tokens(trimmed) > input_budget
            and len(trimmed) > 1 + self.preserve_recent_messages
        ):
            removable_end = len(trimmed) - self.preserve_recent_messages
            drop_count = min(2, max(0, removable_end - 1))
            if drop_count <= 0:
                break
            del trimmed[1:1 + drop_count]
            removed_messages += drop_count

        while self._estimate_messages_tokens(trimmed) > input_budget:
            candidate_indices = [idx for idx in range(1, len(trimmed)) if trimmed[idx].get("content")]
            if not candidate_indices:
                candidate_indices = [0] if trimmed else []
            if not candidate_indices:
                break
            longest_idx = max(
                candidate_indices,
                key=lambda idx: self._estimate_text_tokens(trimmed[idx].get("content", "")),
            )
            current_text = trimmed[longest_idx].get("content", "")
            current_tokens = self._estimate_text_tokens(current_text)
            new_budget = max(32, current_tokens // 2)
            new_text = self._truncate_text_middle(current_text, new_budget)
            if new_text == current_text:
                break
            trimmed[longest_idx]["content"] = new_text
            truncated_messages += 1

        estimated_tokens = self._estimate_messages_tokens(trimmed)
        if removed_messages or truncated_messages:
            self.trim_event_count += 1
            self.trimmed_messages_total += removed_messages
            self.truncated_messages_total += truncated_messages
            self.max_removed_messages = max(self.max_removed_messages, removed_messages)
            self.max_truncated_messages = max(self.max_truncated_messages, truncated_messages)
            self.max_estimated_input_tokens = max(self.max_estimated_input_tokens, estimated_tokens)

        return trimmed

    @staticmethod
    def clean_model_response(text: str) -> str:
        """Remove chat-template/thinking tags before reusing a response as history."""
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"<\|im_start\|>assistant\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<\|im_end\|>", "", cleaned)
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    @staticmethod
    def _extract_thought(cleaned_response: str) -> str:
        thought_match = re.search(
            r"Thought:\s*(.*?)(?:\n\s*Action\s*:|\[\[|$)",
            cleaned_response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = re.split(r"\n\s*Action\s*:|\[\[", cleaned_response, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        thought = re.sub(r"\s+", " ", thought).strip()
        if not thought or thought.lower().startswith("action:"):
            return "I will take the next valid action based on the observation."
        return thought[:800]

    def build_history_response(self, raw_response: str, parsed_action: Optional[str]) -> str:
        """Prepare assistant response for the next history turn."""
        raw = str(raw_response or "").strip()
        if self.history_sanitization == "raw":
            return raw

        cleaned = self.clean_model_response(raw)
        if self.history_sanitization == "strip_tags":
            history_response = cleaned
        elif self.history_sanitization == "canonicalize":
            if parsed_action:
                thought = self._extract_thought(cleaned)
                history_response = f"Thought:\n{thought}\n\nAction:\n[[ {parsed_action} ]]"
            else:
                history_response = "Thought:\nI need to provide exactly one valid action wrapped in [[ ]].\n\nAction:\n"
        else:
            history_response = raw

        if history_response.strip() != raw:
            self.sanitized_messages_total += 1
        return history_response

    def get_trim_stats(self) -> Dict[str, int]:
        return {
            "trim_event_count": self.trim_event_count,
            "trimmed_messages_total": self.trimmed_messages_total,
            "truncated_messages_total": self.truncated_messages_total,
            "max_removed_messages": self.max_removed_messages,
            "max_truncated_messages": self.max_truncated_messages,
            "max_estimated_input_tokens": self.max_estimated_input_tokens,
            "sanitized_messages_total": self.sanitized_messages_total,
            "history_sanitization": self.history_sanitization,
        }
    
    async def generate(self, messages: List[Dict[str, str]], session: aiohttp.ClientSession) -> str:
        trimmed_messages = self._trim_messages_for_context(messages)
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": self.system_prompt}, *trimmed_messages],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False
        }
        if not self.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        
        try:
            async with session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status}: {await response.text()}")
                    return ""
                result = await response.json()
                return result["choices"][0]["message"]["content"].strip() if result.get("choices") else ""
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return ""


async def evaluate_one_episode(
    agent: AsyncSciWorldAgent,
    interaction: SciWorldInteraction,
    session_id: int,
    interaction_kwargs: Dict[str, Any],
    http_session: aiohttp.ClientSession,
    max_rounds: int = 30,
    max_consecutive_invalid_actions: int = 3,
) -> Dict[str, Any]:
    instance_id = f"eval_{session_id}_{uuid.uuid4().hex[:8]}"
    messages, conversations = [], []
    done, total_reward = False, 0.0
    initial_prompt = None
    data_idx = session_id
    invalid_action_count = 0
    consecutive_invalid_action_count = 0
    sanitized_response_count = 0
    early_stop_reason = None

    try:
        try:
            await interaction.start_interaction(instance_id, **interaction_kwargs)
            done, initial_obs, reward, _ = await interaction.generate_response(instance_id, messages)
            total_reward += reward
            messages.append({"role": "user", "content": initial_obs})
            conversations.append({"role": "user", "content": initial_obs})
            initial_prompt = initial_obs
        except Exception as e:
            logger.error(f"Start failed for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "reward": 0.0,
                "success": False,
                "num_turns": 0,
                "conversations": [],
                "initial_prompt": None,
                "data_idx": data_idx,
                "error": str(e),
                "error_type": "start_interaction",
            }

        for turn in range(max_rounds):
            if done:
                logger.info(f"Session {session_id} done at turn {turn}, total_reward={total_reward}")
                break
            try:
                response = await agent.generate(messages, http_session)
                if 'task completed' in response.lower():
                    done = True
                parsed_action = interaction.extract_action(response)
                history_response = agent.build_history_response(response, parsed_action)
                if not parsed_action:
                    invalid_action_count += 1
                    consecutive_invalid_action_count += 1
                else:
                    consecutive_invalid_action_count = 0
                if history_response.strip() != response.strip():
                    sanitized_response_count += 1

                messages.append({"role": "assistant", "content": history_response})
                conversation_msg = {"role": "assistant", "content": history_response}
                if history_response.strip() != response.strip():
                    conversation_msg["raw_content"] = response
                conversations.append(conversation_msg)
            except Exception as e:
                logger.error(f"Generation error session {session_id} turn {turn}: {e}")
                return {
                    "session_id": session_id,
                    "reward": total_reward,
                    "success": total_reward == 1.0 or total_reward == 100.0,
                    "num_turns": len([m for m in messages if m["role"] == "assistant"]),
                    "conversations": conversations,
                    "initial_prompt": initial_prompt,
                    "data_idx": data_idx,
                    "invalid_action_count": invalid_action_count,
                    "sanitized_response_count": sanitized_response_count,
                    "error": str(e),
                    "error_type": "generation",
                }

            try:
                done, observation, step_reward, _ = await interaction.generate_response(instance_id, messages)
                # 注意：sciworld的step_reward就是当前任务的完成度分数，不需要累加
                # 直接使用最后一步的score作为最终reward
                total_reward = step_reward
                messages.append({"role": "user", "content": observation})
                conversations.append({"role": "user", "content": observation})
                
                if done:
                    logger.warning(f"Session {session_id}: ENV returned done=True at turn {turn}, step_reward={step_reward}, obs='{observation[:100]}...'")
            except Exception as e:
                logger.error(f"Environment error session {session_id} turn {turn}: {e}")
                return {
                    "session_id": session_id,
                    "reward": total_reward,
                    "success": total_reward == 1.0 or total_reward == 100.0,
                    "num_turns": len([m for m in messages if m["role"] == "assistant"]),
                    "conversations": conversations,
                    "initial_prompt": initial_prompt,
                    "data_idx": data_idx,
                    "invalid_action_count": invalid_action_count,
                    "sanitized_response_count": sanitized_response_count,
                    "error": str(e),
                    "error_type": "environment",
                }

            if not parsed_action and consecutive_invalid_action_count >= max_consecutive_invalid_actions:
                early_stop_reason = "max_consecutive_invalid_actions"
                logger.warning(
                    f"Session {session_id}: early stop after {consecutive_invalid_action_count} consecutive invalid actions"
                )
                break
        
        return {
            "session_id": session_id,
            "reward": total_reward,
            "success": total_reward == 1.0 or total_reward == 100.0,
            "num_turns": len([m for m in messages if m["role"] == "assistant"]),
            "conversations": conversations,
            "initial_prompt": initial_prompt,
            "data_idx": data_idx,
            "invalid_action_count": invalid_action_count,
            "sanitized_response_count": sanitized_response_count,
            "early_stop_reason": early_stop_reason,
        }
    finally:
        if instance_id in interaction.instance_sessions:
            try:
                await interaction.finalize_interaction(instance_id)
            except Exception as e:
                logger.warning(f"Finalization failed for session {session_id}: {e}")


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k: return None
    if c == 0: return 0.0
    if n - c < k: return 1.0
    prob = 1.0
    for i in range(k):
        prob *= (n - c - i) / (n - i)
    return 1.0 - prob


def safe_write_record(output_file: str, record: Dict):
    try:
        with open(output_file, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record) + '\n')
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Failed to write record to {output_file}: {e}")


async def fetch_model_name(server_url: str) -> str:
    """Fetches the actual model name from vLLM /v1/models endpoint"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{server_url.rstrip('/')}/v1/models",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    if models:
                        model_info = models[0]
                        model_id = model_info.get("id", "unknown")
                        model_root = model_info.get("root", "unknown")
                        return f"{model_id} ({model_root})"
                    return "unknown (no models found)"
                else:
                    logger.warning(f"Failed to query /v1/models: HTTP {response.status}")
                    return f"unknown (HTTP {response.status})"
    except Exception as e:
        logger.warning(f"Failed to fetch model name from vLLM: {e}")
        return f"unknown ({e})"


async def run_evaluation(args):
    interaction = SciWorldInteraction({
        'env_server_base': args.env_server,
        'timeout': 600, 'max_retries': 3
    })
    agent = AsyncSciWorldAgent(
        args.vllm_server_url,
        args.model_name,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.max_context_tokens,
        args.context_safety_margin,
        args.preserve_recent_messages,
        args.history_sanitization,
        args.enable_thinking,
    )
    
    logger.info(f"Loading dataset: {args.data_path}")
    df = pq.read_table(args.data_path).to_pandas()
    
    if 0 < args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)

    dataset_meta = build_sciworld_dataset_metadata(df)
    args._dataset_meta = dataset_meta
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.jsonl")
    summary_file = os.path.join(args.output_dir, f"eval_summary_{timestamp}.txt")
    open(output_file, 'w').close()
    
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    
    async def worker(item_id, session_id, interaction_kwargs, sample_idx, task_meta):
        async with sem:
            try:
                result = await evaluate_one_episode(
                    agent,
                    interaction,
                    session_id,
                    interaction_kwargs,
                    session,
                    args.max_rounds,
                    args.max_consecutive_invalid_actions,
                )
                record = {
                    "item_id": item_id,
                    "session_id": session_id,
                    "sample_idx": sample_idx,
                    "reward": result['reward'],
                    "success": result['success'],
                    "num_turns": result['num_turns'],
                    "data_idx": result.get('data_idx', session_id),
                    "invalid_action_count": result.get("invalid_action_count", 0),
                    "sanitized_response_count": result.get("sanitized_response_count", 0),
                    "history_sanitization": args.history_sanitization,
                    "enable_thinking": args.enable_thinking,
                }
                if result.get("early_stop_reason"):
                    record["early_stop_reason"] = result["early_stop_reason"]
                merge_task_metadata(record, task_meta)
                if not args.no_save_trajectories:
                    record["conversations"] = result.get('conversations', [])
                    record["initial_prompt"] = result.get('initial_prompt')
                if 'error' in result:
                    record["error"] = result['error']
                    record["error_type"] = result.get('error_type', 'unknown')
                safe_write_record(output_file, record)
                return result
            except Exception as e:
                logger.error(f"[CRITICAL] Worker exception for session {session_id}, sample {sample_idx}: {str(e)}")
                error_record = {
                    "item_id": item_id,
                    "session_id": session_id,
                    "sample_idx": sample_idx,
                    "reward": 0.0,
                    "success": False,
                    "num_turns": 0,
                    "data_idx": session_id,
                    "error": str(e),
                    "error_type": "worker_exception",
                    "history_sanitization": args.history_sanitization,
                    "enable_thinking": args.enable_thinking,
                }
                merge_task_metadata(error_record, task_meta)
                safe_write_record(output_file, error_record)
                return error_record
    
    results = []
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for _, row in df.iterrows():
            item_id, session_id, interaction_kwargs = resolve_eval_row(row, "sciworld")
            task_meta = dataset_meta["session_meta"].get(session_id, {"session_id": session_id})
            for sample_idx in range(args.num_samples_per_task):
                tasks.append(worker(item_id, session_id, interaction_kwargs, sample_idx, task_meta))
        pbar = tqdm(total=len(tasks), desc="Evaluating")
        for f in asyncio.as_completed(tasks):
            res = await f
            if res: results.append(res)
            pbar.update(1)
        pbar.close()
    
    args._df_rows = len(df)
    args._history_trim_stats = agent.get_trim_stats()
    await generate_summary(output_file, summary_file, args)
    logger.info(f"Results saved to {output_file}")


async def generate_summary(results_file: str, summary_file: str, args):
    """Generates evaluation summary from results file with categorized statistics."""
    results = []
    invalid_json_lines = 0
    try:
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    invalid_json_lines += 1
                    logger.warning(f"Skipping invalid JSON line: {e}")
    except Exception as e:
        logger.error(f"Failed to read results file: {e}")
        return

    if not results:
        logger.error("No valid results found to summarize!")
        return

    model_name = await fetch_model_name(args.vllm_server_url)
    logger.info(f"Model name from vLLM: {model_name}")

    dataset_meta = getattr(args, "_dataset_meta", None)
    if not dataset_meta:
        try:
            df_summary = pq.read_table(args.data_path).to_pandas()
            dataset_meta = build_sciworld_dataset_metadata(df_summary)
        except Exception as e:
            logger.warning(f"Failed to rebuild SciWorld dataset metadata: {e}")
            dataset_meta = {
                "session_meta": {},
                "split_episode_counts_by_category": {},
                "split_episode_counts_by_task_id": {},
                "split_task_ids_by_category": {},
                "official_categories": set(),
                "official_task_ids": set(),
                "paper_categories": set(),
                "paper_task_ids": set(),
                "split_categories": set(),
                "split_task_ids": set(),
            }

    session_meta = dataset_meta.get("session_meta", {})
    for res in results:
        meta = session_meta.get(int(res.get("session_id", -1)), {})
        merge_task_metadata(res, meta)

    grouped_results = defaultdict(list)
    for res in results:
        grouped_results[res['session_id']].append(res)

    expected_total_tasks = args.num_samples_per_task * args._df_rows if hasattr(args, '_df_rows') else None
    if expected_total_tasks is None:
        try:
            table = pq.read_table(args.data_path)
            df_summary = table.to_pandas()
            if args.max_samples > 0 and args.max_samples < len(df_summary):
                expected_total_tasks = args.num_samples_per_task * args.max_samples
            else:
                expected_total_tasks = args.num_samples_per_task * len(df_summary)
        except Exception:
            expected_total_tasks = len(grouped_results)

    total_expected_samples = expected_total_tasks
    total_finished_samples = len(results)
    total_missing_samples = total_expected_samples - total_finished_samples

    goal_mismatch_samples = []
    error_samples = []
    normal_samples = []

    for res in results:
        if res.get('error_type') == 'start_interaction' and 'Goal mismatch' in str(res.get('error', '')):
            goal_mismatch_samples.append(res)
        elif 'error' in res:
            error_samples.append(res)
        else:
            normal_samples.append(res)

    grouped_normal = defaultdict(list)
    for res in normal_samples:
        grouped_normal[res['session_id']].append(res)

    trim_stats = getattr(args, "_history_trim_stats", {}) or {}
    total_invalid_actions = sum(int(res.get("invalid_action_count", 0) or 0) for res in results)
    total_sanitized_responses = sum(int(res.get("sanitized_response_count", 0) or 0) for res in results)
    early_stop_counts = defaultdict(int)
    for res in results:
        reason = res.get("early_stop_reason")
        if reason:
            early_stop_counts[reason] += 1

    k_values = [1, 2, 4, 8]
    global_metrics = compute_group_metrics(grouped_normal, k_values)
    num_metric_tasks = len(grouped_results)
    num_normal_tasks = global_metrics["num_tasks"]

    split_categories = dataset_meta.get("split_categories", set())
    split_task_ids = dataset_meta.get("split_task_ids", set())
    official_categories = dataset_meta.get("official_categories", set())
    official_task_ids = dataset_meta.get("official_task_ids", set())
    paper_categories = dataset_meta.get("paper_categories", set())
    paper_task_ids = dataset_meta.get("paper_task_ids", set())

    missing_official_categories = ordered_missing_categories = sorted(official_categories - split_categories, key=sciworld_category_sort_key)
    missing_official_task_ids = sorted(official_task_ids - split_task_ids, key=sciworld_task_id_sort_key)
    missing_paper_task_ids = sorted(paper_task_ids - split_task_ids, key=sciworld_task_id_sort_key)

    category_results = defaultdict(list)
    category_normal = defaultdict(lambda: defaultdict(list))
    for res in results:
        task_category = res.get("task_category")
        if task_category:
            category_results[task_category].append(res)
    for res in normal_samples:
        task_category = res.get("task_category")
        if task_category:
            category_normal[task_category][res["session_id"]].append(res)

    task_results = defaultdict(list)
    task_normal = defaultdict(lambda: defaultdict(list))
    for res in results:
        task_id = res.get("task_id")
        if task_id:
            task_results[task_id].append(res)
    for res in normal_samples:
        task_id = res.get("task_id")
        if task_id:
            task_normal[task_id][res["session_id"]].append(res)

    metrics = [
        "=" * 60,
        "SciWorld Evaluation Summary",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"Model: {model_name}",
        f"Server: {args.vllm_server_url}",
        f"Dataset: {args.data_path}",
        "-" * 60,
        "TASK STATISTICS:",
        f"  Expected Samples:    {total_expected_samples}",
        f"  Finished Samples:    {total_finished_samples}",
        f"  Missing Samples:     {total_missing_samples} {'(ok)' if total_missing_samples == 0 else 'WARNING'}",
        f"  Invalid JSON Lines:  {invalid_json_lines} {'(ok)' if invalid_json_lines == 0 else 'WARNING'}",
        f"  Expected Episodes:   {expected_total_tasks // args.num_samples_per_task}",
        f"  Finished Episodes:   {len(grouped_results)}",
        "-" * 60,
        "SAMPLE BREAKDOWN:",
        f"  Normal samples:      {len(normal_samples)}",
        f"  Error samples:       {len(error_samples)} {'(ok)' if not error_samples else 'WARNING'}",
        f"    - start_interaction errors: {sum(1 for r in error_samples if r.get('error_type') == 'start_interaction')}",
        f"    - generation errors:         {sum(1 for r in error_samples if r.get('error_type') == 'generation')}",
        f"    - environment errors:        {sum(1 for r in error_samples if r.get('error_type') == 'environment')}",
        f"    - worker exceptions:         {sum(1 for r in error_samples if r.get('error_type') == 'worker_exception')}",
        f"  Goal mismatch:       {len(goal_mismatch_samples)} {'(ok)' if not goal_mismatch_samples else 'WARNING'}",
        "-" * 60,
        "HISTORY GUARDRAILS:",
        f"  History sanitization mode: {getattr(args, 'history_sanitization', 'raw')}",
        f"  Qwen thinking enabled:     {getattr(args, 'enable_thinking', False)}",
        f"  Invalid action generations: {total_invalid_actions}",
        f"  Sanitized assistant turns:  {total_sanitized_responses}",
        f"  Early-stopped samples:      {sum(early_stop_counts.values())}",
        f"    - max_consecutive_invalid_actions: {early_stop_counts.get('max_consecutive_invalid_actions', 0)}",
        f"  Context trim events:        {trim_stats.get('trim_event_count', 0)}",
        f"    - removed messages total:   {trim_stats.get('trimmed_messages_total', 0)}",
        f"    - truncated messages total: {trim_stats.get('truncated_messages_total', 0)}",
        f"    - max removed in one req:   {trim_stats.get('max_removed_messages', 0)}",
        f"    - max truncated in one req: {trim_stats.get('max_truncated_messages', 0)}",
        f"    - max estimated input toks: {trim_stats.get('max_estimated_input_tokens', 0)}",
        "-" * 60,
        "METRICS (based on normal samples only):",
        f"  Episodes Evaluated:  {num_metric_tasks}",
        f"  Normal Episodes:     {num_normal_tasks}",
        f"  Average Reward:      {global_metrics['avg_reward']:.4f}",
        f"  Average Success (Avg@1): {global_metrics['avg_success']:.4f}",
        "-" * 60,
        "SPLIT COVERAGE (official SciWorld layout vs current eval split):",
        f"  Official categories: {format_ratio(len(split_categories), len(official_categories))}",
        f"  Official task_ids:   {format_ratio(len(split_task_ids), len(official_task_ids))}",
        f"  Paper categories:    {format_ratio(len(split_categories & paper_categories), len(paper_categories))}",
        f"  Paper task_ids:      {format_ratio(len(split_task_ids & paper_task_ids), len(paper_task_ids))}",
        f"  Missing official categories: {', '.join(missing_official_categories) if missing_official_categories else '(none)'}",
        f"  Missing official task_ids:   {', '.join(missing_official_task_ids) if missing_official_task_ids else '(none)'}",
        f"  Missing paper task_ids:      {', '.join(missing_paper_task_ids) if missing_paper_task_ids else '(none)'}",
        "-" * 60,
    ]

    for k in sorted(k_values):
        values = global_metrics["pass_at_k_values"][k]
        if values:
            avg_pass_k = sum(values) / len(values)
            metrics.append(f"Pass@{k:<2}: {avg_pass_k:.4f} (episodes: {len(values)}/{num_normal_tasks})")
        else:
            metrics.append(f"Pass@{k:<2}: N/A    (insufficient samples)")

    if split_categories:
        metrics.extend([
            "-" * 60,
            "CATEGORY BREAKDOWN (split categories; metrics based on normal samples only):",
        ])
        for category in sorted(split_categories, key=sciworld_category_sort_key):
            official_task_ids_in_category = SCIWORLD_OFFICIAL_TASK_IDS_BY_CATEGORY.get(category, [])
            split_task_ids_in_category = dataset_meta["split_task_ids_by_category"].get(category, set())
            split_episode_count = dataset_meta["split_episode_counts_by_category"].get(category, 0)
            finished_samples = len(category_results.get(category, []))
            expected_samples = split_episode_count * args.num_samples_per_task
            category_metrics = compute_group_metrics(category_normal.get(category, {}), k_values)

            metrics.append(f"  {category}")
            metrics.append(
                f"    Task-id coverage: {len(split_task_ids_in_category)}/{len(official_task_ids_in_category)}"
                f" | Episodes in split: {split_episode_count}"
                f" | Samples: {finished_samples}/{expected_samples}"
            )
            metrics.append(
                f"    AvgReward: {category_metrics['avg_reward']:.4f}"
                f" | AvgSuccess: {category_metrics['avg_success']:.4f}"
                f" | {format_pass_metrics(category_metrics, k_values)}"
            )

    if split_task_ids:
        metrics.extend([
            "-" * 60,
            "TASK-ID BREAKDOWN (split task_ids; metrics based on normal samples only):",
        ])
        for task_id in sorted(split_task_ids, key=sciworld_task_id_sort_key):
            official_meta = SCIWORLD_LAYOUT_BY_TASK_ID.get(task_id, {})
            task_name = official_meta.get("task_name")
            task_category = official_meta.get("task_category")
            split_episode_count = dataset_meta["split_episode_counts_by_task_id"].get(task_id, 0)
            finished_samples = len(task_results.get(task_id, []))
            expected_samples = split_episode_count * args.num_samples_per_task
            task_metrics = compute_group_metrics(task_normal.get(task_id, {}), k_values)

            metrics.append(f"  {task_id} | {task_category} | {task_name}")
            metrics.append(
                f"    Episodes: {split_episode_count}"
                f" | Samples: {finished_samples}/{expected_samples}"
                f" | AvgReward: {task_metrics['avg_reward']:.4f}"
                f" | AvgSuccess: {task_metrics['avg_success']:.4f}"
            )
            metrics.append(f"    {format_pass_metrics(task_metrics, k_values)}")

    metrics.append("=" * 60)
    summary_text = "\n".join(metrics)
    print("\n" + summary_text + "\n")

    try:
        with open(summary_file, 'w') as f:
            f.write(summary_text)
    except Exception as e:
        logger.error(f"Failed to write summary file: {e}")


def main():
    parser = argparse.ArgumentParser(description='SciWorld Evaluation')
    parser.add_argument('--data_path', default=str(DEFAULT_DATA_PATH))
    parser.add_argument('--output_dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--env_server', default='http://127.0.0.1:36002')
    parser.add_argument('--vllm_server_url', default='http://localhost:8000')
    parser.add_argument('--model_name', default='qwen3')
    parser.add_argument('--max_rounds', type=int, default=30)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--num_samples_per_task', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--max_context_tokens', type=int, default=12288,
                        help='Estimated context budget used to trim SciWorld dialogue history')
    parser.add_argument('--context_safety_margin', type=int, default=256,
                        help='Reserved token margin when trimming dialogue history')
    parser.add_argument('--preserve_recent_messages', type=int, default=8,
                        help='How many recent messages to preserve when trimming dialogue history')
    parser.add_argument('--max_consecutive_invalid_actions', type=int, default=3,
                        help='Early-stop a sample after this many consecutive generations without a parseable action')
    parser.add_argument('--history_sanitization', choices=('raw', 'strip_tags', 'canonicalize'), default='raw',
                        help='How assistant responses are written into the next history turn. Use raw for formal eval.')
    parser.add_argument('--enable_thinking', action='store_true',
                        help='Allow Qwen thinking mode. Default is disabled via chat_template_kwargs.enable_thinking=false.')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_save_trajectories', action='store_true',
                        help='Do not save conversation trajectories')
    
    args = parser.parse_args()
    log_dir = configure_logging(args.output_dir)
    logger.info(f"Resolved project root: {project_root}")
    logger.info(f"Eval logs will be written to: {log_dir}")
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
