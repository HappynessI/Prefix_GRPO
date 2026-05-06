#!/usr/bin/env python3
"""
BabyAI Evaluation Client - vLLM Service Mode
直接调用已部署的vLLM服务，无需本地加载模型
"""

import os
import sys
import json
import logging
import argparse
import uuid
import asyncio
import random
import math
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pyarrow.parquet as pq
from tqdm import tqdm
import aiohttp
import fcntl

def discover_project_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "verl").exists():
            return candidate
    raise RuntimeError(f"verl not found from {resolved}")


def default_data_path(root: Path) -> Path:
    local_data = root / "data" / "babyai" / "test.parquet"
    shared_data = Path("/Data/wyh/datasets/Verl-Data/eval/babyai/test.parquet")
    return local_data if local_data.exists() else shared_data


def default_output_dir(root: Path) -> Path:
    shared_output_root = Path("/Data/wyh/datasets/Verl-Data/outputs")
    if shared_output_root.exists():
        return shared_output_root / "babyai_eval"
    return root / "outputs" / "babyai_eval"


def bootstrap_import_paths(root: Path) -> None:
    candidate_paths = [root / "verl", root]
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


project_root = discover_project_root(Path(__file__))
if not (project_root / "verl").exists():
    raise RuntimeError(f"verl not found in {project_root}")
bootstrap_import_paths(project_root)

DEFAULT_DATA_PATH = default_data_path(project_root)
DEFAULT_OUTPUT_DIR = default_output_dir(project_root)

from verl.interactions.babyai_interaction import BabyAIInteraction
from agentgym_eval_utils import resolve_eval_row

logger = logging.getLogger("BabyAIEval")
_RESULT_WRITE_LOCK = threading.Lock()


BABYAI_CATEGORY_ORDER = (
    "GoTo",
    "Pickup",
    "AOD",
    "Find Room",
    "SLoc",
)

BABYAI_OFFICIAL_CATEGORY_COUNTS = {
    "GoTo": {"total": 550, "train": 495, "test": 55},
    "Pickup": {"total": 150, "train": 135, "test": 15},
    "AOD": {"total": 50, "train": 45, "test": 5},
    "Find Room": {"total": 100, "train": 90, "test": 10},
    "SLoc": {"total": 50, "train": 45, "test": 5},
}

BABYAI_OFFICIAL_LEVELS_BY_CATEGORY = {
    "GoTo": tuple(f"level_{level:02d}" for level in range(1, 12)),
    "Pickup": ("level_19", "level_20", "level_21"),
    "AOD": ("level_30",),
    "Find Room": ("level_31", "level_33"),
    "SLoc": ("level_36",),
}

BABYAI_LEVEL_TO_CATEGORY = {
    level_name: category
    for category, level_names in BABYAI_OFFICIAL_LEVELS_BY_CATEGORY.items()
    for level_name in level_names
}


def babyai_category_sort_key(category: Optional[str]) -> tuple[int, str]:
    order = {name: idx for idx, name in enumerate(BABYAI_CATEGORY_ORDER)}
    category = category or ""
    return (order.get(category, len(order)), category)


def babyai_level_sort_key(level_name: Optional[str]) -> tuple[int, str]:
    if not level_name:
        return (10**9, "")
    try:
        return (int(str(level_name).split("_")[-1]), str(level_name))
    except Exception:
        return (10**9, str(level_name))


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return f"{numerator}/{denominator}"
    return f"{numerator}/{denominator} ({(numerator / denominator) * 100:.1f}%)"


def build_babyai_dataset_metadata(df) -> Dict[str, Any]:
    session_meta = {}
    split_episode_counts_by_category = defaultdict(int)
    split_episode_counts_by_level = defaultdict(int)
    split_levels_by_category = defaultdict(set)

    for _, row in df.iterrows():
        extra_info = row.get("extra_info") or {}
        interaction_kwargs = dict(extra_info.get("interaction_kwargs") or {})
        session_id = interaction_kwargs.get("session_id", row.get("session_id"))
        if session_id is None:
            continue
        session_id = int(session_id)

        task_category = row.get("task_category") or interaction_kwargs.get("task_category")
        task_subcategory = row.get("task_subcategory") or interaction_kwargs.get("task_subcategory")
        babyai_level = row.get("babyai_level")
        item_id = row.get("item_id") or f"babyai_{session_id}"

        meta = {
            "item_id": str(item_id),
            "session_id": session_id,
            "task_category": task_category,
            "task_subcategory": task_subcategory,
            "babyai_level": babyai_level,
        }
        session_meta[session_id] = meta

        if task_category:
            split_episode_counts_by_category[task_category] += 1
        if task_subcategory:
            split_episode_counts_by_level[task_subcategory] += 1
            if task_category:
                split_levels_by_category[task_category].add(task_subcategory)

    official_categories = set(BABYAI_OFFICIAL_CATEGORY_COUNTS)
    official_levels = set(BABYAI_LEVEL_TO_CATEGORY)
    split_categories = set(split_episode_counts_by_category)
    split_levels = set(split_episode_counts_by_level)

    return {
        "session_meta": session_meta,
        "split_episode_counts_by_category": dict(split_episode_counts_by_category),
        "split_episode_counts_by_level": dict(split_episode_counts_by_level),
        "split_levels_by_category": {k: set(v) for k, v in split_levels_by_category.items()},
        "official_categories": official_categories,
        "official_levels": official_levels,
        "split_categories": split_categories,
        "split_levels": split_levels,
    }


def merge_task_metadata(record: Dict[str, Any], task_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not task_meta:
        return record
    for key in ("task_category", "task_subcategory", "babyai_level"):
        value = task_meta.get(key)
        if value is not None:
            record[key] = value
    return record


def record_group_key(record: Dict[str, Any]) -> Any:
    item_id = record.get("item_id")
    return item_id if item_id is not None else record.get("session_id")


def is_babyai_env_completed(done: bool, observation: Any) -> bool:
    return bool(done) or "task is completed" in str(observation).lower()


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


# =============================================================================
# Async Agent Class (HTTP Client)
# =============================================================================

class AsyncBabyAIAgent:
    """BabyAI Agent - Uses HTTP calls to vLLM service"""
    
    def __init__(
        self,
        server_url: str,
        model_name: str = "qwen3",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_context_tokens: int = 10240,
        context_safety_margin: int = 256,
        preserve_recent_messages: int = 8,
    ):
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_context_tokens = max_context_tokens
        self.context_safety_margin = context_safety_margin
        self.preserve_recent_messages = preserve_recent_messages
        self.system_prompt = self._build_prompt()
        self.trim_event_count = 0
        self.trimmed_messages_total = 0
        self.truncated_messages_total = 0
        self.max_removed_messages = 0
        self.max_truncated_messages = 0
        self.max_estimated_input_tokens = 0

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
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

    def get_trim_stats(self) -> Dict[str, int]:
        return {
            "trim_event_count": self.trim_event_count,
            "trimmed_messages_total": self.trimmed_messages_total,
            "truncated_messages_total": self.truncated_messages_total,
            "max_removed_messages": self.max_removed_messages,
            "max_truncated_messages": self.max_truncated_messages,
            "max_estimated_input_tokens": self.max_estimated_input_tokens,
        }
    
    def _build_prompt(self) -> str:
        return '''You are a BabyAI Navigation Agent. Your goal is to complete navigation tasks in a grid-world environment.

**CORE PROTOCOL (Strictly Follow):**
1. **THINK FIRST**: Before any action, analyze the current observation and goal.
2. **ONE ACTION**: Output exactly ONE action per turn.
3. **BOX FORMAT**: Wrap your command in `[[ ]]`. Example: `Action: [[ turn left ]]`
4. **NO HALLUCINATION**: Do NOT simulate the Environment's response. Stop immediately after outputting the Action.

**REASONING LOGIC:**
1. **Understand Goal**: Parse the goal (e.g., "go to the red ball", "pick up the blue key")
2. **Observe Environment**: Note visible objects, their positions, and available actions
3. **Plan Path**: Determine the sequence of actions to reach the goal
4. **Execute**: Take one action at a time, adjusting based on feedback

**CORE COMMAND SET (API):**
* `turn left` / `turn right` - Rotate 90 degrees
* `move forward` - Move one step forward (if not blocked)
* `pickup [color] [object] [n]` - Pick up an object (e.g., "pickup red ball 1")
* `drop` - Drop the carried object
* `toggle` - Toggle a door (open/close)
* `go to [object]` - Navigate to an object (e.g., "go to blue key 1")
* `go through [door]` - Pass through an open door
* `check available actions` - List all valid actions

**INTERACTION EXAMPLE:**

[Environment]
Your goal: go to the red ball
In front of you in this room, you can see several objects: There is a red ball 1 3 steps in front of you and 1 steps to your left.
The room has walls around you. You are facing a wall 5 steps away.
You are not carrying anything.
Available actions: ["turn left", "turn right", "move forward", "go to red ball 1", "check available actions"]

[You]
Think: My goal is to reach the red ball. It's 3 steps ahead and 1 step to my left. I should use "go to red ball 1" to navigate there.
Action: [[ go to red ball 1 ]]

[Environment]
In front of you in this room, you can see several objects: There is a red ball 1 right in front of you 1 steps away.
You are not carrying anything.
Available actions: ["turn left", "turn right", "pickup red ball 1", "check available actions"]

[You]
Think: I'm now facing the red ball. The goal was to "go to" it, so I've completed the task.
Action: [[ Task Completed! ]]
'''
    
    async def generate(self, messages: List[Dict[str, str]], session: aiohttp.ClientSession) -> str:
        """Generates response by calling vLLM service via HTTP"""
        trimmed_messages = self._trim_messages_for_context(messages)
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *trimmed_messages
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False
        }
        
        try:
            async with session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"HTTP {response.status} from vLLM: {error_text}")
                    return ""
                
                result = await response.json()
                if "choices" not in result or len(result["choices"]) == 0:
                    logger.error(f"Invalid response format: {result}")
                    return ""
                
                return result["choices"][0]["message"]["content"].strip()
                
        except asyncio.TimeoutError:
            logger.error("Request timed out after 120s")
            return ""
        except Exception as e:
            logger.error(f"HTTP request failed: {str(e)}")
            return ""


# =============================================================================
# Evaluation Logic
# =============================================================================

async def evaluate_one_episode(
    agent: AsyncBabyAIAgent,
    interaction: BabyAIInteraction,
    session_id: int,
    interaction_kwargs: Dict[str, Any],
    http_session: aiohttp.ClientSession,
    max_rounds: int = 50,
    initial_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluates a single episode asynchronously."""
    
    instance_id = f"eval_{session_id}_{uuid.uuid4().hex[:8]}"
    messages = []
    conversations = []
    done = False
    total_reward = 0.0
    env_completed = False
    
    try:
        # --- Initialization ---
        if initial_prompt is None:
            try:
                await interaction.start_interaction(instance_id, **interaction_kwargs)
                done, initial_obs, reward, extra = await interaction.generate_response(instance_id, messages)
                total_reward += reward
                env_completed = env_completed or is_babyai_env_completed(done, initial_obs)
                messages.append({"role": "user", "content": initial_obs})
                conversations.append({"role": "user", "content": initial_obs})
                initial_prompt = initial_obs
            except Exception as e:
                logger.error(f"Start interaction failed for session {session_id}: {e}")
                return {
                    "session_id": session_id,
                    "reward": 0.0,
                    "success": False,
                    "num_turns": 0,
                    "initial_prompt": None,
                    "conversations": [],
                    "error": str(e),
                    "error_type": "start_interaction",
                }
        else:
            try:
                await interaction.start_interaction(instance_id, **interaction_kwargs)
                messages.append({"role": "user", "content": initial_prompt})
                conversations.append({"role": "user", "content": initial_prompt})
            except Exception as e:
                logger.error(f"Failed to start interaction for session {session_id}: {e}")
                return {
                    "session_id": session_id,
                    "reward": 0.0,
                    "success": False,
                    "num_turns": 0,
                    "initial_prompt": initial_prompt,
                    "conversations": [],
                    "error": str(e),
                    "error_type": "start_interaction",
                }

        # --- Interaction Loop ---
        consecutive_empty = 0
        max_consecutive_empty = 3  # 连续空响应超过此数则放弃
        
        for turn in range(max_rounds):
            if done:
                break
            
            try:
                response = await agent.generate(messages, http_session)
                
                # 处理空响应（超时/错误导致）：不加入对话，避免雪崩循环
                if not response.strip():
                    consecutive_empty += 1
                    logger.warning(f"Session {session_id} turn {turn}: empty response ({consecutive_empty}/{max_consecutive_empty})")
                    if consecutive_empty >= max_consecutive_empty:
                        logger.warning(f"Session {session_id}: too many consecutive empty responses, giving up")
                        break
                    continue  # 跳过本轮，不污染对话历史
                
                consecutive_empty = 0  # 收到有效响应，重置计数
                
                response_lower = response.lower()
                if 'task completed' in response_lower or 'task failed' in response_lower:
                    done = True
                messages.append({"role": "assistant", "content": response})
                conversations.append({"role": "assistant", "content": response})
            except Exception as e:
                logger.error(f"Generation error session {session_id} turn {turn}: {e}")
                return {
                    "session_id": session_id,
                    "reward": total_reward,
                    "success": env_completed,
                    "env_completed": env_completed,
                    "num_turns": len([m for m in messages if m["role"] == "assistant"]),
                    "conversations": conversations,
                    "initial_prompt": initial_prompt,
                    "error": str(e),
                    "error_type": "generation",
                }
            
            try:
                done, observation, step_reward, extra = await interaction.generate_response(instance_id, messages)
                total_reward += step_reward
                env_completed = env_completed or is_babyai_env_completed(done, observation)
                messages.append({"role": "user", "content": observation})
                conversations.append({"role": "user", "content": observation})
            except Exception as e:
                logger.error(f"Environment error session {session_id} turn {turn}: {e}")
                return {
                    "session_id": session_id,
                    "reward": total_reward,
                    "success": env_completed,
                    "env_completed": env_completed,
                    "num_turns": len([m for m in messages if m["role"] == "assistant"]),
                    "conversations": conversations,
                    "initial_prompt": initial_prompt,
                    "error": str(e),
                    "error_type": "environment",
                }
        
        success = env_completed
        num_turns = len([m for m in messages if m["role"] == "assistant"])
        
        return {
            "session_id": session_id,
            "reward": total_reward,
            "success": success,
            "env_completed": env_completed,
            "num_turns": num_turns,
            "conversations": conversations,
            "initial_prompt": initial_prompt
        }
    finally:
        if instance_id in interaction.instance_sessions:
            try:
                await interaction.finalize_interaction(instance_id)
            except Exception as e:
                logger.warning(f"Finalization failed for session {session_id}: {e}")


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Estimates pass@k metric (k ≤ num_samples)"""
    if num_samples < k:
        return None
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    
    prob_all_failure = 1.0
    for i in range(k):
        prob_all_failure *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - prob_all_failure


# =============================================================================
# Safe File Writing with Locking
# =============================================================================

def safe_write_record(output_file: str, record: Dict[str, Any]):
    """Safely appends a JSON record to a file using file locking."""
    try:
        with _RESULT_WRITE_LOCK:
            with open(output_file, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(record) + '\n')
                    f.flush()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Failed to write record to {output_file}: {e}")


# =============================================================================
# Main Evaluation Loop
# =============================================================================

async def run_evaluation(args: argparse.Namespace):
    """Main evaluation function with async HTTP client"""
    interaction = BabyAIInteraction({
        'env_server_base': args.babyai_server,
        'timeout': 600,
        'max_retries': 3
    })

    agent = AsyncBabyAIAgent(
        server_url=args.vllm_server_url,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_context_tokens=args.max_context_tokens,
    )

    logger.info(f"Loading dataset from: {args.data_path}")
    table = pq.read_table(args.data_path)
    df = table.to_pandas()

    if args.max_samples > 0 and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
        logger.info(f"Using {args.max_samples} random samples")
    else:
        logger.info(f"Using full dataset ({len(df)} samples)")

    dataset_meta = build_babyai_dataset_metadata(df)
    args._dataset_meta = dataset_meta

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.jsonl")
    summary_file = os.path.join(args.output_dir, f"eval_summary_{timestamp}.txt")

    logger.info(f"Output file: {output_file}")
    logger.info(f"Summary file: {summary_file}")

    open(output_file, 'w').close()

    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(
        item_id: str,
        session_id: int,
        interaction_kwargs: Dict[str, Any],
        sample_idx: int,
        task_meta: Dict[str, Any],
    ):
        async with sem:
            try:
                result = await evaluate_one_episode(
                    agent=agent,
                    interaction=interaction,
                    session_id=session_id,
                    interaction_kwargs=interaction_kwargs,
                    http_session=session,
                    max_rounds=args.max_rounds
                )
                
                # Prepare record
                record = {
                    "item_id": item_id,
                    "session_id": session_id,
                    "sample_idx": sample_idx,
                    "reward": result['reward'],
                    "success": result['success'],
                    "env_completed": result.get("env_completed", result["success"]),
                    "num_turns": result['num_turns'],
                }
                merge_task_metadata(record, task_meta)
                if not args.no_save_trajectories:
                    record["conversations"] = result['conversations']
                    record["initial_prompt"] = result['initial_prompt']
                if 'error' in result:
                    record["error"] = result['error']
                    record["error_type"] = result.get('error_type', 'unknown')
                await asyncio.to_thread(safe_write_record, output_file, record)
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
                    "error": str(e),
                    "error_type": "worker_exception",
                }
                merge_task_metadata(error_record, task_meta)
                await asyncio.to_thread(safe_write_record, output_file, error_record)
                return error_record

    results = []
    total_tasks = len(df) * args.num_samples_per_task

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for _, row in df.iterrows():
            item_id, real_session_id, interaction_kwargs = resolve_eval_row(row, "babyai")
            task_meta = dataset_meta["session_meta"].get(real_session_id, {"session_id": real_session_id, "item_id": item_id})
            for sample_idx in range(args.num_samples_per_task):
                tasks.append(worker(item_id, real_session_id, interaction_kwargs, sample_idx, task_meta))

        pbar = tqdm(total=total_tasks, desc="Evaluating", unit="sample")
        for f in asyncio.as_completed(tasks):
            res = await f
            if res:
                results.append(res)
            pbar.update(1)
        pbar.close()

    args._df_rows = len(df)
    args._trim_stats = agent.get_trim_stats()
    await generate_summary(output_file, summary_file, args)

    logger.info(f"✅ Evaluation completed! Results saved to:\n- {output_file}\n- {summary_file}")
    return output_file, summary_file


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


async def generate_summary(results_file: str, summary_file: str, args: argparse.Namespace):
    """Generates evaluation summary from results file with categorized statistics."""
    results = []
    try:
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
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
            dataset_meta = build_babyai_dataset_metadata(df_summary)
        except Exception as e:
            logger.warning(f"Failed to rebuild BabyAI dataset metadata: {e}")
            dataset_meta = {
                "session_meta": {},
                "split_episode_counts_by_category": {},
                "split_episode_counts_by_level": {},
                "split_levels_by_category": {},
                "official_categories": set(),
                "official_levels": set(),
                "split_categories": set(),
                "split_levels": set(),
            }

    session_meta = dataset_meta.get("session_meta", {})
    for res in results:
        meta = session_meta.get(int(res.get("session_id", -1)), {})
        merge_task_metadata(res, meta)

    grouped_results = defaultdict(list)
    for res in results:
        grouped_results[record_group_key(res)].append(res)
    
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

    k_values = [1, 2, 4, 8]

    grouped_normal = defaultdict(list)
    for res in normal_samples:
        grouped_normal[record_group_key(res)].append(res)

    global_metrics = compute_group_metrics(grouped_normal, k_values)
    num_metric_tasks = len(grouped_results)
    num_normal_tasks = global_metrics["num_tasks"]

    split_categories = dataset_meta.get("split_categories", set())
    split_levels = dataset_meta.get("split_levels", set())
    official_categories = dataset_meta.get("official_categories", set())
    official_levels = dataset_meta.get("official_levels", set())

    missing_categories = sorted(official_categories - split_categories, key=babyai_category_sort_key)
    missing_levels = sorted(official_levels - split_levels, key=babyai_level_sort_key)

    category_results = defaultdict(list)
    category_normal = defaultdict(lambda: defaultdict(list))
    for res in results:
        task_category = res.get("task_category")
        if task_category:
            category_results[task_category].append(res)
    for res in normal_samples:
        task_category = res.get("task_category")
        if task_category:
            category_normal[task_category][record_group_key(res)].append(res)

    level_results = defaultdict(list)
    level_normal = defaultdict(lambda: defaultdict(list))
    for res in results:
        task_subcategory = res.get("task_subcategory")
        if task_subcategory:
            level_results[task_subcategory].append(res)
    for res in normal_samples:
        task_subcategory = res.get("task_subcategory")
        if task_subcategory:
            level_normal[task_subcategory][record_group_key(res)].append(res)

    metrics = [
        "=" * 60,
        "BabyAI Evaluation Summary (vLLM Service Mode)",
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
        f"  Expected Tasks:      {expected_total_tasks // args.num_samples_per_task}",
        f"  Finished Tasks:      {len(grouped_results)}",
        "-" * 60,
        "CONTEXT TRIMMING:",
        f"  max_context_tokens:  {args.max_context_tokens}",
        f"  max_new_tokens:      {args.max_new_tokens}",
        f"  trim events:         {getattr(args, '_trim_stats', {}).get('trim_event_count', 0)}",
        f"  removed messages:    {getattr(args, '_trim_stats', {}).get('trimmed_messages_total', 0)}",
        f"  truncated messages:  {getattr(args, '_trim_stats', {}).get('truncated_messages_total', 0)}",
        f"  max est input toks:  {getattr(args, '_trim_stats', {}).get('max_estimated_input_tokens', 0)}",
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
        "METRICS (based on normal samples only):",
        f"  Tasks Evaluated:     {num_metric_tasks}",
        f"  Normal Tasks:        {num_normal_tasks}",
        f"  Average Reward:      {global_metrics['avg_reward']:.4f}",
        f"  Average Success (Avg@1): {global_metrics['avg_success']:.4f}",
        "-" * 60,
        "SPLIT COVERAGE (paper-aligned BabyAI layout vs current eval split):",
        f"  Official categories: {format_ratio(len(split_categories), len(official_categories))}",
        f"  Official levels:     {format_ratio(len(split_levels), len(official_levels))}",
        f"  Missing categories:  {', '.join(missing_categories) if missing_categories else '(none)'}",
        f"  Missing levels:      {', '.join(missing_levels) if missing_levels else '(none)'}",
        "-" * 60,
    ]

    for k in sorted(k_values):
        values = global_metrics["pass_at_k_values"][k]
        if values:
            avg_pass_k = sum(values) / len(values)
            metrics.append(f"Pass@{k:<2}: {avg_pass_k:.4f} (tasks: {len(values)}/{num_normal_tasks})")
        else:
            metrics.append(f"Pass@{k:<2}: N/A    (insufficient samples)")

    if official_categories:
        metrics.extend([
            "-" * 60,
            "CATEGORY BREAKDOWN (paper-aligned categories; metrics based on normal samples only):",
        ])
        for category in BABYAI_CATEGORY_ORDER:
            official_counts = BABYAI_OFFICIAL_CATEGORY_COUNTS.get(category, {})
            official_levels_in_category = BABYAI_OFFICIAL_LEVELS_BY_CATEGORY.get(category, ())
            split_levels_in_category = dataset_meta["split_levels_by_category"].get(category, set())
            split_episode_count = dataset_meta["split_episode_counts_by_category"].get(category, 0)
            finished_samples = len(category_results.get(category, []))
            expected_samples = split_episode_count * args.num_samples_per_task
            category_metrics = compute_group_metrics(category_normal.get(category, {}), k_values)
            visible_levels = ", ".join(sorted(split_levels_in_category, key=babyai_level_sort_key)) if split_levels_in_category else "(none)"

            metrics.append(f"  {category}")
            metrics.append(
                "    Released split: "
                f"total/train/test = {official_counts.get('total', 0)}/{official_counts.get('train', 0)}/{official_counts.get('test', 0)}"
            )
            metrics.append(
                f"    Level coverage: {len(split_levels_in_category)}/{len(official_levels_in_category)}"
                f" | Episodes in split: {split_episode_count}"
                f" | Samples: {finished_samples}/{expected_samples}"
            )
            metrics.append(f"    Visible levels: {visible_levels}")
            metrics.append(
                f"    AvgReward: {category_metrics['avg_reward']:.4f}"
                f" | AvgSuccess: {category_metrics['avg_success']:.4f}"
                f" | {format_pass_metrics(category_metrics, k_values)}"
            )

    if split_levels:
        metrics.extend([
            "-" * 60,
            "LEVEL BREAKDOWN (split levels; metrics based on normal samples only):",
        ])
        for level_name in sorted(split_levels, key=babyai_level_sort_key):
            task_category = BABYAI_LEVEL_TO_CATEGORY.get(level_name, "")
            split_episode_count = dataset_meta["split_episode_counts_by_level"].get(level_name, 0)
            finished_samples = len(level_results.get(level_name, []))
            expected_samples = split_episode_count * args.num_samples_per_task
            level_metrics = compute_group_metrics(level_normal.get(level_name, {}), k_values)

            metrics.append(f"  {level_name} | {task_category}")
            metrics.append(
                f"    Episodes: {split_episode_count}"
                f" | Samples: {finished_samples}/{expected_samples}"
                f" | AvgReward: {level_metrics['avg_reward']:.4f}"
                f" | AvgSuccess: {level_metrics['avg_success']:.4f}"
            )
            metrics.append(f"    {format_pass_metrics(level_metrics, k_values)}")

    metrics.append("=" * 60)
    summary_text = "\n".join(metrics)
    print("\n" + summary_text + "\n")

    try:
        with open(summary_file, 'w') as f:
            f.write(summary_text)
    except Exception as e:
        logger.error(f"Failed to write summary file: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BabyAI Evaluation Client (vLLM Service Mode)')

    parser.add_argument('--data_path', type=str,
                        default=str(DEFAULT_DATA_PATH),
                        help='Path to test dataset (parquet format)')
    parser.add_argument('--output_dir', type=str,
                        default=str(DEFAULT_OUTPUT_DIR),
                        help='Directory to save evaluation results')

    parser.add_argument('--babyai_server', type=str,
                        default='http://127.0.0.1:36005',
                        help='BabyAI environment server URL')
    parser.add_argument('--model_name', type=str, default='qwen3',
                        help='Model name as registered in vLLM (default: qwen3)')
    parser.add_argument('--vllm_server_url', type=str,
                        default='http://localhost:8000',
                        help='vLLM service URL (e.g., http://localhost:8000)')

    parser.add_argument('--max_rounds', type=int, default=50,
                        help='Maximum interaction rounds per episode')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Maximum samples to evaluate (-1 for all)')
    parser.add_argument('--num_samples_per_task', type=int, default=8,
                        help='Number of samples per task (for pass@k)')
    parser.add_argument('--concurrency', type=int, default=128,
                        help='Maximum concurrent requests to vLLM service')

    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum tokens to generate per response')
    parser.add_argument('--max_context_tokens', type=int, default=10240,
                        help='Maximum total context window; history is trimmed before generation')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0.0 for greedy)')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p sampling parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_save_trajectories', action='store_true',
                        help='Do not save conversation trajectories (saves disk space)')

    args = parser.parse_args()

    if not args.vllm_server_url.startswith(('http://', 'https://')):
        args.vllm_server_url = f"http://{args.vllm_server_url}"

    if args.num_samples_per_task > 1 and args.temperature == 0.0:
        logger.warning("WARNING: num_samples_per_task > 1 but temperature=0.0. "
                      "Results will be identical for all samples. Consider increasing temperature.")

    log_dir = configure_logging(args.output_dir)
    logger.info(f"Resolved project root: {project_root}")
    logger.info(f"Eval logs will be written to: {log_dir}")
    logger.info("Evaluation Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    try:
        asyncio.run(run_evaluation(args))
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.exception(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
