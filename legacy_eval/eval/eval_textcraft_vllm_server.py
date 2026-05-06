

#!/usr/bin/env python3
"""
TextCraft Evaluation Client - vLLM Service Mode
直接调用已部署的vLLM服务，无需本地加载模型
"""

import os
import sys
import json
import math
import logging
import argparse
import uuid
import asyncio
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pyarrow.parquet as pq
from tqdm import tqdm
import aiohttp
import fcntl

# Add verl to path
# 文件路径: /Data/wyh/verl/examples/sglang_multiturn/my_exp/eval/eval_textcraft_vllm_server.py
# 需要向上5级到达项目根目录: /Data/wyh/verl
project_root = Path(__file__).resolve().parent.parent.parent
if not (project_root / "verl").exists():
    raise RuntimeError(f"verl not found in {project_root}")
sys.path.insert(0, str(project_root))



from verl.interactions.textcraft_interaction import TextCraftInteraction

# 确保日志目录存在
log_dir = Path("/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"eval_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("TextCraftEval")

OFFICIAL_TEXTCRAFT_DEPTH_BANDS = {
    1: range(0, 31),
    2: range(140, 181),
    3: range(420, 445),
    4: range(533, 536),
}


def _safe_int(value: Any) -> Optional[int]:
    """Best-effort int conversion for task binding metadata."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_textcraft_binding(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the stable TextCraft task binding from a parquet row.

    Priority:
    1. top-level `item_id`
    2. `extra_info.interaction_kwargs.item_id`
    3. `extra_info.interaction_kwargs.session_id`
    4. top-level `original_index`
    5. `extra_info.index`
    """
    item_id = None
    goal = None
    data_idx = None
    session_id = None

    if hasattr(row, 'get') and callable(row.get):
        item_id = row.get('item_id')
        extra_info = row.get('extra_info', {}) or {}
        interaction_kwargs = extra_info.get('interaction_kwargs', {}) or {}

        if item_id is None:
            item_id = interaction_kwargs.get('item_id')
        goal = interaction_kwargs.get('goal')
        data_idx = _safe_int(interaction_kwargs.get('data_idx'))

        if isinstance(item_id, str) and item_id.startswith("textcraft_"):
            session_id = _safe_int(item_id.split("_")[-1])

        if session_id is None:
            session_id = _safe_int(interaction_kwargs.get('session_id'))

        if session_id is None:
            session_id = _safe_int(row.get('original_index'))

        if session_id is None:
            session_id = _safe_int(extra_info.get('index'))

    if session_id is None:
        raise ValueError(f"Could not determine stable TextCraft task id from row: {row}")

    if data_idx is None:
        data_idx = session_id

    if item_id is None:
        item_id = f"textcraft_{session_id}"

    return {
        "item_id": item_id,
        "session_id": session_id,
        "data_idx": data_idx,
        "goal": goal,
    }


def session_id_to_textcraft_depth(session_id: int) -> Optional[int]:
    for depth, band in OFFICIAL_TEXTCRAFT_DEPTH_BANDS.items():
        if session_id in band:
            return depth
    return None


class GenerationRequestError(RuntimeError):
    """Raised when a vLLM generation request fails and the sample should be marked as error."""

# =============================================================================
# Async Agent Class (HTTP Client)
# =============================================================================

class AsyncTextCraftAgent:
    """TextCraft Agent - Uses HTTP calls to vLLM service"""
    
    def __init__(
        self,
        server_url: str,
        model_name: str = "qwen3",
        max_new_tokens: int = 150,
        temperature: float = 0.0,
        top_p: float = 1.0,
        request_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        max_context_tokens: int = 8192,
        context_safety_margin: int = 256,
        preserve_recent_messages: int = 6,
    ):
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.request_retries = request_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_context_tokens = max_context_tokens
        self.context_safety_margin = context_safety_margin
        self.preserve_recent_messages = preserve_recent_messages
        self.system_prompt = self._build_adapt_prompt()
        self.trim_event_count = 0
        self.trimmed_messages_total = 0
        self.truncated_messages_total = 0
        self.max_removed_messages = 0
        self.max_truncated_messages = 0
        self.max_estimated_input_tokens = 0

    def _retry_delay(self, attempt: int) -> float:
        return self.retry_backoff_seconds * (2 ** attempt)

    @staticmethod
    def _is_retryable_status(status: int) -> bool:
        return status in {408, 409, 429, 500, 502, 503, 504}

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """Conservative token estimate for context trimming."""
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
        """
        Keep the initial task prompt plus the most recent turns, and drop stale
        middle history when the estimated context would exceed the model budget.
        """
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
    
    def _build_adapt_prompt(self) -> str:
        return '''You are a Minecraft Assistant. Your goal is to craft items by managing resources and recipes.

**CORE PROTOCOL (Strictly Follow):**
1. **THINK FIRST**: Before any action, analyze the current state using the "Reasoning Logic" below.
2. **ONE ACTION**: Output exactly ONE action per turn.
3. **BOX FORMAT**: Wrap your command in `[[ ]]`. Example: `Action: [[ inventory ]]`
4. **NO HALLUCINATION**: Do NOT simulate the Environment's response. Stop immediately after outputting the Action.

**REASONING LOGIC (The Algorithm):**
When trying to acquire an item [Target]:
1. **Check Inventory**: Do you already have [Target]?
   - If YES -> Task Complete / Proceed to next step.
   - If NO -> Go to step 2.
2. **Check Recipe**: Is there a crafting recipe for [Target]?
   - If YES -> Check if you have the [Ingredients].
     - If you have Ingredients -> `craft [Target] ...`
     - If you miss Ingredients -> **NEW SUB-GOAL**: Get the missing [Ingredient] (Repeat Logic from Step 1).
   - If NO (Base Material) -> `get [Target]` directly from environment.
3. **Handle Errors**:
   - If `get` fails -> The item might require crafting from a sub-ingredient (e.g., gold ingot needs gold nuggets). Check recipes again.
   - If `craft` fails -> Check if you have the exact items.

**CORE COMMAND SET (API):**
* `craft [target] using [ingredients]` (e.g., "craft 1 stick using 2 bamboo")
* `get [item]` (e.g., "get 3 log")
* `inventory` (Check what you have)

**INTERACTION EXAMPLES:**

[Environment]
Goal: craft powered rail
Recipes:
- craft 6 powered rail using 1 redstone, 1 stick, 6 gold ingot
- craft 1 gold ingot using 9 gold nugget
- craft 1 stick using 2 bamboo

[You]
Think: Recipe for powered rail needs redstone, stick, gold ingot. I need to check what I have.
Action: [[ inventory ]]

[Environment]
Inventory: [redstone] (1)

[You]
Think: I have redstone. I miss stick and gold ingot. Gold ingot has a recipe (needs gold nugget). I should get gold nugget first.
Action: [[ get 54 gold nugget ]]

[Environment]
Got 54 gold nugget.

[You]
Think: Now I have nuggets. I can craft gold ingots.
Action: [[ craft 6 gold ingot using 54 gold nugget ]]

[Environment]
Crafted 6 gold ingot.

[You]
Think: Now I need a stick. Recipe says stick needs bamboo. I need to get bamboo.
Action: [[ get 2 bamboo ]]

[Environment]
Got 2 bamboo.

[You]
Think: I have bamboo. I can craft the stick.
Action: [[ craft 1 stick using 2 bamboo ]]

[Environment]
Crafted 1 stick.

[You]
Think: I have redstone, gold ingots, and stick. I can finally craft the powered rail.
Action: [[ craft 6 powered rail using 1 redstone, 1 stick, 6 gold ingot ]]

[Environment]
Crafted 6 powered rail.

[You]
Think: Goal achieved.
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

        for attempt in range(self.request_retries + 1):
            try:
                async with session.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        if attempt < self.request_retries and self._is_retryable_status(response.status):
                            delay = self._retry_delay(attempt)
                            logger.warning(
                                f"Transient HTTP {response.status} from vLLM on attempt "
                                f"{attempt + 1}/{self.request_retries + 1}; retrying in {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise GenerationRequestError(f"HTTP {response.status} from vLLM: {error_text}")
                    
                    result = await response.json()
                    if "choices" not in result or len(result["choices"]) == 0:
                        raise GenerationRequestError(f"Invalid response format: {result}")

                    content = result["choices"][0]["message"]["content"].strip()
                    if not content:
                        if attempt < self.request_retries:
                            delay = self._retry_delay(attempt)
                            logger.warning(
                                f"Empty completion from vLLM on attempt "
                                f"{attempt + 1}/{self.request_retries + 1}; retrying in {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise GenerationRequestError("vLLM returned an empty completion")
                    return content
            except (asyncio.TimeoutError, aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError, aiohttp.ClientOSError) as e:
                if attempt < self.request_retries:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        f"Transient vLLM request failure on attempt {attempt + 1}/"
                        f"{self.request_retries + 1}: {e}. Retrying in {delay:.1f}s"
                            )
                    await asyncio.sleep(delay)
                    continue
                if isinstance(e, asyncio.TimeoutError):
                    raise GenerationRequestError("Request timed out after 120s") from e
                raise GenerationRequestError(f"HTTP request failed after retries: {str(e)}") from e
            except GenerationRequestError:
                raise
            except Exception as e:
                raise GenerationRequestError(f"HTTP request failed: {str(e)}") from e

        raise GenerationRequestError("vLLM request exhausted retries without a valid completion")


# =============================================================================
# Evaluation Logic
# =============================================================================

async def evaluate_one_episode(
    agent: AsyncTextCraftAgent,
    interaction: TextCraftInteraction,
    session_id: int,
    http_session: aiohttp.ClientSession,
    max_rounds: int = 25,
    initial_prompt: Optional[str] = None,
    goal: Optional[str] = None,
    data_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluates a single episode asynchronously."""

    instance_id = f"eval_{session_id}_{uuid.uuid4().hex[:8]}"
    messages = []
    conversations = []
    done = False
    total_reward = 0.0
    try:
        # --- Initialization ---
        if initial_prompt is None:
            try:
                await interaction.start_interaction(instance_id, session_id=session_id, goal=goal, data_idx=data_idx)
                done, initial_obs, reward, extra = await interaction.generate_response(instance_id, messages)
                total_reward += reward
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
                    "goal": goal,
                    "data_idx": data_idx,
                }
        else:
            try:
                await interaction.start_interaction(instance_id, session_id=session_id, goal=goal, data_idx=data_idx)
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
                    "goal": goal,
                    "data_idx": data_idx,
                }

        # --- Interaction Loop ---
        for turn in range(max_rounds):
            if done:
                break

            try:
                response = await agent.generate(messages, http_session)
                response_lower = response.lower()
                if 'task completed' in response_lower or 'task failed' in response_lower:
                    done = True
                messages.append({"role": "assistant", "content": response})
                conversations.append({"role": "assistant", "content": response})
            except GenerationRequestError as e:
                logger.error(f"Generation error session {session_id} turn {turn}: {e}")
                return {
                    "session_id": session_id,
                    "reward": total_reward,
                    "success": total_reward > 0.0,
                    "num_turns": len([m for m in messages if m["role"] == "assistant"]),
                    "conversations": conversations,
                    "initial_prompt": initial_prompt,
                    "error": str(e),
                    "error_type": "generation",
                    "goal": goal,
                    "data_idx": data_idx,
                }
            except Exception as e:
                logger.error(f"Generation error session {session_id} turn {turn}: {e}")
                return {
                    "session_id": session_id,
                    "reward": total_reward,
                    "success": total_reward > 0.0,
                    "num_turns": len([m for m in messages if m["role"] == "assistant"]),
                    "conversations": conversations,
                    "initial_prompt": initial_prompt,
                    "error": str(e),
                    "error_type": "generation",
                    "goal": goal,
                    "data_idx": data_idx,
                }

            try:
                done, observation, step_reward, extra = await interaction.generate_response(instance_id, messages)
                total_reward += step_reward
                messages.append({"role": "user", "content": observation})
                conversations.append({"role": "user", "content": observation})
            except Exception as e:
                logger.error(f"Environment error session {session_id} turn {turn}: {e}")
                return {
                    "session_id": session_id,
                    "reward": total_reward,
                    "success": total_reward > 0.0,
                    "num_turns": len([m for m in messages if m["role"] == "assistant"]),
                    "conversations": conversations,
                    "initial_prompt": initial_prompt,
                    "error": str(e),
                    "error_type": "environment",
                    "goal": goal,
                    "data_idx": data_idx,
                }

        success = total_reward > 0.0
        num_turns = len([m for m in messages if m["role"] == "assistant"])

        return {
            "session_id": session_id,
            "reward": total_reward,
            "success": success,
            "num_turns": num_turns,
            "conversations": conversations,
            "initial_prompt": initial_prompt,
            "goal": goal,
            "data_idx": data_idx,
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
    with open(output_file, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + '\n')
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# =============================================================================
# Main Evaluation Loop
# =============================================================================

async def run_evaluation(args: argparse.Namespace):
    """Main evaluation function with async HTTP client"""
    
    # Initialize components
    interaction = TextCraftInteraction({
        'env_server_base': args.textcraft_server,
        'timeout': 600,
        'max_retries': 3
    })
    
    agent = AsyncTextCraftAgent(
        server_url=args.vllm_server_url,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        request_retries=args.request_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        max_context_tokens=args.max_context_tokens,
        context_safety_margin=args.context_safety_margin,
        preserve_recent_messages=args.preserve_recent_messages,
    )
    
    # Load dataset
    logger.info(f"Loading dataset from: {args.data_path}")
    table = pq.read_table(args.data_path)
    df = table.to_pandas()
    bindings = [extract_textcraft_binding(row) for _, row in df.iterrows()]
    df['item_id'] = [binding['item_id'] for binding in bindings]
    df['original_index'] = [binding['session_id'] for binding in bindings]
    df['task_data_idx'] = [binding['data_idx'] for binding in bindings]
    df['task_goal'] = [binding['goal'] for binding in bindings]
    logger.info(
        "Resolved stable TextCraft task ids from parquet: "
        f"{sorted(df['original_index'].tolist())[:5]} ... {sorted(df['original_index'].tolist())[-5:]}"
    )
    
    # Shuffle and sample
    if args.max_samples > 0 and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
        logger.info(f"Using {args.max_samples} random samples")
    else:
        logger.info(f"Using full dataset ({len(df)} samples)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.jsonl")
    summary_file = os.path.join(args.output_dir, f"eval_summary_{timestamp}.txt")
    
    logger.info(f"Output file: {output_file}")
    logger.info(f"Summary file: {summary_file}")
    
    # Clear output file
    open(output_file, 'w').close()
    
    # Setup HTTP client
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)
    
    # Semaphore for concurrency control
    sem = asyncio.Semaphore(args.concurrency)
    
    async def worker(session_id: int, row: Dict, sample_idx: int, goal: Optional[str] = None, data_idx: Optional[int] = None):
        async with sem:
            try:
                result = await evaluate_one_episode(
                    agent=agent,
                    interaction=interaction,
                    session_id=session_id,
                    http_session=session,
                    max_rounds=args.max_rounds,
                    goal=goal,
                    data_idx=data_idx,
                )

                # Prepare record (包括成功和失败的 episode)
                record = {
                    "item_id": f"textcraft_{session_id}",
                    "session_id": session_id,
                    "sample_idx": sample_idx,
                    "reward": result['reward'],
                    "success": result['success'],
                    "num_turns": result['num_turns'],
                    "goal": result.get('goal', goal),
                    "data_idx": result.get('data_idx', data_idx),
                }
                if not args.no_save_trajectories:
                    record["conversations"] = result['conversations']
                    record["initial_prompt"] = result['initial_prompt']
                if 'error' in result:
                    record["error"] = result['error']
                    record["error_type"] = result.get('error_type', 'unknown')

                # Write to file
                await asyncio.to_thread(safe_write_record, output_file, record)
                return result
            except Exception as e:
                # 不再静默返回 None，将失败写入结果文件
                logger.error(f"[CRITICAL] Worker exception for session {session_id}, sample {sample_idx}: {str(e)}")
                error_record = {
                    "item_id": f"textcraft_{session_id}",
                    "session_id": session_id,
                    "sample_idx": sample_idx,
                    "reward": 0.0,
                    "success": False,
                    "num_turns": 0,
                    "goal": goal,
                    "data_idx": data_idx,
                    "error": str(e),
                    "error_type": "worker_exception",
                }
                await asyncio.to_thread(safe_write_record, output_file, error_record)
                return error_record
    
    # Run evaluation
    results = []
    total_tasks = len(df) * args.num_samples_per_task
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for _, row in df.iterrows():
            real_session_id = int(row['original_index'])
            goal = row.get('task_goal')
            data_idx = _safe_int(row.get('task_data_idx'))
            if data_idx is None:
                data_idx = real_session_id
            for sample_idx in range(args.num_samples_per_task):
                tasks.append(worker(real_session_id, row, sample_idx, goal=goal, data_idx=data_idx))
        
        # Progress bar
        pbar = tqdm(total=total_tasks, desc="Evaluating", unit="sample")
        for f in asyncio.as_completed(tasks):
            res = await f
            if res:
                results.append(res)
            pbar.update(1)
        pbar.close()
    
    # Generate summary (pass df row count for expected task calculation)
    args._df_rows = len(df)
    args._history_trim_stats = agent.get_trim_stats()
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
    # Load all results
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

    # Fetch model name from vLLM service
    model_name = await fetch_model_name(args.vllm_server_url)
    logger.info(f"Model name from vLLM: {model_name}")

    # Group by session_id
    grouped_results = defaultdict(list)
    for res in results:
        grouped_results[res['session_id']].append(res)

    # Categorize results
    expected_total_tasks = args.num_samples_per_task * args._df_rows if hasattr(args, '_df_rows') else None
    # Fallback: compute expected from data_path
    if expected_total_tasks is None:
        try:
            table = pq.read_table(args.data_path)
            df_summary = table.to_pandas()
            if args.max_samples > 0 and args.max_samples < len(df_summary):
                expected_total_tasks = args.num_samples_per_task * args.max_samples
            else:
                expected_total_tasks = args.num_samples_per_task * len(df_summary)
        except:
            expected_total_tasks = len(grouped_results)  # fallback to actual

    total_expected_samples = expected_total_tasks  # = num_tasks * num_samples_per_task
    total_finished_samples = len(results)
    total_missing_samples = total_expected_samples - total_finished_samples
    trim_stats = getattr(args, "_history_trim_stats", {}) or {}

    # Categorize by error type
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

    # Categorize by goal_mismatch vs normal (in session-level)
    sessions_with_goal_mismatch = set(r['session_id'] for r in goal_mismatch_samples)
    sessions_with_error = set(r['session_id'] for r in error_samples)
    sessions_normal = set(grouped_results.keys()) - sessions_with_goal_mismatch - sessions_with_error

    # Pass@k calculation using only normal (non-error) samples
    grouped_normal = defaultdict(list)
    for res in normal_samples:
        grouped_normal[res['session_id']].append(res)

    # Define k values for pass@k
    k_values = [1, 2, 4, 8]

    # Calculate metrics on normal samples
    sum_avg_reward = 0.0
    sum_avg_success = 0.0
    pass_at_k_values = {k: [] for k in k_values}

    for session_id, task_runs in grouped_normal.items():
        n = len(task_runs)
        c = sum(1 for r in task_runs if r.get('success', False))
        total_r = sum(r.get('reward', 0.0) for r in task_runs)

        if n > 0:
            avg_reward = total_r / n
            avg_success = c / n
            sum_avg_reward += avg_reward
            sum_avg_success += avg_success

        for k in k_values:
            if n >= k:
                pass_k = estimate_pass_at_k(n, c, k)
                pass_at_k_values[k].append(pass_k)

    num_normal_tasks = len(grouped_normal)
    global_avg_reward = sum_avg_reward / num_normal_tasks if num_normal_tasks > 0 else 0.0
    global_avg_success = sum_avg_success / num_normal_tasks if num_normal_tasks > 0 else 0.0

    # Prepare summary text
    metrics = [
        "=" * 60,
        "TextCraft Evaluation Summary (vLLM Service Mode)",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"Model: {model_name}",
        f"Server: {args.vllm_server_url}",
        f"Dataset: {args.data_path}",
        "-" * 60,
        "TASK STATISTICS:",
        f"  Expected Samples:    {total_expected_samples}",
        f"  Finished Samples:    {total_finished_samples}",
        f"  Missing Samples:     {total_missing_samples} {'⚠️ (check logs!)' if total_missing_samples > 0 else '(ok)'}",
        f"  Expected Tasks:      {expected_total_tasks // args.num_samples_per_task}",
        f"  Finished Tasks:      {len(grouped_results)}",
        "-" * 60,
        "SAMPLE BREAKDOWN:",
        f"  Normal samples:      {len(normal_samples)}",
        f"  Error samples:       {len(error_samples)} {'⚠️' if error_samples else '(ok)'}",
        f"    - start_interaction errors: {sum(1 for r in error_samples if r.get('error_type') == 'start_interaction')}",
        f"    - generation errors:         {sum(1 for r in error_samples if r.get('error_type') == 'generation')}",
        f"    - environment errors:        {sum(1 for r in error_samples if r.get('error_type') == 'environment')}",
        f"    - worker exceptions:         {sum(1 for r in error_samples if r.get('error_type') == 'worker_exception')}",
        f"  Goal mismatch:       {len(goal_mismatch_samples)} {'⚠️ (goal in parquet != server assigned)' if goal_mismatch_samples else '(ok)'}",
        f"  History trimmed:     {trim_stats.get('trim_event_count', 0)} request(s)",
        f"    - removed messages total:    {trim_stats.get('trimmed_messages_total', 0)}",
        f"    - truncated messages total:  {trim_stats.get('truncated_messages_total', 0)}",
        f"    - max removed in one req:    {trim_stats.get('max_removed_messages', 0)}",
        f"    - max truncated in one req:  {trim_stats.get('max_truncated_messages', 0)}",
        f"    - max estimated input toks:  {trim_stats.get('max_estimated_input_tokens', 0)}",
        "-" * 60,
        "METRICS (based on normal samples only):",
        f"  Normal Tasks:        {num_normal_tasks}",
        f"  Average Reward:      {global_avg_reward:.4f}",
        f"  Average Success (Avg@1): {global_avg_success:.4f}",
        "-" * 60,
    ]

    for k in sorted(k_values):
        values = pass_at_k_values[k]
        if values:
            avg_pass_k = sum(values) / len(values)
            metrics.append(f"Pass@{k:<2}: {avg_pass_k:.4f} (tasks: {len(values)}/{num_normal_tasks})")
        else:
            metrics.append(f"Pass@{k:<2}: N/A    (insufficient samples)")

    # Depth breakdown using official aligned sparse task ids
    depth_grouped_results = defaultdict(list)
    depth_grouped_normal = defaultdict(lambda: defaultdict(list))
    for res in results:
        depth = session_id_to_textcraft_depth(int(res["session_id"]))
        if depth is not None:
            depth_grouped_results[depth].append(res)
    for res in normal_samples:
        depth = session_id_to_textcraft_depth(int(res["session_id"]))
        if depth is not None:
            depth_grouped_normal[depth][res["session_id"]].append(res)

    if depth_grouped_results:
        metrics.extend([
            "-" * 60,
            "DEPTH BREAKDOWN (official aligned sparse ids):",
        ])
        for depth in sorted(depth_grouped_results):
            grouped_depth_results = defaultdict(list)
            for res in depth_grouped_results[depth]:
                grouped_depth_results[res["session_id"]].append(res)

            grouped_depth_normal = depth_grouped_normal.get(depth, {})
            expected_depth_tasks = len(OFFICIAL_TEXTCRAFT_DEPTH_BANDS[depth])
            expected_depth_samples = expected_depth_tasks * args.num_samples_per_task
            finished_depth_samples = len(depth_grouped_results[depth])
            num_depth_normal_tasks = len(grouped_depth_normal)

            depth_avg_reward_sum = 0.0
            depth_avg_success_sum = 0.0
            depth_pass_at_k_values = {k: [] for k in k_values}

            for task_runs in grouped_depth_normal.values():
                n = len(task_runs)
                c = sum(1 for r in task_runs if r.get("success", False))
                total_r = sum(r.get("reward", 0.0) for r in task_runs)
                depth_avg_reward_sum += total_r / n
                depth_avg_success_sum += c / n
                for k in k_values:
                    if n >= k:
                        depth_pass_at_k_values[k].append(estimate_pass_at_k(n, c, k))

            depth_avg_reward = depth_avg_reward_sum / num_depth_normal_tasks if num_depth_normal_tasks > 0 else 0.0
            depth_avg_success = depth_avg_success_sum / num_depth_normal_tasks if num_depth_normal_tasks > 0 else 0.0

            metrics.extend([
                f"  Depth {depth}:",
                f"    Tasks:               {len(grouped_depth_results)}/{expected_depth_tasks}",
                f"    Samples:             {finished_depth_samples}/{expected_depth_samples}",
                f"    Average Reward:      {depth_avg_reward:.4f}",
                f"    Average Success:     {depth_avg_success:.4f}",
            ])
            for k in sorted(k_values):
                values = depth_pass_at_k_values[k]
                if values:
                    avg_pass_k = sum(values) / len(values)
                    metrics.append(
                        f"    Pass@{k:<2}:            {avg_pass_k:.4f} "
                        f"(tasks: {len(values)}/{num_depth_normal_tasks})"
                    )
                else:
                    metrics.append(f"    Pass@{k:<2}:            N/A")

    metrics.append("=" * 60)

    summary_text = "\n".join(metrics)
    print("\n" + summary_text + "\n")

    # Write to summary file
    try:
        with open(summary_file, 'w') as f:
            f.write(summary_text)
    except Exception as e:
        logger.error(f"Failed to write summary file: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TextCraft Evaluation Client (vLLM Service Mode)')
    
    # Paths
    parser.add_argument('--data_path', type=str, 
                        default='/Data/wyh/datasets/Verl-Data/eval/textcraft/test.parquet',
                        help='Path to test dataset (parquet format)')
    parser.add_argument('--output_dir', type=str,
                        default='/Data/wyh/datasets/Verl-Data/outputs/textcraft_eval',
                        help='Directory to save evaluation results')
    
    # Environment
    parser.add_argument('--textcraft_server', type=str,
                        default='http://127.0.0.1:3222',
                        help='TextCraft environment server URL')
    parser.add_argument('--model_name', type=str, default='qwen3',
                        help='Model name as registered in vLLM')
    parser.add_argument('--vllm_server_url', type=str,
                        default='http://localhost:8000',
                        help='vLLM service URL (e.g., http://localhost:8000)')
    
    # Evaluation Config
    parser.add_argument('--max_rounds', type=int, default=30,
                        help='Maximum interaction rounds per episode')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Maximum samples to evaluate (-1 for all)')
    parser.add_argument('--num_samples_per_task', type=int, default=8,
                        help='Number of samples per task (for pass@k)')
    parser.add_argument('--concurrency', type=int, default=8,
                        help='Maximum concurrent requests to vLLM service')
    
    # Generation Params
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum tokens to generate per response')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0.0 for greedy)')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='Top-p sampling parameter')
    parser.add_argument('--request_retries', type=int, default=2,
                        help='Number of retries for transient vLLM request failures')
    parser.add_argument('--retry_backoff_seconds', type=float, default=1.0,
                        help='Base backoff in seconds for vLLM request retries')
    parser.add_argument('--max_context_tokens', type=int, default=32768,
                        help='Maximum total context tokens accepted by the model')
    parser.add_argument('--context_safety_margin', type=int, default=256,
                        help='Reserved token margin for chat formatting and estimation error')
    parser.add_argument('--preserve_recent_messages', type=int, default=6,
                        help='How many most recent chat messages to keep when trimming history')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_save_trajectories', action='store_true',
                        help='Do not save conversation trajectories (saves disk space)')
    
    args = parser.parse_args()
    
    # Validate vLLM server URL
    if not args.vllm_server_url.startswith(('http://', 'https://')):
        args.vllm_server_url = f"http://{args.vllm_server_url}"
    
    # Safety check for temperature
    if args.num_samples_per_task > 1 and args.temperature == 0.0:
        logger.warning("⚠️ WARNING: num_samples_per_task > 1 but temperature=0.0. "
                      "Results will be identical for all samples. Consider increasing temperature.")
    
    logger.info("📋 Evaluation Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Run evaluation
    try:
        asyncio.run(run_evaluation(args))
    except KeyboardInterrupt:
        logger.info("🛑 Evaluation interrupted by user")
    except Exception as e:
        logger.exception(f"❌ Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
