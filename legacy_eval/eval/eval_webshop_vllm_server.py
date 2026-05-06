#!/usr/bin/env python3
"""
Webshop Evaluation Client - vLLM Service Mode
网页购物环境评估脚本
"""

import os
import sys
import json
import logging
import argparse
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pyarrow.parquet as pq
from tqdm import tqdm
import aiohttp
import fcntl

# Add verl to path
project_root = Path(__file__).parent.parent.parent.parent.parent
if not (project_root / "verl").exists():
    raise RuntimeError(f"verl not found in {project_root}")
sys.path.insert(0, str(project_root))

from verl.interactions.webshop_interaction import WebshopInteraction
from agentgym_eval_utils import resolve_eval_row

# 日志配置
log_dir = Path("/Data/wyh/datasets/Verl-Data/outputs/webshop_eval/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"eval_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("WebshopEval")


class AsyncWebshopAgent:
    """Webshop Agent - Uses HTTP calls to vLLM service"""
    
    def __init__(self, server_url: str, model_name: str = "qwen3", max_new_tokens: int = 512, temperature: float = 1.0, top_p: float = 1.0):
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = self._build_prompt()
    
    def _build_prompt(self) -> str:
        return '''You are a Webshop Agent. Your goal is to find and purchase products that match the given requirements.

**CORE PROTOCOL:**
1. **THINK FIRST**: Analyze the product requirements and current page.
2. **ONE ACTION**: Output exactly ONE action per turn.
3. **FORMAT**: Use `search[keywords]` or `click[element]` format.

**CORE COMMAND SET:**
* `search[keywords]` - Search for products with keywords
* `click[Back to Search]` - Return to search page
* `click[Next >]` - Go to next page
* `click[< Prev]` - Go to previous page
* `click[product title]` - Click on a product to view details
* `click[option]` - Select a product option (size, color, etc.)
* `click[Buy Now]` - Purchase the product

**EXAMPLE:**
[Environment]
Instruction: Find a red cotton t-shirt under $30.

[You]
Think: I need to search for red cotton t-shirts.
Action: search[red cotton t-shirt]

[Environment]
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B09XYZ123] Red Cotton T-Shirt - $25.99
[B09ABC456] Blue Cotton T-Shirt - $22.99

[You]
Think: The first product matches - red cotton t-shirt for $25.99 which is under $30.
Action: click[B09XYZ123]

[Environment]
[Back to Search]
Red Cotton T-Shirt
Price: $25.99
Size: [S] [M] [L] [XL]
[Buy Now]

[You]
Think: I need to select a size before buying.
Action: click[M]

[Environment]
You have selected: Size M
[Buy Now]

[You]
Think: Size selected. Now I can purchase.
Action: click[Buy Now]
'''
    
    async def generate(self, messages: List[Dict[str, str]], session: aiohttp.ClientSession) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": self.system_prompt}, *messages],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False
        }
        
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
    agent: AsyncWebshopAgent,
    interaction: WebshopInteraction,
    session_id: int,
    interaction_kwargs: Dict[str, Any],
    http_session: aiohttp.ClientSession,
    max_rounds: int = 15
) -> Dict[str, Any]:
    instance_id = f"eval_{session_id}_{uuid.uuid4().hex[:8]}"
    messages, conversations = [], []
    done, total_reward = False, 0.0
    initial_prompt = None

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
                "error": str(e),
                "error_type": "start_interaction",
            }

        for turn in range(max_rounds):
            if done:
                break
            try:
                response = await agent.generate(messages, http_session)
                if 'task completed' in response.lower():
                    done = True
                messages.append({"role": "assistant", "content": response})
                conversations.append({"role": "assistant", "content": response})
            except Exception as e:
                logger.error(f"Generation error session {session_id} turn {turn}: {e}")
                return {
                    "session_id": session_id,
                    "reward": total_reward,
                    "success": total_reward > 0.5,
                    "num_turns": len([m for m in messages if m["role"] == "assistant"]),
                    "conversations": conversations,
                    "initial_prompt": initial_prompt,
                    "error": str(e),
                    "error_type": "generation",
                }

            try:
                done, observation, step_reward, _ = await interaction.generate_response(instance_id, messages)
                total_reward += step_reward
                messages.append({"role": "user", "content": observation})
                conversations.append({"role": "user", "content": observation})
            except Exception as e:
                logger.error(f"Environment error session {session_id} turn {turn}: {e}")
                return {
                    "session_id": session_id,
                    "reward": total_reward,
                    "success": total_reward > 0.5,
                    "num_turns": len([m for m in messages if m["role"] == "assistant"]),
                    "conversations": conversations,
                    "initial_prompt": initial_prompt,
                    "error": str(e),
                    "error_type": "environment",
                }
        
        return {
            "session_id": session_id,
            "reward": total_reward,
            "success": total_reward > 0.5,
            "num_turns": len([m for m in messages if m["role"] == "assistant"]),
            "conversations": conversations,
            "initial_prompt": initial_prompt,
        }
    finally:
        if instance_id in interaction.instance_sessions:
            try:
                await interaction.finalize_interaction(instance_id)
            except Exception as e:
                logger.warning(f"Finalization failed for session {session_id}: {e}")


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
    interaction = WebshopInteraction({
        'env_server_base': args.env_server,
        'timeout': 600, 'max_retries': 3
    })
    agent = AsyncWebshopAgent(args.vllm_server_url, args.model_name, args.max_new_tokens, args.temperature, args.top_p)
    
    logger.info(f"Loading dataset: {args.data_path}")
    df = pq.read_table(args.data_path).to_pandas()
    
    if 0 < args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.jsonl")
    summary_file = os.path.join(args.output_dir, f"eval_summary_{timestamp}.txt")
    open(output_file, 'w').close()
    
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    
    async def worker(item_id, session_id, interaction_kwargs, sample_idx):
        async with sem:
            try:
                result = await evaluate_one_episode(
                    agent,
                    interaction,
                    session_id,
                    interaction_kwargs,
                    session,
                    args.max_rounds,
                )
                record = {
                    "item_id": item_id,
                    "session_id": session_id,
                    "sample_idx": sample_idx,
                    "reward": result['reward'],
                    "success": result['success'],
                    "num_turns": result['num_turns'],
                }
                if not args.no_save_trajectories:
                    record["conversations"] = result.get('conversations', [])
                    record["initial_prompt"] = result.get('initial_prompt')
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
                await asyncio.to_thread(safe_write_record, output_file, error_record)
                return error_record
    
    results = []
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for _, row in df.iterrows():
            item_id, session_id, interaction_kwargs = resolve_eval_row(row, "webshop")
            for sample_idx in range(args.num_samples_per_task):
                tasks.append(worker(item_id, session_id, interaction_kwargs, sample_idx))
        pbar = tqdm(total=len(tasks), desc="Evaluating")
        for f in asyncio.as_completed(tasks):
            res = await f
            if res: results.append(res)
            pbar.update(1)
        pbar.close()

    args._df_rows = len(df)
    await generate_summary(output_file, summary_file, args)
    logger.info(f"Results saved to {output_file}")


async def generate_summary(results_file: str, summary_file: str, args):
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

    grouped = defaultdict(list)
    for r in results:
        grouped[r['session_id']].append(r)

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
            expected_total_tasks = len(grouped)

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
    sum_avg_reward = 0.0
    sum_avg_success = 0.0
    pass_at_k_values = {k: [] for k in k_values}

    for session_id, task_runs in grouped.items():
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

    num_metric_tasks = len(grouped)
    num_normal_tasks = len({res['session_id'] for res in normal_samples})
    global_avg_reward = sum_avg_reward / num_metric_tasks if num_metric_tasks else 0.0
    global_avg_success = sum_avg_success / num_metric_tasks if num_metric_tasks else 0.0

    metrics = [
        "=" * 60,
        "Webshop Evaluation Summary",
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
        f"  Finished Tasks:      {len(grouped)}",
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
        "METRICS (all finished samples; errors count as failures):",
        f"  Tasks Evaluated:     {num_metric_tasks}",
        f"  Tasks With Normal Samples: {num_normal_tasks}",
        f"  Average Reward:      {global_avg_reward:.4f}",
        f"  Average Success (Avg@1): {global_avg_success:.4f}",
        "-" * 60,
    ]

    for k in sorted(k_values):
        values = pass_at_k_values[k]
        if values:
            avg_pass_k = sum(values) / len(values)
            metrics.append(f"Pass@{k:<2}: {avg_pass_k:.4f} (tasks: {len(values)}/{num_metric_tasks})")
        else:
            metrics.append(f"Pass@{k:<2}: N/A    (insufficient samples)")

    metrics.append("=" * 60)

    summary_text = "\n".join(metrics)
    print("\n" + summary_text + "\n")

    try:
        with open(summary_file, 'w') as f:
            f.write(summary_text)
    except Exception as e:
        logger.error(f"Failed to write summary file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Webshop Evaluation')
    parser.add_argument('--data_path', default='/Data/wyh/datasets/Verl-Data/eval/webshop/test.parquet')
    parser.add_argument('--output_dir', default='/Data/wyh/datasets/Verl-Data/outputs/webshop_eval')
    parser.add_argument('--env_server', default='http://127.0.0.1:36001')
    parser.add_argument('--vllm_server_url', default='http://localhost:8000')
    parser.add_argument('--model_name', default='qwen3')
    parser.add_argument('--max_rounds', type=int, default=15)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--num_samples_per_task', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_save_trajectories', action='store_true',
                        help='Do not save conversation trajectories')
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
