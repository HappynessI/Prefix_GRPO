# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import numpy as np

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score


@register("naive")
class NaiveRewardLoopManager(RewardLoopManagerBase):
    """The reward manager."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        # print(f"[DEBUG] tool_extra_fields type: {type(tool_extra_fields)}, content: {tool_extra_fields}")  # 注释: 冗余DEBUG日志
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores
        
        # For environment interaction tasks (e.g., webshop), extract and normalize turn_scores
        # turn_scores from tool_agent_loop is stored in tool_extra_fields
        # Check both locations: non_tensor_batch directly and tool_extra_fields
        if "turn_scores" in data_item.non_tensor_batch:
            turn_scores_raw = data_item.non_tensor_batch["turn_scores"]
            # print(f"[DEBUG] Found turn_scores in non_tensor_batch: {turn_scores_raw}")  # 注释: 冗余DEBUG日志
        elif "turn_scores" in extra_info:
            turn_scores_raw = extra_info["turn_scores"]
            # print(f"[DEBUG] Found turn_scores in extra_info (from tool_extra_fields): {turn_scores_raw}")  # 注释: 冗余DEBUG日志
        else:
            turn_scores_raw = None
            # print(f"[DEBUG] No turn_scores found anywhere")  # 注释: 冗余DEBUG日志
        
        if turn_scores_raw is not None:
            # Normalize: extract the list from numpy object array and convert to Python list
            if isinstance(turn_scores_raw, np.ndarray) and turn_scores_raw.size > 0:
                turn_scores = turn_scores_raw[0] if turn_scores_raw.ndim > 0 else turn_scores_raw
                # print(f"[DEBUG] Extracted from numpy: {turn_scores}, type: {type(turn_scores)}")  # 注释: 冗余DEBUG日志
            else:
                turn_scores = turn_scores_raw
                # print(f"[DEBUG] Direct use: {turn_scores}, type: {type(turn_scores)}")  # 注释: 冗余DEBUG日志
                
            if isinstance(turn_scores, (list, np.ndarray)):
                extra_info["turn_scores"] = [float(s) for s in turn_scores]
                # print(f"[DEBUG] Final turn_scores (list): {extra_info['turn_scores']}")  # 注释: 冗余DEBUG日志
            elif isinstance(turn_scores, (int, float)):
                extra_info["turn_scores"] = [float(turn_scores)]
                # print(f"[DEBUG] Final turn_scores (scalar): {extra_info['turn_scores']}")  # 注释: 冗余DEBUG日志
            else:
                extra_info["turn_scores"] = []
                # print(f"[DEBUG] Cannot parse turn_scores, set to empty: {turn_scores}")  # 注释: 冗余DEBUG日志

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
