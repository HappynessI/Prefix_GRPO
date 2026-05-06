# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

import requests

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))  # 改为INFO以便调试


class WebshopInteraction(BaseInteraction):
    """Webshop environment interaction for online RL training.
    
    This class handles interaction with the Webshop environment via HTTP API.
    The Webshop environment server should be running at env_server_base.
    
    Interaction flow:
    1. start_interaction: Create and reset environment to a specific task
    2. generate_response: Agent takes action, environment returns observation
    3. calculate_score: Get final reward
    4. finalize_interaction: Clean up environment resources
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.env_server_base = config.get("env_server_base", "http://127.0.0.1:36003")
        self.timeout = config.get("timeout", 600)
        self.max_retries = config.get("max_retries", 3)
        self._envs = {}  # instance_id -> {"env_idx": int, "total_reward": float, "step_count": int}
        
        logger.info(f"WebshopInteraction initialized with server: {self.env_server_base}")
        
        # Test server connectivity
        try:
            response = requests.get(f"{self.env_server_base}/", timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully connected to Webshop server at {self.env_server_base}")
            else:
                logger.warning(f"Webshop server responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Webshop server: {e}")
            raise ConnectionError(f"Cannot connect to Webshop server at {self.env_server_base}. "
                                f"Please ensure the server is running.")

    async def start_interaction(
        self, 
        instance_id: Optional[str] = None, 
        session_id: Optional[int] = None,
        **kwargs
    ) -> str:
        """Create a new Webshop environment and reset to a specific task.
        
        Args:
            instance_id: Unique identifier for this interaction instance
            session_id: Webshop task ID to reset to
            **kwargs: Additional arguments
        
        Returns:
            instance_id: The instance ID for this interaction
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        if session_id is None:
            session_id = kwargs.get("session_id")
        if session_id is None:
            session_id = kwargs.get("task_id")
        if session_id is None:
            raise ValueError("WebShop interaction requires `session_id` in interaction_kwargs.")
        session_id = int(session_id)
        
        logger.info(f"Starting Webshop interaction {instance_id} with session_id={session_id}")
        
        try:
            # 1. Create environment instance
            response = requests.post(
                f"{self.env_server_base}/create",
                timeout=self.timeout
            )
            response.raise_for_status()
            env_idx = response.json()
            logger.debug(f"Created environment with env_idx={env_idx}")
            
            # 2. Reset to specific task
            reset_data = {"env_idx": env_idx, "session_id": session_id}
            logger.debug(f"Reset request data: {reset_data}")
            response = requests.post(
                f"{self.env_server_base}/reset",
                json=reset_data,
                timeout=self.timeout
            )
            if response.status_code != 200:
                logger.error(f"Reset failed with status {response.status_code}: {response.text}")
            response.raise_for_status()
            
            # 3. Get initial observation
            response = requests.get(
                f"{self.env_server_base}/observation",
                params={"env_idx": env_idx},
                timeout=self.timeout
            )
            response.raise_for_status()
            initial_observation = response.json()
            
            # 4. Store environment state
            self._envs[instance_id] = {
                "env_idx": env_idx,
                "total_reward": 0.0,
                "step_count": 0,
                "session_id": session_id,
                "initial_observation": initial_observation
            }
            
            logger.info(f"Successfully started interaction {instance_id} with env_idx={env_idx}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to start interaction {instance_id}: {e}")
            raise

    async def generate_response(
        self, 
        instance_id: str, 
        messages: list[dict[str, Any]], 
        **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Execute agent's action in the environment and get observation.
        
        Args:
            instance_id: The instance ID for this interaction
            messages: Conversation history including agent's latest action
            **kwargs: Additional arguments
        
        Returns:
            should_terminate_sequence: Whether the episode is done
            response_content: Environment's observation (next state)
            current_turn_score: Reward for this step
            additional_data: Extra information
        """
        if instance_id not in self._envs:
            raise ValueError(f"Instance {instance_id} not found. Call start_interaction first.")
        
        env_state = self._envs[instance_id]
        env_idx = env_state["env_idx"]
        step_count = env_state["step_count"]
        
        # On first call, return initial observation without expecting action
        if step_count == 0:
            initial_obs = env_state["initial_observation"]
            # logger.info(f"[WEBSHOP] Instance {instance_id} Turn 1: returning initial observation (reward=0.0): {initial_obs[:200]}...")
            env_state["step_count"] += 1
            return False, initial_obs, 0.0, {"step_count": 1, "is_initial": True}
        
        # Extract agent's last action from messages
        action = self._extract_action(messages)
        
        if action is None:
            # logger.warning(f"[WEBSHOP] Instance {instance_id} Turn {step_count+1}: NO ACTION FOUND in messages!")
            # logger.warning(f"[WEBSHOP] Last 3 messages: {messages[-3:] if len(messages) >= 3 else messages}")
            return False, "Please provide a valid action (search[...] or click[...]).", 0.0, {}
        
        # logger.info(f"[WEBSHOP] Instance {instance_id} Turn {step_count+1}: executing action: {action}")
        
        try:
            # Execute action in environment
            response = requests.post(
                f"{self.env_server_base}/step",
                json={"env_idx": env_idx, "action": action},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            observation = result.get("state", "")
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            info = result.get("info", {})
            
            # Update environment state
            env_state["total_reward"] += reward
            env_state["step_count"] += 1
            
            # logger.info(f"[WEBSHOP] Instance {instance_id} Turn {env_state['step_count']}: "
            #            f"action={action}, reward={reward:.3f}, done={done}, cumulative={env_state['total_reward']:.3f}")
            # logger.info(f"[WEBSHOP] Observation: {observation[:300]}...")
            
            additional_data = {
                "step_count": env_state["step_count"],
                "cumulative_reward": env_state["total_reward"],
                "info": info
            }
            
            return done, observation, reward, additional_data
            
        except Exception as e:
            logger.error(f"Failed to execute action for instance {instance_id}: {e}")
            # Return error message as observation with zero reward
            return False, f"Error executing action: {str(e)}", 0.0, {}

    def _extract_action(self, messages: list[dict[str, Any]]) -> Optional[str]:
        """Extract the last action from conversation messages.
        
        参考AgentGym-RL的parse_react逻辑：
        1. 优先从"Action:"后提取（标准格式）
        2. 容错机制：如果没有"Action:"，但文本中有search[或click[，仍然提取
        3. 使用rsplit和findall取最后一个action（避免提取历史action）
        
        Args:
            messages: Conversation history
        
        Returns:
            action: Extracted action string or None
        """
        # Find the last assistant message
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                
                # Method 1: 标准格式 - 从"Action:"后提取（使用rsplit取最后一个）
                split_result = content.rsplit("Action:", 1)
                if len(split_result) == 2:
                    action_text = split_result[1].strip()
                    # 提取第一个search[...]或click[...]
                    search_match = re.search(r'search\[([^\]]*)\]', action_text, re.IGNORECASE)
                    if search_match:
                        return f"search[{search_match.group(1)}]"
                    click_match = re.search(r'click\[([^\]]*)\]', action_text, re.IGNORECASE)
                    if click_match:
                        return f"click[{click_match.group(1)}]"
                
                # Method 2: 容错机制（参考AgentGym-RL的parse_react）
                # 即使格式不对，只要文本中有search[或click[就提取（取最后一个）
                if "search[" in content or "click[" in content:
                    # 使用findall找所有匹配，取最后一个（最新的action）
                    search_matches = re.findall(r'search\[([^\]]*)\]', content, re.IGNORECASE)
                    if search_matches:
                        # logger.debug(f"Action format incorrect, using fallback extraction (search)")
                        return f"search[{search_matches[-1]}]"
                    
                    click_matches = re.findall(r'click\[([^\]]*)\]', content, re.IGNORECASE)
                    if click_matches:
                        # logger.debug(f"Action format incorrect, using fallback extraction (click)")
                        return f"click[{click_matches[-1]}]"
                
                # logger.debug(f"No valid action pattern found in: {content[:100]}...")
                return None
        
        return None

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate the final score for this interaction.
        
        Args:
            instance_id: The instance ID for this interaction
        
        Returns:
            score: The cumulative reward
        """
        if instance_id not in self._envs:
            logger.warning(f"Instance {instance_id} not found when calculating score")
            return 0.0
        
        total_reward = self._envs[instance_id]["total_reward"]
        logger.info(f"Instance {instance_id} final score: {total_reward:.3f}")
        return total_reward

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up environment resources.
        
        Args:
            instance_id: The instance ID for this interaction
        """
        if instance_id not in self._envs:
            logger.warning(f"Instance {instance_id} not found for finalization")
            return
        
        env_state = self._envs[instance_id]
        env_idx = env_state["env_idx"]
        
        try:
            # Close environment
            response = requests.post(
                f"{self.env_server_base}/close",
                json={"env_idx": env_idx},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            logger.info(f"Successfully closed environment {env_idx} for instance {instance_id}")
            
        except Exception as e:
            logger.error(f"Failed to close environment {env_idx}: {e}")
        
        finally:
            # Remove from tracking
            del self._envs[instance_id]
