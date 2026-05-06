"""
AgentGym环境的通用Base Interaction类
所有AgentGym环境（webshop, babyai, alfworld等）都继承此类
"""

import re
import asyncio
import logging
import requests
import random
from typing import Dict, List, Tuple, Any, Optional
from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)


class AgentGymBaseInteraction(BaseInteraction):
    """
    AgentGym环境的通用基类
    封装了HTTP请求、action提取、错误处理等通用逻辑
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典，必须包含:
                - env_server_base: 环境服务器地址 (e.g., 'http://127.0.0.1:8000')
                - timeout: HTTP请求超时时间（秒）
                - max_retries: 最大重试次数
        """
        self.env_server_base = config.get('env_server_base', 'http://127.0.0.1:8000')
        self.timeout = config.get('timeout', 600)
        self.max_retries = int(config.get('max_retries', 3))
        self.retry_backoff = float(config.get('retry_backoff', 0.5))
        self.retry_status_codes = {
            int(code)
            for code in config.get('retry_status_codes', [408, 429, 500, 502, 503, 504])
        }
        
        # 存储每个instance的session信息
        self.instance_sessions = {}
        
        # 环境特定配置（子类可覆盖）
        self.env_name = "agentgym"  # 子类应设置具体环境名
        self.max_rounds = 30  # 默认最大轮数
    
    # =========================================================================
    # 异步HTTP辅助方法 - 避免阻塞event loop
    # =========================================================================
    
    async def _async_post(self, url: str, **kwargs) -> requests.Response:
        """异步包装 requests.post，不阻塞event loop，并重试瞬时服务不可用。"""
        kwargs.setdefault('timeout', self.timeout)
        attempts = max(1, self.max_retries)
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                response = await asyncio.to_thread(requests.post, url, **kwargs)
                if response.status_code not in self.retry_status_codes or attempt == attempts:
                    return response
                logger.warning(
                    "Transient HTTP %s from %s; retrying %s/%s",
                    response.status_code,
                    url,
                    attempt,
                    attempts,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == attempts:
                    raise
                logger.warning("Transient request error from %s: %s; retrying %s/%s", url, exc, attempt, attempts)

            sleep_s = self.retry_backoff * (2 ** (attempt - 1)) + random.uniform(0, self.retry_backoff)
            await asyncio.sleep(sleep_s)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Failed to POST {url}")
    
    async def _async_get(self, url: str, **kwargs) -> requests.Response:
        """异步包装 requests.get，不阻塞event loop"""
        kwargs.setdefault('timeout', self.timeout)
        return await asyncio.to_thread(requests.get, url, **kwargs)
    
    async def start_interaction(self, instance_id: str, **kwargs) -> None:
        """
        启动一个新的交互session
        
        Args:
            instance_id: 实例唯一标识
            **kwargs: 环境特定参数（如session_id, task_id等）
        """
        # 创建环境实例
        create_url = f"{self.env_server_base}/create"
        try:
            response = await self._async_post(create_url, json=kwargs)
            response.raise_for_status()
            data = response.json()
            self._raise_env_error(data, f"create {self.env_name} environment")
            # 兼容不同环境的返回格式：有些返回'env_id'，有些返回'id'
            env_id = data.get('env_id') if data.get('env_id') is not None else data.get('id')
            if env_id is None:
                raise ValueError(f"No env_id found in response: {data}")
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            raise
        
        # 重置环境获取初始observation
        reset_url = f"{self.env_server_base}/reset"
        try:
            response = await self._async_post(reset_url, json={'id': env_id})
            response.raise_for_status()
            data = response.json()
            self._raise_env_error(data, f"reset {self.env_name} environment {env_id}")
        except Exception as e:
            logger.error(f"Failed to reset environment {env_id}: {e}")
            raise
        
        # 保存session信息
        self.instance_sessions[instance_id] = {
            'env_id': env_id,
            'env_server_base': self.env_server_base,
            'done': False,
            'step_count': 0,
            'initial_observation': self._format_observation(data.get('observation', ''), data),
            'kwargs': kwargs
        }
        
        # logger.info(f"Started interaction {instance_id} with env_id {env_id}")
    
    async def generate_response(
        self, 
        instance_id: str, 
        messages: List[Dict[str, str]],
        **kwargs  # 接受额外参数（如name等），但不使用
    ) -> Tuple[bool, str, float, Dict]:
        """
        根据对话历史生成环境响应
        
        Args:
            instance_id: 实例ID
            messages: 对话历史 [{"role": "user/assistant", "content": "..."}]
        
        Returns:
            (done, observation, reward, extra_info)
        """
        session = self.instance_sessions.get(instance_id)
        if not session:
            raise ValueError(f"Instance {instance_id} not found")
        
        env_id = session['env_id']
        
        # 如果是第一次调用（messages为空或只有system prompt），返回初始observation
        if len(messages) == 0 or (len(messages) == 1 and messages[0]['role'] == 'system'):
            observation = session['initial_observation']
            return False, observation, 0.0, {}
        
        # 如果已经完成，直接返回
        if session['done']:
            return True, "Episode finished.", 0.0, {}
        
        # 检查是否超过最大轮数
        if session['step_count'] >= self.max_rounds:
            session['done'] = True
            return True, f"Max rounds ({self.max_rounds}) reached.", 0.0, {}
        
        # 提取最后一条assistant消息作为action
        last_assistant_msg = None
        for msg in reversed(messages):
            if msg['role'] == 'assistant':
                last_assistant_msg = msg['content']
                break
        
        if not last_assistant_msg:
            # 没有action，返回初始observation
            # logger.warning(f"[{instance_id}] No assistant message found")
            return False, session['initial_observation'], 0.0, {}
        
        # 提取action（子类可覆盖此方法）
        action = self.extract_action(last_assistant_msg)
        
        # 如果没有提取到有效action，提示用户
        if not action:
            return False, self.get_invalid_action_prompt(), 0.0, {}
        
        # 执行action（异步，不阻塞event loop）
        env_server_base = session.get('env_server_base', self.env_server_base)
        step_url = f"{env_server_base}/step"
        try:
            response = await self._async_post(
                step_url, 
                json=self._build_step_payload(env_id, action)
            )
            response.raise_for_status()
            data = response.json()
            self._raise_env_error(data, f"step {self.env_name} environment {env_id}")
        except Exception as e:
            logger.error(f"[{instance_id}] Failed to step environment {env_id}: {e}")
            raise
        
        # 更新session状态
        session['step_count'] += 1
        session['done'] = data.get('done', False)
        
        observation = self._format_observation(data.get('observation', ''), data)
        reward = data.get('reward', 0.0)
        done = data.get('done', False)
        
        # 仅记录异常/失败终止，避免成功结束时刷屏拖慢训练
        if reward < 0 or (done and reward <= 0):
            logger.warning(f"[{instance_id}] Step {session['step_count']} - Env response: reward={reward}, done={done}, action='{action[:50] if action else 'None'}...', obs='{observation[:100]}...'")
        
        return done, observation, reward, {}
    
    async def calculate_score(self, instance_id: str) -> float:
        """
        计算最终得分
        
        Returns:
            最终reward（通常是0-1之间的归一化分数）
        """
        session = self.instance_sessions.get(instance_id)
        if not session:
            return 0.0
        
        env_id = session['env_id']
        
        # 获取环境信息（包含最终reward）
        try:
            env_server_base = session.get('env_server_base', self.env_server_base)
            observe_url = f"{env_server_base}/observe"
            response = await self._async_get(f"{observe_url}?id={env_id}")
            response.raise_for_status()
            data = response.json()
            return data.get('reward', 0.0)
        except Exception as e:
            logger.error(f"Failed to get final score: {e}")
            return 0.0
    
    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """清理交互 session，并释放远端环境实例。"""
        session = self.instance_sessions.get(instance_id)
        if not session:
            return

        env_id = session.get('env_id')
        env_server_base = session.get('env_server_base', self.env_server_base)
        try:
            if env_id is not None:
                response = await self._async_post(f"{env_server_base}/close", json={'id': env_id})
                response.raise_for_status()
                self._raise_env_error(response.json(), f"close {self.env_name} environment {env_id}")
        except Exception as e:
            logger.warning(f"Failed to close {self.env_name} environment {env_id}: {e}")
        finally:
            self.instance_sessions.pop(instance_id, None)
            # logger.info(f"Finalized interaction {instance_id}")
    
    def _build_step_payload(self, env_id: int, action: str) -> dict:
        """构建 /step 请求的 payload，子类可覆盖以适配不同的字段名"""
        return {'id': env_id, 'action': action}

    def _raise_env_error(self, data: Dict[str, Any], context: str) -> None:
        """Turn AgentGym JSON error payloads into explicit rollout failures."""
        if isinstance(data, dict) and data.get('error'):
            raise RuntimeError(f"{context} failed: {data['error']}")

    def _format_observation(self, observation: str, data: Dict[str, Any]) -> str:
        """Hook for env-specific observation formatting."""
        return observation
    
    def extract_action(self, text: str) -> Optional[str]:
        """
        从模型输出中提取action
        子类应根据具体环境的action格式覆盖此方法
        
        默认实现：提取最后一行非空文本
        """
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('<think>') and not line.startswith('</think>'):
                return line
        return None
    
    def get_invalid_action_prompt(self) -> str:
        """
        返回无效action时的提示信息
        子类可覆盖以提供环境特定的提示
        """
        return "Please provide a valid action."
