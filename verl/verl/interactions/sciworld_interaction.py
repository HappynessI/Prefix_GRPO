"""SciWorld环境的Interaction实现"""

import re
import logging
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction

logger = logging.getLogger(__name__)

CHAT_TEMPLATE_ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\s*\n?", re.IGNORECASE)
CHAT_TEMPLATE_END_RE = re.compile(r"<\|im_end\|>")
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
BOXED_ACTION_RE = re.compile(r"\[\[\s*(.*?)\s*\]\]", re.DOTALL)
ACTION_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


class SciWorldInteraction(AgentGymBaseInteraction):
    """SciWorld环境交互类
    
    SciWorld action格式示例（科学实验环境）：
    - "open door to kitchen"
    - "move to kitchen"
    - "take thermometer from table"
    - "use thermometer on water"
    - "read thermometer"
    - "pour water from beaker to container"
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.env_name = "sciworld"
        self.max_rounds = 100  # 科学实验可能需要很多步骤
    
    async def start_interaction(self, instance_id: str, **kwargs) -> None:
        """
        SciWorld环境特殊处理：
        1. reset需要显式 data_idx/session_id 参数
        2. 初始observation需要包含task_description
        """
        # 创建环境实例
        create_url = f"{self.env_server_base}/create"
        try:
            response = await self._async_post(create_url, json={})
            response.raise_for_status()
            data = response.json()
            
            # SciWorld返回 {"id": int}
            if isinstance(data, dict):
                env_id = data.get('id', None)
                if env_id is None:
                    env_id = data.get('env_id', None)
            elif isinstance(data, int):
                env_id = data
            else:
                env_id = data
            if env_id is None:
                raise ValueError(f"No env_id found in response: {data}")
        except Exception as e:
            logger.error(f"Failed to create SciWorld environment: {e}")
            raise
        
        data_idx = kwargs.get('data_idx')
        session_id = kwargs.get('session_id', data_idx)
        if data_idx is None:
            if session_id is None:
                raise ValueError("SciWorld interaction requires `data_idx` or `session_id` in interaction_kwargs.")
            data_idx = session_id
        data_idx = int(data_idx)
        session_id = int(session_id if session_id is not None else data_idx)

        reset_url = f"{self.env_server_base}/reset"
        try:
            reset_response = await self._async_post(
                reset_url,
                json={"id": env_id, "data_idx": data_idx}
            )
            reset_response.raise_for_status()
            reset_data = reset_response.json()
        except Exception as e:
            logger.error(f"Failed to reset SciWorld environment: {e}")
            raise
        
        # 构建完整的初始observation（包含任务描述）
        task_description = reset_data.get('task_description', '')
        observation = reset_data.get('observation', '')
        
        # 组合任务描述和初始观察
        if task_description:
            initial_observation = f"Task: {task_description}\n\n{observation}"
        else:
            initial_observation = observation
        
        # 保存session信息
        self.instance_sessions[instance_id] = {
            'env_id': env_id,
            'done': reset_data.get('done', False),
            'step_count': 0,
            'initial_observation': initial_observation,
            'task_description': task_description,
            'session_id': session_id,
            'data_idx': data_idx,
            'kwargs': kwargs
        }
        
        logger.info(f"Started SciWorld interaction {instance_id} with env_id {env_id}, task: {task_description[:50]}...")
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取SciWorld action
        
        支持的格式：
        1. [[ action ]] 格式（推荐）
        2. Action: action 格式

        注意：不再支持“最后一行短文本”fallback，避免把 </think>
        或普通解释文本误当作环境 action。
        """
        text = str(text or "").strip()

        def normalize_action(action_text: str) -> Optional[str]:
            action = re.sub(r'^\[\[\s*|\s*\]\]$', '', str(action_text or "").strip())
            action = " ".join(action.split()).strip().lower()
            if not action:
                return None
            if "<" in action or ">" in action:
                return None
            if action in {"thought:", "think:", "action:"}:
                return None
            if len(action) > 200:
                return None
            return action

        # First try the raw text. Some Qwen generations can place the final
        # Action block inside a closed <think>...</think> span.
        action_matches = BOXED_ACTION_RE.findall(text)
        if action_matches:
            action = normalize_action(action_matches[-1])
            if action:
                return action

        action_match = ACTION_RE.search(text)
        if action_match:
            action = normalize_action(action_match.group(1))
            if action:
                return action

        # 移除chat template标记
        text = CHAT_TEMPLATE_ASSISTANT_RE.sub("", text)
        text = CHAT_TEMPLATE_END_RE.sub("", text)
        
        # 移除思考标签
        text = THINK_BLOCK_RE.sub("", text)
        text = THINK_TAG_RE.sub("", text)
        
        # 格式1: [[ action ]]
        action_matches = BOXED_ACTION_RE.findall(text)
        if action_matches:
            action = normalize_action(action_matches[-1])
            if action:
                return action
        
        # 格式2: Action: action
        action_match = ACTION_RE.search(text)
        if action_match:
            action = normalize_action(action_match.group(1))
            if action:
                return action
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action wrapped in [[ ]].\n"
                "Example actions:\n"
                "- [[ move to kitchen ]]\n"
                "- [[ take thermometer from table ]]\n"
                "- [[ use thermometer on water ]]\n"
                "- [[ read thermometer ]]\n"
                "- [[ look around ]]")

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Close the remote SciWorld env before dropping local session state."""
        session = self.instance_sessions.get(instance_id)
        if not session:
            return

        env_id = session.get('env_id')
        close_url = f"{self.env_server_base}/close"
        try:
            response = await self._async_post(close_url, json={"id": env_id})
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to close SciWorld environment {env_id}: {e}")
        finally:
            self.instance_sessions.pop(instance_id, None)
