"""BabyAI环境的Interaction实现"""

import re
import logging
import hashlib
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction

logger = logging.getLogger(__name__)


class BabyAIInteraction(AgentGymBaseInteraction):
    """BabyAI环境交互类
    
    BabyAI是一个基于网格世界的导航和任务完成环境。
    动作是高级别的语义指令，如：
    - "turn left" / "turn right"
    - "move forward"
    - "pickup [object]"
    - "drop"
    - "toggle"
    - "go to [object]"
    - "go through [door]"
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.env_name = "babyai"
        self.max_rounds = 50  # BabyAI可能需要较多步数
        env_server_bases = config.get("env_server_bases", None)
        if env_server_bases is None:
            self.env_server_bases = [self.env_server_base]
        else:
            self.env_server_bases = [str(base).rstrip("/") for base in list(env_server_bases) if str(base).strip()]
            if not self.env_server_bases:
                self.env_server_bases = [self.env_server_base]
            self.env_server_base = self.env_server_bases[0]

    def _select_env_server_base(self, instance_id: str) -> str:
        if len(self.env_server_bases) == 1:
            return self.env_server_bases[0]
        digest = hashlib.md5(str(instance_id).encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % len(self.env_server_bases)
        return self.env_server_bases[index]
    
    async def start_interaction(self, instance_id: str, **kwargs) -> None:
        """
        BabyAI环境特殊处理：需要使用显式 data_idx/session_id 来 reset 环境。
        """
        prefix_actions = kwargs.get("prefix_actions", None)
        if prefix_actions is not None and not isinstance(prefix_actions, list):
            prefix_actions = list(prefix_actions)

        # 创建环境实例
        env_server_base = self._select_env_server_base(instance_id)
        create_url = f"{env_server_base}/create"
        try:
            response = await self._async_post(create_url, json={})
            response.raise_for_status()
            data = response.json()
            
            # BabyAI返回的是'id'字段
            if 'id' in data:
                env_id = data['id']
            elif 'env_id' in data:
                env_id = data['env_id']
            else:
                raise ValueError(f"No env_id found in response: {data}")
        except Exception as e:
            logger.error(f"Failed to create BabyAI environment: {e}")
            raise
        
        # data_idx 是真正的 reset 主键；旧数据只保留了 session_id 时回退兼容。
        data_idx = kwargs.get('data_idx')
        session_id = kwargs.get('session_id', data_idx)
        if data_idx is None:
            if session_id is None:
                raise ValueError("BabyAI interaction requires `data_idx` or `session_id` in interaction_kwargs.")
            data_idx = session_id
        data_idx = int(data_idx)
        session_id = int(session_id if session_id is not None else data_idx)

        reset_url = f"{env_server_base}/reset"
        try:
            reset_response = await self._async_post(
                reset_url,
                json={"id": env_id, "data_idx": data_idx}
            )
            reset_response.raise_for_status()
            reset_data = reset_response.json()
        except Exception as e:
            logger.error(f"Failed to reset BabyAI environment: {e}")
            raise
        
        # 保存session信息
        self.instance_sessions[instance_id] = {
            'env_id': env_id,
            'env_server_base': env_server_base,
            'done': reset_data.get('done', False),
            'step_count': 0,
            'initial_observation': reset_data.get('observation', ''),
            'session_id': session_id,
            'data_idx': data_idx,
            'kwargs': kwargs
        }

        if prefix_actions:
            await self._replay_prefix_actions(instance_id, prefix_actions)
        
        logger.info(f"Started BabyAI interaction {instance_id} with env_id {env_id} on {env_server_base}")

    async def _replay_prefix_actions(self, instance_id: str, actions: list[str]) -> None:
        """Replay dataset-provided prefix actions so rollout starts from the cut state."""
        session = self.instance_sessions.get(instance_id)
        if not session:
            raise ValueError(f"Instance {instance_id} not found")

        env_id = session["env_id"]
        env_server_base = session.get("env_server_base", self.env_server_base)
        step_url = f"{env_server_base}/step"

        for replay_idx, action in enumerate(actions):
            if session.get("done", False):
                logger.warning(
                    "[%s] BabyAI prefix replay stopped early at action %s/%s because env is done.",
                    instance_id,
                    replay_idx,
                    len(actions),
                )
                break

            action = str(action).strip().lower()
            if not action:
                continue

            response = await self._async_post(step_url, json=self._build_step_payload(env_id, action))
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise RuntimeError(f"BabyAI prefix replay failed at action {replay_idx}: {data['error']}")

            session["step_count"] = int(session.get("step_count", 0)) + 1
            session["done"] = bool(data.get("done", False))
            session["initial_observation"] = data.get("observation", session.get("initial_observation", ""))
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取BabyAI action
        
        支持的格式：
        1. [[ action ]] 格式（推荐）
        2. <action>action</action> 格式
        3. Action: action 格式
        4. 直接action（最后一行非空文本）
        
        BabyAI的动作空间：
        - turn left, turn right
        - move forward
        - pickup [color] [object] [n]
        - drop
        - toggle
        - go to [object]
        - go through [door]
        - check available actions
        """
        text = text.strip()
        
        # 移除chat template标记
        text = re.sub(r'<\|im_start\|>assistant\s*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\|im_end\|>', '', text)
        
        # 移除思考标签
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 格式1: [[ action ]] - 推荐格式
        action_matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
        if action_matches:
            action = action_matches[-1].strip()
            action = " ".join(action.split())
            if action:
                return action.lower()
        
        # 格式2: <action>action</action>
        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        if action_match:
            return action_match.group(1).strip().lower()
        
        # 格式3: Action: action
        action_line = re.search(r'Action:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if action_line:
            action = action_line.group(1).strip()
            # 移除可能的 [[ ]] 包裹（如果前面没匹配到）
            action = re.sub(r'^\[\[\s*|\s*\]\]$', '', action)
            if action:
                return action.lower()
        
        # 格式4: 提取最后一行非空文本作为action
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip().lower()
            # 排除思考内容和太长的行
            if line and len(line) < 100:
                # 检查是否像一个有效的action
                if any(keyword in line for keyword in [
                    'turn', 'move', 'pickup', 'drop', 'toggle', 
                    'go to', 'go through', 'check'
                ]):
                    return line
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return (
            "Please provide a valid action wrapped in [[ ]].\n"
            "Example actions:\n"
            "- [[ turn left ]] or [[ turn right ]]\n"
            "- [[ move forward ]]\n"
            "- [[ pickup red ball 1 ]]\n"
            "- [[ go to blue key 1 ]]\n"
            "- [[ toggle ]]\n"
            "- [[ check available actions ]]"
        )
