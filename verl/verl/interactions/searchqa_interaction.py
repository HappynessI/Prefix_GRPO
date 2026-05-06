"""SearchQA环境的Interaction实现"""

import re
import logging
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction

logger = logging.getLogger(__name__)


class SearchQAInteraction(AgentGymBaseInteraction):
    """SearchQA环境交互类（搜索问答）
    
    SearchQA action格式示例：
    - search[machine learning]
    - click[3]
    - answer[deep learning is a subset of machine learning]
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.env_name = "searchqa"
        self.max_rounds = 15  # 搜索问答步数适中
    
    async def start_interaction(self, instance_id: str, **kwargs) -> None:
        """
        SearchQA环境特殊处理：reset需要env_idx和id参数
        """
        # 创建环境实例
        session_id = kwargs.get('session_id', 0)
        create_url = f"{self.env_server_base}/create"
        try:
            response = await self._async_post(create_url, json={"id": session_id})
            response.raise_for_status()
            data = response.json()
            
            # SearchQA返回的格式
            if isinstance(data, int):
                env_id = data
            elif isinstance(data, dict):
                env_id = data.get('env_idx') or data.get('id') or data.get('env_id')
            else:
                env_id = data
        except Exception as e:
            logger.error(f"Failed to create SearchQA environment: {e}")
            raise
        
        # 使用session_id作为item_id来reset
        reset_url = f"{self.env_server_base}/reset"
        try:
            reset_response = await self._async_post(
                reset_url,
                json={"env_idx": env_id, "id": session_id}
            )
            reset_response.raise_for_status()
            reset_data = reset_response.json()
        except Exception as e:
            logger.error(f"Failed to reset SearchQA environment: {e}")
            raise
        
        # 保存session信息
        # SearchQA的/reset直接返回observation字符串，不是dict
        if isinstance(reset_data, str):
            observation = reset_data
            done = False
        elif isinstance(reset_data, dict):
            observation = reset_data.get('observation', '')
            done = reset_data.get('done', False)
        else:
            observation = str(reset_data)
            done = False
        
        self.instance_sessions[instance_id] = {
            'env_id': env_id,
            'done': done,
            'step_count': 0,
            'initial_observation': observation,
            'kwargs': kwargs
        }
        
        logger.info(f"Started SearchQA interaction {instance_id} with env_id {env_id}")
    
    def _build_step_payload(self, env_id: int, action: str) -> dict:
        """SearchQA 的 /step 接口使用 env_idx 而非 id"""
        return {'env_idx': env_id, 'action': action}
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取SearchQA action
        
        支持的格式：
        1. [[ search[keywords] ]] 或 [[ click[n] ]] 或 [[ answer[text] ]]
        2. 直接 search[...], click[...], answer[...]
        """
        text = text.strip()
        
        # 移除chat template标记
        text = re.sub(r'<\|im_start\|>assistant\s*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\|im_end\|>', '', text)
        
        # 移除思考标签
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 尝试从 [[ ]] 中提取
        box_matches = re.findall(r'\[\[\s*(.*?)\s*\]\]', text, re.DOTALL)
        if box_matches:
            text = box_matches[-1]
        
        # 提取search[...], click[...], answer[...]格式
        action_match = re.search(r'\b(search|click|answer)\s*\[([^\]]+)\]', text, re.IGNORECASE)
        if action_match:
            action_type = action_match.group(1).lower()
            action_arg = action_match.group(2).strip()
            return f"{action_type}[{action_arg}]"
        
        # 提取"Action:"后的内容
        action_label_match = re.search(r'Action:\s*(.+)', text, re.IGNORECASE)
        if action_label_match:
            action_line = action_label_match.group(1).strip()
            action_match = re.search(r'\b(search|click|answer)\s*\[([^\]]+)\]', action_line, re.IGNORECASE)
            if action_match:
                action_type = action_match.group(1).lower()
                action_arg = action_match.group(2).strip()
                return f"{action_type}[{action_arg}]"
        
        # 提取最后一行包含action格式的文本
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            action_match = re.search(r'\b(search|click|answer)\s*\[([^\]]+)\]', line, re.IGNORECASE)
            if action_match:
                action_type = action_match.group(1).lower()
                action_arg = action_match.group(2).strip()
                return f"{action_type}[{action_arg}]"
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action.\n"
                "Valid actions: search[keywords], click[number], answer[your answer]")
