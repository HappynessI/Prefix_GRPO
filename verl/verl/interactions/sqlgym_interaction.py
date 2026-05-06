"""SQLGym环境的Interaction实现"""

import re
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction


class SQLGymInteraction(AgentGymBaseInteraction):
    """SQLGym环境交互类
    
    SQLGym action格式：SQL查询语句
    示例：
    - "SELECT * FROM users WHERE age > 18"
    - "SELECT name, count(*) FROM orders GROUP BY name"
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.env_name = "sqlgym"
        self.max_rounds = 10  # SQL任务通常步数较少
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取SQL查询
        
        支持的格式：
        1. 直接SQL: SELECT * FROM users
        2. 代码块: ```sql\nSELECT...\n```
        3. 带思考: <think>...</think>\nSELECT...
        4. 带标签: Action: SELECT...
        """
        text = text.strip()
        
        # 移除思考标签
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 提取SQL代码块
        sql_block = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if sql_block:
            return sql_block.group(1).strip()
        
        # 提取任意代码块
        code_block = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_block:
            sql = code_block.group(1).strip()
            if sql.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE')):
                return sql
        
        # 提取"Action:"后的内容
        action_match = re.search(r'Action:\s*(.+)', text, re.IGNORECASE)
        if action_match:
            return action_match.group(1).strip()
        
        # 查找SQL关键词开头的语句
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')):
                return line
        
        # 返回最后一行非空文本
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) > 5:  # SQL语句至少有几个字符
                return line
        
        return None
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid SQL query. "
                "Example: SELECT column FROM table WHERE condition")

