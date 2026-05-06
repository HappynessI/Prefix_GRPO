"""TextCraft环境的Interaction实现"""

import os
import re
import logging
from typing import Optional
from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction
from verl.interactions.textcraft_action_parser import extract_textcraft_action_loose

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Debug mode flag
DEBUG_MODE = os.getenv("VERL_DEBUG_MODE", "0") == "1" or os.getenv("DEBUG_MODE", "0") == "1"


class TextCraftInteraction(AgentGymBaseInteraction):
    """TextCraft环境交互类（文本版Minecraft）
    
    TextCraft action格式示例：
    - "craft(wood_pickaxe)"
    - "mine(stone)"
    - "get(wood)"
    - "goto(forest)"
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.env_name = "textcraft"
        self.max_rounds = 100  # 制作任务可能需要很多步骤
    
    async def start_interaction(self, instance_id: str, **kwargs) -> None:
        """
        TextCraft环境特殊处理：API返回的是'id'而非'env_id'
        支持 prefix_actions：创建环境后 replay 指定的 actions 来同步状态
        """
        prefix_actions = kwargs.pop('prefix_actions', None)
        if prefix_actions is not None and not isinstance(prefix_actions, list):
            prefix_actions = list(prefix_actions)

        # 提取 goal：优先从 interaction_kwargs 读（修复后的 parquet），兜底从 prompt 解析
        expected_goal = kwargs.pop('goal', None)
        if expected_goal is None:
            # 兜底：从 prompt（prompt 字段是 messages 列表）里解析 Goal: 字段
            prompt = kwargs.get('prompt')
            if prompt is not None:
                prompt_list = prompt.tolist() if hasattr(prompt, 'tolist') else (prompt if isinstance(prompt, list) else [])
                for msg in prompt_list:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        content = msg.get('content', '')
                        m = re.search(r'Goal:\s*craft\s+(.+?)\.?$', content, re.IGNORECASE | re.MULTILINE)
                        if m:
                            expected_goal = m.group(1).strip()
                            logger.warning(
                                f"[{instance_id}] goal not in interaction_kwargs — "
                                f"extracted from prompt: {expected_goal!r}"
                            )
                            break

        # 提取 data_idx（数据集索引，用于确定性任务分配）
        data_idx = kwargs.pop('data_idx', None)

        # 创建环境实例：显式传入 goal 和 data_idx
        create_body = {}
        if expected_goal is not None:
            create_body['goal'] = expected_goal
        if data_idx is not None:
            create_body['data_idx'] = data_idx

        create_url = f"{self.env_server_base}/create"
        try:
            response = await self._async_post(create_url, json=create_body)
            response.raise_for_status()
            data = response.json()
            # TextCraft返回的是'id'字段，不是'env_id'
            # 注意：id可能为0，所以不能用 or 判断
            if 'id' in data:
                env_id = data['id']
            elif 'env_id' in data:
                env_id = data['env_id']
            else:
                raise ValueError(f"No env_id found in response: {data}")
        except Exception as e:
            logger.error(f"Failed to create TextCraft environment: {e}")
            raise

        # === 提取实际 goal（用于后续校验和记录） ===
        actual_obs = data.get('observation', '')
        actual_goal_in_obs = None
        m = re.search(r'Goal:\s*craft\s+(.+?)\.?$', actual_obs, re.IGNORECASE | re.MULTILINE)
        if m:
            actual_goal_in_obs = m.group(1).strip()

        # === FAIL-FAST: 当提供了 expected_goal 时校验一致性 ===
        if expected_goal is not None:
            if actual_goal_in_obs is None:
                raise ValueError(
                    f"[{instance_id}] FAIL-FAST: Could not extract goal from server observation. "
                    f"expected_goal={expected_goal!r}, obs={actual_obs[:200]!r}"
                )

            def _norm(g):
                return g.lower().replace('_', ' ').replace("'", '').strip()

            if _norm(actual_goal_in_obs) != _norm(expected_goal):
                raise ValueError(
                    f"[{instance_id}] FAIL-FAST: Goal mismatch! "
                    f"expected_goal={expected_goal!r}, "
                    f"actual_goal_in_env={actual_goal_in_obs!r}. "
                    f"This means the task binding is broken — stop the run immediately."
                )
            logger.info(f"[{instance_id}] Goal validation passed: expected={expected_goal!r}, actual={actual_goal_in_obs!r}")
        else:
            # 没有 expected_goal 时，记录实际分配到的 goal（便于调试和理解随机分配情况）
            if actual_goal_in_obs is not None:
                logger.warning(
                    f"[{instance_id}] No expected_goal provided — server assigned goal={actual_goal_in_obs!r}. "
                    f"This run will use server-side random goal assignment (data_idx={data_idx}). "
                    f"Consider adding goal to test.parquet for deterministic evaluation."
                )
            else:
                logger.warning(
                    f"[{instance_id}] No expected_goal provided and could not extract goal from observation. "
                    f"data_idx={data_idx}, obs={actual_obs[:200]!r}"
                )

        # TextCraft的create接口已经返回了initial observation，不需要再调用reset
        self.instance_sessions[instance_id] = {
            'env_id': env_id,
            'done': data.get('done', False),
            'step_count': 0,
            'initial_observation': actual_obs,
            'kwargs': kwargs,
            'expected_goal': expected_goal,
            'actual_goal_in_obs': actual_goal_in_obs,
            'data_idx': data_idx,
        }

        logger.info(f"Started TextCraft interaction {instance_id} with env_id {env_id}")

        # ==================== DEBUG: Replay 开始 ====================
        if DEBUG_MODE:
            print(f"\n{'='*60}")
            print(f"DEBUG: TextCraft start_interaction for {instance_id}")
            print(f"  - env_id: {env_id}")
            print(f"  - expected_goal: {expected_goal}")
            print(f"  - actual_goal_in_obs: {actual_goal_in_obs}")
            print(f"  - data_idx: {data_idx}")
            print(f"  - prefix_actions 数量: {len(prefix_actions) if prefix_actions else 0}")
            if prefix_actions:
                print(f"  - prefix_actions 前3项: {prefix_actions[:3]}")
            print(f"{'='*60}\n")

        # Replay prefix actions to sync environment state
        if prefix_actions:
            await self._replay_prefix_actions(instance_id, prefix_actions)

    async def _replay_prefix_actions(self, instance_id: str, actions: list[str]) -> None:
        """Replay a sequence of actions to bring the environment to the post-prefix state."""
        session = self.instance_sessions.get(instance_id)
        if not session:
            raise ValueError(f"Instance {instance_id} not found for prefix replay")

        env_id = session['env_id']
        step_url = f"{self.env_server_base}/step"

        # ==================== DEBUG: Replay 过程 ====================
        if DEBUG_MODE:
            print(f"\n{'='*60}")
            print(f"DEBUG: Replay 开始 for {instance_id}")
            print(f"  - 总共 {len(actions)} 个 actions 需要 replay")
            print(f"{'='*60}\n")

        for i, action in enumerate(actions):
            if session['done']:
                print(f"[REPLAY_STEP] instance={instance_id}, action={action!r} — SKIPPED (env already done)", flush=True)
                break
            try:
                response = await self._async_post(
                    step_url,
                    json=self._build_step_payload(env_id, action),
                )
                response.raise_for_status()
                data = response.json()

                obs = data.get('observation', '')
                raw_reward = data.get('reward', 0.0)
                done = data.get('done', False)

                session['latest_observation'] = obs
                session['step_count'] += 1
                session['done'] = done

                # Unconditional: print every replay step response
                print(f"[REPLAY_STEP] instance={instance_id}, action={action!r}, reward={raw_reward}, done={done}, obs={obs[:100] if obs else 'None'!r}", flush=True)
            except Exception as e:
                print(f"[REPLAY_STEP] instance={instance_id}, action={action!r} — ERROR: {e}", flush=True)
                raise

        # Unconditional: print final replay state
        final_obs = session.get('latest_observation', session.get('initial_observation', ''))
        print(f"[REPLAY_DONE] instance={instance_id}, steps={session['step_count']}, done={session['done']}, final_obs={final_obs[:200] if final_obs else 'None'!r}", flush=True)

        logger.info(
            f"[{instance_id}] Replayed {len(actions)} prefix actions, "
            f"step_count={session['step_count']}, done={session['done']}"
        )

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Close the remote TextCraft env before dropping local session state."""
        session = self.instance_sessions.get(instance_id)
        if not session:
            return

        env_id = session['env_id']
        close_url = f"{self.env_server_base}/close"
        try:
            response = await self._async_post(close_url, json={"id": env_id})
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to close TextCraft environment {env_id}: {e}")
        finally:
            self.instance_sessions.pop(instance_id, None)
    
    def extract_action(self, text: str) -> Optional[str]:
        """从模型输出中提取TextCraft action
        
        支持两种格式：
        1. [[ ... ]] 格式（评估脚本要求）: Action: [[ inventory ]]
        2. Action: 格式（训练数据格式）: Action: \n inventory
        
        格式示例：
        - Action: [[ inventory ]]
        - Action: [[ get 3 logs ]]
        - Action: [[ craft 4 stick using 2 oak planks ]]
        - Action: \n inventory
        - Action: \n get 3 logs
        """
        return extract_textcraft_action_loose(text)
    
    def get_invalid_action_prompt(self) -> str:
        return ("Please provide a valid action. "
                "Example actions:\n"
                "- craft 1 blue dye using 1 lapis lazuli\n"
                "- get 9 slime ball\n"
                "- inventory")
