# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

# Debug mode flag - controlled by environment variable
DEBUG_MODE = os.getenv("VERL_DEBUG_MODE", "0") == "1" or os.getenv("DEBUG_MODE", "0") == "1"
_DEBUG_SAMPLE_COUNTER = 0  # Counter for limiting debug output to first few samples

# 模块级别打印，确认模块加载和环境变量
print(f"\n{'='*60}")
print(f"DEBUG: tool_agent_loop.py module loaded - VERL_DEBUG_MODE={os.getenv('VERL_DEBUG_MODE', 'not set')}, DEBUG_MODE={DEBUG_MODE}")
print(f"{'='*60}\n")

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BABYAI_INITIAL_PROMPT_PLACEHOLDER = (
    "Please help me complete the BabyAI task. I will provide you with the task description."
)
ALFWORLD_INITIAL_PROMPT_PLACEHOLDER = (
    "Please help me complete the household task. I will provide you with the task description."
)


def _is_agentgym_initial_placeholder(messages: list[dict[str, Any]], interaction_kwargs: dict[str, Any]) -> bool:
    if len(messages) != 2:
        return False
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        return False

    placeholder_by_env = {
        "babyai": BABYAI_INITIAL_PROMPT_PLACEHOLDER,
        "alfworld": ALFWORLD_INITIAL_PROMPT_PLACEHOLDER,
    }
    expected_placeholder = placeholder_by_env.get(interaction_kwargs.get("name"))
    return expected_placeholder is not None and str(messages[1].get("content", "")).strip() == expected_placeholder


def _needs_initial_observation(messages: list[dict[str, Any]], interaction_kwargs: dict[str, Any]) -> bool:
    if len(messages) == 0:
        return True
    if len(messages) == 1 and messages[0].get("role") == "system":
        return True
    return _is_agentgym_initial_placeholder(messages, interaction_kwargs)


def _initial_observation_request_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if messages and messages[0].get("role") == "system":
        return [messages[0]]
    return []


def _apply_initial_observation(
    messages: list[dict[str, Any]],
    interaction_kwargs: dict[str, Any],
    initial_obs: str,
) -> None:
    if _is_agentgym_initial_placeholder(messages, interaction_kwargs):
        messages[1] = {"role": "user", "content": initial_obs}
    else:
        messages.append({"role": "user", "content": initial_obs})


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # Debug: sample index for logging
        self.sample_idx: int = -1
        
        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.completed_interaction_rounds = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        round_caps = [cap for cap in (cls.max_assistant_turns, cls.max_user_turns) if cap is not None]
        cls.max_interaction_rounds = min(round_caps) if round_caps else None
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = initialize_system_prompt(cls.tokenizer, **cls.apply_chat_template_kwargs)

        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        global _DEBUG_SAMPLE_COUNTER
        
        # ==================== DEBUG: 入口确认 ====================
        sample_idx = _DEBUG_SAMPLE_COUNTER
        _DEBUG_SAMPLE_COUNTER += 1
        request_id = uuid4().hex  # 先定义 request_id
        
        if DEBUG_MODE and sample_idx < 3:
            extra_info = kwargs.get("extra_info", {})
            interaction_kwargs = extra_info.get("interaction_kwargs", {})
            prefix_actions = interaction_kwargs.get("prefix_actions", [])
            
            print(f"\n{'='*60}")
            print(f"DEBUG: Sample #{sample_idx} - ToolAgentLoop.run()")
            print(f"  - request_id: {request_id}")
            print(f"  - prefix_actions 数量: {len(prefix_actions) if prefix_actions else 0}")
            print(f"{'='*60}\n")
        
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_started = False
        interaction_kwargs = {}
        try:
            if self.interaction_config_file:
                interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
                if "name" not in interaction_kwargs:
                    raise ValueError("'name' key is required in interaction_kwargs")
                interaction_name = interaction_kwargs["name"]
                if interaction_name not in self.interaction_map:
                    raise ValueError(
                        f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                        f"{list(self.interaction_map.keys())}"
                    )
                interaction = self.interaction_map[interaction_name]
                interaction_kwargs_with_prompt = {**interaction_kwargs, "prompt": kwargs.get("raw_prompt")}
                await interaction.start_interaction(request_id, **interaction_kwargs_with_prompt)
                interaction_started = True
            # Create AgentData instance to encapsulate all state
            agent_data = AgentData(
                messages=messages,
                image_data=image_data,
                metrics=metrics,
                request_id=request_id,
                tools_kwargs=tools_kwargs,
                interaction=interaction,
                interaction_kwargs=interaction_kwargs,
            )
            # Store sample_idx for debug logging
            agent_data.sample_idx = sample_idx

            # State machine loop
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)
                else:
                    logger.error(f"Invalid state: {state}")
                    state = AgentState.TERMINATED

            # Finalize output
            response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
            prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
            multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=agent_data.response_mask[: self.response_length],
                multi_modal_data=multi_modal_data,
                response_logprobs=agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs
                else None,
                num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                metrics=agent_data.metrics,
                extra_fields={},
            )
            output.extra_fields.update(agent_data.extra_fields)
            output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
            return output
        finally:
            if interaction is not None and interaction_started:
                try:
                    await interaction.finalize_interaction(request_id)
                except Exception as exc:
                    logger.warning(f"Failed to finalize interaction {request_id}: {exc}")

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        # Check if we need to get initial observation from interaction.
        # BabyAI GRPO data historically used a generic user placeholder; replace it
        # with the real env reset observation before the first assistant action.
        if agent_data.interaction is not None and _needs_initial_observation(
            agent_data.messages, agent_data.interaction_kwargs
        ):
            # Get initial observation before first generation
            (
                should_terminate,
                initial_obs,
                reward,
                metrics,
            ) = await agent_data.interaction.generate_response(
                agent_data.request_id,
                _initial_observation_request_messages(agent_data.messages),
                **agent_data.interaction_kwargs,
            )
            agent_data.user_turns += 1
            
            # Add or replace the first user message with the real reset observation.
            _apply_initial_observation(agent_data.messages, agent_data.interaction_kwargs, initial_obs)
            
            if reward is not None:
                agent_data.turn_scores.append(reward)
            
            # If environment already terminated (shouldn't happen), stop here
            if should_terminate:
                return AgentState.TERMINATED
        
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        # For environment interactions, terminate after a completed assistant->env round
        # instead of after the model has already produced the next assistant message.
        if not self.interaction_config_file and self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if not self.interaction_config_file and self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # ==================== DEBUG: Student 生成结果 ====================
        if DEBUG_MODE and agent_data.sample_idx < 3:
            assistant_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            print(f"\n{'='*60}")
            print(f"DEBUG: Student 生成 for sample {agent_data.sample_idx}")
            print(f"  - 原始文本 (前300字符): {assistant_text[:300]}...")
            
            # 尝试提取 action
            if agent_data.interaction and hasattr(agent_data.interaction, 'extract_action'):
                extracted_action = agent_data.interaction.extract_action(assistant_text)
                print(f"  - 提取的 action: {extracted_action}")
                
                # 注意：不进行 fail-fast，让训练继续
                # 如果无法提取 action，后续状态机会处理
            print(f"{'='*60}\n")

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)
        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if self.tool_parser_name == "gpt-oss":
                logger.info("manually format tool responses for gpt-oss")
                # Format tool responses manually
                tool_response_texts = []
                for i, tool_msg in enumerate(add_messages):
                    actual_tool_name = tool_call_names[i]
                    formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
                    tool_response_texts.append(formatted)

                tool_response_text = add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
                )
                response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        
        # ==================== DEBUG: Student continuation 开始 ====================
        if DEBUG_MODE and agent_data.sample_idx < 3:
            print(f"\n{'='*60}")
            print(f"DEBUG: Student continuation 开始 for request_id={agent_data.request_id}")
            print(f"  - 当前 messages 数量: {len(agent_data.messages)}")
            print(f"  - 当前 assistant_turns: {agent_data.assistant_turns}")
            print(f"{'='*60}\n")
            
            # Fail-fast: 检查 student 是否真的进入环境交互
            if not agent_data.interaction:
                print(f"❌ FAIL-FAST: No interaction object - student not in environment!")
                raise ValueError(f"Student failed to enter environment interaction!")

        with simple_timer("tool_calls", agent_data.metrics):
            (
                should_terminate_sequence,
                interaction_responses,
                reward,
                metrics,
            ) = await agent_data.interaction.generate_response(
                agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
            )
        
        # ==================== DEBUG: Student continuation 结果 ====================
        if DEBUG_MODE and agent_data.sample_idx < 3:
            print(f"\n{'='*60}")
            print(f"DEBUG: Student continuation 结果 for request_id={agent_data.request_id}")
            print(f"  - 环境返回的 response (前300字符): {interaction_responses[:300] if interaction_responses else 'None'}...")
            print(f"  - reward: {reward}")
            print(f"  - should_terminate: {should_terminate_sequence}")
            print(f"  - 新的 user message 已添加到 messages" if len(agent_data.messages) > 0 else "  - 无新消息")
            print(f"{'='*60}\n")
            
            # Fail-fast: 检查 student continuation 是否成功
            if not interaction_responses:
                print(f"❌ FAIL-FAST: No interaction response - environment replay/continuation failed!")
                raise ValueError(f"Student continuation failed - no response from environment!")

        agent_data.user_turns += 1
        agent_data.completed_interaction_rounds += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)
        if metrics:
            for key, value in metrics.items():
                if isinstance(key, str) and key.startswith("textcraft/"):
                    agent_data.extra_fields[key] = value

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        if (
            self.interaction_config_file
            and self.max_interaction_rounds is not None
            and agent_data.completed_interaction_rounds >= self.max_interaction_rounds
        ):
            return AgentState.TERMINATED
        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
