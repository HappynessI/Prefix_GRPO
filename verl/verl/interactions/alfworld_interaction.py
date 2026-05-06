"""ALFWorld环境的Interaction实现"""

import asyncio
import hashlib
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

from verl.interactions.agentgym_base_interaction import AgentGymBaseInteraction

logger = logging.getLogger(__name__)


class ALFWorldInteraction(AgentGymBaseInteraction):
    """ALFWorld环境交互类"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.env_name = "alfworld"
        self.max_rounds = 50
        self.world_type = config.get("world_type", "Text")
        self.create_retry_timeout = float(config.get("create_retry_timeout", self.timeout))
        self.create_retry_backoff = float(config.get("create_retry_backoff", 0.5))
        client_max_active_envs = config.get(
            "client_max_active_envs",
            os.environ.get("ALFWORLD_CLIENT_MAX_ACTIVE_ENVS", 0),
        )
        try:
            client_max_active_envs = int(client_max_active_envs)
        except (TypeError, ValueError):
            client_max_active_envs = 0
        self.client_max_active_envs = max(0, client_max_active_envs)
        self._client_active_sem = (
            asyncio.Semaphore(self.client_max_active_envs) if self.client_max_active_envs > 0 else None
        )
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
        """ALFWorld reset 需要显式传入 game 与 world_type。"""
        prefix_actions = kwargs.get("prefix_actions", None)
        if prefix_actions is not None and not isinstance(prefix_actions, list):
            prefix_actions = list(prefix_actions)

        game = kwargs.get("game")
        session_id = kwargs.get("session_id", game)
        if game is None:
            if session_id is None:
                raise ValueError("ALFWorld interaction requires `game` or `session_id` in interaction_kwargs.")
            game = session_id

        game = int(game)
        session_id = int(session_id if session_id is not None else game)
        world_type = kwargs.get("world_type", self.world_type)

        env_server_base = self._select_env_server_base(instance_id)
        env_id = None
        client_slot_acquired = False
        try:
            if self._client_active_sem is not None:
                await self._client_active_sem.acquire()
                client_slot_acquired = True

            data, env_server_base = await self._create_with_capacity_retry(env_server_base)
            env_id = data.get("env_id") if data.get("env_id") is not None else data.get("id")
            if env_id is None:
                raise ValueError(f"No env_id found in response: {data}")

            reset_url = f"{env_server_base}/reset"
            reset_payload = {"id": env_id, "game": game, "world_type": world_type}
            response = await self._async_post(reset_url, json=reset_payload)
            response.raise_for_status()
            data = response.json()
            self._raise_env_error(data, f"reset ALFWorld environment {env_id}")

            self.instance_sessions[instance_id] = {
                "env_id": env_id,
                "env_server_base": env_server_base,
                "done": False,
                "step_count": 0,
                "initial_observation": self._format_observation(data.get("observation", ""), data),
                "session_id": session_id,
                "game": game,
                "world_type": world_type,
                "kwargs": kwargs,
                "client_slot_acquired": client_slot_acquired,
            }

            if prefix_actions:
                await self._replay_prefix_actions(instance_id, prefix_actions)
        except Exception as e:
            logger.error(f"Failed to start ALFWorld interaction {instance_id} env_id={env_id}: {e}")
            self.instance_sessions.pop(instance_id, None)
            if env_id is not None:
                try:
                    response = await self._async_post(f"{env_server_base}/close", json={"id": env_id})
                    response.raise_for_status()
                    self._raise_env_error(response.json(), f"close failed ALFWorld environment {env_id}")
                except Exception as close_exc:
                    logger.warning(
                        "Failed to close partially-started ALFWorld environment %s: %s",
                        env_id,
                        close_exc,
                    )
            if client_slot_acquired:
                self._release_client_active_slot()
            raise

    def _release_client_active_slot(self) -> None:
        if self._client_active_sem is not None:
            self._client_active_sem.release()

    async def _create_with_capacity_retry(self, preferred_env_server_base: str) -> Tuple[Dict[str, Any], str]:
        """Create an ALFWorld env, rotating across servers when active-env slots are full."""
        deadline = time.monotonic() + max(1.0, self.create_retry_timeout)
        attempts = 0
        last_capacity_error = None
        try:
            preferred_index = self.env_server_bases.index(preferred_env_server_base)
        except ValueError:
            preferred_index = 0

        while True:
            attempts += 1
            round_start = (preferred_index + attempts - 1) % len(self.env_server_bases)
            for offset in range(len(self.env_server_bases)):
                env_server_base = self.env_server_bases[(round_start + offset) % len(self.env_server_bases)]
                create_url = f"{env_server_base}/create"
                response = await self._async_post(create_url, json={})
                response.raise_for_status()
                data = response.json()

                if data.get("retryable") and data.get("error_code") == "active_capacity":
                    last_capacity_error = data.get("error", data)
                    continue

                self._raise_env_error(data, "create ALFWorld environment")
                return data, env_server_base

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for ALFWorld server capacity after {attempts} create rounds: "
                    f"{last_capacity_error}"
                )
            if attempts == 1 or attempts % 20 == 0:
                logger.info(
                    "ALFWorld create waiting for server capacity across %s servers: %s",
                    len(self.env_server_bases),
                    last_capacity_error,
                )
            await asyncio.sleep(max(0.05, self.create_retry_backoff))

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
                    "[%s] ALFWorld prefix replay stopped early at action %s/%s because env is done.",
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
            self._raise_env_error(data, f"ALFWorld prefix replay at action {replay_idx}")

            session["step_count"] = int(session.get("step_count", 0)) + 1
            session["done"] = bool(data.get("done", False))
            session["initial_observation"] = self._format_observation(
                data.get("observation", session.get("initial_observation", "")),
                data,
            )

    def _format_observation(self, observation: str, data: Dict[str, Any]) -> str:
        """Append ALFWorld admissible commands to match SFT/eval prompt format."""
        available_actions = data.get("available_actions") or data.get("admissible_commands")
        if not available_actions:
            return observation

        observation = str(observation or "").rstrip()
        if "available actions:" in observation.lower():
            return observation

        if isinstance(available_actions, str):
            actions_text = available_actions.strip()
        else:
            actions_text = ",".join(str(action).strip() for action in available_actions if str(action).strip())
        if not actions_text:
            return observation
        return f"{observation}\nAVAILABLE ACTIONS: {actions_text}"

    def extract_action(self, text: str) -> Optional[str]:
        text = text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        bracket_matches = re.findall(r"\[\[([^\]]+)\]\]", text)
        if bracket_matches:
            for match in bracket_matches:
                action = match.strip()
                if action:
                    return action.lower()

        action_match = re.search(r"Action:\s*(.+)", text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            action = re.sub(r"^\[\[\s*", "", action)
            action = re.sub(r"\s*\]\]$", "", action)
            return action.lower()

        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) < 100:
                line = re.sub(r"^\[\[\s*", "", line)
                line = re.sub(r"\s*\]\]$", "", line)
                return line.lower()

        return None

    def get_invalid_action_prompt(self) -> str:
        return (
            "Please provide a valid action. "
            "Example actions: go to <object>, take <object> from <receptacle>, "
            "put <object> in/on <receptacle>, examine <object>"
        )

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        session = self.instance_sessions.get(instance_id)
        client_slot_acquired = bool(session and session.get("client_slot_acquired", False))
        try:
            await super().finalize_interaction(instance_id, **kwargs)
        finally:
            if client_slot_acquired:
                self._release_client_active_slot()


# Backward compatibility for existing interaction YAMLs that used this spelling.
AlfWorldInteraction = ALFWorldInteraction
