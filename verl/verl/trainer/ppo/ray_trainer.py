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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import csv
import json
import logging
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Mapping, Optional, Sequence

# Initialize module_logger for this module
module_logger = logging.getLogger(__name__)

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


class MetricsCSVWriter:
    """Persist scalar training metrics to a CSV file at a fixed step interval."""

    def __init__(self, output_dir: str, frequency: int = 50, filename: str = "training_metrics.csv"):
        self.frequency = max(int(frequency), 1)
        self.output_dir = os.path.join(output_dir, "metrics")
        os.makedirs(self.output_dir, exist_ok=True)
        self.filepath = os.path.join(self.output_dir, filename)
        self.rows: list[dict] = []
        self.fieldnames: list[str] = ["step"]

    @staticmethod
    def _is_scalar(value) -> bool:
        return isinstance(value, (int, float, str, bool, np.integer, np.floating))

    def maybe_log(self, metrics: dict, step: int, force: bool = False) -> None:
        if not force and step % self.frequency != 0:
            return

        row = {"step": step}
        for key, value in metrics.items():
            if self._is_scalar(value):
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                row[key] = value

        self.rows.append(row)

        merged_fields = set(self.fieldnames)
        for existing_row in self.rows:
            merged_fields.update(existing_row.keys())
        self.fieldnames = ["step"] + sorted(field for field in merged_fields if field != "step")

        with open(self.filepath, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.fieldnames)
            writer.writeheader()
            for existing_row in self.rows:
                writer.writerow(existing_row)


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_prefix_mask(data: DataProto, tokenizer=None):
    """Compute the mask for assistant prefix tokens in the prompt.

    This function identifies which tokens in the prompt correspond to assistant messages
    (teacher history from the prompt). Used when optimize_prefix_tokens=True.

    IMPORTANT: Only assistant role tokens are included in the prefix mask.
    System and user tokens are NOT included in prefix optimization.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        tokenizer: Tokenizer for processing chat format prompts. REQUIRED when optimize_prefix_tokens=True.

    Returns:
        torch.Tensor: The mask for assistant prefix tokens (shape: batch_size, seq_len).
                      1 for assistant prefix tokens, 0 for others (including system/user).

    Raises:
        ValueError: If tokenizer is None or cannot parse assistant token spans.
    """
    # Check if we already have precomputed assistant prefix token mask
    if "assistant_prefix_mask" in data.batch:
        return data.batch["assistant_prefix_mask"]

    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    total_length = attention_mask.size(1)
    prompt_length = total_length - response_length

    # FAIL FAST: tokenizer is required for computing assistant token mask
    if tokenizer is None:
        raise ValueError(
            "tokenizer is required for computing assistant prefix mask. "
            "Please ensure compute_prefix_mask() is called with a valid tokenizer."
        )

    # Try to get raw prompt from non_tensor_batch
    # Support both 'prompt' and 'raw_prompt' keys for backwards compatibility
    # (dataset may store as 'raw_prompt' when return_raw_chat=True)
    raw_prompts = data.non_tensor_batch.get(
        "prompt",
        data.non_tensor_batch.get("raw_prompt", None)
    )

    # Debug: log what keys are available in non_tensor_batch
    available_keys = list(data.non_tensor_batch.keys()) if hasattr(data.non_tensor_batch, 'keys') else []
    module_logger.info(f"[PREFIX_MASK] compute_prefix_mask: available keys in non_tensor_batch = {available_keys}")

    # Normalize raw_prompts if needed (handle numpy object arrays from collate_fn)
    if raw_prompts is not None and not isinstance(raw_prompts, list):
        if isinstance(raw_prompts, np.ndarray):
            # Convert numpy object array to list of lists
            raw_prompts = [list(p) for p in raw_prompts]
            module_logger.info(f"[PREFIX_MASK] compute_prefix_mask: converted numpy array to list, length = {len(raw_prompts)}")
        else:
            raw_prompts = list(raw_prompts)
            module_logger.info(f"[PREFIX_MASK] compute_prefix_mask: converted to list, length = {len(raw_prompts)}")

    if raw_prompts is None or len(raw_prompts) == 0:
        # Enhanced error message for debugging
        has_raw_prompt = "raw_prompt" in data.non_tensor_batch if hasattr(data.non_tensor_batch, '__contains__') else False
        raise ValueError(
            f"prompt field is missing or empty in batch.non_tensor_batch. "
            f"Cannot compute assistant prefix mask without prompt data.\n"
            f"  - Available keys: {available_keys}\n"
            f"  - Has 'raw_prompt': {has_raw_prompt}\n"
            f"  - Batch size: {attention_mask.shape[0] if attention_mask is not None else 'unknown'}\n"
            f"  - Attempted to find: 'prompt' or 'raw_prompt'"
        )

    module_logger.info(f"[PREFIX_MASK] compute_prefix_mask: successfully got raw_prompts, type = {type(raw_prompts)}, length = {len(raw_prompts)}")

    # Compute assistant token mask from ChatML format prompts
    batch_size = attention_mask.shape[0]
    device = attention_mask.device

    # FAIL FAST: If we cannot parse assistant token spans, raise error instead of fallback
    assistant_mask = compute_assistant_token_mask_from_prompt(
        raw_prompts=raw_prompts,
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        batch_size=batch_size,
        device=device
    )

    return assistant_mask


def parse_prefix_span(prefix_span) -> tuple[int, int]:
    """Normalize dataset prefix span metadata into an absolute [start, end) tuple."""
    if isinstance(prefix_span, dict):
        start = prefix_span.get("start", None)
        end = prefix_span.get("end", None)
    elif isinstance(prefix_span, (list, tuple)) and len(prefix_span) == 2:
        start, end = prefix_span
    elif isinstance(prefix_span, np.ndarray) and prefix_span.size == 2:
        start, end = prefix_span.reshape(-1).tolist()
    elif isinstance(prefix_span, torch.Tensor) and prefix_span.numel() == 2:
        start, end = prefix_span.reshape(-1).tolist()
    else:
        raise ValueError(f"Unsupported assistant_prefix_span format: {type(prefix_span)}")

    start = int(start)
    end = int(end)
    if start < 0 or end < start:
        raise ValueError(f"Invalid assistant_prefix_span [{start}, {end})")
    return start, end


def _as_python_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return value
    return list(value)


def _normalize_chat_messages(value) -> list[dict]:
    messages = _as_python_list(value)
    normalized = []
    for msg in messages:
        if hasattr(msg, "tolist"):
            msg = msg.tolist()
        if not isinstance(msg, dict):
            continue
        normalized.append({k: v for k, v in msg.items() if k in {"role", "content", "reasoning_content"}})
    return normalized


def _same_message(left: dict, right: dict) -> bool:
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("role") == right.get("role")
        and str(left.get("content", "")).strip() == str(right.get("content", "")).strip()
    )


def _is_raw_variant_value(value, variant_label=None) -> bool:
    if variant_label is not None and str(variant_label).lower() == "raw":
        return True
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "raw"}


def _config_get(config: Optional[AlgoConfig], primary: str, legacy: str | None = None, default=None):
    if config is None:
        return default
    value = config.get(primary, None)
    if value is not None:
        return value
    if legacy is not None:
        value = config.get(legacy, None)
        if value is not None:
            return value
    return default


def _config_enabled(config: Optional[AlgoConfig], primary: str, legacy: str | None = None) -> bool:
    value = _config_get(config, primary, legacy, False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _use_bc_aux(config: Optional[AlgoConfig]) -> bool:
    return _config_enabled(config, "use_bc_aux", "use_textcraft_bc_aux")


def _use_teacher_demo(config: Optional[AlgoConfig]) -> bool:
    return _config_enabled(config, "use_teacher_demo", "use_textcraft_teacher_demo")


def _textcraft_teacher_demo_labels(config: Optional[AlgoConfig]) -> set[str]:
    raw_labels = _config_get(config, "teacher_demo_labels", "textcraft_teacher_demo_labels", None)
    if raw_labels is None:
        raw_labels = ["teacher_demo", "demo"]
    if isinstance(raw_labels, str):
        labels = [item.strip() for item in raw_labels.split(",")]
    else:
        labels = [str(item).strip() for item in raw_labels]
    return {label.lower() for label in labels if label}


def _is_textcraft_teacher_demo_label(value, labels: set[str]) -> bool:
    return str(value).strip().lower() in labels


def split_textcraft_teacher_demo_batch(
    batch: DataProto,
    config: Optional[AlgoConfig],
) -> tuple[DataProto | None, DataProto | None, dict[str, float]]:
    if not _use_teacher_demo(config):
        return batch, None, {}
    if "variant_label" not in batch.non_tensor_batch:
        return batch, None, {"teacher_demo/source_rows": 0.0, "teacher_demo/split_available": 0.0}

    labels = _textcraft_teacher_demo_labels(config)
    variant_labels = np.asarray(batch.non_tensor_batch["variant_label"], dtype=object).reshape(-1)
    is_demo = np.asarray(
        [_is_textcraft_teacher_demo_label(value, labels) for value in variant_labels],
        dtype=bool,
    )
    demo_count = int(is_demo.sum())
    if demo_count == 0:
        return batch, None, {"teacher_demo/source_rows": 0.0, "teacher_demo/split_available": 1.0}
    if demo_count == len(is_demo):
        return None, batch, {
            "teacher_demo/source_rows": float(demo_count),
            "teacher_demo/online_source_rows": 0.0,
            "teacher_demo/split_available": 1.0,
            "teacher_demo/demo_only_source_batch": 1.0,
        }

    online_batch = batch.select_idxs(~is_demo)
    demo_batch = batch.select_idxs(is_demo)
    return online_batch, demo_batch, {
        "teacher_demo/source_rows": float(demo_count),
        "teacher_demo/online_source_rows": float(len(is_demo) - demo_count),
        "teacher_demo/split_available": 1.0,
        "teacher_demo/demo_only_source_batch": 0.0,
    }


def _as_1d_numpy(value, *, dtype, key: str, row_idx: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{key} for demo row {row_idx} must be 1-D, got shape={array.shape}.")
    return array


def _get_demo_sidecar(demo_source: DataProto, row_idx: int, names: Sequence[str], *, required: bool = True):
    for name in names:
        if name in demo_source.non_tensor_batch:
            return demo_source.non_tensor_batch[name][row_idx]
        if name in demo_source.batch.keys():
            return demo_source.batch[name][row_idx]
    if required:
        raise ValueError(f"Missing teacher demo sidecar; expected one of {list(names)}.")
    return None


def _compute_textcraft_raw_reward_means(batch: DataProto) -> dict[str, float]:
    if "sample_uid" not in batch.non_tensor_batch or "variant_label" not in batch.non_tensor_batch:
        return {}
    if "seq_level_rewards" not in batch.batch:
        return {}

    sample_uids = np.asarray(batch.non_tensor_batch["sample_uid"], dtype=object).reshape(-1)
    variant_labels = np.asarray(batch.non_tensor_batch["variant_label"], dtype=object).reshape(-1)
    raw_flags = np.asarray(
        batch.non_tensor_batch.get("is_raw_variant", np.zeros_like(sample_uids, dtype=object)),
        dtype=object,
    ).reshape(-1)
    rewards = batch.batch["seq_level_rewards"].detach().float().cpu().numpy().reshape(-1)

    uid_to_rewards: dict[str, list[float]] = defaultdict(list)
    for idx, sample_uid in enumerate(sample_uids):
        if _is_raw_variant_value(raw_flags[idx], variant_labels[idx]):
            uid_to_rewards[str(sample_uid)].append(float(rewards[idx]))
    return {
        sample_uid: float(np.mean(values))
        for sample_uid, values in uid_to_rewards.items()
        if values
    }


def _make_zero_like_for_demo(value: torch.Tensor, demo_count: int) -> torch.Tensor:
    shape = (demo_count, *tuple(value.shape[1:]))
    return torch.zeros(shape, dtype=value.dtype, device=value.device)


def _align_demo_batch_to_online_keys(online_batch: DataProto, demo_batch: DataProto) -> DataProto:
    demo_count = len(demo_batch)
    aligned_tensors = {}
    for key, online_value in online_batch.batch.items():
        if key in demo_batch.batch.keys():
            aligned_tensors[key] = demo_batch.batch[key]
        else:
            aligned_tensors[key] = _make_zero_like_for_demo(online_value, demo_count)

    aligned_non_tensors = {}
    for key, online_value in online_batch.non_tensor_batch.items():
        if key in demo_batch.non_tensor_batch:
            aligned_non_tensors[key] = demo_batch.non_tensor_batch[key]
        else:
            aligned_non_tensors[key] = np.array([None] * demo_count, dtype=object)

    return DataProto.from_dict(tensors=aligned_tensors, non_tensors=aligned_non_tensors, meta_info=demo_batch.meta_info)


def build_textcraft_teacher_demo_batch(
    demo_source: DataProto,
    online_batch: DataProto | None,
    *,
    config: Optional[AlgoConfig],
    pad_token_id: int,
    raw_means: Mapping[str, float] | None = None,
    response_length: int | None = None,
    prefix_width: int | None = None,
) -> tuple[DataProto | None, dict[str, float]]:
    current_raw_means = _compute_textcraft_raw_reward_means(online_batch) if online_batch is not None else {}
    if raw_means is None:
        raw_mean_map = dict(current_raw_means)
    else:
        raw_mean_map = {str(key): float(value) for key, value in raw_means.items()}
        raw_mean_map.update(current_raw_means)
    if not raw_mean_map:
        return None, {
            "teacher_demo/available": 0.0,
            "teacher_demo/raw_baseline_available": 0.0,
            "teacher_demo/kept_rows": 0.0,
        }

    if response_length is None:
        if online_batch is None:
            raise ValueError("response_length is required when building a demo-only TextCraft batch.")
        response_length = int(online_batch.batch["responses"].shape[1])
    response_length = int(response_length)

    device = (
        online_batch.batch["responses"].device
        if online_batch is not None
        else demo_source.batch["input_ids"].device
    )
    demo_weight = float(_config_get(config, "teacher_demo_weight", "textcraft_teacher_demo_weight", 1.0))
    skip_overlong = bool(_config_get(config, "teacher_demo_skip_overlong", "textcraft_teacher_demo_skip_overlong", False))

    source_count = len(demo_source)
    prompt_input_ids = demo_source.batch["input_ids"].to(device)
    prompt_attention_mask = demo_source.batch["attention_mask"].to(device)
    attention_dtype = (
        online_batch.batch["attention_mask"].dtype
        if online_batch is not None
        else prompt_attention_mask.dtype
    )

    kept_source_indices: list[int] = []
    response_id_rows: list[np.ndarray] = []
    response_attention_rows: list[np.ndarray] = []
    response_loss_rows: list[np.ndarray] = []
    old_logprob_rows: list[np.ndarray] = []
    reward_values: list[float] = []
    raw_mean_values: list[float] = []
    advantage_values: list[float] = []
    missing_raw = 0
    overlong = 0

    sample_uids = np.asarray(demo_source.non_tensor_batch.get("sample_uid", []), dtype=object).reshape(-1)
    for row_idx in range(source_count):
        sample_uid = str(sample_uids[row_idx]) if row_idx < len(sample_uids) else ""
        raw_mean = raw_mean_map.get(sample_uid)
        if raw_mean is None:
            missing_raw += 1
            continue

        response_ids = _as_1d_numpy(
            _get_demo_sidecar(demo_source, row_idx, ["teacher_demo_response_ids", "demo_response_ids"]),
            dtype=np.int64,
            key="teacher_demo_response_ids",
            row_idx=row_idx,
        )
        response_attention = _get_demo_sidecar(
            demo_source,
            row_idx,
            [
                "teacher_demo_response_attention_mask",
                "teacher_demo_attention_mask",
                "demo_response_attention_mask",
                "demo_attention_mask",
            ],
            required=False,
        )
        if response_attention is None:
            response_attention_array = np.ones_like(response_ids, dtype=np.int64)
        else:
            response_attention_array = _as_1d_numpy(
                response_attention,
                dtype=np.int64,
                key="teacher_demo_response_attention_mask",
                row_idx=row_idx,
            )

        response_loss_mask = _as_1d_numpy(
            _get_demo_sidecar(
                demo_source,
                row_idx,
                [
                    "teacher_demo_response_loss_mask",
                    "teacher_demo_loss_mask",
                    "demo_response_loss_mask",
                    "demo_loss_mask",
                ],
            ),
            dtype=np.float32,
            key="teacher_demo_response_loss_mask",
            row_idx=row_idx,
        )
        old_log_probs = _as_1d_numpy(
            _get_demo_sidecar(
                demo_source,
                row_idx,
                [
                    "teacher_demo_old_log_probs",
                    "teacher_demo_response_old_log_probs",
                    "demo_old_log_probs",
                    "demo_response_old_log_probs",
                ],
            ),
            dtype=np.float32,
            key="teacher_demo_old_log_probs",
            row_idx=row_idx,
        )
        reward = float(
            _get_demo_sidecar(
                demo_source,
                row_idx,
                ["teacher_demo_reward", "demo_reward"],
            )
        )

        lengths = {
            "response_ids": len(response_ids),
            "response_attention_mask": len(response_attention_array),
            "response_loss_mask": len(response_loss_mask),
            "old_log_probs": len(old_log_probs),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"teacher demo row {row_idx} sidecar length mismatch: {lengths}.")
        if len(response_ids) > response_length:
            overlong += 1
            if skip_overlong:
                continue
            raise ValueError(
                f"teacher demo row {row_idx} has response length {len(response_ids)} "
                f"> configured response length {response_length}; refusing to truncate logged demo."
            )
        if float(response_loss_mask.sum()) <= 0:
            raise ValueError(f"teacher demo row {row_idx} has no assistant loss tokens.")

        advantage = float(np.clip(reward - raw_mean, 0.0, 1.0))
        kept_source_indices.append(row_idx)
        response_id_rows.append(response_ids)
        response_attention_rows.append(response_attention_array)
        response_loss_rows.append(response_loss_mask)
        old_logprob_rows.append(old_log_probs)
        reward_values.append(reward)
        raw_mean_values.append(raw_mean)
        advantage_values.append(advantage)

    kept_count = len(kept_source_indices)
    if kept_count == 0:
        return None, {
            "teacher_demo/available": 1.0,
            "teacher_demo/raw_baseline_available": 1.0,
            "teacher_demo/source_rows": float(source_count),
            "teacher_demo/kept_rows": 0.0,
            "teacher_demo/missing_raw_baseline_rows": float(missing_raw),
            "teacher_demo/overlong_rows": float(overlong),
        }

    responses = torch.full((kept_count, response_length), int(pad_token_id), dtype=torch.long, device=device)
    response_attention = torch.zeros((kept_count, response_length), dtype=attention_dtype, device=device)
    response_mask = torch.zeros((kept_count, response_length), dtype=torch.float32, device=device)
    old_log_probs = torch.zeros((kept_count, response_length), dtype=torch.float32, device=device)
    token_level_scores = torch.zeros((kept_count, response_length), dtype=torch.float32, device=device)

    for out_idx, response_ids in enumerate(response_id_rows):
        valid_len = len(response_ids)
        responses[out_idx, :valid_len] = torch.as_tensor(response_ids, dtype=torch.long, device=device)
        response_attention[out_idx, :valid_len] = torch.as_tensor(
            response_attention_rows[out_idx],
            dtype=response_attention.dtype,
            device=device,
        )
        response_mask[out_idx, :valid_len] = torch.as_tensor(
            response_loss_rows[out_idx],
            dtype=torch.float32,
            device=device,
        )
        old_log_probs[out_idx, :valid_len] = torch.as_tensor(
            old_logprob_rows[out_idx],
            dtype=torch.float32,
            device=device,
        )
        valid_positions = np.nonzero(response_attention_rows[out_idx] > 0)[0]
        if len(valid_positions) > 0:
            token_level_scores[out_idx, int(valid_positions[-1])] = float(reward_values[out_idx])

    kept_indices_tensor = torch.as_tensor(kept_source_indices, dtype=torch.long, device=prompt_input_ids.device)
    prompt_input_ids = prompt_input_ids.index_select(0, kept_indices_tensor)
    prompt_attention_mask = prompt_attention_mask.index_select(0, kept_indices_tensor).to(response_attention.dtype)
    input_ids = torch.cat([prompt_input_ids, responses], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention], dim=1)
    position_ids = compute_position_id_with_mask(attention_mask)

    demo_advantages = torch.as_tensor(advantage_values, dtype=torch.float32, device=device).reshape(-1, 1)
    advantages = demo_advantages * float(demo_weight) * response_mask
    returns = advantages.clone()
    token_level_rewards = token_level_scores.clone()
    seq_level_rewards = torch.as_tensor(reward_values, dtype=torch.float32, device=device)

    if prefix_width is None:
        prefix_width = (
            int(online_batch.batch["prefix_mask"].shape[1])
            if online_batch is not None and "prefix_mask" in online_batch.batch.keys()
            else 0
        )
    tensors = {
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "returns": returns,
        "token_level_scores": token_level_scores,
        "token_level_rewards": token_level_rewards,
        "seq_level_rewards": seq_level_rewards,
        "is_textcraft_teacher_demo": torch.ones((kept_count,), dtype=torch.float32, device=device),
        "is_teacher_demo": torch.ones((kept_count,), dtype=torch.float32, device=device),
    }
    include_prefix_tensors = prefix_width > 0 or bool(config and config.get("optimize_prefix_tokens", False))
    if include_prefix_tensors:
        tensors["prefix_mask"] = torch.zeros((kept_count, prefix_width), dtype=torch.float32, device=device)
        tensors["assistant_prefix_old_log_probs"] = torch.zeros((kept_count, prefix_width), dtype=torch.float32, device=device)
        tensors["prefix_token_count"] = torch.zeros((kept_count,), dtype=torch.long, device=device)
        tensors["assistant_prefix_span"] = torch.zeros((kept_count, 2), dtype=torch.long, device=device)

    non_tensors = {}
    kept_indices_np = np.asarray(kept_source_indices, dtype=np.int64)
    for key, value in demo_source.non_tensor_batch.items():
        non_tensors[key] = value[kept_indices_np]
    if "variant_label" in non_tensors:
        non_tensors["variant_label"] = np.array(["teacher_demo"] * kept_count, dtype=object)
    if "is_raw_variant" in non_tensors:
        non_tensors["is_raw_variant"] = np.array([False] * kept_count, dtype=object)

    demo_batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={})
    metrics = {
        "teacher_demo/available": 1.0,
        "teacher_demo/source_rows": float(source_count),
        "teacher_demo/kept_rows": float(kept_count),
        "teacher_demo/coverage_rate": float(kept_count / max(source_count, 1)),
        "teacher_demo/missing_raw_baseline_rows": float(missing_raw),
        "teacher_demo/overlong_rows": float(overlong),
        "teacher_demo/reward_mean": float(np.mean(reward_values)),
        "teacher_demo/raw_mean": float(np.mean(raw_mean_values)),
        "teacher_demo/advantage_mean": float(np.mean(advantage_values)),
        "teacher_demo/advantage_positive_rate": float(np.mean(np.asarray(advantage_values) > 0.0)),
        "teacher_demo/loss_weight": float(demo_weight),
    }
    return demo_batch, metrics


def _textcraft_bc_should_train(source: str, is_raw: bool) -> bool:
    source = str(source).strip().lower()
    if source == "all":
        return True
    if source in {"prefix", "non_raw", "prefix_only"}:
        return not is_raw
    if source in {"raw", "raw_only"}:
        return is_raw
    raise ValueError(f"Unsupported bc_source={source!r}; expected all/prefix/raw.")


def _build_textcraft_bc_conversation(prompt, continuation_messages, *, source: str, is_raw: bool) -> list[dict]:
    if not _textcraft_bc_should_train(source, is_raw):
        return []

    prompt_messages = _normalize_chat_messages(prompt)
    continuation = _normalize_chat_messages(continuation_messages)
    if not continuation:
        return []

    # Prefix rows store the cut-state user observation both at the prompt tail and as the first
    # continuation message. Drop the duplicate before tokenizing the BC target.
    if prompt_messages and continuation and _same_message(prompt_messages[-1], continuation[0]):
        continuation = continuation[1:]
    if not any(msg.get("role") == "assistant" for msg in continuation):
        return []

    messages: list[dict] = []
    for msg in prompt_messages:
        item = dict(msg)
        item["_bc_train"] = False
        messages.append(item)
    for msg in continuation:
        item = dict(msg)
        item["_bc_train"] = item.get("role") == "assistant"
        messages.append(item)
    return messages


def _encode_textcraft_bc_messages(tokenizer, messages: list[dict], max_length: int, apply_chat_template_kwargs: dict):
    if not messages:
        return None

    tokens: list[int] = []
    loss_mask: list[int] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        clean_messages = [{k: v for k, v in item.items() if not k.startswith("_")} for item in messages]
        prev_messages = clean_messages[:idx]
        cur_messages = clean_messages[: idx + 1]
        prev_text = (
            tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=False,
                **apply_chat_template_kwargs,
            )
            if idx > 0
            else ""
        )
        cur_text = tokenizer.apply_chat_template(
            cur_messages,
            tokenize=False,
            add_generation_prompt=False,
            **apply_chat_template_kwargs,
        )

        if role == "assistant":
            prev_gen_text = tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=True,
                **apply_chat_template_kwargs,
            )
            gen_prompt_text = prev_gen_text[len(prev_text) :]
            assistant_text = cur_text[len(prev_gen_text) :]
            gen_prompt_tokens = tokenizer.encode(gen_prompt_text, add_special_tokens=False)
            assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
            tokens.extend(gen_prompt_tokens)
            loss_mask.extend([0] * len(gen_prompt_tokens))
            tokens.extend(assistant_tokens)
            loss_mask.extend([1 if msg.get("_bc_train", False) else 0] * len(assistant_tokens))
        else:
            msg_tokens = tokenizer.encode(cur_text[len(prev_text) :], add_special_tokens=False)
            tokens.extend(msg_tokens)
            loss_mask.extend([0] * len(msg_tokens))

    if len(tokens) < 2 or sum(loss_mask) == 0:
        return None
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        loss_mask = loss_mask[:max_length]
    if len(tokens) < 2 or sum(loss_mask) == 0:
        return None

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    attention_mask = [1] * len(tokens)
    if len(tokens) < max_length:
        pad_len = max_length - len(tokens)
        tokens = tokens + [pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        loss_mask = loss_mask + [0] * pad_len
    position_ids = [idx if mask else 0 for idx, mask in enumerate(attention_mask)]
    return tokens, attention_mask, position_ids, loss_mask


def build_textcraft_bc_aux_tensors(
    restore_non_tensor_batch: dict,
    tokenizer,
    *,
    max_length: int,
    source: str,
    apply_chat_template_kwargs: dict,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    prompts = restore_non_tensor_batch.get("prompt", restore_non_tensor_batch.get("raw_prompt", None))
    continuations = restore_non_tensor_batch.get("continuation_messages", None)
    if prompts is None or continuations is None:
        return {}, {"bc/build_available": 0.0, "textcraft_bc/build_available": 0.0}

    prompts = _as_python_list(prompts)
    continuations = _as_python_list(continuations)
    variant_labels = _as_python_list(restore_non_tensor_batch.get("variant_label", [None] * len(prompts)))
    raw_flags = _as_python_list(restore_non_tensor_batch.get("is_raw_variant", [False] * len(prompts)))
    max_length = max(int(max_length), 2)

    encoded_rows = []
    valid_rows = 0
    valid_tokens = 0
    for idx, prompt in enumerate(prompts):
        variant_label = variant_labels[idx] if idx < len(variant_labels) else None
        is_raw = _is_raw_variant_value(raw_flags[idx] if idx < len(raw_flags) else False, variant_label)
        continuation = continuations[idx] if idx < len(continuations) else []
        messages = _build_textcraft_bc_conversation(prompt, continuation, source=source, is_raw=is_raw)
        encoded = _encode_textcraft_bc_messages(tokenizer, messages, max_length, apply_chat_template_kwargs)
        if encoded is None:
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            encoded = (
                [pad_token_id] * max_length,
                [0] * max_length,
                [0] * max_length,
                [0] * max_length,
            )
        else:
            valid_rows += 1
            valid_tokens += int(sum(encoded[3]))
        encoded_rows.append(encoded)

    input_ids, attention_mask, position_ids, loss_mask = zip(*encoded_rows, strict=True)
    tensors = {
        "bc_input_ids": torch.tensor(input_ids, dtype=torch.long),
        "bc_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "bc_position_ids": torch.tensor(position_ids, dtype=torch.long),
        "bc_loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
    }
    metrics = {
        "bc/build_available": 1.0,
        "bc/build_rows_with_loss": float(valid_rows),
        "bc/build_loss_tokens": float(valid_tokens),
        "bc/build_source_prefix": float(str(source).strip().lower() in {"prefix", "non_raw", "prefix_only"}),
        "bc/build_source_raw": float(str(source).strip().lower() in {"raw", "raw_only"}),
        "textcraft_bc/build_available": 1.0,
        "textcraft_bc/build_rows_with_loss": float(valid_rows),
        "textcraft_bc/build_loss_tokens": float(valid_tokens),
        "textcraft_bc/build_source_prefix": float(str(source).strip().lower() in {"prefix", "non_raw", "prefix_only"}),
        "textcraft_bc/build_source_raw": float(str(source).strip().lower() in {"raw", "raw_only"}),
    }
    return tensors, metrics


def compute_prefix_family_lift_advantage(
    batch: DataProto,
    *,
    clip_value: float = 1.0,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    required_keys = {"sample_uid", "variant_label"}
    if not required_keys.issubset(batch.non_tensor_batch.keys()) or "seq_level_rewards" not in batch.batch:
        return None, {"family_lift/available": 0.0}

    sample_uids = np.asarray(batch.non_tensor_batch["sample_uid"], dtype=object).reshape(-1)
    variant_labels = np.asarray(batch.non_tensor_batch["variant_label"], dtype=object).reshape(-1)
    raw_flags = np.asarray(
        batch.non_tensor_batch.get("is_raw_variant", np.zeros_like(sample_uids, dtype=object)),
        dtype=object,
    ).reshape(-1)
    rewards = batch.batch["seq_level_rewards"].detach().float().cpu().numpy().reshape(-1)
    if sample_uids.shape[0] != rewards.shape[0]:
        return None, {"family_lift/available": 0.0, "family_lift/shape_mismatch": 1.0}

    success = (rewards > 0).astype(np.float32)
    lift = np.zeros_like(success, dtype=np.float32)
    covered = np.zeros_like(success, dtype=np.float32)
    prefix_mask = np.zeros_like(success, dtype=np.float32)
    family_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, sample_uid in enumerate(sample_uids):
        family_to_indices[str(sample_uid)].append(idx)

    family_count = 0
    family_with_raw = 0
    family_with_prefix = 0
    family_complete = 0
    per_variant_lifts: dict[str, list[float]] = defaultdict(list)
    for indices in family_to_indices.values():
        family_count += 1
        raw_indices = [
            idx for idx in indices
            if _is_raw_variant_value(raw_flags[idx], variant_labels[idx])
        ]
        variant_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in indices:
            variant = str(variant_labels[idx])
            if variant != "raw" and not _is_raw_variant_value(raw_flags[idx], variant):
                variant_to_indices[variant].append(idx)
                prefix_mask[idx] = 1.0
        if raw_indices:
            family_with_raw += 1
        if variant_to_indices:
            family_with_prefix += 1
        if raw_indices and len(variant_to_indices) >= 3:
            family_complete += 1
        if not raw_indices:
            continue
        raw_avg = float(success[raw_indices].mean())
        for variant, variant_indices in variant_to_indices.items():
            prefix_avg = float(success[variant_indices].mean())
            value = prefix_avg - raw_avg
            if float(clip_value) > 0:
                value = max(-float(clip_value), min(float(clip_value), value))
            for idx in variant_indices:
                lift[idx] = value
                covered[idx] = 1.0
            per_variant_lifts[variant].append(value)

    prefix_count = float(prefix_mask.sum())
    covered_count = float((covered * prefix_mask).sum())
    lift_tensor = torch.from_numpy(lift).to(batch.batch["seq_level_rewards"].device)
    metrics = {
        "family_lift/available": 1.0,
        "family_lift/family_count": float(family_count),
        "family_lift/family_with_raw_rate": float(family_with_raw / max(family_count, 1)),
        "family_lift/family_with_prefix_rate": float(family_with_prefix / max(family_count, 1)),
        "family_lift/family_complete_rate": float(family_complete / max(family_count, 1)),
        "family_lift/prefix_coverage_rate": float(covered_count / max(prefix_count, 1.0)),
        "family_lift/mean": float(lift[covered > 0.5].mean()) if covered_count > 0 else 0.0,
        "family_lift/positive_rate": float((lift[covered > 0.5] > 0).mean()) if covered_count > 0 else 0.0,
    }
    for variant, values in per_variant_lifts.items():
        safe_variant = variant.replace("/", "_")
        metrics[f"family_lift/{safe_variant}_mean"] = float(np.mean(values))
    return lift_tensor, metrics


def compute_assistant_token_mask_from_prompt(raw_prompts, tokenizer, prompt_length, batch_size, device):
    """Compute mask for assistant role tokens in ChatML format prompts.

    This function parses the ChatML format prompt and identifies which tokens
    correspond to assistant role messages (excluding system and user).

    Args:
        raw_prompts: List of prompts (each is a list of chat messages with 'role' and 'content')
        tokenizer: Tokenizer for processing
        prompt_length: Expected prompt length
        batch_size: Batch size
        device: Device to place the mask tensor

    Returns:
        torch.Tensor: Mask with 1 for assistant tokens, 0 for others
    """
    mask = torch.zeros((batch_size, prompt_length), dtype=torch.float32, device=device)

    for i, raw_prompt in enumerate(raw_prompts):
        if i >= batch_size:
            break

        try:
            # raw_prompt is a numpy array or list of message dicts
            if hasattr(raw_prompt, 'tolist'):
                chat_messages = raw_prompt.tolist()
            else:
                chat_messages = list(raw_prompt)

            if not isinstance(chat_messages, (list, tuple)) or len(chat_messages) == 0:
                continue

            # Check if it's in chat format (list of dicts with 'role' and 'content')
            if isinstance(chat_messages[0], dict) and 'role' in chat_messages[0]:
                # Apply chat template to get full text
                full_text = tokenizer.apply_chat_template(
                    chat_messages,
                    add_generation_prompt=False,
                    tokenize=False
                )

                # Tokenize to get token IDs
                tokens = tokenizer(full_text, add_special_tokens=True, return_tensors="pt")
                token_ids = tokens.input_ids[0]

                # Now find assistant token positions using the chat template structure
                # We need to identify where assistant messages start and end
                assistant_start_pos = 0

                for msg in chat_messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')

                    # Tokenize role prefix (e.g., "<|im_start|>assistant\n")
                    role_text = f"<|im_start|>{role}\n"
                    role_tokens = tokenizer(role_text, add_special_tokens=False).input_ids
                    role_len = len(role_tokens)

                    # Tokenize content + end token
                    content_text = f"{content}<|im_end|>\n"
                    content_tokens = tokenizer(content_text, add_special_tokens=False).input_ids
                    content_len = len(content_tokens)

                    if role == 'assistant':
                        # Mark assistant tokens (role + content, excluding the trailing <|im_end|>\n)
                        # Actually, we need to include the content but not the <|im_end|> as it belongs to next message
                        # The role tokens + content tokens (excluding trailing <|im_end|> are assistant tokens)
                        # But simpler: we mark from role start to content end

                        assistant_end_pos = assistant_start_pos + role_len + content_len

                        # Ensure we don't exceed prompt_length
                        end_idx = min(assistant_end_pos, prompt_length)
                        start_idx = min(assistant_start_pos, prompt_length)

                        if start_idx < end_idx:
                            mask[i, start_idx:end_idx] = 1.0

                        assistant_start_pos = assistant_end_pos
                    else:
                        # For system/user, just advance past them
                        assistant_start_pos += role_len + content_len
        except Exception as e:
            # Skip this sample if there's an error
            continue

    # FAIL FAST: If no assistant tokens found, raise error
    # Check if any sample has assistant tokens
    if mask.sum() == 0:
        raise ValueError(
            "No assistant tokens found in any prompts. "
            "Please ensure prompt contains at least one assistant message with role='assistant'."
        )

    return mask


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_module_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured module_logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured module_logger
        self.validation_generations_module_logger.log(self.config.trainer.module_logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        metrics_csv_writer = MetricsCSVWriter(
            output_dir=self.config.trainer.default_local_dir,
            frequency=self.config.trainer.get("metrics_csv_freq", 50),
            filename=self.config.trainer.get("metrics_csv_filename", "training_metrics.csv"),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            tracking_logger.log(data=val_metrics, step=self.global_steps)
            metrics_csv_writer.maybe_log(val_metrics, step=self.global_steps, force=True)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
        textcraft_teacher_demo_raw_means: dict[str, float] = {}

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.meta_info["logprob_temperature"] = 1.0

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                pending_teacher_demo_batch = None
                if _use_teacher_demo(self.config.algorithm):
                    batch, pending_teacher_demo_batch, demo_split_metrics = split_textcraft_teacher_demo_batch(
                        batch,
                        self.config.algorithm,
                    )
                    metrics.update(demo_split_metrics)

                    if batch is None:
                        if self.use_critic:
                            raise ValueError("teacher demo training currently supports critic-free GRPO.")
                        is_last_step = self.global_steps >= self.total_training_steps
                        reward_extra_infos_dict = {}
                        with marked_timer("step", timing_raw):
                            with marked_timer("adv", timing_raw, color="brown"):
                                demo_batch, demo_metrics = build_textcraft_teacher_demo_batch(
                                    pending_teacher_demo_batch,
                                    None,
                                    config=self.config.algorithm,
                                    pad_token_id=(
                                        self.tokenizer.pad_token_id
                                        if self.tokenizer.pad_token_id is not None
                                        else 0
                                    ),
                                    raw_means=textcraft_teacher_demo_raw_means,
                                    response_length=int(self.config.data.max_response_length),
                                    prefix_width=0,
                                )
                                metrics.update(demo_metrics)
                                metrics["teacher_demo/raw_baseline_cache_size"] = float(
                                    len(textcraft_teacher_demo_raw_means)
                                )

                            if demo_batch is not None:
                                demo_repeat_times = (
                                    int(self.config.actor_rollout_ref.rollout.n)
                                    if _config_get(
                                        self.config.algorithm,
                                        "teacher_demo_repeat_to_rollout_n",
                                        "textcraft_teacher_demo_repeat_to_rollout_n",
                                        True,
                                    )
                                    else 1
                                )
                                if demo_repeat_times > 1:
                                    demo_batch = demo_batch.repeat(repeat_times=demo_repeat_times, interleave=True)
                                metrics["teacher_demo/training_rows"] = float(len(demo_batch))
                                batch = demo_batch
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                batch.meta_info["logprob_temperature"] = 1.0
                                batch.meta_info["global_token_num"] = torch.sum(
                                    batch.batch["attention_mask"], dim=-1
                                ).tolist()

                                if self.config.algorithm.get("optimize_prefix_tokens", False):
                                    batch.meta_info["optimize_prefix_tokens"] = True
                                    batch.meta_info["prefix_loss_weight"] = self.config.algorithm.get(
                                        "prefix_loss_weight", 1.0
                                    )
                                    batch.meta_info["prefix_loss_mode"] = self.config.algorithm.get(
                                        "prefix_loss_mode", "split"
                                    )
                                    batch.meta_info["prefix_advantage_mode"] = self.config.algorithm.get(
                                        "prefix_advantage_mode", "cont_mean"
                                    )
                                    batch.meta_info["prefix_advantage_constant"] = self.config.algorithm.get(
                                        "prefix_advantage_constant", 1.0
                                    )
                                    batch.meta_info["prefix_cont_adv_weight"] = self.config.algorithm.get(
                                        "prefix_cont_adv_weight", 1.0
                                    )
                                    batch.meta_info["prefix_family_lift_weight"] = self.config.algorithm.get(
                                        "prefix_family_lift_weight", 1.0
                                    )
                                    batch.meta_info["prefix_family_lift_clip"] = self.config.algorithm.get(
                                        "prefix_family_lift_clip", 1.0
                                    )

                                with marked_timer("update_actor", timing_raw, color="red"):
                                    actor_output = self.actor_rollout_wg.update_actor(batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(actor_output_metrics)
                            else:
                                metrics["teacher_demo/demo_only_skipped_no_raw_baseline"] = 1.0

                        with marked_timer("stop_profile", timing_raw):
                            next_step_profile = (
                                self.global_steps + 1 in self.config.global_profiler.steps
                                if self.config.global_profiler.steps is not None
                                else False
                            )
                            self._stop_profiling(
                                curr_step_profile and not next_step_profile
                                if self.config.global_profiler.profile_continuous_steps
                                else curr_step_profile
                            )
                            prev_step_profile = curr_step_profile
                            curr_step_profile = next_step_profile

                        metrics.update(
                            {
                                "training/global_step": self.global_steps,
                                "training/epoch": epoch,
                            }
                        )
                        if demo_batch is not None:
                            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                            n_gpus = self.resource_pool_manager.get_n_gpus()
                            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                        else:
                            metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})

                        tracking_logger.log(data=metrics, step=self.global_steps)
                        metrics_csv_writer.maybe_log(metrics, step=self.global_steps, force=is_last_step)
                        progress_bar.update(1)
                        self.global_steps += 1
                        if is_last_step:
                            if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                                self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                            progress_bar.close()
                            return
                        continue

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                # Preserve the repeated pre-rollout non-tensor sidecars because the rollout output
                # replaces non_tensor_batch with reward/interaction fields only.
                restore_non_tensor_batch = gen_batch_output.non_tensor_batch.copy()

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # Repeat the original batch first so non-tensor prefix sidecars are duplicated
                    # consistently with rollout.n, then materialize them into batch.batch before union.
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    if self.config.algorithm.get("optimize_prefix_tokens", False):
                        prefix_keys_to_restore = [
                            "assistant_prefix_old_log_probs",
                            "assistant_prefix_old_logprobs",
                            "prefix_token_count",
                            "prefix_mask",
                            "assistant_prefix_span",
                            "prompt",
                            "raw_prompt",
                        ]
                        for key in prefix_keys_to_restore:
                            if key in restore_non_tensor_batch:
                                batch.batch[key] = restore_non_tensor_batch[key]

                    batch = batch.union(gen_batch_output)

                    for key in ["sample_uid", "variant_label", "is_raw_variant", "record_uid"]:
                        if key in restore_non_tensor_batch:
                            batch.non_tensor_batch[key] = restore_non_tensor_batch[key]

                    if _use_bc_aux(self.config.algorithm):
                        bc_max_length = _config_get(
                            self.config.algorithm,
                            "bc_max_length",
                            "textcraft_bc_max_length",
                            None,
                        )
                        if bc_max_length is None:
                            bc_max_length = self.config.actor_rollout_ref.rollout.get(
                                "max_model_len",
                                self.config.data.max_prompt_length + self.config.data.max_response_length,
                            )
                        bc_tensors, bc_build_metrics = build_textcraft_bc_aux_tensors(
                            restore_non_tensor_batch,
                            self.tokenizer,
                            max_length=int(bc_max_length),
                            source=_config_get(
                                self.config.algorithm,
                                "bc_source",
                                "textcraft_bc_source",
                                "prefix",
                            ),
                            apply_chat_template_kwargs=self.config.data.get("apply_chat_template_kwargs", {}),
                        )
                        for key, value in bc_tensors.items():
                            batch.batch[key] = value
                        metrics.update(bc_build_metrics)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # Compute prefix_mask when optimize_prefix_tokens is enabled
                    # This is used to include prefix tokens in the GRPO loss
                    if self.config.algorithm.get("optimize_prefix_tokens", False):
                        if "prefix_mask" not in batch.batch.keys():
                            raise ValueError(
                                "prefix optimization requires dataset-provided prefix_mask under the rebuilt "
                                "stage7 data contract; runtime compute_prefix_mask fallback is disabled."
                            )

                        prefix_logprobs_key = None
                        for key in ["assistant_prefix_old_log_probs", "assistant_prefix_old_logprobs"]:
                            if key in batch.batch:
                                prefix_logprobs_key = key
                                break

                        if prefix_logprobs_key is None:
                            raise ValueError(
                                "prefix optimization requires cached SFT old logprobs. "
                                "Missing 'assistant_prefix_old_log_probs' in the repeated training batch."
                            )
                        if "assistant_prefix_span" not in batch.batch:
                            raise ValueError(
                                "prefix optimization requires 'assistant_prefix_span' under the rebuilt stage7 "
                                "data contract."
                            )

                        device = batch.batch["attention_mask"].device
                        raw_prefix_mask = batch.batch["prefix_mask"]
                        cached_olp = batch.batch[prefix_logprobs_key]
                        raw_prefix_span = batch.batch["assistant_prefix_span"]

                        if "prefix_token_count" in batch.batch:
                            ptc = batch.batch["prefix_token_count"]
                            if isinstance(ptc, torch.Tensor):
                                prefix_lens = ptc.detach().cpu().numpy().astype(np.int64)
                            else:
                                prefix_lens = np.asarray(ptc, dtype=np.int64)
                        else:
                            if isinstance(raw_prefix_mask, torch.Tensor):
                                prefix_lens = raw_prefix_mask.sum(dim=1).detach().cpu().numpy().astype(np.int64)
                            else:
                                prefix_lens = np.asarray(
                                    [int(np.asarray(raw_prefix_mask[b], dtype=np.float32).sum()) for b in range(len(raw_prefix_mask))],
                                    dtype=np.int64,
                                )
                        prefix_lens = np.asarray(prefix_lens, dtype=np.int64).reshape(-1)

                        batch_size = len(prefix_lens)
                        prefix_spans = np.zeros((batch_size, 2), dtype=np.int64)
                        prefix_window_lens = np.zeros((batch_size,), dtype=np.int64)
                        for b in range(batch_size):
                            span_start, span_end = parse_prefix_span(raw_prefix_span[b])
                            prefix_spans[b, 0] = span_start
                            prefix_spans[b, 1] = span_end
                            prefix_window_lens[b] = span_end - span_start

                        if isinstance(cached_olp, torch.Tensor):
                            cached_prefix_old_log_probs = cached_olp.float().to(device)
                            max_prefix_window_len = int(prefix_window_lens.max()) if batch_size > 0 else 0
                            if cached_prefix_old_log_probs.shape[1] < max_prefix_window_len:
                                raise ValueError(
                                    "[PREFIX_OPT] cached old_logprobs tensor width is smaller than the required "
                                    "prefix window length."
                                )
                        else:
                            max_prefix_window_len = int(prefix_window_lens.max()) if batch_size > 0 else 0
                            dense_olp = np.zeros((batch_size, max_prefix_window_len), dtype=np.float32)
                            for b in range(batch_size):
                                sample_lp = np.asarray(cached_olp[b], dtype=np.float32).reshape(-1)
                                expected_len = int(prefix_window_lens[b])
                                if sample_lp.shape[0] != expected_len:
                                    raise ValueError(
                                        f"[PREFIX_OPT] sample {b}: cached old_logprobs len={sample_lp.shape[0]} "
                                        f"!= prefix_window_len={expected_len}."
                                    )
                                dense_olp[b, :expected_len] = sample_lp
                            cached_prefix_old_log_probs = torch.from_numpy(dense_olp).to(device)

                        if isinstance(raw_prefix_mask, torch.Tensor):
                            prefix_mask = raw_prefix_mask.float().to(device)
                            max_prefix_window_len = int(prefix_window_lens.max()) if batch_size > 0 else 0
                            if prefix_mask.shape[1] < max_prefix_window_len:
                                raise ValueError(
                                    "[PREFIX_OPT] prefix_mask tensor width is smaller than the required "
                                    "prefix window length."
                                )
                            for b in range(batch_size):
                                sample_mask = prefix_mask[b, : int(prefix_window_lens[b])]
                                if int(sample_mask.sum().item()) != int(prefix_lens[b]):
                                    raise ValueError(
                                        f"[PREFIX_OPT] sample {b}: prefix_mask.sum()={int(sample_mask.sum().item())} "
                                        f"!= prefix_token_count={int(prefix_lens[b])}."
                                    )
                        else:
                            prefix_mask_lens = np.asarray([len(raw_prefix_mask[b]) for b in range(batch_size)], dtype=np.int64)
                            max_prefix_window_len = int(prefix_mask_lens.max()) if batch_size > 0 else 0
                            dense_prefix_mask = np.zeros((batch_size, max_prefix_window_len), dtype=np.float32)
                            for b in range(batch_size):
                                sample_mask = np.asarray(raw_prefix_mask[b], dtype=np.float32).reshape(-1)
                                expected_window_len = int(prefix_window_lens[b])
                                if sample_mask.shape[0] != expected_window_len:
                                    raise ValueError(
                                        f"[PREFIX_OPT] sample {b}: prefix_mask len={sample_mask.shape[0]} "
                                        f"!= prefix_window_len={expected_window_len}."
                                    )
                                dense_prefix_mask[b, : sample_mask.shape[0]] = sample_mask
                                if int(sample_mask.sum()) != int(prefix_lens[b]):
                                    raise ValueError(
                                        f"[PREFIX_OPT] sample {b}: prefix_mask.sum()={int(sample_mask.sum())} "
                                        f"!= prefix_token_count={int(prefix_lens[b])}."
                                    )
                            prefix_mask = torch.from_numpy(dense_prefix_mask).to(device)

                        batch.batch["assistant_prefix_old_log_probs"] = cached_prefix_old_log_probs
                        batch.batch["prefix_mask"] = prefix_mask
                        batch.batch["prefix_token_count"] = torch.from_numpy(prefix_lens).to(device)
                        batch.batch["assistant_prefix_span"] = torch.from_numpy(prefix_spans).to(device)

                        metrics["actor/use_cached_prefix_old_logprob"] = True
                        metrics["actor/prefix_mask_sum"] = int(prefix_mask.sum().item())
                    
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                        actor_config = self.config.actor_rollout_ref.actor
                        if actor_config.get("calculate_entropy", False):
                            # MIS/TIS bypass keeps rollout old_log_probs for training, but we may still
                            # want actor entropy as a pure diagnostic metric.
                            with marked_timer("old_log_prob_entropy", timing_raw, color="blue"):
                                entropy_output = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = entropy_output.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                metrics.update({"actor/entropy": entropy_agg.detach().item()})
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute seq_level_rewards for DRPO
                        # DRPO requires uid and seq_level_rewards to compute the loss
                        # This is safe for GRPO/PPO as they don't use these fields
                        batch.batch['seq_level_rewards'] = batch.batch['token_level_scores'].sum(dim=-1)
                        if 'uid' in batch.non_tensor_batch:
                            # Keep uid as numpy array of strings - GRPO can use it directly for grouping
                            batch.batch['uid'] = np.array(
                                [str(u) for u in batch.non_tensor_batch['uid']], dtype=object
                            )
                        if _use_teacher_demo(self.config.algorithm):
                            current_raw_means = _compute_textcraft_raw_reward_means(batch)
                            textcraft_teacher_demo_raw_means.update(current_raw_means)
                            metrics["teacher_demo/raw_baseline_cache_updates"] = float(len(current_raw_means))
                            metrics["teacher_demo/raw_baseline_cache_size"] = float(
                                len(textcraft_teacher_demo_raw_means)
                            )

                        prefix_advantage_mode = str(
                            self.config.algorithm.get("prefix_advantage_mode", "")
                        ).strip().lower().replace("-", "_").replace(" ", "")
                        if prefix_advantage_mode in {
                            "family_lift",
                            "family_lift_pos",
                            "family_lift_positive",
                            "cont_abs_plus_family_lift",
                            "cont_abs_plus_family_lift_pos",
                            "cont_abs_family_lift",
                            "cont_abs_family_lift_pos",
                            "family_lift_pos_cont_abs",
                        }:
                            family_lift, family_lift_metrics = compute_prefix_family_lift_advantage(
                                batch,
                                clip_value=float(self.config.algorithm.get("prefix_family_lift_clip", 1.0)),
                            )
                            metrics.update(family_lift_metrics)
                            if family_lift is None:
                                raise ValueError(
                                    "family-lift prefix advantage requires sample_uid, variant_label, and "
                                    "seq_level_rewards in the rollout batch."
                                )
                            batch.batch["prefix_family_lift_advantage"] = family_lift

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        if pending_teacher_demo_batch is not None:
                            if self.use_critic:
                                raise ValueError("teacher demo training currently supports critic-free GRPO.")
                            demo_batch, demo_metrics = build_textcraft_teacher_demo_batch(
                                pending_teacher_demo_batch,
                                batch,
                                config=self.config.algorithm,
                                pad_token_id=(
                                    self.tokenizer.pad_token_id
                                    if self.tokenizer.pad_token_id is not None
                                    else 0
                                ),
                                raw_means=textcraft_teacher_demo_raw_means,
                            )
                            metrics.update(demo_metrics)
                            if demo_batch is not None:
                                demo_repeat_times = (
                                    int(self.config.actor_rollout_ref.rollout.n)
                                    if _config_get(
                                        self.config.algorithm,
                                        "teacher_demo_repeat_to_rollout_n",
                                        "textcraft_teacher_demo_repeat_to_rollout_n",
                                        True,
                                    )
                                    else 1
                                )
                                if demo_repeat_times > 1:
                                    demo_batch = demo_batch.repeat(repeat_times=demo_repeat_times, interleave=True)
                                    metrics["teacher_demo/training_rows"] = float(len(demo_batch))
                                if "is_textcraft_teacher_demo" not in batch.batch.keys():
                                    batch.batch["is_textcraft_teacher_demo"] = torch.zeros(
                                        (len(batch),),
                                        dtype=torch.float32,
                                        device=batch.batch["responses"].device,
                                    )
                                if "is_teacher_demo" not in batch.batch.keys():
                                    batch.batch["is_teacher_demo"] = torch.zeros(
                                        (len(batch),),
                                        dtype=torch.float32,
                                        device=batch.batch["responses"].device,
                                    )
                                demo_batch = _align_demo_batch_to_online_keys(batch, demo_batch)
                                batch = DataProto.concat([batch, demo_batch])

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature
                            batch.meta_info["logprob_temperature"] = 1.0
                            
                            # Pass prefix optimization config to actor update
                            if self.config.algorithm.get("optimize_prefix_tokens", False):
                                batch.meta_info["optimize_prefix_tokens"] = True
                                batch.meta_info["prefix_loss_weight"] = self.config.algorithm.get("prefix_loss_weight", 1.0)
                                batch.meta_info["prefix_loss_mode"] = self.config.algorithm.get("prefix_loss_mode", "split")
                                batch.meta_info["prefix_advantage_mode"] = self.config.algorithm.get(
                                    "prefix_advantage_mode", "cont_mean"
                                )
                                batch.meta_info["prefix_advantage_constant"] = self.config.algorithm.get(
                                    "prefix_advantage_constant", 1.0
                                )
                                batch.meta_info["prefix_cont_adv_weight"] = self.config.algorithm.get(
                                    "prefix_cont_adv_weight", 1.0
                                )
                                batch.meta_info["prefix_family_lift_weight"] = self.config.algorithm.get(
                                    "prefix_family_lift_weight", 1.0
                                )
                                batch.meta_info["prefix_family_lift_clip"] = self.config.algorithm.get(
                                    "prefix_family_lift_clip", 1.0
                                )
                            if _use_bc_aux(self.config.algorithm):
                                batch.meta_info["use_textcraft_bc_aux"] = True
                                batch.meta_info["use_bc_aux"] = True
                                batch.meta_info["textcraft_bc_weight"] = _config_get(
                                    self.config.algorithm,
                                    "bc_weight",
                                    "textcraft_bc_weight",
                                    0.0,
                                )
                                batch.meta_info["bc_weight"] = _config_get(
                                    self.config.algorithm,
                                    "bc_weight",
                                    "textcraft_bc_weight",
                                    0.0,
                                )
                            
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical module_logger that supports various backend
                tracking_logger.log(data=metrics, step=self.global_steps)
                metrics_csv_writer.maybe_log(metrics, step=self.global_steps, force=is_last_step)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
