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
Single Process Actor
"""

import logging
import os
from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SUPPORTED_PREFIX_ADVANTAGE_MODES = frozenset(
    {
        "cont_mean",
        "cont_mean_abs",
        "cont_mean_pos",
        "constant",
        "success_gate_constant",
        "family_lift",
        "family_lift_pos",
        "cont_abs_plus_family_lift",
        "cont_abs_plus_family_lift_pos",
    }
)
PREFIX_ADVANTAGE_MODE_ALIASES = {
    "contstant": "constant",
    "family_lift_positive": "family_lift_pos",
    "cont_abs_family_lift": "cont_abs_plus_family_lift",
    "cont_abs_family_lift_pos": "cont_abs_plus_family_lift_pos",
    "family_lift_pos_cont_abs": "cont_abs_plus_family_lift_pos",
}


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _build_prefix_loss_config(self):
        return replace(
            self.config,
            clip_ratio=(
                self.config.prefix_clip_ratio
                if self.config.prefix_clip_ratio is not None
                else self.config.clip_ratio
            ),
            clip_ratio_low=(
                self.config.prefix_clip_ratio_low
                if self.config.prefix_clip_ratio_low is not None
                else self.config.clip_ratio_low
            ),
            clip_ratio_high=(
                self.config.prefix_clip_ratio_high
                if self.config.prefix_clip_ratio_high is not None
                else self.config.clip_ratio_high
            ),
            clip_ratio_c=(
                self.config.prefix_clip_ratio_c
                if self.config.prefix_clip_ratio_c is not None
                else self.config.clip_ratio_c
            ),
        )

    def _prefix_clip_override_requested(self) -> bool:
        return any(
            value is not None
            for value in (
                self.config.prefix_clip_ratio,
                self.config.prefix_clip_ratio_low,
                self.config.prefix_clip_ratio_high,
                self.config.prefix_clip_ratio_c,
            )
        )

    @staticmethod
    def _resolve_effective_clip_config(config) -> dict[str, float]:
        clip_ratio = float(config.clip_ratio)
        clip_ratio_low = float(config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio)
        clip_ratio_high = float(config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio)
        clip_ratio_c = float(config.clip_ratio_c if config.clip_ratio_c is not None else 3.0)
        return {
            "clip_ratio": clip_ratio,
            "clip_ratio_low": clip_ratio_low,
            "clip_ratio_high": clip_ratio_high,
            "clip_ratio_c": clip_ratio_c,
        }

    def _build_prefix_clip_metrics(self, prefix_loss_mode: str) -> dict[str, float]:
        clip_config = self._build_prefix_loss_config() if prefix_loss_mode == "split" else self.config
        effective_clip = self._resolve_effective_clip_config(clip_config)
        return {
            "actor/prefix_clip_override_active": float(
                prefix_loss_mode == "split" and self._prefix_clip_override_requested()
            ),
            "actor/prefix_clip_ratio": effective_clip["clip_ratio"],
            "actor/prefix_clip_ratio_low": effective_clip["clip_ratio_low"],
            "actor/prefix_clip_ratio_high": effective_clip["clip_ratio_high"],
            "actor/prefix_clip_ratio_c": effective_clip["clip_ratio_c"],
        }

    @staticmethod
    def _masked_mean_item(values: torch.Tensor, mask: torch.Tensor) -> float:
        if mask is None or mask.sum().item() <= 0:
            return 0.0
        return float(verl_F.masked_mean(values, mask).detach().item())

    @staticmethod
    def _teacher_demo_token_mask(
        teacher_demo_mask: torch.Tensor | None,
        response_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if teacher_demo_mask is None:
            return None
        if not isinstance(teacher_demo_mask, torch.Tensor):
            teacher_demo_mask = torch.as_tensor(teacher_demo_mask, device=response_mask.device)
        teacher_demo_mask = teacher_demo_mask.to(device=response_mask.device, dtype=response_mask.dtype).reshape(-1, 1)
        token_mask = response_mask * (teacher_demo_mask > 0.5).to(response_mask.dtype)
        if token_mask.sum().item() <= 0:
            return None
        return token_mask

    def _mix_teacher_demo_old_log_prob(
        self,
        old_log_prob: torch.Tensor,
        cached_old_log_prob: torch.Tensor,
        teacher_demo_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        token_mask = self._teacher_demo_token_mask(teacher_demo_mask, torch.ones_like(old_log_prob))
        if token_mask is None:
            return old_log_prob
        return torch.where(token_mask > 0.5, cached_old_log_prob, old_log_prob)

    def _build_teacher_demo_diagnostic_metrics(
        self,
        *,
        log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        teacher_demo_mask: torch.Tensor | None,
    ) -> dict[str, float]:
        token_mask = self._teacher_demo_token_mask(teacher_demo_mask, response_mask)
        if token_mask is None:
            return {}

        logprob_delta = log_prob - old_log_prob
        ratio = torch.exp(torch.clamp(logprob_delta, min=-20.0, max=20.0))
        effective_clip = self._resolve_effective_clip_config(self.config)
        clipped = (
            (ratio > (1.0 + effective_clip["clip_ratio_high"]))
            | (ratio < (1.0 - effective_clip["clip_ratio_low"]))
        ).to(dtype=token_mask.dtype)
        return {
            "teacher_demo/token_count": float(token_mask.sum().detach().item()),
            "teacher_demo/ratio_mean": self._masked_mean_item(ratio, token_mask),
            "teacher_demo/logprob_delta_mean": self._masked_mean_item(logprob_delta, token_mask),
            "teacher_demo/advantage_mean": self._masked_mean_item(advantages, token_mask),
            "teacher_demo/clipfrac": self._masked_mean_item(clipped, token_mask),
        }

    def _build_prefix_diagnostic_metrics(
        self,
        current_prefix_log_probs: torch.Tensor,
        cached_prefix_valid_log_probs: torch.Tensor,
        prefix_advantages: torch.Tensor,
        prefix_loss_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, float]:
        prefix_logprob_delta = current_prefix_log_probs - cached_prefix_valid_log_probs
        prefix_ratio = torch.exp(torch.clamp(prefix_logprob_delta, min=-20.0, max=20.0))

        prefix_token_total = float(prefix_loss_mask.sum().item())
        continuation_token_total = float(response_mask.sum().item())
        total_token_total = prefix_token_total + continuation_token_total

        return {
            "actor/prefix_logprob_delta_mean": self._masked_mean_item(prefix_logprob_delta, prefix_loss_mask),
            "actor/prefix_logprob_delta_abs_mean": self._masked_mean_item(prefix_logprob_delta.abs(), prefix_loss_mask),
            "actor/prefix_ratio_mean": self._masked_mean_item(prefix_ratio, prefix_loss_mask),
            "actor/prefix_advantage_mean": self._masked_mean_item(prefix_advantages, prefix_loss_mask),
            "actor/prefix_advantage_abs_mean": self._masked_mean_item(prefix_advantages.abs(), prefix_loss_mask),
            "actor/prefix_token_frac": (prefix_token_total / total_token_total) if total_token_total > 0 else 0.0,
        }

    @staticmethod
    def _build_prefix_loss_contribution_metrics(
        prefix_loss: float | None,
        continuation_loss: float,
        prefix_loss_weight: float,
        exact: bool,
    ) -> dict[str, float]:
        if not exact or prefix_loss is None:
            return {
                "actor/prefix_loss_contribution": float("nan"),
                "actor/prefix_loss_share": float("nan"),
            }

        prefix_loss_contribution = float(prefix_loss_weight) * float(prefix_loss)
        denom = abs(prefix_loss_contribution) + abs(float(continuation_loss)) + 1e-12
        return {
            "actor/prefix_loss_contribution": prefix_loss_contribution,
            "actor/prefix_loss_share": abs(prefix_loss_contribution) / denom,
        }

    @staticmethod
    def _normalize_prefix_advantage_mode(mode: str) -> str:
        raw_mode = str(mode)
        normalized_mode = raw_mode.strip().lower().replace("-", "_").replace(" ", "")
        aliased_mode = PREFIX_ADVANTAGE_MODE_ALIASES.get(normalized_mode, normalized_mode)
        if aliased_mode != normalized_mode:
            logger.warning(
                "Normalizing prefix_advantage_mode=%r to %r.",
                raw_mode,
                aliased_mode,
            )
        if aliased_mode not in SUPPORTED_PREFIX_ADVANTAGE_MODES:
            supported_modes = ", ".join(sorted(SUPPORTED_PREFIX_ADVANTAGE_MODES))
            raise ValueError(
                f"Unsupported prefix_advantage_mode: {mode}. Supported modes: {supported_modes}"
            )
        return aliased_mode

    @staticmethod
    def _compute_prefix_sequence_advantage(
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        mode: str,
        constant_value: float,
        seq_level_rewards: torch.Tensor | None = None,
        family_lift_advantages: torch.Tensor | None = None,
        cont_adv_weight: float = 1.0,
        family_lift_weight: float = 1.0,
        family_lift_clip: float = 1.0,
    ) -> torch.Tensor:
        mode = DataParallelPPOActor._normalize_prefix_advantage_mode(mode)
        denom = response_mask.sum(dim=1, keepdim=True).clamp(min=1)
        seq_advantage = (advantages * response_mask).sum(dim=1, keepdim=True) / denom

        if mode == "cont_mean":
            return seq_advantage
        if mode == "cont_mean_abs":
            return seq_advantage.abs()
        if mode == "cont_mean_pos":
            return seq_advantage.clamp_min(0.0)
        if mode == "constant":
            return torch.full_like(seq_advantage, float(constant_value))
        if mode == "success_gate_constant":
            if seq_level_rewards is None:
                raise ValueError("success_gate_constant requires seq_level_rewards to be provided.")
            success_mask = seq_level_rewards.reshape(-1, 1).to(seq_advantage.dtype) > 0
            constant_advantage = torch.full_like(seq_advantage, float(constant_value))
            return torch.where(success_mask, constant_advantage, torch.zeros_like(seq_advantage))
        if mode in {
            "family_lift",
            "family_lift_pos",
            "cont_abs_plus_family_lift",
            "cont_abs_plus_family_lift_pos",
        }:
            if family_lift_advantages is None:
                raise ValueError(f"{mode} requires prefix_family_lift_advantage to be provided.")
            lift = family_lift_advantages.reshape(-1, 1).to(seq_advantage.device, dtype=seq_advantage.dtype)
            if float(family_lift_clip) > 0:
                lift = lift.clamp(min=-float(family_lift_clip), max=float(family_lift_clip))
            if mode.endswith("_pos"):
                lift = lift.clamp_min(0.0)
            if mode in {"family_lift", "family_lift_pos"}:
                return float(family_lift_weight) * lift
            return float(cont_adv_weight) * seq_advantage.abs() + float(family_lift_weight) * lift
        raise ValueError(f"Unsupported prefix_advantage_mode: {mode}")

    def _compute_textcraft_bc_aux_loss(
        self,
        model_inputs: dict,
        temperature: float,
        loss_agg_mode: str,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        required = {"bc_input_ids", "bc_attention_mask", "bc_position_ids", "bc_loss_mask"}
        if not required.issubset(model_inputs.keys()):
            return None, {}

        bc_input_ids = model_inputs["bc_input_ids"]
        if bc_input_ids.shape[1] < 2:
            return None, {}

        bc_loss_mask = model_inputs["bc_loss_mask"].float()
        shifted_loss_mask = bc_loss_mask[:, 1:]
        valid_tokens = shifted_loss_mask.sum()
        if int(valid_tokens.detach().item()) <= 0:
            return None, {
                "actor/bc_token_count": 0.0,
                "actor/bc_sequence_count": 0.0,
                "actor/textcraft_bc_token_count": 0.0,
                "actor/textcraft_bc_sequence_count": 0.0,
            }

        bc_model_inputs = {
            "input_ids": bc_input_ids,
            "attention_mask": model_inputs["bc_attention_mask"],
            "position_ids": model_inputs["bc_position_ids"],
            "responses": bc_input_ids[:, 1:],
        }
        _, bc_log_prob = self._forward_micro_batch(
            bc_model_inputs,
            temperature=temperature,
            calculate_entropy=False,
            return_full_seq=False,
        )
        bc_loss = agg_loss(
            loss_mat=-bc_log_prob,
            loss_mask=shifted_loss_mask,
            loss_agg_mode=loss_agg_mode,
        )
        sequence_count = (shifted_loss_mask.sum(dim=1) > 0).float().sum()
        return bc_loss, {
            "actor/bc_loss": bc_loss.detach().item(),
            "actor/bc_token_count": valid_tokens.detach().item(),
            "actor/bc_sequence_count": sequence_count.detach().item(),
            "actor/textcraft_bc_loss": bc_loss.detach().item(),
            "actor/textcraft_bc_token_count": valid_tokens.detach().item(),
            "actor/textcraft_bc_sequence_count": sequence_count.detach().item(),
        }

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, return_full_seq=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len) or (bs, prompt_len) if return_full_seq=True
        """
        assert temperature > 0, f"temperature must be > 0 for logprob computation, got {temperature}"
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if return_full_seq:
                    prompt_length = seqlen - response_length
                    if calculate_entropy:
                        entropy = full_entropy.squeeze(-1)[:, :prompt_length]
                    log_probs = full_log_probs.squeeze(-1)[:, :prompt_length]
                else:
                    if calculate_entropy:
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    if return_full_seq:
                        logits = output.logits
                        logits.div_(temperature)
                        prompt_length = seqlen - response_length
                        logits_prompt = logits[:, :prompt_length, :]
                        log_probs = logprobs_from_logits(
                            logits=logits_prompt,
                            labels=input_ids[:, 1 : 1 + prompt_length],
                        )
                        if calculate_entropy:
                            if not self.config.entropy_checkpointing:
                                entropy = verl_F.entropy_from_logits(logits_prompt)
                            else:
                                entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits_prompt)
                    else:
                        log_probs = output.log_probs[:, -response_length - 1 : -1]
                        entropy = output.entropy[:, -response_length - 1 : -1] if calculate_entropy else None

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    if return_full_seq:
                        prompt_length = seqlen - response_length
                        logits_prompt = logits[:, :prompt_length, :]
                        log_probs = logprobs_from_logits(logits_prompt, input_ids[:, 1 : 1 + prompt_length])
                        if calculate_entropy:
                            if not self.config.entropy_checkpointing:
                                entropy = verl_F.entropy_from_logits(logits_prompt)
                            else:
                                entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits_prompt)
                    else:
                        logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                        if calculate_entropy:
                            if not self.config.entropy_checkpointing:
                                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                            else:
                                entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        logprob_temperature = data.meta_info.get("logprob_temperature", 1.0)
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=logprob_temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        logprob_temperature = data.meta_info.get("logprob_temperature", 1.0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        # Add keys for DRPO support
        if "uid" in data.batch.keys():
            select_keys.append("uid")
        if "seq_level_rewards" in data.batch.keys():
            select_keys.append("seq_level_rewards")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        if "is_textcraft_teacher_demo" in data.batch.keys():
            select_keys.append("is_textcraft_teacher_demo")
        if "is_teacher_demo" in data.batch.keys():
            select_keys.append("is_teacher_demo")
        
        # Include prefix_mask for prefix optimization
        # This is used when algorithm.optimize_prefix_tokens=True
        optimize_prefix_tokens = data.meta_info.get("optimize_prefix_tokens", False)
        if optimize_prefix_tokens and "prefix_mask" in data.batch.keys():
            select_keys.append("prefix_mask")
            # Include cached prefix old_log_probs if available (from SFT model)
            if "assistant_prefix_old_log_probs" in data.batch.keys():
                select_keys.append("assistant_prefix_old_log_probs")
            if "prefix_token_count" in data.batch.keys():
                select_keys.append("prefix_token_count")
            if "assistant_prefix_span" in data.batch.keys():
                select_keys.append("assistant_prefix_span")
            if "prefix_family_lift_advantage" in data.batch.keys():
                select_keys.append("prefix_family_lift_advantage")

        if data.meta_info.get("use_bc_aux", False) or data.meta_info.get("use_textcraft_bc_aux", False):
            for key in ["bc_input_ids", "bc_attention_mask", "bc_position_ids", "bc_loss_mask"]:
                if key in data.batch.keys():
                    select_keys.append(key)

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    teacher_demo_mask = model_inputs.get(
                        "is_teacher_demo",
                        model_inputs.get("is_textcraft_teacher_demo", None),
                    )

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # DRPO uses a different architecture - check loss_type from config
                    # DRPO: Decoupled Reward Policy Optimization
                    drpo_loss_type = getattr(self.config, 'loss_type', None)

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # DRPO support: check if loss_type is 'drpo'
                    if drpo_loss_type == "drpo":
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs, temperature=logprob_temperature, calculate_entropy=calculate_entropy
                        )
                        if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                            old_log_prob = model_inputs["old_log_probs"]
                        else:
                            if on_policy:
                                old_log_prob = log_prob.detach()
                            else:
                                old_log_prob = model_inputs["old_log_probs"]
                        # DRPO requires uid and seq_level_rewards
                        uid = model_inputs.get("uid", None)
                        seq_level_rewards = model_inputs.get("seq_level_rewards", None)
                        delta = getattr(self.config, 'delta', 1e-4)
                        beta = getattr(self.config, 'beta', 1e3)
                        tau = getattr(self.config, 'tau', 10.0)
                        Lambda = getattr(self.config, 'Lambda', 0.1)
                        kl_type = getattr(self.config, 'ppo_kl_type', 'low_var_kl')

                        from verl.trainer.ppo.core_algos import compute_policy_loss_drpo
                        pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss_drpo(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            eos_mask=response_mask,
                            uid=uid,
                            seq_level_rewards=seq_level_rewards,
                            delta=delta,
                            beta=beta,
                            tau=tau,
                            Lambda=Lambda,
                            kl_type=kl_type
                        )
                        pg_metrics = {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                        }
                        micro_batch_metrics.update(pg_metrics)
                    else:
                        if not optimize_prefix_tokens:
                            entropy, log_prob = self._forward_micro_batch(
                                model_inputs, temperature=logprob_temperature, calculate_entropy=calculate_entropy
                            )

                            if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                                old_log_prob = model_inputs["old_log_probs"]
                            else:
                                if on_policy:
                                    old_log_prob = log_prob.detach()
                                    old_log_prob = self._mix_teacher_demo_old_log_prob(
                                        old_log_prob,
                                        model_inputs["old_log_probs"],
                                        teacher_demo_mask,
                                    )
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]

                            policy_loss_fn = get_policy_loss_fn(loss_mode)
                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            micro_batch_metrics.update(pg_metrics)
                            micro_batch_metrics.update(
                                self._build_teacher_demo_diagnostic_metrics(
                                    log_prob=log_prob,
                                    old_log_prob=old_log_prob,
                                    advantages=advantages,
                                    response_mask=response_mask,
                                    teacher_demo_mask=teacher_demo_mask,
                                )
                            )
                            rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                            if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs
                                rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                    log_prob=log_prob,
                                    rollout_log_prob=rollout_log_prob,
                                    response_mask=response_mask,
                                )
                                micro_batch_metrics.update(rollout_corr_metrics)

                            policy_loss = pg_loss
                            micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                        else:
                            prefix_mask = model_inputs.get("prefix_mask", None)
                            prefix_loss_weight = data.meta_info.get("prefix_loss_weight", 1.0)
                            prefix_loss_mode = data.meta_info.get("prefix_loss_mode", "split")
                            prefix_advantage_mode = self._normalize_prefix_advantage_mode(
                                data.meta_info.get("prefix_advantage_mode", "cont_mean")
                            )
                            prefix_advantage_constant = data.meta_info.get("prefix_advantage_constant", 1.0)
                            if prefix_mask is None:
                                raise ValueError(
                                    "prefix_mask is required when optimize_prefix_tokens=True."
                                )

                            cached_prefix_old_log_probs = model_inputs.get("assistant_prefix_old_log_probs", None)
                            if cached_prefix_old_log_probs is None:
                                raise ValueError(
                                    "assistant_prefix_old_log_probs is required when optimize_prefix_tokens=True."
                                )
                            assistant_prefix_span = model_inputs.get("assistant_prefix_span", None)
                            if assistant_prefix_span is None:
                                raise ValueError(
                                    "assistant_prefix_span is required when optimize_prefix_tokens=True."
                                )
                            if prefix_loss_mode not in {"split", "joint"}:
                                raise ValueError(f"Unsupported prefix_loss_mode: {prefix_loss_mode}")

                            entropy_resp, log_prob_resp = self._forward_micro_batch(
                                model_inputs,
                                temperature=logprob_temperature,
                                calculate_entropy=calculate_entropy,
                                return_full_seq=False,
                            )
                            log_prob = log_prob_resp
                            _, log_prob_prefix = self._forward_micro_batch(
                                model_inputs,
                                temperature=logprob_temperature,
                                calculate_entropy=False,
                                return_full_seq=True,
                            )

                            if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                                cont_old_log_prob = model_inputs["old_log_probs"]
                            else:
                                if on_policy:
                                    cont_old_log_prob = log_prob_resp.detach()
                                    cont_old_log_prob = self._mix_teacher_demo_old_log_prob(
                                        cont_old_log_prob,
                                        model_inputs["old_log_probs"],
                                        teacher_demo_mask,
                                    )
                                else:
                                    cont_old_log_prob = model_inputs["old_log_probs"]

                            cont_policy_loss_fn = get_policy_loss_fn(loss_mode)
                            cont_pg_loss, cont_metrics = cont_policy_loss_fn(
                                old_log_prob=cont_old_log_prob,
                                log_prob=log_prob_resp,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            teacher_demo_metrics = self._build_teacher_demo_diagnostic_metrics(
                                log_prob=log_prob_resp,
                                old_log_prob=cont_old_log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                teacher_demo_mask=teacher_demo_mask,
                            )

                            batch_size = log_prob_prefix.shape[0]
                            actual_prompt_len = log_prob_prefix.shape[1]
                            num_prefix_tokens = []
                            gathered_current_log_probs = []
                            gathered_cached_old_log_probs = []
                            for sample_idx in range(batch_size):
                                span_start = int(assistant_prefix_span[sample_idx, 0].item())
                                span_end = int(assistant_prefix_span[sample_idx, 1].item())
                                if span_end < span_start:
                                    raise ValueError(
                                        f"sample {sample_idx}: invalid assistant_prefix_span [{span_start}, {span_end})."
                                    )
                                if span_end > actual_prompt_len:
                                    raise ValueError(
                                        f"sample {sample_idx}: assistant_prefix_span end={span_end} "
                                        f"exceeds prompt length {actual_prompt_len}."
                                    )
                                span_len = span_end - span_start
                                if prefix_mask.shape[1] < span_len:
                                    raise ValueError(
                                        f"sample {sample_idx}: prefix_mask width {prefix_mask.shape[1]} "
                                        f"is smaller than span_len {span_len}."
                                    )
                                if cached_prefix_old_log_probs.shape[1] < span_len:
                                    raise ValueError(
                                        f"sample {sample_idx}: cached old_log_probs width "
                                        f"{cached_prefix_old_log_probs.shape[1]} is smaller than span_len {span_len}."
                                    )

                                window_mask = prefix_mask[sample_idx, :span_len] > 0.5
                                window_offsets = window_mask.nonzero(as_tuple=True)[0]
                                gather_indices = span_start + window_offsets - 1
                                if gather_indices.numel() > 0:
                                    if gather_indices.min().item() < 0 or gather_indices.max().item() >= actual_prompt_len:
                                        raise ValueError(
                                            f"sample {sample_idx}: invalid prefix gather range "
                                            f"[{gather_indices.min().item()}, {gather_indices.max().item()}] "
                                            f"for prompt length {actual_prompt_len}."
                                        )
                                    current_gathered = log_prob_prefix[sample_idx].gather(0, gather_indices)
                                    cached_gathered = cached_prefix_old_log_probs[sample_idx, :span_len].gather(
                                        0, window_offsets
                                    )
                                else:
                                    current_gathered = log_prob_prefix[sample_idx].new_zeros((0,))
                                    cached_gathered = cached_prefix_old_log_probs[sample_idx].new_zeros((0,))
                                num_prefix_tokens.append(int(window_offsets.numel()))
                                gathered_current_log_probs.append(current_gathered)
                                gathered_cached_old_log_probs.append(cached_gathered)

                            max_prefix_tokens = max(num_prefix_tokens) if num_prefix_tokens else 0
                            current_prefix_log_probs = log_prob_prefix.new_zeros((batch_size, max_prefix_tokens))
                            cached_prefix_valid_log_probs = cached_prefix_old_log_probs.new_zeros(
                                (batch_size, max_prefix_tokens)
                            )
                            prefix_loss_mask = prefix_mask.new_zeros((batch_size, max_prefix_tokens))

                            for sample_idx, gathered in enumerate(gathered_current_log_probs):
                                valid_len = gathered.shape[0]
                                if valid_len > 0:
                                    current_prefix_log_probs[sample_idx, :valid_len] = gathered
                                    cached_prefix_valid_log_probs[sample_idx, :valid_len] = gathered_cached_old_log_probs[
                                        sample_idx
                                    ]
                                    prefix_loss_mask[sample_idx, :valid_len] = 1.0

                            if "prefix_token_count" in model_inputs:
                                expected_prefix_counts = model_inputs["prefix_token_count"]
                                if isinstance(expected_prefix_counts, torch.Tensor):
                                    expected_prefix_counts = expected_prefix_counts.reshape(-1).tolist()
                                else:
                                    expected_prefix_counts = np.asarray(expected_prefix_counts).reshape(-1).tolist()
                                if len(expected_prefix_counts) != batch_size:
                                    raise ValueError(
                                        "prefix_token_count batch size mismatch: "
                                        f"counts={len(expected_prefix_counts)} vs prefix_batch={batch_size}."
                                    )
                                for sample_idx, expected_count in enumerate(expected_prefix_counts):
                                    if int(expected_count) != num_prefix_tokens[sample_idx]:
                                        raise ValueError(
                                            f"sample {sample_idx}: prefix_token_count={int(expected_count)} "
                                            f"!= gathered_prefix_tokens={num_prefix_tokens[sample_idx]}."
                                        )

                            prefix_seq_advantage = self._compute_prefix_sequence_advantage(
                                advantages=advantages,
                                response_mask=response_mask,
                                mode=prefix_advantage_mode,
                                constant_value=float(prefix_advantage_constant),
                                seq_level_rewards=model_inputs.get("seq_level_rewards", None),
                                family_lift_advantages=model_inputs.get("prefix_family_lift_advantage", None),
                                cont_adv_weight=float(data.meta_info.get("prefix_cont_adv_weight", 1.0)),
                                family_lift_weight=float(data.meta_info.get("prefix_family_lift_weight", 1.0)),
                                family_lift_clip=float(data.meta_info.get("prefix_family_lift_clip", 1.0)),
                            )
                            prefix_advantages = prefix_seq_advantage.expand(-1, max_prefix_tokens)
                            prefix_metric_config = self._build_prefix_loss_config() if prefix_loss_mode == "split" else self.config
                            prefix_policy_loss_fn = get_policy_loss_fn("vanilla")
                            if int(prefix_loss_mask.sum().item()) <= 0:
                                # Keep the prefix full-sequence forward graph attached to backward even when this
                                # micro-batch contains only raw / zero-prefix samples. Without this zero-gradient
                                # surrogate, some ranks may entirely skip the prefix branch in backward while other
                                # ranks still backprop through it, which can trigger FSDP/NCCL collective mismatch.
                                prefix_pg_loss = log_prob_prefix.sum() * 0.0
                                prefix_metrics = {
                                    "actor/pg_clipfrac": 0.0,
                                    "actor/ppo_kl": 0.0,
                                    "actor/pg_clipfrac_lower": 0.0,
                                }
                            else:
                                prefix_pg_loss, prefix_metrics = prefix_policy_loss_fn(
                                    old_log_prob=cached_prefix_valid_log_probs,
                                    log_prob=current_prefix_log_probs,
                                    advantages=prefix_advantages,
                                    response_mask=prefix_loss_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    config=prefix_metric_config,
                                    rollout_is_weights=None,
                                )
                            prefix_diagnostic_metrics = self._build_prefix_diagnostic_metrics(
                                current_prefix_log_probs=current_prefix_log_probs,
                                cached_prefix_valid_log_probs=cached_prefix_valid_log_probs,
                                prefix_advantages=prefix_advantages,
                                prefix_loss_mask=prefix_loss_mask,
                                response_mask=response_mask,
                            )

                            entropy = entropy_resp
                            log_prob = log_prob_resp

                            if prefix_loss_mode == "split":
                                policy_loss = cont_pg_loss + float(prefix_loss_weight) * prefix_pg_loss
                                micro_batch_metrics.update(cont_metrics)
                                micro_batch_metrics.update(teacher_demo_metrics)
                                continuation_loss_value = cont_pg_loss.detach().item() * loss_scale_factor
                                prefix_loss_value = prefix_pg_loss.detach().item() * loss_scale_factor
                                micro_batch_metrics["actor/continuation_loss"] = (
                                    continuation_loss_value
                                )
                                micro_batch_metrics["actor/prefix_loss"] = (
                                    prefix_loss_value
                                )
                                micro_batch_metrics["actor/pg_loss"] = policy_loss.detach().item() * loss_scale_factor
                                micro_batch_metrics["actor/prefix_loss_weight"] = float(prefix_loss_weight)
                                micro_batch_metrics["actor/prefix_token_count"] = float(sum(num_prefix_tokens))
                                micro_batch_metrics["actor/prefix_old_logprob_source"] = 1.0
                                micro_batch_metrics["actor/prefix_loss_mode_joint"] = 0.0
                                micro_batch_metrics.update(prefix_diagnostic_metrics)
                                micro_batch_metrics.update(
                                    self._build_prefix_loss_contribution_metrics(
                                        prefix_loss=prefix_loss_value,
                                        continuation_loss=continuation_loss_value,
                                        prefix_loss_weight=float(prefix_loss_weight),
                                        exact=True,
                                    )
                                )
                                micro_batch_metrics.update(
                                    {
                                        key.replace("actor/", "actor/continuation_", 1): value
                                        for key, value in cont_metrics.items()
                                        if key.startswith("actor/")
                                    }
                                )
                                micro_batch_metrics.update(
                                    {
                                        key.replace("actor/", "actor/prefix_", 1): value
                                        for key, value in prefix_metrics.items()
                                        if key.startswith("actor/")
                                    }
                                )
                            else:
                                prefix_graph_anchor = log_prob_prefix.sum() * 0.0
                                joint_old_log_prob = torch.cat(
                                    [cached_prefix_valid_log_probs, cont_old_log_prob], dim=1
                                )
                                joint_log_prob = torch.cat([current_prefix_log_probs, log_prob_resp], dim=1)
                                joint_advantages = torch.cat([prefix_advantages, advantages], dim=1)
                                joint_response_mask = torch.cat([prefix_loss_mask, response_mask], dim=1)
                                if rollout_is_weights is not None:
                                    prefix_rollout_is_weights = rollout_is_weights.new_ones(prefix_loss_mask.shape)
                                    joint_rollout_is_weights = torch.cat(
                                        [prefix_rollout_is_weights, rollout_is_weights], dim=1
                                    )
                                else:
                                    joint_rollout_is_weights = None

                                joint_policy_loss_fn = get_policy_loss_fn(loss_mode)
                                joint_pg_loss, joint_metrics = joint_policy_loss_fn(
                                    old_log_prob=joint_old_log_prob,
                                    log_prob=joint_log_prob,
                                    advantages=joint_advantages,
                                    response_mask=joint_response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    config=self.config,
                                    rollout_is_weights=joint_rollout_is_weights,
                                )
                                policy_loss = joint_pg_loss + prefix_graph_anchor
                                micro_batch_metrics.update(joint_metrics)
                                micro_batch_metrics.update(teacher_demo_metrics)
                                continuation_loss_value = cont_pg_loss.detach().item() * loss_scale_factor
                                micro_batch_metrics["actor/joint_loss"] = joint_pg_loss.detach().item() * loss_scale_factor
                                micro_batch_metrics["actor/continuation_loss"] = (
                                    continuation_loss_value
                                )
                                micro_batch_metrics["actor/pg_loss"] = joint_pg_loss.detach().item() * loss_scale_factor
                                micro_batch_metrics["actor/prefix_token_count"] = float(sum(num_prefix_tokens))
                                micro_batch_metrics["actor/prefix_old_logprob_source"] = 1.0
                                micro_batch_metrics["actor/prefix_loss_mode_joint"] = 1.0
                                micro_batch_metrics["actor/prefix_loss_weight_ignored"] = float(prefix_loss_weight)
                                micro_batch_metrics.update(prefix_diagnostic_metrics)
                                micro_batch_metrics.update(
                                    self._build_prefix_loss_contribution_metrics(
                                        prefix_loss=None,
                                        continuation_loss=continuation_loss_value,
                                        prefix_loss_weight=float(prefix_loss_weight),
                                        exact=False,
                                    )
                                )
                                micro_batch_metrics.update(
                                    {
                                        key.replace("actor/", "actor/continuation_", 1): value
                                        for key, value in cont_metrics.items()
                                        if key.startswith("actor/")
                                    }
                                )
                            micro_batch_metrics.update(self._build_prefix_clip_metrics(prefix_loss_mode))
                    
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        if not optimize_prefix_tokens:
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        else:
                            micro_batch_metrics["actor/kl_disabled"] = 1.0

                    if data.meta_info.get("use_bc_aux", False) or data.meta_info.get("use_textcraft_bc_aux", False):
                        bc_weight = float(data.meta_info.get("bc_weight", data.meta_info.get("textcraft_bc_weight", 0.0)))
                        if bc_weight > 0:
                            bc_loss, bc_metrics = self._compute_textcraft_bc_aux_loss(
                                model_inputs=model_inputs,
                                temperature=logprob_temperature,
                                loss_agg_mode=loss_agg_mode,
                            )
                            if bc_loss is not None:
                                policy_loss = policy_loss + bc_weight * bc_loss
                                micro_batch_metrics.update(bc_metrics)
                                micro_batch_metrics["actor/bc_weight"] = bc_weight
                                micro_batch_metrics["actor/bc_loss_contribution"] = (
                                    bc_weight * bc_loss.detach().item() * loss_scale_factor
                                )
                                micro_batch_metrics["actor/textcraft_bc_weight"] = bc_weight
                                micro_batch_metrics["actor/textcraft_bc_loss_contribution"] = (
                                    bc_weight * bc_loss.detach().item() * loss_scale_factor
                                )

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
