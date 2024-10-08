from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

from moe_peft.common import LLMFeedForward, LLMModelInput, LLMMoeBlock, slice_tensor

from .config import MixLoraConfig


def _mixlora_compatible_forward(
    ffn_layer: LLMFeedForward,
    moe_name: str,
    act_fn: torch.nn.Module,
    expert_mask: torch.Tensor,
    hidden_states: torch.Tensor,
    input_dtype: torch.device,
):
    final_expert_states = []
    for expert_idx in range(expert_mask.shape[0]):
        _, top_x = torch.where(expert_mask[expert_idx])
        lora_name = f"moe.{moe_name}.experts.{expert_idx}"
        lora_data = slice_tensor(hidden_states, top_x, input_dtype)
        final_expert_states.append(
            ffn_layer._lora_forward(lora_name, act_fn, lora_data)
        )

    return final_expert_states


def _mixtral_load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = routing_weights.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(routing_weights.device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(routing_weights.device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class MixtralRouterLoss(torch.nn.Module):
    def __init__(self, config: MixLoraConfig) -> None:
        super().__init__()
        self.aux_loss_coef = config.router_aux_loss_coef_
        self.experts = config.num_experts_
        self.topk = config.top_k_

    def forward(self, gate_logits, attention_mask) -> torch.Tensor:
        return self.aux_loss_coef * _mixtral_load_balancing_loss_func(
            gate_logits, self.experts, self.topk, attention_mask
        )


class MixtralSparseMoe(LLMMoeBlock):
    def __init__(
        self,
        in_features: int,
        device: torch.device,
        config: MixLoraConfig,
        gate: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            in_features,
            config.num_experts_,
            bias=False,
            device=device,
            dtype=self.dtype_,
        )
        self.act_ = (
            ACT2FN[config.act_fn_]
            if isinstance(config.act_fn_, str)
            else config.act_fn_
        )
        self.experts_: int = config.num_experts_
        self.topk_: int = config.top_k_
        self.jitter_noise_: float = config.jitter_noise_
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

        if gate is None:
            torch.nn.init.normal_(
                self.gate_.weight,
                mean=0.0,
                std=config.router_init_range_,
            )
        else:
            with torch.no_grad():
                self.gate_.weight.copy_(gate)

    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {"gate": self.gate_.weight}

    def _profiling(
        self, batch_size: int, sequence_length: int, selected_experts: torch.Tensor
    ) -> None:
        if not self.router_profile_:
            return

        router_statistic_ = list(0 for _ in range(self.experts_))
        for selected in selected_experts.tolist():
            for idx in selected:
                router_statistic_[idx] += 1

        if self.profiler_ is None:
            self.profiler_ = list(0 for _ in range(self.experts_))
            for idx in range(self.experts_):
                self.profiler_[idx] = (
                    router_statistic_[idx] / batch_size
                ) / sequence_length
        else:
            for idx in range(self.experts_):
                pressure = (router_statistic_[idx] / batch_size) / sequence_length
                self.profiler_[idx] = (self.profiler_[idx] + pressure) / 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        ffn_layer: LLMFeedForward,
        input_args: LLMModelInput,
    ) -> Tuple:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if not input_args.inference_mode_ and self.jitter_noise_ > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_
            )

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate_(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=self.dtype_)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.topk_, dim=-1
        )

        self._profiling(batch_size, sequence_length, selected_experts)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=self.dtype_,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.experts_
        ).permute(2, 1, 0)

        # Perform the computation on each expert
        if input_args.efficient_operator_ and hasattr(ffn_layer, "_mixlora_forward"):
            expert_states = ffn_layer._mixlora_forward(
                self.adapter_name_, self.act_, expert_mask, hidden_states, input_dtype
            )
        else:
            expert_states = _mixlora_compatible_forward(
                ffn_layer,
                self.adapter_name_,
                self.act_,
                expert_mask,
                hidden_states,
                input_dtype,
            )

        # Unpack
        for expert_idx in range(self.experts_):
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_hidden_states = (
                expert_states[expert_idx] * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(self.dtype_)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        ).to(input_dtype)

        return final_hidden_states, router_logits


def _dynamic_top_p(router_logits: torch.Tensor, top_p: float, temperature: float = 0.0):
    if temperature > 0.0:
        router_logits = router_logits / temperature
    sorted_logits, sorted_indices = torch.sort(router_logits, dim=-1, descending=True)
    cumulative_probs = sorted_logits.cumsum(dim=-1)
    expert_mask = cumulative_probs > top_p
    threshold_indices = expert_mask.long().argmax(dim=-1)
    threshold_mask = torch.nn.functional.one_hot(
        threshold_indices, num_classes=sorted_indices.size(-1)
    ).bool()
    expert_mask = expert_mask & ~threshold_mask
    sorted_logits = sorted_logits.masked_fill(expert_mask, 0.0)
    sorted_indices = sorted_indices.masked_fill(expert_mask, -1)
    return sorted_logits, sorted_indices


def _dynamic_load_balancing_loss_func(
    routing_weights: torch.Tensor,
    num_experts: int,
    top_p: float,
    temperature: float,
) -> float:
    _, selected_experts = _dynamic_top_p(routing_weights, top_p, temperature)

    expert_mask = torch.empty(
        (num_experts, num_experts, routing_weights.size(0)),
        dtype=routing_weights.dtype,
        device=routing_weights.device,
    )

    for expert_idx in range(num_experts):
        expert_mask[expert_idx] = (selected_experts == expert_idx).transpose(0, 1)

    expert_mask = expert_mask.permute(2, 1, 0)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class DynamicRouterLoss(torch.nn.Module):
    def __init__(self, config: MixLoraConfig) -> None:
        super().__init__()
        self.aux_loss_coef = config.router_aux_loss_coef_
        self.experts = config.num_experts_
        self.top_p = config.top_p_
        self.temperature = config.temperature_

    def forward(self, gate_logits, attention_mask) -> torch.Tensor:
        routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1)
        return self.aux_loss_coef * _dynamic_load_balancing_loss_func(
            routing_weights,
            self.experts,
            self.top_p,
            self.temperature,
        )


class DynamicSparseMoe(LLMMoeBlock):
    def __init__(
        self,
        in_features: int,
        device: torch.device,
        config: MixLoraConfig,
        gate: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            in_features,
            config.num_experts_,
            bias=False,
            device=device,
            dtype=self.dtype_,
        )
        self.act_ = (
            ACT2FN[config.act_fn_]
            if isinstance(config.act_fn_, str)
            else config.act_fn_
        )
        self.experts_: int = config.num_experts_
        self.top_p_: float = config.top_p_
        self.temperature_: float = config.temperature_
        self.jitter_noise_: float = config.jitter_noise_
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

        if gate is None:
            torch.nn.init.normal_(
                self.gate_.weight,
                mean=0.0,
                std=config.router_init_range_,
            )
        else:
            with torch.no_grad():
                self.gate_.weight.copy_(gate)

    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {"gate": self.gate_.weight}

    def _profiling(
        self, batch_size: int, sequence_length: int, selected_experts: torch.Tensor
    ) -> None:
        if not self.router_profile_:
            return

        router_statistic_ = list(0 for _ in range(self.experts_))
        for selected in selected_experts.tolist():
            for idx in selected:
                router_statistic_[idx] += 1

        if self.profiler_ is None:
            self.profiler_ = list(0 for _ in range(self.experts_))
            for idx in range(self.experts_):
                self.profiler_[idx] = (
                    router_statistic_[idx] / batch_size
                ) / sequence_length
        else:
            for idx in range(self.experts_):
                pressure = (router_statistic_[idx] / batch_size) / sequence_length
                self.profiler_[idx] = (self.profiler_[idx] + pressure) / 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        ffn_layer: LLMFeedForward,
        input_args: LLMModelInput,
    ) -> Tuple:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if not input_args.inference_mode_ and self.jitter_noise_ > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_
            )

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate_(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=self.dtype_)
        routing_weights, selected_experts = _dynamic_top_p(
            routing_weights, self.top_p_, self.temperature_
        )

        self._profiling(batch_size, sequence_length, selected_experts)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=self.dtype_,
            device=hidden_states.device,
        )

        expert_mask = torch.empty(
            (self.experts_, self.experts_, batch_size * sequence_length),
            dtype=self.dtype_,
            device=hidden_states.device,
        )

        for expert_idx in range(self.experts_):
            expert_mask[expert_idx] = (selected_experts == expert_idx).transpose(0, 1)

        # Perform the computation on each expert
        if input_args.efficient_operator_ and hasattr(ffn_layer, "_mixlora_forward"):
            expert_states = ffn_layer._mixlora_forward(
                self.adapter_name_, self.act_, expert_mask, hidden_states, input_dtype
            )
        else:
            expert_states = _mixlora_compatible_forward(
                ffn_layer,
                self.adapter_name_,
                self.act_,
                expert_mask,
                hidden_states,
                input_dtype,
            )

        # Unpack
        for expert_idx in range(self.experts_):
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_hidden_states = (
                expert_states[expert_idx] * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(self.dtype_)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        ).to(input_dtype)

        return final_hidden_states, router_logits


def _switch_router_z_loss_func(router_logits: torch.Tensor) -> float:
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (router_logits.size(0))


def _switch_load_balancing_loss_func(router_probs: torch.Tensor) -> float:
    num_experts = router_probs.size(-1)

    expert_mask = torch.argmax(router_probs, dim=-1)
    expert_mask = torch.nn.functional.one_hot(expert_mask, num_classes=num_experts)

    tokens_per_group_and_expert = torch.mean(expert_mask.float(), dim=0)

    router_prob_per_group_and_expert = torch.mean(router_probs, dim=0)
    return torch.mean(
        tokens_per_group_and_expert * router_prob_per_group_and_expert
    ) * (num_experts**2)


class SwitchRouterLoss(torch.nn.Module):
    def __init__(self, config: MixLoraConfig) -> None:
        super().__init__()
        self.experts = config.num_experts_
        self.expert_capacity_ = config.expert_capacity_
        self.z_loss_coef = config.router_z_loss_coef_
        self.aux_loss_coef = config.router_aux_loss_coef_

    def forward(self, router_logits, attention_mask) -> torch.Tensor:
        z_loss = _switch_router_z_loss_func(router_logits)
        router_probs = F.softmax(router_logits, dim=-1)
        # recompute expert indexes due to MoE-PEFT constraints
        aux_loss = _switch_load_balancing_loss_func(router_probs)
        return self.z_loss_coef * z_loss + self.aux_loss_coef * aux_loss


class SwitchSparseMoe(LLMMoeBlock):
    def __init__(
        self,
        in_features: int,
        device: torch.device,
        config: MixLoraConfig,
        gate: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            in_features,
            config.num_experts_,
            bias=False,
            device=device,
            dtype=self.dtype_,
        )
        self.act_ = (
            ACT2FN[config.act_fn_]
            if isinstance(config.act_fn_, str)
            else config.act_fn_
        )
        self.experts_: int = config.num_experts_
        self.dropout_ = (
            torch.nn.Dropout(config.ffn_dropout_)
            if config.ffn_dropout_ > 0
            else torch.nn.Identity()
        )
        self.expert_capacity_: int = config.expert_capacity_
        self.jitter_noise_: float = config.jitter_noise_
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

        if gate is None:
            torch.nn.init.normal_(
                self.gate_.weight,
                mean=0.0,
                std=config.router_init_range_,
            )
        else:
            with torch.no_grad():
                self.gate_.weight.copy_(gate)

    def _profiling(
        self, batch_size: int, sequence_length: int, router_mask: torch.Tensor
    ) -> None:
        if not self.router_profile_:
            return

        selected_experts = torch.argmax(router_mask, dim=-1)

        router_statistic_ = list(0 for _ in range(self.experts_))
        for selected in selected_experts.tolist():
            for idx in selected:
                router_statistic_[idx] += 1

        if self.profiler_ is None:
            self.profiler_ = list(0 for _ in range(self.experts_))
            for idx in range(self.experts_):
                self.profiler_[idx] = (
                    router_statistic_[idx] / batch_size
                ) / sequence_length
        else:
            for idx in range(self.experts_):
                pressure = (router_statistic_[idx] / batch_size) / sequence_length
                self.profiler_[idx] = (self.profiler_[idx] + pressure) / 2

    def route(self, hidden_states: torch.Tensor, input_args: LLMModelInput) -> Tuple:
        if not input_args.inference_mode_ and self.jitter_noise_ > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_
            )

        # Apply Softmax
        router_logits = self.gate_(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1, dtype=self.dtype_)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(
            expert_index, num_classes=self.experts_
        )

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity_
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        ffn_layer: LLMFeedForward,
        input_args: LLMModelInput,
    ) -> Tuple:
        batch_size, sequence_length, _ = hidden_states.shape

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype_)

        router_mask, router_probs, router_logits = self.route(hidden_states, input_args)

        self._profiling(batch_size, sequence_length, router_mask)

        next_states = hidden_states.clone()
        for expert_idx in range(self.experts_):
            token_indices = router_mask[:, :, expert_idx].bool()
            lora_name = f"moe.{self.adapter_name_}.experts.{expert_idx}"
            next_states[token_indices] = ffn_layer._lora_forward(
                lora_name, self.act_, hidden_states[token_indices].to(input_dtype)
            ).to(next_states.dtype)

        if input_args.inference_mode_:
            hidden_states = hidden_states.to(input_dtype)
        else:
            hidden_states = self.dropout_(router_probs * next_states).to(input_dtype)

        return hidden_states, router_logits.reshape(-1, self.experts_)
