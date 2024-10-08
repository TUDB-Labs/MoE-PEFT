import math
from typing import Optional

import torch
import torch.nn.functional as F

from moe_peft.common import Linear, LLMMoeBlock

from .config import MolaConfig


# copied from mixlora.model._mixtral_load_balancing_loss_func
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


class MolaRouterLoss(torch.nn.Module):
    def __init__(self, config: MolaConfig) -> None:
        super().__init__()
        self.aux_loss_coef = config.router_aux_loss_coef_
        self.experts = config.num_experts_
        self.topk = config.top_k_

    def forward(self, gate_logits, attention_mask) -> torch.Tensor:
        return self.aux_loss_coef * _mixtral_load_balancing_loss_func(
            gate_logits, self.experts, self.topk, attention_mask
        )


class MolaSparseMoe(LLMMoeBlock):
    def __init__(
        self,
        in_features: int,
        device: torch.device,
        config: MolaConfig,
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
            dtype=torch.float32,
        )
        self.experts_ = config.num_experts_
        self.topk_ = config.top_k_
        self.router_logits_: torch.Tensor = None

        if gate is None:
            torch.nn.init.kaiming_uniform_(
                self.gate_.weight, a=math.sqrt(config.router_init_range_)
            )
        else:
            with torch.no_grad():
                self.gate_.weight.copy_(gate)

    def forward(
        self,
        residual: torch.Tensor,
        hidden_states: torch.Tensor,
        lora_linear: Optional[Linear] = None,
    ):
        assert lora_linear is not None
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)
        router_logits = self.gate_(hidden_states)
        self.router_logits_ = router_logits.reshape(-1, self.experts_)
        routing_weights_before = F.softmax(router_logits, dim=1, dtype=self.dtype_)

        routing_weights, selected_experts = torch.topk(
            routing_weights_before, self.topk_, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.experts_
        ).permute(2, 1, 0)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, lora_linear.out_features_),
            dtype=self.dtype_,
            device=hidden_states.device,
        )

        for expert_idx in range(self.experts_):
            expert_lora = lora_linear.loras_[
                f"moe.{self.adapter_name_}.experts.{expert_idx}"
            ]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_lora.lora_forward(current_state)
                * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(self.dtype_)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, lora_linear.out_features_
        ).to(input_dtype)

        return residual + final_hidden_states
