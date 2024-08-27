from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_moe import modeling_qwen2_moe

from moe_peft.backends import backend
from moe_peft.models.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaEmbedding,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
)
from moe_peft.models.modeling_mistral import MistralFlashAttention
from moe_peft.modules import FeedForward, LLMFeedForward, LLMModelInput
from moe_peft.modules.lora_linear import dequantize_module_weight
from moe_peft.utils import copy_parameters


def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: torch.Tensor = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
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
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


@dataclass
class Qwen2MoeConfig(LlamaConfig):
    use_sliding_window_: bool = False
    max_window_layers_: int = None
    sliding_window_: int = None
    # MoE arguments
    decoder_sparse_step_: int = None
    moe_intermediate_size_: int = None
    shared_expert_intermediate_size_: int = None
    num_experts_per_tok_: int = None
    num_experts_: int = None
    norm_topk_prob_: bool = None
    router_aux_loss_coef_: float = None
    mlp_only_layers_: list = None


QWEN2MOE_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attn": MistralFlashAttention,
}


class Qwen2MoeSparseMoeBlock(LLMFeedForward):
    def __init__(
        self,
        moe_block: modeling_qwen2_moe.Qwen2MoeSparseMoeBlock,
        config: Qwen2MoeConfig,
    ):
        super().__init__()
        self.dtype_: torch.dtype = torch.float32
        self.num_experts = config.num_experts_
        self.top_k = config.num_experts_per_tok_
        self.norm_topk_prob = config.norm_topk_prob_

        # gating
        self.gate = nn.Linear(
            config.dim_,
            config.num_experts_,
            bias=False,
            device=config.device_,
            dtype=self.dtype_,
        )
        self.experts: List[LlamaMLP] = []
        for idx in range(self.num_experts):
            expert_module: modeling_qwen2_moe.Qwen2MoeMLP = moe_block.experts[idx]
            self.experts.append(
                LlamaMLP(
                    expert_module.gate_proj,
                    expert_module.down_proj,
                    expert_module.up_proj,
                    config,
                )
            )

        self.shared_expert = LlamaMLP(
            moe_block.shared_expert.gate_proj,
            moe_block.shared_expert.down_proj,
            moe_block.shared_expert.up_proj,
            config,
        )
        self.shared_expert_gate = torch.nn.Linear(
            config.dim_,
            1,
            bias=False,
            device=config.device_,
            dtype=self.dtype_,
        )
        with torch.no_grad():
            self.gate.weight.copy_(dequantize_module_weight(moe_block.gate))
            self.shared_expert_gate.weight.copy_(
                dequantize_module_weight(moe_block.shared_expert_gate)
            )

    def state_dict(self) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Module]]:
        mlp_state_dict = {}
        for k, v in self.shared_expert.state_dict().items():
            mlp_state_dict[f"shared_expert_gate.{k}"] = v
        for idx in range(self.num_experts):
            for k, v in self.experts[idx].state_dict().items():
                mlp_state_dict[f"experts.{idx}.{k}"] = v
        return mlp_state_dict

    def _batch_forward(
        self, hidden_states: torch.Tensor, input_args: LLMModelInput
    ) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer: LlamaMLP = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.nelement() == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(
                batch_size, -1, hidden_dim
            )
            current_hidden_states = (
                expert_layer._batch_forward(current_state, input_args).reshape(
                    -1, hidden_dim
                )
                * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        shared_expert_output = self.shared_expert._batch_forward(
            hidden_states.reshape(batch_size, -1, hidden_dim), input_args
        ).reshape(-1, hidden_dim)

        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        ).to(input_dtype)

        # return final_hidden_states, router_logits
        return final_hidden_states


class Qwen2MoeForCausalLM(LlamaForCausalLM):
    def __init__(self, config: Qwen2MoeConfig) -> None:
        super().__init__(config)

    @staticmethod
    def from_pretrained(
        llm_model: modeling_qwen2_moe.Qwen2MoeForCausalLM,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = backend.default_device_name(),
    ):
        llm_config: modeling_qwen2_moe.Qwen2MoeConfig = llm_model.config
        llm_args = Qwen2MoeConfig(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            head_dim_=llm_config.hidden_size // llm_config.num_attention_heads,
            intermediate_=llm_config.intermediate_size,
            n_layers_=llm_config.num_hidden_layers,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.num_key_value_heads,
            hidden_act_=llm_config.hidden_act,
            rms_norm_eps_=llm_config.rms_norm_eps,
            max_seq_len_=llm_config.max_position_embeddings,
            rope_theta_=llm_config.rope_theta,
            pad_token_id_=llm_config.pad_token_id,
            attn_implementation_=attn_impl,
            use_sliding_window_=use_sliding_window,
            max_window_layers_=llm_config.max_window_layers,
            sliding_window_=llm_config.sliding_window,
            decoder_sparse_step_=llm_config.decoder_sparse_step,
            moe_intermediate_size_=llm_config.moe_intermediate_size,
            shared_expert_intermediate_size_=llm_config.shared_expert_intermediate_size,
            num_experts_per_tok_=llm_config.num_experts_per_tok,
            num_experts_=llm_config.num_experts,
            norm_topk_prob_=llm_config.norm_topk_prob,
            router_aux_loss_coef_=llm_config.router_aux_loss_coef,
            mlp_only_layers_=llm_config.mlp_only_layers,
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = Qwen2MoeForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = LlamaEmbedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_
        )
        model.norm_ = LlamaRMSNorm(llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer(idx)
            decoder.self_attn_ = QWEN2MOE_ATTENTION_CLASSES[
                llm_args.attn_implementation_
            ](
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                idx,
                llm_args,
            )
            if (idx not in llm_args.mlp_only_layers_) and (
                llm_args.num_experts_ > 0
                and (idx + 1) % llm_args.decoder_sparse_step_ == 0
            ):
                decoder.mlp_ = FeedForward(
                    Qwen2MoeSparseMoeBlock(
                        layer.mlp,
                        llm_args,
                    )
                )
            else:
                decoder.mlp_ = FeedForward(
                    LlamaMLP(
                        layer.mlp.gate_proj,
                        layer.mlp.down_proj,
                        layer.mlp.up_proj,
                        llm_args,
                    )
                )

            decoder.input_layernorm_ = LlamaRMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_
            )
            decoder.post_attention_layernorm_ = LlamaRMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_
            )
            model.layers_.append(decoder)

        return model
