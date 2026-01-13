from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3

from moe_peft.common import FeedForward, LLMCache, LLMModelInput
from moe_peft.executors import executor
from moe_peft.models.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaEmbedding,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
)
from moe_peft.utils import copy_parameters


@dataclass
class QwenConfig(LlamaConfig):
    use_sliding_window_: bool = False
    max_window_layers_: int = None
    sliding_window_: int = None


class QwenAttention(LlamaAttention):
    def __init__(
        self,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        o_proj: nn.Module,
        q_norm: Optional[nn.Module],
        k_norm: Optional[nn.Module],
        idx: int,
        config: QwenConfig,
    ):
        super().__init__(q_proj, k_proj, v_proj, o_proj, idx, config)
        self.q_norm_: nn.Module = nn.Identity() if q_norm is None else q_norm
        self.k_norm_: nn.Module = nn.Identity() if k_norm is None else k_norm
        self.config_ = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[LLMCache] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj_(hidden_states, input_args)
        key_states = self.k_proj_(hidden_states, input_args)
        value_states = self.v_proj_(hidden_states, input_args)

        query_states = self.q_norm_(
            query_states.view(bsz, q_len, self.n_heads_, self.head_dim_)
        ).transpose(1, 2)
        key_states = self.k_norm_(
            key_states.view(bsz, q_len, self.n_kv_heads_, self.head_dim_)
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx_, cache_kwargs
            )

        sliding_window = None
        if (
            self.config_.use_sliding_window_
            and self.config_.sliding_window_ is not None
            and self.config_.max_window_layers_ is not None
            and self.layer_idx_ >= self.config_.max_window_layers_
        ):
            sliding_window = self.config_.sliding_window_

        input_dtype = query_states.dtype
        target_dtype = None
        if input_dtype == torch.float32:
            if executor.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16

        attn_output = self.attention_interface_(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling_,
            # eager attention arguments
            model_config=self.config_,
            # flash attention arguments
            query_length=q_len,
            is_causal=self.is_causal_,
            target_dtype=target_dtype,
            sliding_window=sliding_window,  # main diff with Llama
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        return self.o_proj_(attn_output, input_args)


class QwenForCausalLM(LlamaForCausalLM):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__(config)

    @staticmethod
    def from_pretrained(
        llm_model: Union[
            modeling_qwen2.Qwen2ForCausalLM, modeling_qwen3.Qwen3ForCausalLM
        ],
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = executor.default_device_name(),
    ):
        llm_config: Union[modeling_qwen2.Qwen2Config, modeling_qwen3.Qwen3Config] = (
            llm_model.config
        )
        llm_args = QwenConfig(
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
            sliding_window_=llm_config.sliding_window,
            max_window_layers_=getattr(llm_config, "max_window_layers", None),
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        # Prefer config-provided sliding window flag when caller does not override
        if hasattr(llm_config, "use_sliding_window") and not use_sliding_window:
            llm_args.use_sliding_window_ = llm_config.use_sliding_window

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = QwenForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = LlamaEmbedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_
        )
        model.norm_ = LlamaRMSNorm(llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer(idx)
            decoder.self_attn_ = QwenAttention(
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.self_attn.q_norm if hasattr(layer.self_attn, "q_norm") else None,
                layer.self_attn.k_norm if hasattr(layer.self_attn, "k_norm") else None,
                idx,
                llm_args,
            )
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
