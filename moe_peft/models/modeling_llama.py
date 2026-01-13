from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from moe_peft.common import (
    ATTENTION_FUNCTIONS,
    ROPE_INIT_FUNCTIONS,
    FeedForward,
    Linear,
    LLMAttention,
    LLMCache,
    LLMDecoder,
    LLMFeedForward,
    LLMForCausalLM,
    LLMModelConfig,
    LLMModelInput,
    collect_plugin_router_logtis,
    prepare_4d_causal_attention_mask,
    slice_tensor,
)
from moe_peft.executors import executor
from moe_peft.utils import copy_parameters


@dataclass
class LlamaConfig(LLMModelConfig):
    rms_norm_eps_: float = 1e-6
    rope_scaling_: Optional[Dict[str, Any]] = None


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: Optional[LlamaConfig],
        scaling_factor=1.0,
        rope_type="default",
    ):
        super().__init__()
        self.rope_kwargs = {
            "rope_type": rope_type,
            "factor": scaling_factor,
            "dim": config.head_dim_,
            "base": config.rope_theta_,
            "max_position_embeddings": config.max_seq_len_,
        }
        if config is None:
            self.rope_type = rope_type
            self.max_seq_len_cached = config.max_seq_len_
            self.original_max_seq_len = config.max_seq_len_
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling_ is not None:
                self.rope_type = config.rope_scaling_.get(
                    "rope_type", config.rope_scaling_.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_seq_len_
            self.original_max_seq_len = config.max_seq_len_

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, config.device_, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Multi-headed attention from 'Attention Is All You Need' paper.
class LlamaAttention(LLMAttention):
    def __init__(
        self,
        wq: nn.Module,
        wk: nn.Module,
        wv: nn.Module,
        wo: nn.Module,
        idx: int,
        config: LlamaConfig,
    ):
        super().__init__()
        # attention
        self.q_proj_: Linear = Linear(wq, config.device_)  # dim * dim
        self.k_proj_: Linear = Linear(wk, config.device_)  # dim * dim
        self.v_proj_: Linear = Linear(wv, config.device_)  # dim * dim
        self.o_proj_: Linear = Linear(wo, config.device_)  # dim * dim
        # config
        self.layer_idx_ = idx
        self.config_ = config
        self.dim_ = config.dim_
        self.n_heads_ = config.n_heads_
        self.n_kv_heads_ = config.n_kv_heads_
        self.head_dim_ = config.head_dim_
        self.dtype_ = config.dtype_
        self.scaling_ = self.head_dim_**-0.5
        self.is_causal_ = True
        self.attention_interface_: Callable = ATTENTION_FUNCTIONS[
            config.attn_implementation_
        ]

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "q_proj": self.q_proj_,
            "k_proj": self.k_proj_,
            "v_proj": self.v_proj_,
            "o_proj": self.o_proj_,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[LLMCache] = None,
    ):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.q_proj_.forward(hidden_states, input_args)
        xk = self.k_proj_.forward(hidden_states, input_args)
        xv = self.v_proj_.forward(hidden_states, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_).transpose(
            1, 2
        )
        xk = xk.view(
            batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        xv = xv.view(
            batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        # apply rotary embedding
        cos, sin = rotary_emb
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            xk, xv = past_key_value.update(xk, xv, self.layer_idx_, cache_kwargs)

        input_dtype = xq.dtype
        target_dtype = None
        if input_dtype == torch.float32:
            if executor.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16

        attention_score = self.attention_interface_(
            xq,
            xk,
            xv,
            attention_mask,
            scaling=self.scaling_,
            # eager attention arguments
            model_config=self.config_,
            # flash attention arguments
            query_length=max_seq_len,
            is_causal=self.is_causal_,
            target_dtype=target_dtype,
        )
        attention_score = attention_score.reshape(
            batch_size, max_seq_len, -1
        ).contiguous()

        # get output attention score
        return self.o_proj_.forward(attention_score, input_args)


class LlamaMLP(LLMFeedForward):
    def __init__(
        self, w1: nn.Module, w2: nn.Module, w3: nn.Module, args: LlamaConfig
    ) -> None:
        super().__init__()
        # feed forward
        self.w1_: Linear = Linear(w1, args.device_)
        self.w2_: Linear = Linear(w2, args.device_)
        self.w3_: Linear = Linear(w3, args.device_)
        self.act_ = ACT2FN[args.hidden_act_]

    def state_dict(self) -> Dict[str, nn.Module]:
        return {
            "gate_proj": self.w1_,
            "down_proj": self.w2_,
            "up_proj": self.w3_,
        }

    def _batch_forward(
        self, data: torch.Tensor, input_args: LLMModelInput
    ) -> torch.Tensor:
        w1 = self.w1_.forward(data, input_args)
        w3 = self.w3_.forward(data, input_args)
        return self.w2_.forward(self.act_(w1) * w3, input_args)

    def _lora_forward(
        self, lora_name: str, act_fn: nn.Module, data: torch.Tensor
    ) -> torch.Tensor:
        # Applying LoRA weights to FFN weights
        if lora_name in self.w1_.loras_:
            w1 = self.w1_.loras_[lora_name].forward(
                self.w1_.base_layer_.forward(data), data
            )
        else:
            w1 = self.w1_.base_layer_.forward(data)

        if lora_name in self.w3_.loras_:
            w3 = self.w3_.loras_[lora_name].forward(
                self.w3_.base_layer_.forward(data), data
            )
        else:
            w3 = self.w3_.base_layer_.forward(data)

        act_result = act_fn(w1) * w3
        if lora_name in self.w2_.loras_:
            return self.w2_.loras_[lora_name].forward(
                self.w2_.base_layer_.forward(act_result), act_result
            )
        else:
            return self.w2_.base_layer_.forward(act_result)

    def _mixlora_forward(
        self, moe_name, act_fn, expert_mask, hidden_states, input_dtype
    ):
        common_w1 = self.w1_.base_layer_.forward(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        common_w3 = self.w3_.base_layer_.forward(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        final_expert_states = []
        for expert_idx in range(expert_mask.shape[0]):
            _, top_x = torch.where(expert_mask[expert_idx])

            lora_name = f"moe.{moe_name}.experts.{expert_idx}"
            if lora_name in self.w1_.loras_:
                lora_data = slice_tensor(hidden_states, top_x, input_dtype)
                w1 = self.w1_.loras_[lora_name].forward(
                    slice_tensor(common_w1, top_x, input_dtype), lora_data
                )
            else:
                lora_data = None
                w1 = slice_tensor(common_w1, top_x, input_dtype)

            if lora_name in self.w3_.loras_:
                w3 = self.w3_.loras_[lora_name].forward(
                    slice_tensor(common_w3, top_x, input_dtype),
                    slice_tensor(hidden_states, top_x, input_dtype, lora_data),
                )
            else:
                w3 = slice_tensor(common_w3, top_x, input_dtype)

            act_result = act_fn(w1) * w3
            if lora_name in self.w2_.loras_:
                final_expert_states.append(
                    self.w2_.loras_[lora_name].forward(
                        self.w2_.base_layer_.forward(act_result), act_result
                    )
                )
            else:
                final_expert_states.append(self.w2_.base_layer_.forward(act_result))

        return final_expert_states


class LlamaRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(LLMDecoder):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id_: int = layer_id
        self.self_attn_: LlamaAttention = None
        self.mlp_: FeedForward = None
        self.input_layernorm_: LlamaRMSNorm = None
        self.post_attention_layernorm_: LlamaRMSNorm = None

    def state_dict(self) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Module]]:
        return self.self_attn_.state_dict(), self.mlp_.state_dict()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[LLMCache] = None,
    ):

        residual = hidden_states
        hidden_states = self.input_layernorm_(hidden_states)
        # Self Attention
        hidden_states = self.self_attn_.forward(
            hidden_states,
            input_args,
            rotary_emb,
            attention_mask,
            cache_position,
            past_key_value,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm_(hidden_states)
        hidden_states, router_logits = self.mlp_.forward(hidden_states, input_args)
        hidden_states = residual + hidden_states

        if input_args.output_router_logits_:
            router_logits = collect_plugin_router_logtis(
                router_logits, input_args, self
            )

        return hidden_states, *router_logits


class LlamaEmbedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_, padding_idx=self.padding_idx_)
        return data


class LlamaForCausalLM(LLMForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_: LlamaEmbedding = None
        self.norm_: LlamaRMSNorm = None
        self.rotary_emb_ = LlamaRotaryEmbedding(config)
        self.lm_head_ = nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=False,
            dtype=config.dtype_,
            device=config.device_,
        )
        self.layers_: List[LlamaDecoderLayer] = []

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens_(input_ids)

    def rotary_embed(
        self, input_tensor: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb_(input_tensor, position_ids)

    def decoder_stack(self) -> List[LLMDecoder]:
        return self.layers_

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm_(hidden_states)

    def causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[LLMCache],
    ) -> torch.Tensor:

        return prepare_4d_causal_attention_mask(
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
        )

    def model_config(self) -> LlamaConfig:
        return self.config_

    @staticmethod
    def from_pretrained(
        llm_model: modeling_llama.LlamaForCausalLM,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = executor.default_device_name(),
    ):
        assert not use_sliding_window, "Llama model does not support SWA."
        llm_config: modeling_llama.LlamaConfig = llm_model.config
        llm_args = LlamaConfig(
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
            rope_scaling_=llm_config.rope_scaling,
            pad_token_id_=llm_config.pad_token_id,
            attn_implementation_=attn_impl,
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = LlamaForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = LlamaEmbedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_
        )
        model.norm_ = LlamaRMSNorm(llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer(idx)
            decoder.self_attn_ = LlamaAttention(
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
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
