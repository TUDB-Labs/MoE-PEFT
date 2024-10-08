import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.phi3 import modeling_phi3
from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb, repeat_kv
from transformers.utils import is_flash_attn_2_available

from moe_peft.common import (
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
    eager_attention_forward,
    flash_attention_forward,
    prepare_4d_causal_attention_mask,
    slice_tensor,
)
from moe_peft.executors import executor
from moe_peft.utils import copy_parameters

from .modeling_gemma2 import Gemma2RotaryEmbedding as Phi3RotaryEmbedding
from .modeling_llama import LlamaEmbedding as Phi3Embedding
from .modeling_llama import LlamaRMSNorm as Phi3RMSNorm


@dataclass
class Phi3Config(LLMModelConfig):
    rms_norm_eps_: float = 1e-6
    original_max_position_embeddings_: int = 4096
    rope_scaling_: Optional[Dict[str, Any]] = None
    use_sliding_window_: bool = False
    sliding_window_: int = 4096
    resid_pdrop_: float = 0.0


class Phi3LongRoPEScaledRotaryEmbedding(Phi3RotaryEmbedding):
    def __init__(self, dim, config: Phi3Config, device=None):
        super().__init__(dim, config.max_seq_len_, config.rope_theta_, device)

        self.short_factor = config.rope_scaling_["short_factor"]
        self.long_factor = config.rope_scaling_["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings_

    @torch.no_grad()
    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(
                self.long_factor, dtype=torch.float32, device=x.device
            )
        else:
            ext_factors = torch.tensor(
                self.short_factor, dtype=torch.float32, device=x.device
            )

        inv_freq_shape = (
            torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float()
            / self.dim
        )
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
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

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(
                    1
                    + math.log(scale) / math.log(self.original_max_position_embeddings)
                )

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3Attention(LLMAttention):
    def __init__(
        self, qkv_proj: nn.Module, o_proj: nn.Module, layer_idx: int, args: Phi3Config
    ) -> None:
        super().__init__()
        # attention
        self.qkv_proj_ = Linear(qkv_proj, args.device_)
        self.o_proj_ = Linear(o_proj, args.device_)
        # config
        self.layer_idx_ = layer_idx
        self.args_ = args
        self.dim_ = args.dim_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.rope_theta_ = args.rope_theta_
        self.head_dim_ = self.dim_ // self.n_heads_
        self.dtype_ = args.dtype_
        self.is_causal_ = True

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "qkv_proj": self.qkv_proj_,
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
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj_.forward(hidden_states, input_args)
        query_pos = self.n_heads_ * self.head_dim_
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.n_kv_heads_ * self.head_dim_]
        value_states = qkv[..., query_pos + self.n_kv_heads_ * self.head_dim_ :]

        query_states = query_states.view(
            bsz, q_len, self.n_heads_, self.head_dim_
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        # apply rotary embedding
        cos, sin = rotary_emb
        assert query_states.dtype == key_states.dtype
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, cache_position.unsqueeze(0)
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

        value_states = repeat_kv(value_states, self.n_rep_)
        key_states = repeat_kv(key_states, self.n_rep_)

        attn_output = eager_attention_forward(
            query_states, key_states, value_states, attention_mask
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)

        return self.o_proj_(attn_output, input_args)


class Phi3FlashAttention2(Phi3Attention):
    def __init__(
        self, qkv_proj: nn.Module, o_proj: nn.Module, layer_idx: int, args: Phi3Config
    ) -> None:
        assert is_flash_attn_2_available(), "Flash Attention is not available"
        super().__init__(qkv_proj, o_proj, layer_idx, args)
        self.sliding_window_ = args.sliding_window_

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

        # cutting
        qkv = self.qkv_proj_.forward(hidden_states, input_args)
        query_pos = self.n_heads_ * self.head_dim_
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.n_kv_heads_ * self.head_dim_]
        value_states = qkv[..., query_pos + self.n_kv_heads_ * self.head_dim_ :]

        # viewing
        query_states = query_states.view(
            bsz, q_len, self.n_heads_, self.head_dim_
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0]

        # apply rotary embedding
        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Activate slicing cache
        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx_) > 0
            if (
                self.sliding_window_ is not None
                and kv_seq_len > self.sliding_window_
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.sliding_window_

                past_key = past_key_value[self.layer_idx_][0]
                past_value = past_key_value[self.layer_idx_][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.sliding_window_ - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.sliding_window - 1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])],
                        dim=-1,
                    )

            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx_, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.n_rep_)
        value_states = repeat_kv(value_states, self.n_rep_)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if executor.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            is_causal=self.is_causal_,
            sliding_window=self.sliding_window_,
        ).to(input_dtype)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj_(attn_output, input_args)

        return attn_output


PHI3_ATTENTION_CLASSES = {
    "eager": Phi3Attention,
    "flash_attn": Phi3FlashAttention2,
}


class Phi3MLP(LLMFeedForward):
    def __init__(self, gate: nn.Module, down: nn.Module, args: Phi3Config) -> None:
        super().__init__()
        # feed forward
        self.gate_up_proj_ = Linear(gate, args.device_)
        self.down_proj_ = Linear(down, args.device_)
        self.act_ = ACT2FN[args.hidden_act_]

    def state_dict(self) -> Dict[str, nn.Module]:
        return {
            "gate_up_proj": self.gate_up_proj_,
            "down_proj": self.down_proj_,
        }

    def _batch_forward(
        self, hidden_states: torch.Tensor, input_args: LLMModelInput
    ) -> torch.Tensor:
        up_states = self.gate_up_proj_(hidden_states, input_args)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.act_(gate)

        return self.down_proj_(up_states, input_args)

    def _lora_forward(
        self, lora_name: str, act_fn: nn.Module, data: torch.Tensor
    ) -> torch.Tensor:
        # Applying LoRA weights to FFN weights
        if lora_name in self.gate_up_proj_.loras_:
            gate_up_states = self.gate_up_proj_.loras_[lora_name].forward(
                self.gate_up_proj_.base_layer_.forward(data), data
            )
        else:
            gate_up_states = self.gate_up_proj_.base_layer_.forward(data)

        gate_states, up_states = gate_up_states.chunk(2, dim=-1)
        act_result = act_fn(gate_states) * up_states

        if lora_name in self.down_proj_.loras_:
            return self.down_proj_.loras_[lora_name].forward(
                self.down_proj_.base_layer_.forward(act_result), act_result
            )
        else:
            return self.down_proj_.base_layer_.forward(act_result)

    def _mixlora_forward(
        self, moe_name, act_fn, expert_mask, hidden_states, input_dtype
    ):
        common_gate_up = self.gate_up_proj_.base_layer_.forward(
            hidden_states.to(input_dtype)
        ).to(hidden_states.dtype)

        final_expert_states = []
        for expert_idx in range(expert_mask.shape[0]):
            _, top_x = torch.where(expert_mask[expert_idx])

            lora_name = f"moe.{moe_name}.experts.{expert_idx}"
            if lora_name in self.gate_up_proj_.loras_:
                gate_up_states = self.gate_up_proj_.loras_[lora_name].forward(
                    slice_tensor(common_gate_up, top_x, input_dtype),
                    slice_tensor(hidden_states, top_x, input_dtype),
                )
            else:
                gate_up_states = slice_tensor(common_gate_up, top_x, input_dtype)

            gate_states, up_states = gate_up_states.chunk(2, dim=-1)
            act_result = up_states * act_fn(gate_states)

            if lora_name in self.down_proj_.loras_:
                final_expert_states.append(
                    self.down_proj_.loras_[lora_name].forward(
                        self.down_proj_.base_layer_.forward(act_result),
                        act_result,
                    )
                )
            else:
                final_expert_states.append(
                    self.down_proj_.base_layer_.forward(act_result)
                )

        return final_expert_states


class Phi3DecoderLayer(LLMDecoder):
    def __init__(self, layer_id: int, config: Phi3Config) -> None:
        super().__init__()
        self.layer_id_: int = layer_id
        self.self_attn_: Phi3Attention = None
        self.mlp_: FeedForward = None
        self.input_layernorm_: Phi3RMSNorm = None

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop_)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop_)
        self.post_attention_layernorm_: Phi3RMSNorm = None

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
        attn_outputs = self.self_attn_.forward(
            hidden_states,
            input_args,
            rotary_emb,
            attention_mask,
            cache_position,
            past_key_value,
        )
        hidden_states = residual + self.resid_attn_dropout(attn_outputs)
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm_(hidden_states)
        hidden_states, router_logits = self.mlp_.forward(hidden_states, input_args)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        if input_args.output_router_logits_:
            router_logits = collect_plugin_router_logtis(
                router_logits, input_args, self
            )

        return hidden_states, *router_logits


class Phi3ForCausalLM(LLMForCausalLM):
    def _init_rope(self):
        if self.config_.rope_scaling_ is None:
            return Phi3RotaryEmbedding(
                self.config_.head_dim_,
                max_position_embeddings=self.config_.max_seq_len_,
                base=self.config_.rope_theta_,
                device=self.config_.device_,
            )
        else:
            scaling_type = self.config_.rope_scaling_["type"]
            assert scaling_type == "longrope", ValueError(
                f"Unknown RoPE scaling type {scaling_type}"
            )
            return Phi3LongRoPEScaledRotaryEmbedding(
                self.config_.head_dim_,
                config=self.config_,
                device=self.config_.device_,
            )

    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_: Phi3Embedding = None
        self.norm_: Phi3Embedding = None
        self.rotary_emb_ = self._init_rope()
        self.lm_head_ = nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=False,
            dtype=config.dtype_,
            device=config.device_,
        )
        self.layers_: List[Phi3DecoderLayer] = []

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

    def model_config(self) -> Phi3Config:
        return self.config_

    @staticmethod
    def from_pretrained(
        llm_model: modeling_phi3.Phi3ForCausalLM,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = executor.default_device_name(),
    ):
        llm_config: modeling_phi3.Phi3Config = llm_model.config
        llm_args = Phi3Config(
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
            resid_pdrop_=llm_config.resid_pdrop,
            max_seq_len_=llm_config.max_position_embeddings,
            rope_theta_=llm_config.rope_theta,
            rope_scaling_=llm_config.rope_scaling,
            original_max_position_embeddings_=llm_config.original_max_position_embeddings,
            pad_token_id_=llm_config.pad_token_id,
            attn_implementation_=attn_impl,
            use_sliding_window_=use_sliding_window,
            sliding_window_=llm_config.sliding_window,
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = Phi3ForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = Phi3Embedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_
        )
        model.norm_ = Phi3RMSNorm(llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = Phi3DecoderLayer(idx, llm_args)
            decoder.self_attn_ = PHI3_ATTENTION_CLASSES[llm_args.attn_implementation_](
                layer.self_attn.qkv_proj,
                layer.self_attn.o_proj,
                idx,
                llm_args,
            )
            decoder.mlp_ = FeedForward(
                Phi3MLP(
                    layer.mlp.gate_up_proj,
                    layer.mlp.down_proj,
                    llm_args,
                )
            )
            decoder.input_layernorm_ = Phi3RMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_
            )
            decoder.post_attention_layernorm_ = Phi3RMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_
            )
            model.layers_.append(decoder)

        return model
