from abc import ABCMeta
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import LLMModelConfig, LLMModelInput


class LLMCache(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError(
            "Make sure to implement `get_seq_length` in a subclass."
        )

    def get_max_length(self) -> Optional[int]:
        raise NotImplementedError(
            "Make sure to implement `get_max_length` in a subclass."
        )

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx.to(device)
            )


class LLMAttention(metaclass=ABCMeta):
    @classmethod
    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {}

    @classmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[LLMCache] = None,
    ):
        pass


class LLMFeedForward(metaclass=ABCMeta):
    @classmethod
    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {}

    @classmethod
    def _batch_forward(
        self, hidden_states: torch.Tensor, input_args: LLMModelInput
    ) -> torch.Tensor:
        pass

    @classmethod
    def _lora_forward(
        self, lora_name: str, act_fn: torch.nn.Module, data: torch.Tensor
    ) -> torch.Tensor:
        pass


class LLMMoeBlock(metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

        self.adapter_name_: str = None
        self.dtype_: torch.dtype = None
        self.gate_: torch.nn.Linear = None
        self.experts_: int = None
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

    @classmethod
    def forward(
        self,
        residual: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple:
        pass


class LLMDecoder(metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn_: LLMAttention = None
        self.mlp_: LLMFeedForward = None

    @classmethod
    def state_dict(
        self,
    ) -> Tuple[Dict[str, torch.nn.Module], Dict[str, torch.nn.Module]]:
        return {}

    @classmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[LLMCache] = None,
    ):
        pass


class LLMOutput(metaclass=ABCMeta):
    @classmethod
    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {}

    @classmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def loss(
        self,
        input_ids: torch.Tensor,
        output_logits: torch.Tensor,
        labels: List[List[int]],
    ) -> torch.Tensor:
        pass


class LLMForCausalLM(metaclass=ABCMeta):
    @classmethod
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def rotary_embed(
        self, input_tensor: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @classmethod
    def decoder_stack(self) -> List[LLMDecoder]:
        pass

    @classmethod
    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[LLMCache],
    ) -> torch.Tensor:
        pass

    @classmethod
    def cache_implementation(self) -> str:
        return "dynamic"

    @classmethod
    def model_config(self) -> LLMModelConfig:
        pass

    @staticmethod
    def from_pretrained(llm_model, **kwargs):
        pass
