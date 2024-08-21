import math
from typing import Optional, Tuple

import torch

from .config import LLMModelConfig


def _compute_default_rope_parameters(
    config: Optional[LLMModelConfig] = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta_
        partial_rotary_factor = (
            config.partial_rotary_factor_
            if config.partial_rotary_factor_ is not None
            else 1.0
        )
        head_dim = (
            config.dim_ // config.n_heads_
            if config.head_dim_ is None
            else config.head_dim_
        )
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: LLMModelConfig,
    device: torch.device,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(
        config, device, seq_len, **rope_kwargs
    )

    factor = config.rope_scaling_["factor"]  # `8` in the original implementation
    low_freq_factor = config.rope_scaling_[
        "low_freq_factor"
    ]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling_[
        "high_freq_factor"
    ]  # `4` in the original implementation
    old_context_len = config.rope_scaling_[
        "original_max_position_embeddings"
    ]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "llama3": _compute_llama3_parameters,
}
