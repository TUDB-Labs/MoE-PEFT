import math
from functools import wraps
from typing import Optional, Tuple

import torch

from .config import LLMModelConfig


def dynamic_rope_update(rope_forward):
    """Decorator to update RoPE frequencies when using dynamic rope types."""

    def longrope_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        original_max = getattr(self.config, "original_max_position_embeddings", None)
        original_max = original_max or self.config.max_seq_len_
        if seq_len > original_max:
            if not hasattr(self, "long_inv_freq"):
                self.long_inv_freq, _ = self.rope_init_fn(
                    self.config, device, seq_len=original_max + 1
                )
            self.register_buffer("inv_freq", self.long_inv_freq, persistent=False)
        else:
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)

    def dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @wraps(rope_forward)
    def wrapper(self, x, position_ids):
        if "dynamic" in getattr(self, "rope_type", ""):
            dynamic_frequency_update(self, position_ids, device=x.device)
        elif getattr(self, "rope_type", "") == "longrope":
            longrope_frequency_update(self, position_ids, device=x.device)
        return rope_forward(self, x, position_ids)

    return wrapper


def _base_rope_dims(config: LLMModelConfig) -> Tuple[float, int, float, int, int]:
    base = config.rope_theta_
    partial_rotary_factor = config.partial_rotary_factor_ or 1.0
    head_dim = config.head_dim_ or config.dim_ // config.n_heads_
    dim = int(head_dim * partial_rotary_factor)
    max_pos = config.max_seq_len_
    return base, dim, partial_rotary_factor, head_dim, max_pos


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
        base, dim, _, _, _ = _base_rope_dims(config)
    else:
        raise ValueError("Config or rope_kwargs must be provided for RoPE init.")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, 1.0


def _compute_linear_scaling_rope_parameters(
    config: LLMModelConfig,
    device: torch.device,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    inv_freq, attention_factor = _compute_default_rope_parameters(
        config, device, seq_len, **rope_kwargs
    )
    factor = config.rope_scaling_["factor"]
    inv_freq = inv_freq / factor
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: LLMModelConfig,
    device: torch.device,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    base, dim, _, _, max_pos = _base_rope_dims(config)
    factor = config.rope_scaling_["factor"]

    if seq_len is None:
        seq_len = max_pos
    elif isinstance(seq_len, torch.Tensor):
        seq_len = torch.maximum(
            seq_len,
            torch.tensor(max_pos, dtype=seq_len.dtype, device=seq_len.device),
        )
    else:
        seq_len = max(seq_len, max_pos)

    base = base * ((factor * seq_len / max_pos) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


def _compute_yarn_parameters(
    config: LLMModelConfig,
    device: torch.device,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    base, dim, _, _, max_pos = _base_rope_dims(config)
    rope_scaling = config.rope_scaling_
    factor = rope_scaling["factor"]
    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")
    original_max = rope_scaling.get("original_max_position_embeddings") or max_pos

    def get_mscale(scale, mscale_val=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale_val * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = get_mscale(factor)

    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1

    def find_correction_dim(num_rotations, dim_val, base_val, max_position_embeddings):
        return (
            dim_val * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base_val))

    def find_correction_range(
        low_rot, high_rot, dim_val, base_val, max_position_embeddings, truncate
    ):
        low = find_correction_dim(low_rot, dim_val, base_val, max_position_embeddings)
        high = find_correction_dim(high_rot, dim_val, base_val, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim_val - 1)

    def linear_ramp_factor(min_val, max_val, dim_val):
        if min_val == max_val:
            max_val += 0.001

        linear_func = (torch.arange(dim_val, dtype=torch.float32) - min_val) / (
            max_val - min_val
        )
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    pos_freqs = base ** (
        torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = rope_scaling.get("truncate", True)
    low, high = find_correction_range(
        beta_fast, beta_slow, dim, base, original_max, truncate
    )

    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(
        device=device, dtype=torch.float
    )
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor


def _compute_longrope_parameters(
    config: LLMModelConfig,
    device: torch.device,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    base, dim, _, _, max_pos = _base_rope_dims(config)
    rope_scaling = config.rope_scaling_
    long_factor = rope_scaling["long_factor"]
    short_factor = rope_scaling["short_factor"]
    factor = rope_scaling.get("factor")
    attention_factor = rope_scaling.get("attention_factor")

    original_max = getattr(config, "original_max_position_embeddings", None)
    if original_max:
        factor = max_pos / original_max
    else:
        original_max = max_pos

    if attention_factor is None:
        if factor is not None and factor > 1.0:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max))
        else:
            attention_factor = 1.0

    ext_factors = (
        torch.tensor(long_factor, dtype=torch.float32, device=device)
        if seq_len and seq_len > original_max
        else torch.tensor(short_factor, dtype=torch.float32, device=device)
    )
    inv_freq_shape = (
        torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    )
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: LLMModelConfig,
    device: torch.device,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    inv_freq, attention_factor = _compute_default_rope_parameters(
        config, device, seq_len, **rope_kwargs
    )

    factor = config.rope_scaling_["factor"]
    low_freq_factor = config.rope_scaling_["low_freq_factor"]
    high_freq_factor = config.rope_scaling_["high_freq_factor"]
    old_context_len = config.rope_scaling_["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
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
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}
