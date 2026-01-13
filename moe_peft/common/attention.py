import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.utils import is_flash_attn_2_available, is_torchdynamo_compiling

from .abstracts import LLMModelConfig
from .cache import LLMCache

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )


def prepare_4d_causal_attention_mask(
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: LLMCache,
) -> torch.Tensor:
    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    if past_seen_tokens is None:
        past_seen_tokens = 0

    # Infer past length from cache_position to stay aligned with static caches
    if cache_position is not None and cache_position.numel() > 0:
        inferred = int(cache_position.max().item() + 1 - input_tensor.shape[1])
        past_seen_tokens = max(past_seen_tokens, inferred)

    input_shape = input_tensor.shape[:2]
    key_value_length = input_shape[-1] + past_seen_tokens

    converter = AttentionMaskConverter(is_causal=True, sliding_window=None)

    # 2D mask → 4D
    if attention_mask is not None and len(attention_mask.shape) == 2:
        return converter.to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=input_tensor.dtype,
        )
    # 4D mask passthrough with validation + invert
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        inverted_mask = 1.0 - attention_mask
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(input_tensor.dtype).min
        )
    # No mask provided → causal only
    else:
        return converter.to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )


@dataclass
class AttentionMaskConverter:
    """Local copy of HF mask converter to avoid internal imports."""

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"sliding_window must be a positive integer, got `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, "str"] = "cpu",
    ) -> Optional[torch.Tensor]:
        if not self.is_causal:
            raise ValueError("Use `to_causal_4d` only when `is_causal=True`.")

        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        input_shape = (attention_mask_2d.shape[0], query_length)

        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "Causal converter needs `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is only implemented for causal masking"
            )

        expanded_attn_mask = self._expand_mask(
            attention_mask_2d, dtype, tgt_len=input_shape[-1]
        ).to(attention_mask_2d.device)

        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(
                expanded_attn_mask.bool(), torch.finfo(dtype).min
            )

        return expanded_attn_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [
                    torch.zeros(
                        tgt_len, past_key_values_length, dtype=dtype, device=device
                    ),
                    mask,
                ],
                dim=-1,
            )

        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1
            context_mask = torch.tril(
                torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal
            )
            if is_torchdynamo_compiling():
                mask = mask.clone()
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    @staticmethod
    def _expand_mask(
        mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
    ):
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = (
            mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        )

        inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask

        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    model_config: LLMModelConfig,
    scaling: Optional[float] = None,
    softcap: Optional[
        float
    ] = None,  # Softcap value for attention logits; applies tanh capping when provided
    **kwargs,
) -> torch.Tensor:
    if scaling is None:
        scaling = model_config.head_dim_**-0.5

    if model_config.n_kv_heads_ is not None:
        n_rep = model_config.n_heads_ // model_config.n_kv_heads_
        key_states = repeat_kv(key_states, n_rep)
        value_states = repeat_kv(value_states, n_rep)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        value_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


def _get_unpad_data(
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
):
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    if attention_mask.shape[0] != batch_size:
        raise ValueError(
            f"attention_mask batch {attention_mask.shape[0]} != key batch {batch_size}"
        )

    # Normalize attention_mask length against kv_seq_len (assumes left padding)
    if attention_mask.shape[-1] > kv_seq_len:
        attention_mask = attention_mask[:, -kv_seq_len:]
    elif attention_mask.shape[-1] < kv_seq_len:
        # If the mask is all ones but shorter (typical for cache-only masks), expand to kv length.
        if attention_mask.sum() == attention_mask.numel():
            attention_mask = torch.ones(
                (batch_size, kv_seq_len),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
        else:
            # Pad on the left to preserve padding alignment when possible.
            pad_len = kv_seq_len - attention_mask.shape[-1]
            attention_mask = F.pad(attention_mask, (pad_len, 0), value=0)

    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
            query_layer, attention_mask
        )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def prepare_fa2_from_position_ids(query, key, value, position_ids):
    # Flatten to (total_tokens, heads, head_dim)
    query = query.contiguous().view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))

    # Use cumulative sequence lengths derived from position_ids==0 boundaries
    # This is robust to non-monotonic position ids in packed sequences.
    position_ids = position_ids.view(-1)
    start_indices = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens = torch.cat(
        (
            start_indices.to(dtype=torch.int32, device=position_ids.device),
            torch.tensor(
                position_ids.size(0), dtype=torch.int32, device=position_ids.device
            ),
        )
    )

    # Compute max lengths from cu_seq_lens to handle non-increasing position ids
    max_length = cu_seq_lens.diff().max().item()

    return (
        query,
        key,
        value,
        start_indices.to(dtype=torch.int32, device=position_ids.device),
        (cu_seq_lens, cu_seq_lens),
        (max_length, max_length),
    )


def flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):

    # Align causal behavior with eager path
    causal = is_causal

    # Repeat K/V heads to match Q heads when using grouped-query attention
    n_heads_q = query_states.shape[1]
    n_kv_heads = key_states.shape[1]
    if n_heads_q % n_kv_heads != 0:
        raise ValueError(
            f"Q heads ({n_heads_q}) not divisible by KV heads ({n_kv_heads}) in flash attention."
        )
    n_rep = n_heads_q // n_kv_heads
    if n_rep > 1:
        key_states = repeat_kv(key_states, n_rep)
        value_states = repeat_kv(value_states, n_rep)

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size
        and sliding_window is not None
        and key_states.shape[1] > sliding_window
    )
    # Flash attention expects window size to be reduced by 1 for proper indexing
    flash_kwargs = (
        {"window_size": (sliding_window - 1, sliding_window - 1)}
        if use_sliding_windows
        else {}
    )

    if deterministic is not None:
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Flash expects (batch, seq_len, heads, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # Default softmax scale to 1/sqrt(head_dim) if not provided
    if softmax_scale is None:
        head_dim = query_states.size(-1)
        softmax_scale = 1.0 / float(head_dim) ** 0.5

        # Normalize attention_mask dtype to {0,1}
        if attention_mask is not None:
            attention_mask = (attention_mask != 0).to(torch.int32)

    # Determine whether to use varlen path by attention_mask or packed position ids
    # Disable position_ids-only padding-free path for now; prefer attention_mask-guided varlen
    is_fa_with_varlen_kwargs = all(
        kwarg is not None
        for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    )

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
            _upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        if isinstance(attn_output_unpad, tuple):
            attn_output_unpad = attn_output_unpad[0]
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    # Padding free path: sequences flattened into one total sequence
    elif is_fa_with_varlen_kwargs:
        batch_size = query_states.size(0)

        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = prepare_fa2_from_position_ids(
                query_states, key_states, value_states, position_ids
            )

            cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
            max_length_q, max_length_k = max_seq_lens

        else:
            query_states = query_states.reshape(
                -1, query_states.size(-2), query_states.size(-1)
            )
            key_states = key_states.reshape(
                -1, key_states.size(-2), key_states.size(-1)
            )
            value_states = value_states.reshape(
                -1, value_states.size(-2), value_states.size(-1)
            )

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        attn_output = attn_output.view(
            batch_size, -1, attn_output.size(-2), attn_output.size(-1)
        )

    else:
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

    return attn_output


ATTENTION_FUNCTIONS = {
    "eager": eager_attention_forward,
    "flash_attn": flash_attention_forward,
}
