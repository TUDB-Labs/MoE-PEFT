import torch

import moe_peft.common.attention as attn


def _ensure_index_first_axis():
    if hasattr(attn, "index_first_axis"):
        return

    def _fallback_index_first_axis(tensor, indices):
        return tensor.index_select(0, indices)

    attn.index_first_axis = _fallback_index_first_axis


def _make_inputs(kv_len, q_len, num_q_heads=4, num_kv_heads=2, head_dim=8):
    torch.manual_seed(0)
    query = torch.randn(1, q_len, num_q_heads, head_dim)
    key = torch.randn(1, kv_len, num_kv_heads, head_dim)
    value = torch.randn(1, kv_len, num_kv_heads, head_dim)
    return query, key, value


def test_upad_input_expands_all_ones_mask_to_kv_len():
    _ensure_index_first_axis()

    query, key, value = _make_inputs(kv_len=4, q_len=4)
    attention_mask = torch.ones(1, 2, dtype=torch.int32)

    (
        query_unpad,
        key_unpad,
        _value_unpad,
        indices_q,
        (cu_q, cu_k),
        (max_q, max_k),
    ) = attn._upad_input(query, key, value, attention_mask, query_length=4)

    assert max_k == 4
    assert max_q == 4
    assert cu_k[-1].item() == 4
    assert cu_q[-1].item() == 4
    assert key_unpad.shape[0] == 4
    assert query_unpad.shape[0] == 4
    assert indices_q.numel() == 4


def test_upad_input_left_pads_short_mask_without_truncation():
    _ensure_index_first_axis()

    query, key, value = _make_inputs(kv_len=5, q_len=5)
    attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.int32)

    (
        query_unpad,
        key_unpad,
        _value_unpad,
        indices_q,
        (cu_q, cu_k),
        (max_q, max_k),
    ) = attn._upad_input(query, key, value, attention_mask, query_length=5)

    assert max_k == 2
    assert max_q == 2
    assert cu_k[-1].item() == 2
    assert cu_q[-1].item() == 2
    assert key_unpad.shape[0] == 2
    assert query_unpad.shape[0] == 2
    assert indices_q.numel() == 2
