import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers.models.mistral import (  # type: ignore  # noqa: E402
    modeling_mistral as hf_mistral,
)
from transformers.models.phi3 import (  # type: ignore  # noqa: E402
    modeling_phi3 as hf_phi3,
)

from moe_peft.common import AdapterConfig, LLMBatchConfig, LLMModelInput  # noqa: E402
from moe_peft.model import LLMModel  # noqa: E402
from moe_peft.models.modeling_mistral import (  # noqa: E402
    MistralForCausalLM as MoeMistral,
)
from moe_peft.models.modeling_phi3 import Phi3ForCausalLM as MoePhi3  # noqa: E402


def build_batch(adapter_name: str, seq_len: int = 16):
    tokens = torch.randint(1, 100, (1, seq_len), dtype=torch.long).tolist()
    masks = torch.ones((1, seq_len), dtype=torch.long).tolist()

    batch_cfg = LLMBatchConfig(
        adapter_name_=adapter_name,
        batch_start_idx_=0,
        batch_end_idx_=1,
    )

    return LLMModelInput(
        batch_configs_=[batch_cfg],
        batch_tokens_=tokens,
        batch_labels_=tokens,
        batch_masks_=masks,
        output_router_logits_=False,
        gradient_checkpoint_="none",
        inference_mode_=False,
    )


@pytest.mark.parametrize("dtype", [torch.float32])
def test_mistral_alignment_minimal_forward(dtype):
    cfg = hf_mistral.MistralConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=172,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        rope_theta=10000.0,
        sliding_window=64,
    )
    hf_model = hf_mistral.MistralForCausalLM(cfg).to(dtype)

    moe = MoeMistral.from_pretrained(
        hf_model, attn_impl="eager", use_sliding_window=False, device="cpu"
    )
    model = LLMModel(moe)
    model.init_adapter(AdapterConfig(adapter_name="base", task_name="causal"))

    batch = build_batch("base", seq_len=12)
    outputs = model(batch)

    assert len(outputs) == 1
    out = outputs[0]
    assert out.logits.shape[:2] == (1, 12)
    assert out.logits.shape[2] == cfg.vocab_size
    assert out.loss is not None and torch.isfinite(out.loss).item()


@pytest.mark.parametrize("dtype", [torch.float32])
def test_phi3_alignment_minimal_forward(dtype):
    cfg = hf_phi3.Phi3Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=172,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        rope_theta=10000.0,
    )
    hf_model = hf_phi3.Phi3ForCausalLM(cfg).to(dtype)

    moe = MoePhi3.from_pretrained(
        hf_model, attn_impl="eager", use_sliding_window=False, device="cpu"
    )
    model = LLMModel(moe)
    model.init_adapter(AdapterConfig(adapter_name="base", task_name="causal"))

    batch = build_batch("base", seq_len=12)
    outputs = model(batch)

    assert len(outputs) == 1
    out = outputs[0]
    assert out.logits.shape[:2] == (1, 12)
    assert out.logits.shape[2] == cfg.vocab_size
    assert out.loss is not None and torch.isfinite(out.loss).item()
