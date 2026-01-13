import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers.models.gemma2 import (  # type: ignore  # noqa: E402
    modeling_gemma2 as hf_gemma2,
)

from moe_peft.common import AdapterConfig, LLMBatchConfig, LLMModelInput  # noqa: E402
from moe_peft.model import LLMModel  # noqa: E402
from moe_peft.models.modeling_gemma2 import Gemma2ForCausalLM as MoeGemma2  # noqa: E402


def build_batch(adapter_name: str, seq_len: int = 12):
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
def test_gemma2_eager_minimal_forward(dtype):
    cfg = hf_gemma2.Gemma2Config()
    cfg.vocab_size = 128
    cfg.hidden_size = 64
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 8
    cfg.intermediate_size = 172
    cfg.max_position_embeddings = 128
    cfg.rms_norm_eps = 1e-6
    cfg.pad_token_id = 0
    cfg.rope_theta = 10000.0

    hf_model = hf_gemma2.Gemma2ForCausalLM(cfg).to(dtype)
    moe = MoeGemma2.from_pretrained(
        hf_model, attn_impl="eager", use_sliding_window=False, device="cpu"
    )
    model = LLMModel(moe)
    model.init_adapter(AdapterConfig(adapter_name="base", task_name="causal"))

    outputs = model(build_batch("base"))
    out = outputs[0]
    assert out.logits.shape[:2] == (1, 12)
    assert out.logits.shape[2] == cfg.vocab_size
    assert out.loss is not None and torch.isfinite(out.loss).item()
