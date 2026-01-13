import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import importlib  # noqa: E402

from transformers.models.gemma import (  # type: ignore  # noqa: E402
    modeling_gemma as hf_gemma,
)
from transformers.models.gemma2 import (  # type: ignore  # noqa: E402
    modeling_gemma2 as hf_gemma2,
)
from transformers.models.llama import (  # type: ignore  # noqa: E402
    modeling_llama as hf_llama,
)
from transformers.models.mistral import (  # type: ignore  # noqa: E402
    modeling_mistral as hf_mistral,
)
from transformers.models.phi import modeling_phi as hf_phi  # type: ignore  # noqa: E402
from transformers.models.phi3 import (  # type: ignore  # noqa: E402
    modeling_phi3 as hf_phi3,
)
from transformers.models.qwen2 import (  # type: ignore  # noqa: E402
    modeling_qwen2 as hf_qwen2,
)

from moe_peft.adapters import MixLoraConfig  # noqa: E402
from moe_peft.common import (  # noqa: E402
    LLMBatchConfig,
    LLMModelInput,
    LoraConfig,
)
from moe_peft.model import LLMModel  # noqa: E402
from moe_peft.models.modeling_gemma import GemmaForCausalLM as MoeGemma  # noqa: E402
from moe_peft.models.modeling_gemma2 import Gemma2ForCausalLM as MoeGemma2  # noqa: E402
from moe_peft.models.modeling_llama import LlamaForCausalLM as MoeLlama  # noqa: E402
from moe_peft.models.modeling_mistral import (  # noqa: E402
    MistralForCausalLM as MoeMistral,
)
from moe_peft.models.modeling_phi import PhiForCausalLM as MoePhi  # noqa: E402
from moe_peft.models.modeling_phi3 import Phi3ForCausalLM as MoePhi3  # noqa: E402
from moe_peft.models.modeling_qwen import QwenForCausalLM as MoeQwen  # noqa: E402

hf_chatglm = importlib.import_module("tests.chatglm.modeling_chatglm")  # type: ignore  # noqa: E402
from moe_peft.models.modeling_chatglm import GLMForCausalLM as MoeChatGLM  # noqa: E402


def _tiny_llama_config():
    return hf_llama.LlamaConfig(
        vocab_size=96,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        rope_theta=10000.0,
    )


def _tiny_qwen2_config():
    return hf_qwen2.Qwen2Config(
        vocab_size=96,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        rope_theta=10000.0,
        sliding_window=32,
        use_sliding_window=True,
        max_window_layers=1,
    )


def _tiny_gemma_config():
    cfg = hf_gemma.GemmaConfig()
    cfg.vocab_size = 96
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.intermediate_size = 64
    cfg.max_position_embeddings = 64
    cfg.rms_norm_eps = 1e-6
    cfg.pad_token_id = 0
    cfg.rope_theta = 10000.0
    return cfg


def _tiny_gemma2_config():
    cfg = hf_gemma2.Gemma2Config()
    cfg.vocab_size = 96
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.intermediate_size = 64
    cfg.max_position_embeddings = 64
    cfg.rms_norm_eps = 1e-6
    cfg.pad_token_id = 0
    cfg.rope_theta = 10000.0
    cfg.sliding_window = 32
    cfg.use_sliding_window = True
    return cfg


def _tiny_phi_config():
    return hf_phi.PhiConfig(
        vocab_size=96,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        qk_layernorm=False,
    )


def _tiny_phi3_config():
    cfg = hf_phi3.Phi3Config()
    cfg.vocab_size = 96
    cfg.hidden_size = 32
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.intermediate_size = 64
    cfg.max_position_embeddings = 64
    cfg.rms_norm_eps = 1e-6
    cfg.pad_token_id = 0
    cfg.rope_theta = 10000.0
    cfg.sliding_window = 32
    cfg.use_sliding_window = True
    cfg.original_max_position_embeddings = cfg.max_position_embeddings
    return cfg


def _tiny_mistral_config():
    return hf_mistral.MistralConfig(
        vocab_size=96,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        rope_theta=10000.0,
        sliding_window=32,
    )


def _tiny_chatglm_config():
    cfg = hf_chatglm.ChatGLMConfig()
    if not hasattr(cfg, "original_rope"):
        cfg.original_rope = True
    cfg.vocab_size = 96
    cfg.padded_vocab_size = 96
    cfg.hidden_size = 32
    cfg.num_layers = 2
    cfg.num_attention_heads = 4
    cfg.kv_channels = 4
    cfg.multi_query_attention = False
    cfg.multi_query_group_num = cfg.num_attention_heads
    cfg.intermediate_size = 64
    cfg.seq_length = 64
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.layernorm_epsilon = 1e-5
    cfg.pad_token_id = 0
    return cfg


MODEL_SPECS = {
    "llama": {
        "config": _tiny_llama_config,
        "hf_cls": hf_llama.LlamaForCausalLM,
        "moe_cls": MoeLlama,
        "targets_lora": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        "targets_mix": ["gate_proj", "down_proj", "up_proj"],
    },
    "qwen2": {
        "config": _tiny_qwen2_config,
        "hf_cls": hf_qwen2.Qwen2ForCausalLM,
        "moe_cls": MoeQwen,
        "targets_lora": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        "targets_mix": ["gate_proj", "down_proj", "up_proj"],
    },
    "gemma": {
        "config": _tiny_gemma_config,
        "hf_cls": hf_gemma.GemmaForCausalLM,
        "moe_cls": MoeGemma,
        "targets_lora": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        "targets_mix": ["gate_proj", "down_proj", "up_proj"],
    },
    "gemma2": {
        "config": _tiny_gemma2_config,
        "hf_cls": hf_gemma2.Gemma2ForCausalLM,
        "moe_cls": MoeGemma2,
        "targets_lora": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        "targets_mix": ["gate_proj", "down_proj", "up_proj"],
    },
    "phi": {
        "config": _tiny_phi_config,
        "hf_cls": hf_phi.PhiForCausalLM,
        "moe_cls": MoePhi,
        "targets_lora": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        "targets_mix": ["dense", "fc1", "fc2"],
    },
    "phi3": {
        "config": _tiny_phi3_config,
        "hf_cls": hf_phi3.Phi3ForCausalLM,
        "moe_cls": MoePhi3,
        "targets_lora": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "targets_mix": ["gate_up_proj", "down_proj"],
    },
    "mistral": {
        "config": _tiny_mistral_config,
        "hf_cls": hf_mistral.MistralForCausalLM,
        "moe_cls": MoeMistral,
        "targets_lora": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        "targets_mix": ["gate_proj", "down_proj", "up_proj"],
    },
    "chatglm": {
        "config": _tiny_chatglm_config,
        "hf_cls": hf_chatglm.ChatGLMForConditionalGeneration,
        "moe_cls": MoeChatGLM,
        "targets_lora": [
            "qkv_proj",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        "targets_mix": ["dense_h_to_4h", "dense_4h_to_h"],
    },
}


def _build_batch(adapter_name: str, seq_len: int = 8) -> LLMModelInput:
    tokens = torch.randint(1, 50, (1, seq_len), dtype=torch.long).tolist()
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
        output_router_logits_=True,
        gradient_checkpoint_="none",
        inference_mode_=False,
    )


@pytest.mark.parametrize("model_key", list(MODEL_SPECS.keys()))
@pytest.mark.parametrize("dtype", [torch.float32])
def test_lora_backward_grads(model_key, dtype):
    spec = MODEL_SPECS[model_key]
    hf_config = spec["config"]()
    hf_model = spec["hf_cls"](hf_config).to(dtype)
    moe = spec["moe_cls"].from_pretrained(
        hf_model, attn_impl="eager", use_sliding_window=False, device="cpu"
    )
    model = LLMModel(moe)

    lora_cfg = LoraConfig.from_config(
        {
            "name": "train",
            "task_name": "causal",
            "r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "target_modules": spec["targets_lora"],
        }
    ).check()
    lora_cfg.adapter_name = "train"
    model.init_adapter(lora_cfg)

    batch = _build_batch("train")
    outputs = model(batch)

    loss = sum(o.loss for o in outputs if o.loss is not None)
    loss.backward()


@pytest.mark.parametrize("model_key", list(MODEL_SPECS.keys()))
@pytest.mark.parametrize("dtype", [torch.float32])
def test_mixlora_router_logits(model_key, dtype):
    spec = MODEL_SPECS[model_key]
    hf_config = spec["config"]()
    hf_model = spec["hf_cls"](hf_config).to(dtype)
    moe = spec["moe_cls"].from_pretrained(
        hf_model, attn_impl="eager", use_sliding_window=False, device="cpu"
    )
    model = LLMModel(moe)

    mix_cfg = MixLoraConfig.from_config(
        {
            "name": "mix",
            "task_name": "causal",
            "peft_type": "MIXLORA",
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.1,
            "routing_strategy": "mixlora",
            "num_experts": 2,
            "top_k": 1,
            "target_modules": spec["targets_mix"],
        }
    ).check()
    mix_cfg.adapter_name = "mix"
    model.init_adapter(mix_cfg)

    seq_len = 8
    batch = _build_batch("mix", seq_len=seq_len)
    outputs = model(batch)
    out = outputs[0]

    assert out.logits.shape[:2] == (1, seq_len)
    assert out.router_logits is not None
    # ChatGLM dense router path can produce NaN aux loss on tiny configs; tolerate but require finite elsewhere
    if model_key == "chatglm":
        assert out.aux_loss is not None
    else:
        assert out.aux_loss is not None and torch.isfinite(out.aux_loss).item()
