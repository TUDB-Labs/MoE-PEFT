import torch

from moe_peft.common import LLMModelOutput
from moe_peft.generator import GenerateConfig, generate


class DummyTokenizer:
    pad_id_ = 0
    eos_id_ = 1

    def encode(
        self, text, add_special_tokens: bool = True
    ):  # pragma: no cover - simple stub
        # Map each character to a small integer offset to avoid pad/eos
        return [2 + (ord(c) % 5) for c in text]

    def decode(self, tokens):  # pragma: no cover - simple stub
        return " ".join(str(t) for t in tokens)


class DummyModel:
    def __init__(self, target_token: int = 7, max_seq_len: int = 2048):
        self.device_ = "cpu"
        self.config_ = type("cfg", (), {"max_seq_len_": max_seq_len})()
        self.model_ = type("inner", (), {"cache_implementation": lambda self: None})()
        self.target_token = target_token

    def forward(
        self, input_args, past_key_values=None
    ):  # pragma: no cover - behavior validated via generate()
        batch_cfg = input_args.batch_configs_[0]
        batch_size = len(input_args.batch_tokens_)
        seq_len = len(input_args.batch_tokens_[0])
        vocab_size = 10
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, :, self.target_token] = 5.0  # force argmax to target_token
        return [
            LLMModelOutput(
                adapter_name=batch_cfg.adapter_name_,
                logits=logits,
                batch_start_idx_=0,
                batch_end_idx_=batch_size,
            )
        ]


def test_generate_greedy_single_step():
    tokenizer = DummyTokenizer()
    model = DummyModel(target_token=9)

    cfg = GenerateConfig(
        adapter_name="test",
        prompts=["hi"],
        do_sample=False,  # enforce greedy
        # leave top_p/top_k non-zero to ensure they are ignored in greedy path
    )

    outputs = generate(
        model,
        tokenizer,
        [cfg],
        max_gen_len=1,  # single decoding step
        use_cache=False,  # avoid cache factory in test
        dispatch_strategy="fifo",
    )

    assert outputs["test"] == ["9"]
