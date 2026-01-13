import copy
from typing import List, Optional

import torch

from .abstracts import LLMDecoder, LLMModelInput


@torch.jit.script
def tsallis_entropy(p: torch.Tensor, q: float, normalize: bool = True) -> torch.Tensor:
    N = p.size(dim=-1)
    if q == 1.0:
        ent_terms = torch.where(p > 0, p * torch.log(p), torch.zeros_like(p))
        entropy = -torch.sum(ent_terms, dim=-1)
        max_entropy = torch.log(torch.tensor(N, dtype=p.dtype, device=p.device))
    else:
        entropy = (1 - torch.sum(p**q, dim=-1)) / (q - 1)
        max_entropy = torch.tensor(
            (1 - N ** (1 - q)) / (q - 1), dtype=p.dtype, device=p.device
        )

    if normalize:
        return entropy / max_entropy
    else:
        return entropy


def shannon_entropy(p: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    return tsallis_entropy(p, 1.0, normalize)


@torch.jit.script
def renyi_entropy(p: torch.Tensor, a: float, normalize: bool = True) -> torch.Tensor:
    N = p.size(dim=-1)
    if a == 1.0:
        ent_terms = torch.where(p > 0, p * torch.log(p), torch.zeros_like(p))
        entropy = -torch.sum(ent_terms, dim=-1)
    else:
        sum_pa = torch.sum(p**a, dim=-1)
        entropy = 1 / (1 - a) * torch.log(sum_pa)

    max_entropy = torch.log(torch.tensor(N, dtype=p.dtype, device=p.device))
    if normalize:
        return entropy / max_entropy
    else:
        return entropy


def slice_tensor(
    data: torch.Tensor,
    slice: torch.Tensor,
    dtype: torch.dtype,
    last_value: Optional[torch.Tensor] = None,
):
    if last_value is None:
        # for macOS debugging, please uncomment this line
        # assert data.dtype in (torch.float, torch.int, torch.bool)
        return data[None, slice].reshape(-1, data.shape[-1]).to(dtype)
    else:
        return last_value


def unpack_router_logits(gate_logits: List[torch.Tensor]) -> torch.Tensor:
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
    )
    return concatenated_gate_logits


def collect_plugin_router_logtis(
    router_logits, input_args: LLMModelInput, decoder_layer: LLMDecoder
):
    if router_logits is None or len(router_logits) == 0:
        router_logits = [None for _ in range(len(input_args.batch_configs_))]

    attn_proj, mlp_proj = decoder_layer.state_dict()
    all_proj = copy.copy(attn_proj)
    all_proj.update(mlp_proj)
    for idx, config in enumerate(input_args.batch_configs_):
        if router_logits[idx] is not None:
            continue
        adapter_name = config.adapter_name_
        for proj in all_proj.values():
            if adapter_name in proj.moes_ and hasattr(
                proj.moes_[adapter_name], "router_logits_"
            ):
                if router_logits[idx] is None:
                    router_logits[idx] = []
                router_logits[idx].append(proj.moes_[adapter_name].router_logits_)

    for proj in all_proj.values():
        for adapter_name in proj.moes_.keys():
            if hasattr(proj.moes_[adapter_name], "router_logits_"):
                proj.moes_[adapter_name].router_logits_ = None

    for idx, logits in enumerate(router_logits):
        if isinstance(logits, list):
            router_logits[idx] = torch.cat(logits, 0)

    return router_logits
