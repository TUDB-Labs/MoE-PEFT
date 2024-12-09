from typing import Dict, Optional, TypeAlias

import torch

from moe_peft.common import AdapterConfig, LoraConfig

from .loramoe import LoraMoe, LoraMoeConfig
from .mixlora import (
    DynamicRouterLoss,
    DynamicSparseMoe,
    MixLoraConfig,
    MixtralRouterLoss,
    MixtralSparseMoe,
    SwitchRouterLoss,
    SwitchSparseMoe,
)
from .mola import MolaConfig, MolaRouterLoss, MolaSparseMoe

peft_type_dict = {
    "LORA": LoraConfig,
    "MIXLORA": MixLoraConfig,
    "LORAMOE": LoraMoeConfig,
    "MOLA": MolaConfig,
}

routing_strategy_dict = {
    "mixlora": MixLoraConfig,
    "mixlora-dynamic": MixLoraConfig,
    "mixlora-switch": MixLoraConfig,
    "loramoe": LoraMoeConfig,
    "mola": MolaConfig,
}

router_loss_dict = {
    "mixlora": MixtralRouterLoss,
    "mixlora-dynamic": DynamicRouterLoss,
    "mixlora-switch": SwitchRouterLoss,
    "mola": MolaRouterLoss,
}

moe_layer_dict = {
    "mixlora": MixtralSparseMoe,
    "mixlora-dynamic": DynamicSparseMoe,
    "mixlora-switch": SwitchSparseMoe,
    "loramoe": LoraMoe,
    "mola": MolaSparseMoe,
}


def lora_config_factory(config: Dict[str, any]) -> LoraConfig:
    if peft_type_dict.get(config.get("peft_type", ""), None) is not None:
        config_class: TypeAlias[AdapterConfig] = peft_type_dict[config["peft_type"]]
    elif (
        routing_strategy_dict.get(config.get("routing_strategy", ""), None) is not None
    ):
        config_class: TypeAlias[AdapterConfig] = routing_strategy_dict[
            config["routing_strategy"]
        ]
    else:
        config_class = LoraConfig

    return config_class.from_config(config).check()


def adapter_factory(peft_type: str, adapter_name: str, **kwargs) -> LoraConfig:
    kwargs["peft_type"] = peft_type
    config = lora_config_factory(kwargs)
    config.adapter_name = adapter_name
    return config


def router_loss_factory(config: MixLoraConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        return None
    if config.router_loss_:
        return router_loss_dict[config.routing_strategy_](config)
    else:
        return None


def moe_layer_factory(
    in_features: int,
    device: torch.device,
    config: MolaConfig,
    gate: Optional[torch.Tensor] = None,
) -> torch.nn.Module:
    if config.routing_strategy_ not in moe_layer_dict:
        raise ValueError(f"Unknown routing strategy {config.routing_strategy_}")
    return moe_layer_dict[config.routing_strategy_](in_features, device, config, gate)


__all__ = [
    "MixLoraConfig",
    "MixtralRouterLoss",
    "MixtralSparseMoe",
    "DynamicRouterLoss",
    "DynamicSparseMoe",
    "SwitchRouterLoss",
    "SwitchSparseMoe",
    "LoraMoeConfig",
    "LoraMoe",
    "MolaConfig",
    "MolaSparseMoe",
    "peft_type_dict",
    "routing_strategy_dict",
    "router_loss_dict",
    "moe_layer_dict",
    "lora_config_factory",
    "router_loss_factory",
    "moe_layer_factory",
    "adapter_factory",
]
