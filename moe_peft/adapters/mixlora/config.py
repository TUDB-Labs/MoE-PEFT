import copy
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from transformers.activations import ACT2FN

from moe_peft.common import LoraConfig

available_routing_strategies = ["mixlora", "mixlora-dynamic", "mixlora-switch"]


@dataclass
class MixLoraConfig(LoraConfig):
    # expert lora
    expert_config_: LoraConfig = None
    # router config
    router_aux_loss_coef_: float = None
    router_init_range_: float = None
    routing_strategy_: str = None
    jitter_noise_: float = None
    router_loss_: bool = True
    num_experts_: int = None
    act_fn_: Optional[Union[str, torch.nn.Module]] = None
    # mixtral config
    top_k_: int = None
    # dynamic config
    top_p_: float = None
    temperature_: float = None
    # switch transformers config
    router_z_loss_coef_: float = None
    expert_capacity_: int = None
    ffn_dropout_: float = None
    sparse_step_: int = None

    def check(self) -> "MixLoraConfig":
        super().check()
        if self.expert_config_ is not None:
            self.expert_config_.check()
        assert (
            isinstance(self.router_aux_loss_coef_, float)
            and self.router_aux_loss_coef_ >= 0
        )
        assert (
            isinstance(self.router_init_range_, float) and self.router_init_range_ >= 0
        )
        assert (
            isinstance(self.routing_strategy_, str)
            and self.routing_strategy_ in available_routing_strategies
        )
        assert isinstance(self.jitter_noise_, float) and self.jitter_noise_ >= 0
        assert isinstance(self.router_loss_, bool)
        assert isinstance(self.num_experts_, int) and self.num_experts_ > 0
        assert self.act_fn_ is None or (
            isinstance(self.act_fn_, str) and self.act_fn_ in ACT2FN
        )
        if self.routing_strategy_ == "mixlora":
            assert isinstance(self.top_k_, int) and self.top_k_ > 0
        elif self.routing_strategy_ == "mixlora-dynamic":
            assert (
                isinstance(self.top_p_, float) and self.top_p_ > 0 and self.top_p_ <= 1
            )
            assert isinstance(self.temperature_, float) and self.temperature_ >= 0
        elif self.routing_strategy_ == "mixlora-switch":
            assert (
                isinstance(self.router_z_loss_coef_, float)
                and self.router_z_loss_coef_ >= 0
            )
            if self.sparse_step_ is not None:
                assert isinstance(self.sparse_step_, int) and self.sparse_step_ > 0
            assert isinstance(self.expert_capacity_, int) and self.expert_capacity_ > 0
            assert isinstance(self.ffn_dropout_, float) and self.ffn_dropout_ >= 0

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "MixLoraConfig":
        lora_config = MixLoraConfig(**LoraConfig.from_config(config).__dict__)
        if "expert_lora" in config:
            expert_config = copy.deepcopy(config)
            expert_config.update(config["expert_lora"])
            lora_config.expert_config_ = LoraConfig().from_config(expert_config)
        lora_config.router_aux_loss_coef_ = config.get(
            "router_aux_loss_coef", 0.001
        )  # for training
        lora_config.routing_strategy_ = config["routing_strategy"]
        lora_config.router_loss_ = config.get("router_loss", True)
        lora_config.num_experts_ = config["num_experts"]
        # silu for mixtral or gelu_new for switch transformers
        # left blank to automatically use the original act_fn of FFN
        lora_config.act_fn_ = config.get("act_fn", None)
        if lora_config.routing_strategy_ == "mixlora":
            lora_config.router_init_range_ = config.get("router_init_range", 0.02)
            lora_config.jitter_noise_ = config.get("jitter_noise", 0.0)
            lora_config.top_k_ = config.get("top_k", 2)
        elif lora_config.routing_strategy_ == "mixlora-dynamic":
            lora_config.router_init_range_ = config.get("router_init_range", 0.02)
            lora_config.jitter_noise_ = config.get("jitter_noise", 0.0)
            lora_config.top_p_ = config.get("top_p", 0.8)
            lora_config.temperature_ = config.get("temperature", 0.0)
        elif lora_config.routing_strategy_ == "mixlora-switch":
            lora_config.router_init_range_ = config.get("router_init_range", 1.0)
            lora_config.jitter_noise_ = config.get("jitter_noise", 0.01)
            lora_config.router_z_loss_coef_ = config.get(
                "router_z_loss_coef", 0.001
            )  # for training
            # expert_capacity = (max_sequence_length / num_experts) * capacity_factor
            # common values of capacity_factor: 1.0, 1.25, 2.0
            lora_config.expert_capacity_ = config.get("expert_capacity", 32)
            lora_config.ffn_dropout_ = config.get("ffn_dropout", 0.0)
            lora_config.sparse_step_ = config.get("sparse_step", None)

        return lora_config

    def export(self) -> Dict[str, any]:
        config = super().export()
        config["peft_type"] = "MIXLORA"
        if self.expert_config_ is not None:
            expert_config = self.expert_config_.export()
            expert_config.pop("peft_type")
            expert_config.pop("target_modules")
            config["expert_lora"] = expert_config
        config["routing_strategy"] = self.routing_strategy_
        config["num_experts"] = self.num_experts_
        if self.act_fn_ is not None and isinstance(self.act_fn_, str):
            config["act_fn"] = self.act_fn_
        if self.routing_strategy_ == "mixlora":
            config["top_k"] = self.top_k_
        elif self.routing_strategy_ == "mixlora-dynamic":
            config["top_p"] = self.top_p_
            config["temperature"] = self.temperature_
        elif self.routing_strategy_ == "mixlora-switch":
            config["expert_capacity"] = self.expert_capacity_
            config["sparse_step"] = self.sparse_step_

        return config

    def expert_config(self, expert_idx: int) -> LoraConfig:
        if self.expert_config_ is None:
            config = copy.deepcopy(super())
        else:
            config = copy.deepcopy(self.expert_config_)
        config.adapter_name = f"moe.{self.adapter_name}.experts.{expert_idx}"
        return config
