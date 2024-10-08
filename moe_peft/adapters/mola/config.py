import copy
from dataclasses import dataclass
from typing import Dict

from moe_peft.common import LoraConfig


@dataclass
class MolaConfig(LoraConfig):
    top_k_: int = None
    num_experts_: int = None
    router_init_range_: float = None
    routing_strategy_: str = "mola"

    def check(self) -> "MolaConfig":
        super().check()
        assert isinstance(self.top_k_, int) and self.top_k_ > 0
        assert isinstance(self.num_experts_, int) and self.num_experts_ > 0
        assert (
            isinstance(self.router_init_range_, float) and self.router_init_range_ >= 0
        )

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "MolaConfig":
        return MolaConfig(
            top_k_=config.get("top_k", 2),
            num_experts_=config["num_experts"],
            router_init_range_=config.get("router_init_range", 5.0),
            **LoraConfig.from_config(config).__dict__,
        )

    def export(self) -> Dict[str, any]:
        config = super().export()
        config["peft_type"] = "MOLA"
        config["routing_strategy"] = self.routing_strategy_
        config["num_experts"] = self.num_experts_
        config["top_k"] = self.top_k_

        return config

    def expert_config(self, expert_idx: int) -> LoraConfig:
        config = copy.deepcopy(super())
        config.adapter_name = f"moe.{self.adapter_name}.experts.{expert_idx}"
        return config
