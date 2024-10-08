import copy
from dataclasses import dataclass
from typing import Dict

from moe_peft.common import LoraConfig


@dataclass
class LoraMoeConfig(LoraConfig):
    num_experts_: int = None
    router_init_range_: float = None
    routing_strategy_: str = "loramoe"

    def check(self) -> "LoraMoeConfig":
        super().check()
        assert isinstance(self.num_experts_, int) and self.num_experts_ > 0
        assert (
            isinstance(self.router_init_range_, float) and self.router_init_range_ >= 0
        )

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "LoraMoeConfig":
        return LoraMoeConfig(
            num_experts_=config["num_experts"],
            router_init_range_=config.get("router_init_range", 5.0),
            **LoraConfig.from_config(config).__dict__,
        )

    def export(self) -> Dict[str, any]:
        config = super().export()
        config["peft_type"] = "LORAMOE"
        config["routing_strategy"] = self.routing_strategy_
        config["num_experts"] = self.num_experts_

        return config

    def expert_config(self, expert_idx: int) -> LoraConfig:
        config = copy.deepcopy(super())
        config.adapter_name = f"moe.{self.adapter_name}.experts.{expert_idx}"
        return config
