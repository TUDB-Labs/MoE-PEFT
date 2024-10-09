import copy
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from moe_peft.adapters import (
    LoraMoeConfig,
    MixLoraConfig,
    MolaConfig,
    lora_config_factory,
    moe_layer_factory,
    router_loss_factory,
)
from moe_peft.common import (
    CHECKPOINT_CLASSES,
    AdapterConfig,
    Linear,
    LLMCache,
    LLMDecoder,
    LLMForCausalLM,
    LLMModelConfig,
    LLMModelInput,
    LLMModelOutput,
    LLMMoeBlock,
    LLMOutput,
    LoraConfig,
    unpack_router_logits,
)
from moe_peft.executors import executor
from moe_peft.models import from_pretrained
from moe_peft.tasks import SequenceClassificationTask, task_dict
from moe_peft.utils import is_package_available

if is_package_available("bitsandbytes"):
    from transformers import BitsAndBytesConfig
else:
    from moe_peft.utils import BitsAndBytesConfig


class CasualOutputLayer(LLMOutput):
    def __init__(self, vocab_size: int, weight: torch.nn.Linear):
        super().__init__()
        self.vocab_size_: int = vocab_size
        self.lm_head_: torch.nn.Module = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.lm_head_(data).float()

    def loss(
        self, input_ids: torch.Tensor, output_logits: torch.Tensor, labels
    ) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            labels = (
                labels.clone()
                .detach()
                .to(dtype=torch.long, device=output_logits.device)
            )
        else:
            labels = torch.tensor(labels, dtype=torch.long, device=output_logits.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(
            output_logits[..., :-1, :].contiguous().view(-1, self.vocab_size_),
            labels[..., 1:].contiguous().view(-1),
        )


class ClassificationOutputLayer(LLMOutput):
    def __init__(
        self,
        task_type: str,
        num_labels: int,
        label_dtype: torch.dtype,
        hidden_size: int,
        pad_token_id: int,
        device: str,
        weight: Optional[torch.Tensor],
    ):
        super().__init__()
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.task_type_ = task_type
        self.pad_id_ = pad_token_id
        self.score_ = torch.nn.Linear(
            hidden_size,
            self.num_labels_,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        if weight is None:
            torch.nn.init.kaiming_normal_(self.score_.weight, a=math.sqrt(5))
        else:
            with torch.no_grad():
                self.score_.weight.copy_(weight["classifier"])

    def state_dict(self):
        return {"classifier": self.score_.weight}

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.score_(data.to(torch.float32))

    def loss(
        self, input_ids: torch.Tensor, output_logits: torch.Tensor, labels
    ) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            labels = (
                labels.clone()
                .detach()
                .to(dtype=self.label_dtype_, device=output_logits.device)
            )
        else:
            labels = torch.tensor(
                labels, dtype=self.label_dtype_, device=output_logits.device
            )
        batch_size = input_ids.shape[0]
        sequence_lengths = (torch.eq(input_ids, self.pad_id_).int().argmax(-1) - 1).to(
            output_logits.device
        )
        pooled_logits = output_logits[
            torch.arange(batch_size, device=output_logits.device), sequence_lengths
        ]
        if self.task_type_ == "single_label_classification":
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(pooled_logits.view(-1, self.num_labels_), labels.view(-1))
        elif self.task_type_ == "multi_label_classification":
            loss_fn = torch.nn.BCEWithLogitsLoss()
            return loss_fn(pooled_logits, labels)
        else:
            raise ValueError(f"unknown task type {self.task_type_}")


class OutputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_: Dict[str, torch.nn.Module] = {}

    def forward(
        self, data: torch.Tensor, input_args: LLMModelInput
    ) -> List[LLMModelOutput]:
        outputs = []
        for lora_config in input_args.batch_configs_:
            adapter_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            assert adapter_name != "" and adapter_name in self.layers_
            layer = self.layers_[adapter_name]
            outputs.append(
                LLMModelOutput(
                    adapter_name=adapter_name,
                    logits=layer.forward(data[start_idx:end_idx]),
                    loss_fn_=layer.loss,
                )
            )

        return outputs


def init_lora_layer_weight(
    transformer_layer: LLMDecoder,
    llm_config: LLMModelConfig,
    lora_config: LoraConfig,
    lora_weights: Optional[Dict[str, torch.Tensor]],
):
    target_modules = lora_config.target_modules_
    attn_state_dict, mlp_state_dict = transformer_layer.state_dict()
    attn_state_dict: Dict[str, torch.Tensor]
    mlp_state_dict: Dict[str, torch.Tensor]
    all_state_dict: Dict[str, torch.Tensor] = copy.copy(attn_state_dict)
    all_state_dict.update(mlp_state_dict)
    moe_init_strategy = "none"
    if isinstance(lora_config, MixLoraConfig):
        model_prefix_name = "mixlora"
        moe_layer_name_list = list(mlp_state_dict.keys())
        moe_init_strategy = "fused_mlp"
    elif isinstance(lora_config, LoraMoeConfig):
        model_prefix_name = "loramoe"
        moe_layer_name_list = list(mlp_state_dict.keys())
        moe_init_strategy = "plugin"
    elif isinstance(lora_config, MolaConfig):
        model_prefix_name = "mola"
        moe_layer_name_list = list(all_state_dict.keys())
        moe_init_strategy = "plugin"
    else:
        model_prefix_name = "base_model.model.model"
        moe_layer_name_list = []

    assert len(moe_layer_name_list) == 0 or moe_init_strategy in ["plugin", "fused_mlp"]

    if moe_init_strategy == "fused_mlp":
        transformer_layer.mlp_.moes_[lora_config.adapter_name] = moe_layer_factory(
            llm_config.dim_,
            llm_config.device_,
            lora_config,
            (
                None
                if lora_weights is None
                else lora_weights[
                    f"{model_prefix_name}.layers.{transformer_layer.layer_id_}.mlp.moe_gate.weight"
                ]
            ),
        )

    for proj_name, lora_linear in all_state_dict.items():
        lora_linear: Linear
        if proj_name not in target_modules or not target_modules[proj_name]:
            continue
        module_name = (
            "self_attn"
            if proj_name in attn_state_dict
            else ("mlp" if proj_name in mlp_state_dict else None)
        )
        module_name = f"{model_prefix_name}.layers.{transformer_layer.layer_id_}.{module_name}.{proj_name}"
        if proj_name in moe_layer_name_list:
            if moe_init_strategy == "plugin":
                # init for gating mechanisms
                lora_linear.moes_[lora_config.adapter_name] = moe_layer_factory(
                    lora_linear.in_features_,
                    llm_config.device_,
                    lora_config,
                    (
                        lora_weights.get(f"{module_name}.moe_gate.weight", None)
                        if lora_weights is not None
                        else None
                    ),
                )

            for expert_idx in range(lora_config.num_experts_):
                if lora_weights is None:
                    lora_a = None
                    lora_b = None
                else:
                    lora_a = lora_weights.get(
                        f"{module_name}.experts.{expert_idx}.lora_A.weight", None
                    )
                    lora_b = lora_weights.get(
                        f"{module_name}.experts.{expert_idx}.lora_B.weight", None
                    )

                lora_linear.init_lora_weight(
                    lora_config.expert_config(expert_idx), (lora_a, lora_b)
                )
        else:
            if lora_weights is None:
                lora_a = None
                lora_b = None
            else:
                lora_a = lora_weights.get(f"{module_name}.lora_A.weight", None)
                lora_b = lora_weights.get(f"{module_name}.lora_B.weight", None)

            lora_linear.init_lora_weight(lora_config, (lora_a, lora_b))


def get_lora_layer_weight(
    transformer_layer: LLMDecoder,
    lora_config: LoraConfig,
    lora_weights: Dict[str, torch.Tensor],
):
    target_modules = lora_config.target_modules_
    attn_state_dict, mlp_state_dict = transformer_layer.state_dict()
    attn_state_dict: Dict[str, torch.Tensor]
    mlp_state_dict: Dict[str, torch.Tensor]
    all_state_dict: Dict[str, torch.Tensor] = copy.copy(attn_state_dict)
    all_state_dict.update(mlp_state_dict)
    if isinstance(lora_config, MixLoraConfig):
        model_prefix_name = "mixlora"
        gate_layer_name = (
            f"mixlora.layers.{transformer_layer.layer_id_}.mlp.moe_gate.weight"
        )
        moe_layer_name_list = list(mlp_state_dict.keys())
    elif isinstance(lora_config, LoraMoeConfig):
        model_prefix_name = "loramoe"
        moe_layer_name_list = list(mlp_state_dict.keys())
    elif isinstance(lora_config, MolaConfig):
        model_prefix_name = "mola"
        moe_layer_name_list = list(all_state_dict.keys())
    else:
        model_prefix_name = "base_model.model.model"
        moe_layer_name_list = []

    # for fused MoEs such as MixLoRA
    mlp_moe_layer: LLMMoeBlock = transformer_layer.mlp_.moes_.get(
        lora_config.adapter_name, None
    )
    if mlp_moe_layer is not None:
        lora_weights[gate_layer_name] = mlp_moe_layer.gate_.weight

    for proj_name, lora_linear in all_state_dict.items():
        lora_linear: Linear
        if proj_name not in target_modules or not target_modules[proj_name]:
            continue
        module_name = (
            "self_attn"
            if proj_name in attn_state_dict
            else ("mlp" if proj_name in mlp_state_dict else None)
        )
        module_name = f"{model_prefix_name}.layers.{transformer_layer.layer_id_}.{module_name}.{proj_name}"
        if proj_name in moe_layer_name_list:
            moe_layer = (
                lora_linear.moes_[lora_config.adapter_name]
                if lora_config.adapter_name in lora_linear.moes_
                else mlp_moe_layer
            )
            # for plugged MoEs such as LoRAMoE, MoLA, etc.
            if lora_config.adapter_name in lora_linear.moes_:
                lora_weights[f"{module_name}.moe_gate.weight"] = lora_linear.moes_[
                    lora_config.adapter_name
                ].gate_.weight

            for expert_idx in range(moe_layer.experts_):
                moe_lora_name = f"moe.{lora_config.adapter_name}.experts.{expert_idx}"
                lora_obj = lora_linear.loras_.get(moe_lora_name, None)
                if lora_obj is not None:
                    lora_weights[
                        f"{module_name}.experts.{expert_idx}.lora_A.weight"
                    ] = lora_obj.lora_a_.weight
                    lora_weights[
                        f"{module_name}.experts.{expert_idx}.lora_B.weight"
                    ] = lora_obj.lora_b_.weight

        else:
            lora_obj = lora_linear.loras_.get(lora_config.adapter_name, None)
            if lora_obj is not None:
                lora_weights[f"{module_name}.lora_A.weight"] = lora_obj.lora_a_.weight
                lora_weights[f"{module_name}.lora_B.weight"] = lora_obj.lora_b_.weight


class LLMModel(torch.nn.Module):
    def __init__(self, model: LLMForCausalLM):
        super().__init__()
        args: LLMModelConfig = model.config_
        if args.vocab_size_ >= torch.finfo(args.dtype_).max:
            logging.warn(
                f"vocab_size >= max({args.dtype_}), consider load model with higher precision."
            )
        self.model_ = model
        self.config_ = args
        # configs
        self.name_or_path_ = args.name_or_path_
        self.vocab_size_ = args.vocab_size_
        self.device_ = args.device_
        self.dtype_ = args.dtype_

        self.output_ = OutputLayer()
        # adapter configs
        self.adapter_configs_: Dict[str, LoraConfig] = {}

    def _prepare_inputs(
        self, input_args: LLMModelInput, past_key_values: Optional[LLMCache] = None
    ):
        assert input_args.batch_tokens_ is not None, "Model have no input."
        assert (
            input_args.gradient_checkpoint_ == "none" or past_key_values is None
        ), "Cache is incompatible with gradient checkpointing."
        assert (
            not input_args.inference_mode_ or input_args.gradient_checkpoint_ == "none"
        ), "Can not use gradient checkpoint when inference."

        # prepare inputs
        if isinstance(input_args.batch_tokens_, torch.Tensor):
            input_ids = input_args.batch_tokens_.to(
                dtype=torch.int64, device=self.device_
            )
        else:
            input_ids = torch.tensor(
                input_args.batch_tokens_, dtype=torch.int64, device=self.device_
            )

        inputs_embeds = self.model_.embed_tokens(input_ids)
        if input_args.gradient_checkpoint_ != "none":
            inputs_embeds.requires_grad_(True)

        # prepare cache
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        if past_seen_tokens is None:
            past_seen_tokens = 0

        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        # prepare mask
        if input_args.batch_masks_ is not None:
            # 2d mask is passed through the layers
            if isinstance(input_args.batch_masks_, torch.Tensor):
                attention_mask = input_args.batch_masks_.to(
                    dtype=torch.int64, device=self.device_
                )
            else:
                attention_mask = torch.tensor(
                    input_args.batch_masks_, dtype=torch.int64, device=self.device_
                )
        else:
            attention_mask = None

        if self.config_.attn_implementation_ != "flash_attn":
            causal_mask = self.model_.causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values
            )
        else:
            causal_mask = attention_mask

        return input_ids, inputs_embeds, attention_mask, causal_mask, cache_position

    def _call_decoder_stack(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[LLMCache] = None,
    ):
        # decoder layers
        num_adapters = len(input_args.batch_configs_)
        all_router_logits = [[] for _ in range(num_adapters)]
        gradient_checkpoint = CHECKPOINT_CLASSES[input_args.gradient_checkpoint_]

        for decoder_layer in self.model_.decoder_stack():
            hidden_states, *router_logits = gradient_checkpoint(
                decoder_layer.forward,
                hidden_states,
                input_args,
                rotary_emb,
                attention_mask,
                cache_position,
                past_key_value,
            )
            if len(router_logits) == 0:
                continue
            # collecting router logits
            assert len(router_logits) == num_adapters
            for idx in range(num_adapters):
                if router_logits[idx] is not None:
                    all_router_logits[idx].append(router_logits[idx])

        hidden_states = self.model_.norm(hidden_states)

        return hidden_states, all_router_logits

    # compute the model: output probs
    def forward(
        self, input_args: LLMModelInput, past_key_values: Optional[LLMCache] = None
    ) -> List[LLMModelOutput]:
        input_ids, inputs_embeds, attention_mask, causal_mask, cache_position = (
            self._prepare_inputs(input_args, past_key_values)
        )

        labels = input_args.batch_labels_

        input_args.batch_labels_ = None
        input_args.batch_tokens_ = None
        input_args.batch_masks_ = None

        # embed positions
        hidden_states = inputs_embeds

        rotary_emb = self.model_.rotary_embed(
            hidden_states, cache_position.unsqueeze(0)
        )

        hidden_states, all_router_logits = self._call_decoder_stack(
            hidden_states,
            input_args,
            rotary_emb,
            causal_mask,
            cache_position,
            past_key_values,
        )

        # calculate loss
        output = self.output_(hidden_states, input_args)
        assert isinstance(output, List)
        for idx, lora_config in enumerate(input_args.batch_configs_):
            output_data = output[idx]
            assert isinstance(output_data, LLMModelOutput)
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            output_data.batch_start_idx_ = start_idx
            output_data.batch_end_idx_ = end_idx
            if input_args.output_router_logits_ and len(all_router_logits[idx]) > 0:
                output_data.router_logits = unpack_router_logits(all_router_logits[idx])
            if labels is None:
                continue
            # compute loss when labels provided
            output_data.loss = output_data.loss_fn_(
                input_ids[start_idx:end_idx],
                output_data.logits,
                labels[start_idx:end_idx],
            )
            output_data.loss_fn_ = None
            if output_data.router_logits is None:
                continue
            # compute router loss when router logits is available
            loss_fn = router_loss_factory(
                self.adapter_configs_[output_data.adapter_name]
            )
            if loss_fn is not None:
                output_data.aux_loss = loss_fn(
                    output_data.router_logits, attention_mask[start_idx:end_idx]
                )

        return output

    def from_pretrained(
        name_or_path: str,
        device: str,
        bits: int = None,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        load_dtype: torch.dtype = torch.bfloat16,
        compute_dtype: torch.dtype = torch.bfloat16,
        double_quant: bool = True,
        quant_type: str = "nf4",
    ) -> "LLMModel":
        # load_dtype will change the precision of LLaMA pre-trained model
        # when loading with quantization (bits = 8 or bits = 4), load_dtype will only influence the actual computing precision
        if load_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"unsupported load dtype {load_dtype}")

        if compute_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"unsupported compute dtype {compute_dtype}")

        if load_dtype in [torch.bfloat16, torch.float16]:
            logging.info("Loading model with half precision.")

        # BFloat16 is only supported after Ampere GPUs
        if not executor.is_bf16_supported():
            if load_dtype == torch.bfloat16:
                logging.warning("bf16 is not available. deprecated to fp16.")
                load_dtype = torch.float16

            if bits in [4, 8] and compute_dtype == torch.bfloat16:
                logging.warning("bf16 is not available. deprecated to fp16.")
                compute_dtype = torch.float16

        if bits in [4, 8]:
            logging.info(f"Loading model with quantization, bits = {bits}.")
            llm_model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                device_map=device,
                trust_remote_code=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=bits == 4,
                    load_in_8bit=bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=double_quant,
                    bnb_4bit_quant_type=quant_type,
                ),
                torch_dtype=load_dtype,
            )
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=load_dtype,
            )

        llm_model.requires_grad_(False)

        model = from_pretrained(
            llm_model,
            attn_impl=attn_impl,
            use_sliding_window=use_sliding_window,
            device=device,
        )

        logging.info(f"Use {attn_impl} as attention implementation.")

        return LLMModel(model)

    def init_adapter(
        self, config: AdapterConfig, weight: Optional[Dict[str, torch.Tensor]] = None
    ):
        # Patch for MixLoRA
        if isinstance(config, MixLoraConfig) and config.act_fn_ is None:
            config.act_fn_ = self.config_.hidden_act_

        self.adapter_configs_[config.adapter_name] = config
        # init output layer
        if config.task_name in task_dict and isinstance(
            task_dict[config.task_name], SequenceClassificationTask
        ):
            output_layer = ClassificationOutputLayer(
                **task_dict[config.task_name].init_kwargs(),
                hidden_size=self.config_.dim_,
                pad_token_id=self.config_.pad_token_id_,
                device=self.device_,
                weight=weight,
            )
        else:
            output_layer = CasualOutputLayer(
                vocab_size=self.config_.vocab_size_, weight=self.model_.lm_head_
            )

        self.output_.layers_[config.adapter_name] = output_layer
        if type(config) is not AdapterConfig:
            # init transformer layers
            for transformer_layer in self.model_.layers_:
                init_lora_layer_weight(transformer_layer, self.config_, config, weight)
        else:
            assert weight is None, "can not load basic adapter with weight"

        return config.adapter_name

    def get_adapter_weight_dict(self, adapter_name: str) -> Dict[str, torch.Tensor]:
        # return the lora weight and target_module's name
        lora_weight_dict = self.output_.layers_[adapter_name].state_dict()
        lora_config = self.adapter_configs_[adapter_name]
        for transformer_layer in self.model_.layers_:
            get_lora_layer_weight(transformer_layer, lora_config, lora_weight_dict)

        return lora_weight_dict

    def unload_adapter(
        self, adapter_name: str
    ) -> Tuple[LoraConfig, Dict[str, torch.Tensor]]:
        assert adapter_name in self.adapter_configs_, "adapter not exist"
        lora_weight = self.get_adapter_weight_dict(adapter_name)
        lora_config = self.adapter_configs_.pop(adapter_name)
        self.output_.layers_.pop(adapter_name)
        for transformer_layer in self.model_.layers_:
            attn_state_dict, mlp_state_dict = transformer_layer.state_dict()
            attn_state_dict: Dict[str, torch.Tensor]
            mlp_state_dict: Dict[str, torch.Tensor]
            lora_layer_list = list(attn_state_dict.values())
            lora_layer_list.extend(mlp_state_dict.values())

            for lora_layer in lora_layer_list:
                if adapter_name in lora_layer.loras_:
                    lora_layer.loras_.pop(adapter_name, None)
                elif adapter_name in transformer_layer.mlp_.moes_:
                    for expert_idx in range(
                        transformer_layer.mlp_.moes_[adapter_name].experts_
                    ):
                        moe_lora_name = f"moe.{adapter_name}.experts.{expert_idx}"
                        lora_layer.loras_.pop(moe_lora_name, None)

                    transformer_layer.mlp_.moes_.pop(adapter_name)
                elif adapter_name in lora_layer.moes_:
                    for expert_idx in range(lora_layer.moes_[adapter_name].experts_):
                        moe_lora_name = f"moe.{adapter_name}.experts.{expert_idx}"
                        lora_layer.loras_.pop(moe_lora_name, None)

                    lora_layer.moes_.pop(lora_config.adapter_name, None)

        return lora_config, lora_weight

    def load_adapter(self, name_or_path: str, adapter_name: Optional[str] = None):
        if adapter_name is None:
            adapter_name = name_or_path

        if not os.path.exists(name_or_path):
            name_or_path = snapshot_download(repo_id=name_or_path, repo_type="model")
        with open(
            name_or_path + os.sep + "adapter_config.json", "r", encoding="utf8"
        ) as fp:
            lora_config = lora_config_factory(json.load(fp))
        lora_config.adapter_name = adapter_name
        lora_weight = torch.load(
            name_or_path + os.sep + "adapter_model.bin",
            map_location=self.device_,
            weights_only=False,
        )

        self.init_adapter(lora_config, lora_weight)
        return adapter_name
