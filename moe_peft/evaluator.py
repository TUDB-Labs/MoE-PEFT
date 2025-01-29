import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import torch

from .adapters import LoraMoeConfig, MixLoraConfig, MolaConfig
from .analysts import SVDProcessor
from .common import InputData, LLMBatchConfig, LLMModelInput, Prompt
from .model import LLMModel
from .tasks import BasicMetric, BasicTask, CommonSenseTask, task_dict
from .tokenizer import Tokenizer


@dataclass
class EvaluateConfig:
    adapter_name: str = None
    task_name: str = None
    data_path: str = None
    batch_size: int = 16
    router_profile: bool = False
    svd_ana: bool = False
    svd_cos_file: bool = False
    moe_flag: bool = False
    target_modules: Dict = None
    # Do not set these manually
    task_: BasicTask = None
    data_: List[InputData] = None
    metric_: BasicMetric = None
    rollback_start_idx_: int = 0
    batch_start_idx_: int = 0
    batch_end_idx_: int = 0

    def _dataload_fn(self, tokenizer: Tokenizer, **tokenizer_kwargs):
        data = self.task_.loading_data(False, self.data_path)
        for idx, data_point in enumerate(data):
            assert not isinstance(data_point.inputs, Prompt)

            data_point.tokens = tokenizer.encode(data_point.inputs, **tokenizer_kwargs)
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return data

    @staticmethod
    def from_config(config: Dict[str, any]) -> List["EvaluateConfig"]:
        adapter_name = config["name"]
        data_path = config.get("data", None)
        task_list = config.get("task_name", "casual").split(";")
        profile = config.get("router_profile", False)
        svd_ana = config.get("svd_analysis", False)
        svd_cos_file = config.get("svd_cos_file", False)
        moe_flag = svd_ana and ("routing_strategy" in config)
        target_modules = config.get("target_modules", None) if svd_ana else None
        path_list = (
            [None] * len(task_list) if data_path is None else data_path.split(";")
        )
        config_list = []
        for task_name_, data_path_ in zip(task_list, path_list):
            if task_name_ not in task_dict:
                continue
            config_list.append(
                EvaluateConfig(
                    adapter_name=adapter_name,
                    task_name=task_name_,
                    data_path=data_path_,
                    batch_size=config["evaluate_batch_size"],
                    router_profile=True if profile else False,
                    svd_ana=True if svd_ana else False,
                    svd_cos_file=True if svd_cos_file else False,
                    moe_flag=True if moe_flag else False,
                    target_modules=target_modules if target_modules else None,
                )
            )

        return config_list

    def prepare(self, tokenizer: Tokenizer, device: str):
        self.reset_parameters()
        assert (
            self.task_name != "casual"
        ), "Auto evaluation is not currently available for casual supervised fine-tuning tasks."
        self.task_ = task_dict[self.task_name]
        self.data_ = self._dataload_fn(tokenizer)
        self.metric_ = self.task_.loading_metric()
        if isinstance(self.task_, CommonSenseTask):
            labels = self.task_.label_list()
            label_indices = [0] * len(labels)
            for idx, label in enumerate(labels):
                ids = tokenizer.encode(" " + label)
                label_indices[idx] = ids[-1]
            self.label_indices_ = torch.tensor(
                label_indices, dtype=torch.int64, device=device
            )
        else:
            self.label_indices_ = None

    def reset_parameters(self):
        self.task_ = None
        self.data_ = None
        self.metric_ = None
        self.rollback_start_idx_ = 0
        self.batch_start_idx_ = 0
        self.batch_end_idx_ = 0


def _prepare_tasks(model, tokenizer, configs):
    for config in configs:
        config.prepare(tokenizer, model.device_)
        if not isinstance(
            model.adapter_configs_[config.adapter_name],
            (MixLoraConfig, MolaConfig, LoraMoeConfig),
        ):
            continue
        for layer in model.model_.layers_:
            if config.adapter_name in layer.mlp_.moes_:
                layer.mlp_.moes_[config.adapter_name].router_profile_ = (
                    config.router_profile
                )


def _dispatch_task_in(tokenizer, configs, concurrent_jobs, max_seq_len):
    batch_data_config = []
    sequence_lengths = []
    current_configs = []
    batch_tokens = []
    batch_labels = []
    atten_masks = []
    max_tokens_len = 0
    for config in configs:
        if len(current_configs) >= concurrent_jobs:
            break
        if config.batch_start_idx_ >= len(config.data_):
            continue
        config.batch_end_idx_ = min(
            config.batch_start_idx_ + config.batch_size, len(config.data_)
        )
        batch_start_idx = len(batch_tokens)
        for idx in range(config.batch_start_idx_, config.batch_end_idx_):
            if idx >= len(config.data_):
                break
            tokens = config.data_[idx].tokens
            labels = config.data_[idx].labels
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            max_tokens_len = max(len(tokens), max_tokens_len)
            batch_tokens.append(tokens)
            batch_labels.append(labels.copy())

        config.batch_start_idx_ = config.batch_end_idx_
        current_configs.append(config)
        batch_data_config.append(
            LLMBatchConfig(
                adapter_name_=config.adapter_name,
                batch_start_idx_=batch_start_idx,
                batch_end_idx_=len(batch_tokens),
            )
        )

    max_seq_len = min(max_seq_len, max_tokens_len)

    for tokens in batch_tokens:
        sequence_lengths.append(len(tokens) - 1)
        while len(tokens) < max_seq_len:
            tokens.append(tokenizer.pad_id_)
        atten_masks.append(tokenizer.mask_from(tokens))

    return (
        current_configs,
        sequence_lengths,
        batch_labels,
        LLMModelInput(
            batch_configs_=batch_data_config,
            batch_tokens_=batch_tokens,
            batch_masks_=atten_masks,
            inference_mode_=True,
        ),
    )


def _accumulate_router_statistic(model, config, reset_profile=False):
    adapter_config = model.adapter_configs_[config.adapter_name]

    # Only accumulate statistics if adapter_config is an instance of (MixLoraConfig, MolaConfig) and not LoraMoeConfig
    if not (
        isinstance(adapter_config, (MixLoraConfig, MolaConfig))
        and not isinstance(adapter_config, LoraMoeConfig)
    ):
        return None

    router_statistic_ = list(0 for _ in range(adapter_config.num_experts_))

    # Iterate through each layer of the model and collect counts from profiler_
    for layer in model.model_.layers_:
        # If the current adapter is in the MLP's moes_
        if config.adapter_name in layer.mlp_.moes_:
            for idx, val in enumerate(layer.mlp_.moes_[config.adapter_name].profiler_):
                router_statistic_[idx] += val

            # Only reset profiler_ if needed and conditions are met
            if reset_profile:
                layer.mlp_.moes_[config.adapter_name].profiler_ = None

        else:
            # self_attn
            for attr in ["wq_", "wk_", "wv_", "wo_"]:
                moes_attr = getattr(layer.self_attn_, attr).moes_
                if config.adapter_name in moes_attr:
                    if moes_attr[config.adapter_name].profiler_ is not None:
                        for idx, val in enumerate(
                            moes_attr[config.adapter_name].profiler_
                        ):
                            router_statistic_[idx] += val
                    # Skip if profiler_ is None or adapter_name is not in moes_attr
                # Skip if adapter_name is not in moes_attr

            # MLP
            for attr in ["w1_", "w2_", "w3_"]:
                moes_attr = getattr(layer.mlp_.mlp_, attr).moes_
                if config.adapter_name in moes_attr:
                    if moes_attr[config.adapter_name].profiler_ is not None:
                        for idx, val in enumerate(
                            moes_attr[config.adapter_name].profiler_
                        ):
                            router_statistic_[idx] += val
                    # Similarly, skip if adapter_name is not in moes_attr or profiler_ is None

    return router_statistic_


def _compute_metrcis(model, current_configs, sequence_lengths, batch_labels, outputs):
    for idx, output in enumerate(outputs):
        config: EvaluateConfig = current_configs[idx]
        task: BasicTask = config.task_
        metric: BasicMetric = config.metric_
        start_idx = output.batch_start_idx_
        end_idx = output.batch_end_idx_
        logits = output.logits

        # Extract router_statistic_ logic and handle it uniformly with _accumulate_router_statistic
        if config.router_profile:
            router_statistic_ = _accumulate_router_statistic(
                model,
                config,
                reset_profile=False,  # In _compute_metrics, resetting profiler_ is not required
            )
            if router_statistic_ is not None and any(router_statistic_):
                for r_idx, val in enumerate(router_statistic_):
                    logging.info(
                        f"{config.adapter_name}: expert {r_idx}, load = {val / 32}"
                    )

        # The following is the original logits processing and metric computation logic
        batch_size = logits.shape[0]
        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device),
            sequence_lengths[start_idx:end_idx],
        ]
        labels = torch.tensor(
            batch_labels[start_idx:end_idx],
            dtype=task.label_dtype_,
            device=logits.device,
        )

        if task.task_type_ == "common_sense":
            pooled_logits = pooled_logits[:, config.label_indices_]
            pooled_logits = pooled_logits.softmax(-1).argmax(-1)
        elif task.task_type_ == "single_label_classification":
            pooled_logits = pooled_logits.softmax(-1).argmax(-1)
            pooled_logits = pooled_logits.to(task.label_dtype_)
        elif task.task_type_ != "multi_label_classification":
            raise ValueError(f"unknown task type {task.task_type_}")

        metric.add_batch(
            predictions=pooled_logits.detach().cpu(), references=labels.detach().cpu()
        )
        logging.info(f"{config.adapter_name} evaluate data:")
        logging.info(f"    step: {config.batch_start_idx_}/{len(config.data_)}")


def _compute_result(model, configs, save_file):
    results = []
    for config in configs:
        result = {
            "adapter_name": config.adapter_name,
            "task_name": (
                config.task_name if config.data_path is None else config.data_path
            ),
            "date_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "metrics": {},
        }
        compute_results = config.metric_.compute()
        result["metrics"] = compute_results

        # Extract router_statistic_ logic and handle it uniformly with _accumulate_router_statistic
        # At this point, according to the original logic, if it's not svd_ana and router_profile is needed, reset profiler_
        if config.router_profile:
            router_statistic_ = _accumulate_router_statistic(
                model,
                config,
                reset_profile=(
                    not getattr(config, "svd_ana", False) and config.router_profile
                ),
            )
            if router_statistic_ is not None and any(router_statistic_):
                result["router_profile"] = list(val / 224 for val in router_statistic_)

        results.append(result)

    if save_file is not None:
        with open(save_file, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"saving evaluation result to {save_file}")
    else:
        print(json.dumps(results, indent=4))

    return results


@torch.inference_mode()
def evaluate(
    model: LLMModel,
    tokenizer: Tokenizer,
    configs: List[EvaluateConfig],
    max_concurrent_jobs: int = None,
    retrying_steps: int = 20,
    max_seq_len: int = 512,
    save_file: str = None,
) -> Dict:

    if max_concurrent_jobs is None:
        max_concurrent_jobs = len(configs)
        logging.info(
            f"Setting max_concurrent_jobs to {max_concurrent_jobs} automatically"
        )

    assert max_concurrent_jobs > 0
    assert retrying_steps > 0

    _prepare_tasks(model, tokenizer, configs)

    concurrent_jobs = max_concurrent_jobs
    retrying_count = 0
    while True:
        if concurrent_jobs < max_concurrent_jobs and retrying_count > 0:
            retrying_count -= 1
            if retrying_count == 0:
                concurrent_jobs += 1
                logging.info(f"recovering concurrent jobs to {concurrent_jobs}")

        current_configs, sequence_lengths, batch_labels, input_args = _dispatch_task_in(
            tokenizer, configs, concurrent_jobs, max_seq_len
        )

        if len(current_configs) == 0:
            break

        try:
            _compute_metrcis(
                model,
                current_configs,
                sequence_lengths,
                batch_labels,
                model.forward(input_args),
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                concurrent_jobs -= 1
                if concurrent_jobs == 0:
                    raise e
                logging.warn(
                    f"deprecating concurrent jobs to {concurrent_jobs} due to OOM."
                )
                # rollback
                retrying_count = retrying_steps
                for config in current_configs:
                    config.batch_start_idx_ = config.rollback_start_idx_
                    logging.info(
                        f"{config.adapter_name}: rollback to {config.batch_start_idx_}/{len(config.data_)}"
                    )
                continue
            else:
                raise e

        for config in current_configs:
            config.rollback_start_idx_ = config.batch_start_idx_

    if config.svd_ana:
        for config in configs:  # call analyst process
            processor = SVDProcessor(model, config)
            if not config.svd_cos_file:
                svd_result = processor.process()
            else:
                svd_result = processor.process(all_result=True)

            file = (
                f"svd_result_{config.adapter_name}_50%_rvt5%.json"
                if not save_file
                else save_file
            )
            with open(file, "w") as f:
                json.dump(svd_result, f, indent=4)
            logging.info(f"saving svd_analysis result to {file}")

        return _compute_result(model, configs, save_file)

    return _compute_result(model, configs, save_file)
