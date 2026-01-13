import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple, Union

import torch
from transformers.utils import is_flash_attn_2_available

from .adapters import lora_config_factory
from .evaluator import EvaluateConfig, evaluate
from .executors import executor as moe_peft_executor
from .generator import GenerateConfig, generate
from .model import LLMModel
from .tokenizer import Tokenizer
from .trainer import TrainConfig, train
from .utils import setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoE-PEFT main program")
    parser.add_argument(
        "--base_model", type=str, required=True, help="Path to or name of base model"
    )
    parser.add_argument(
        "--inference", action="store_true", help="The inference mode (just for test)"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="The evaluate mode (just for test)"
    )
    parser.add_argument(
        "--disable_prompter",
        action="store_true",
        help="Disable prompter when inference",
    )
    parser.add_argument(
        "--load_adapter",
        action="store_true",
        help="Load adapter from file instead of init randomly",
    )
    parser.add_argument(
        "--disable_adapter", action="store_true", help="Disable the adapter modules"
    )
    parser.add_argument(
        "--attn_impl", type=str, help="Specify the implementation of attention"
    )
    parser.add_argument(
        "--sliding_window",
        action="store_true",
        help="Use sliding window attention (requires flash attention)",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="Disable cache when inference",
    )
    parser.add_argument(
        "--cache_implementation",
        type=str,
        help="Specify the implementation of cache",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Load base model in float16 precision"
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Load base model in bfloat16 precision"
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Use tfloat32 instead of float32 if available",
    )
    parser.add_argument(
        "--load_8bit",
        action="store_true",
        help="Load base model with 8bit quantization",
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="Load base model with 4bit quantization",
    )
    parser.add_argument("--device", type=str, help="Specify which GPU to be used")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to finetune configuration"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed in integer, default is 42"
    )
    parser.add_argument(
        "--dir", type=str, default=".", help="Path to read or save checkpoints"
    )
    parser.add_argument("--disable_log", action="store_true", help="Disable logging")
    parser.add_argument("--log_file", type=str, help="Save log to specific file")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show extra informations such as parameters",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite adapter model when older one existed",
    )
    parser.add_argument("--debug", action="store_true", help="Enabling debugging mode")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms to improve the reproducibility",
    )
    return parser


def query_yes_no(question, default="no"):
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def load_base_model(args) -> Tuple[Tokenizer, LLMModel]:
    logging.info("Initializing pre-trained model.")
    model = LLMModel.from_pretrained(
        name_or_path=args.base_model,
        device=args.device,
        attn_impl=args.attn_impl,
        use_sliding_window=args.sliding_window,
        bits=(8 if args.load_8bit else (4 if args.load_4bit else None)),
        load_dtype=(
            torch.bfloat16
            if args.bf16
            else (torch.float16 if args.fp16 else torch.float32)
        ),
    )

    tokenizer = Tokenizer(args.base_model)

    return tokenizer, model


def init_adapter_config(
    args,
    config: Dict[str, any],
    llm_model: LLMModel,
) -> List[Union[GenerateConfig, TrainConfig]]:
    config_list = []

    if config["cutoff_len"] == -1:
        config["cutoff_len"] = llm_model.config_.max_seq_len_
        logging.info(
            f"Setting cutoff_len to {llm_model.config_.max_seq_len_} automatically."
        )

    for lora_config in config["lora"]:
        adapter_name = lora_config["name"]
        adapter_path = f"{args.dir}{os.sep}{adapter_name}"
        if not args.load_adapter and os.path.exists(adapter_path):
            if args.overwrite:
                logging.warning(
                    f"Overwriting existed adapter model file: {adapter_path}"
                )
            elif not query_yes_no(
                f"Existed adapter model file detected: {adapter_path}\n" + "Overwrite?"
            ):
                logging.info("User canceled training due to file conflict.")
                exit(0)

        if args.load_adapter:
            llm_model.load_adapter(adapter_path, adapter_name)
        else:
            llm_model.init_adapter(lora_config_factory(lora_config))

        if args.inference:
            config_class = GenerateConfig(adapter_name=adapter_name)
            if not args.disable_prompter:
                config_class.prompt_template = lora_config.get("prompt", None)
            config_list.append(config_class)
        elif args.evaluate:
            config_list.extend(EvaluateConfig.from_config(lora_config))
        else:
            config_list.append(TrainConfig.from_config(lora_config))

        if args.verbose:
            logging.info(config_list[-1].__dict__)

    return config_list


def inference_callback(cur_pos, outputs):
    print(f"POSITION: {cur_pos}")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} OUTPUT: {output[0]}")


def run_inference(
    args,
    model: LLMModel,
    tokenizer: Tokenizer,
    configs: List[GenerateConfig],
    concurrent_jobs: int,
):
    while True:
        input_raw = input("INPUT WITHOUT PROMPT: ")
        if input_raw == "QUIT":
            return
        for config in configs:
            config.prompts = [input_raw]
        callback = None if args.disable_log else inference_callback
        outputs = generate(
            model,
            tokenizer,
            configs,
            max_gen_len=128,
            use_cache=not args.disable_cache,
            concurrent_jobs=concurrent_jobs,
            cache_implementation=args.cache_implementation,
            stream_callback=callback,
        )
        print(f"\n{'=' * 10}\n")
        print(f"PROMPT: {input_raw}")
        for adapter_name, output in outputs.items():
            print(f"{adapter_name} OUTPUT:")
            print(output[0])
        print(f"\n{'=' * 10}\n")


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    if args.inference or args.evaluate:
        args.load_adapter = True
        inference_mode = True
    else:
        inference_mode = False

    setup_logging("INFO", args.log_file)

    if not moe_peft_executor.check_available():
        return -1

    if args.attn_impl is None:
        if (
            inference_mode
            and moe_peft_executor.device_name() == "cuda"
            and is_flash_attn_2_available()
        ):
            args.attn_impl = "flash_attn"
        else:
            args.attn_impl = "eager"

    if args.device is None:
        args.device = moe_peft_executor.default_device_name()

    moe_peft_executor.use_deterministic_algorithms(args.deterministic)
    moe_peft_executor.allow_tf32(args.tf32)
    moe_peft_executor.manual_seed(args.seed)

    with open(args.config, "r", encoding="utf8") as fp:
        config = json.load(fp)

    tokenizer, model = load_base_model(args)
    adapters = init_adapter_config(args, config, model)

    moe_peft_executor.empty_cache()

    if os.getenv("MOE_PEFT_EVALUATE_MODE") is None:
        logging.info("Using efficient operators.")
    else:
        logging.info("Using deterministic operators.")

    if args.inference:
        run_inference(
            args=args,
            model=model,
            tokenizer=tokenizer,
            configs=adapters,
            concurrent_jobs=config.get("inference_lora_simultaneously_num", 2),
        )
    elif args.evaluate:
        evaluate(
            model=model,
            tokenizer=tokenizer,
            configs=adapters,
            max_concurrent_jobs=config.get("eval_lora_simultaneously_num", None),
            retrying_steps=config.get("eval_rollback_retrying_steps", 20),
            max_seq_len=config["cutoff_len"],
            save_file=config.get("evaluate_result", None),
        )
    else:
        train(
            model=model,
            tokenizer=tokenizer,
            configs=adapters,
            max_concurrent_jobs=config.get("train_lora_simultaneously_num", None),
            strategy=config["train_strategy"],
            cutoff_len=config["cutoff_len"],
            save_step=config["save_step"],
            save_dir=args.dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
