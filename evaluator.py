import fire
import torch

import moe_peft


def main(
    base_model: str,
    task_name: str,
    data_path: str = None,
    lora_weights: str = None,
    load_16bit: bool = True,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
    save_file: str = None,
    batch_size: int = 32,
    router_profile: bool = False,
    device: str = moe_peft.executor.default_device_name(),
):

    moe_peft.setup_logging("INFO")

    if not moe_peft.executor.check_available():
        exit(-1)

    model = moe_peft.LLMModel.from_pretrained(
        base_model,
        device=device,
        attn_impl="flash_attn" if flash_attn else "eager",
        bits=(8 if load_8bit else (4 if load_4bit else None)),
        load_dtype=torch.bfloat16 if load_16bit else torch.float32,
    )
    tokenizer = moe_peft.Tokenizer(base_model)
    if lora_weights:
        adapter_name = model.load_adapter(lora_weights)
    else:
        adapter_name = model.init_adapter(
            moe_peft.AdapterConfig(adapter_name="default")
        )

    evaluate_paramas = moe_peft.EvaluateConfig(
        adapter_name=adapter_name,
        task_name=task_name,
        data_path=data_path,
        batch_size=batch_size,
        router_profile=router_profile,
    )

    moe_peft.evaluate(model, tokenizer, [evaluate_paramas], save_file=save_file)


if __name__ == "__main__":
    fire.Fire(main)
