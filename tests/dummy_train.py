import fire
import torch

import moe_peft


def main(
    base_model: str,
    adapter_name: str = "lora_0",
    train_data: str = "TUDB-Labs/Dummy-MoE-PEFT",
    test_prompt: str = "Could you provide an introduction to MoE-PEFT?",
):
    moe_peft.setup_logging("INFO")

    model: moe_peft.LLMModel = moe_peft.LLMModel.from_pretrained(
        base_model,
        device=moe_peft.executor.default_device_name(),
        load_dtype=torch.bfloat16,
    )
    tokenizer = moe_peft.Tokenizer(base_model)

    lora_config = moe_peft.LoraConfig(
        adapter_name=adapter_name,
        lora_r_=32,
        lora_alpha_=64,
        lora_dropout_=0.05,
        target_modules_={
            "q_proj": True,
            "k_proj": True,
            "v_proj": True,
            "o_proj": True,
        },
    )

    train_config = moe_peft.TrainConfig(
        adapter_name=adapter_name,
        data_path=train_data,
        num_epochs=10,
        batch_size=16,
        micro_batch_size=8,
        learning_rate=1e-4,
    )

    with moe_peft.executors.no_cache():
        model.init_adapter(lora_config)
        moe_peft.train(model=model, tokenizer=tokenizer, configs=[train_config])
        lora_config, lora_weight = model.unload_adapter(adapter_name)

    generate_configs = [
        moe_peft.GenerateConfig(
            adapter_name=adapter_name,
            prompts=[test_prompt],
            stop_token="\n",
        ),
        moe_peft.GenerateConfig(
            adapter_name="default",
            prompts=[test_prompt],
            stop_token="\n",
        ),
    ]

    with moe_peft.executors.no_cache():
        model.init_adapter(lora_config, lora_weight)
        model.init_adapter(moe_peft.AdapterConfig(adapter_name="default"))
        outputs = moe_peft.generate(
            model=model,
            tokenizer=tokenizer,
            configs=generate_configs,
            max_gen_len=128,
        )

    print(f"\n{'='*10}\n")
    print(f"PROMPT: {test_prompt}\n")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} OUTPUT:")
        print(f"{output[0]}\n")
    print(f"\n{'='*10}\n")


if __name__ == "__main__":
    fire.Fire(main)
