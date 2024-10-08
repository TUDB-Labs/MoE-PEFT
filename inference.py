import traceback
from queue import Queue
from threading import Thread

import fire
import gradio as gr
import torch

import moe_peft


class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(seq_pos, output):
            if self.stop_now:
                raise ValueError
            self.q.put(output["default"][0])

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True


placeholder_text = "Could you provide an introduction to MoE-PEFT?"


def main(
    base_model: str,
    template: str = None,
    lora_weights: str = "",
    load_16bit: bool = True,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
    device: str = moe_peft.executor.default_device_name(),
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
):

    model = moe_peft.LLMModel.from_pretrained(
        base_model,
        device=device,
        attn_impl="flash_attn" if flash_attn else "eager",
        bits=(8 if load_8bit else (4 if load_4bit else None)),
        load_dtype=torch.bfloat16 if load_16bit else torch.float32,
    )
    tokenizer = moe_peft.Tokenizer(base_model)

    if lora_weights:
        model.load_adapter(lora_weights, "default")
    else:
        model.init_adapter(moe_peft.AdapterConfig(adapter_name="default"))

    generation_config = moe_peft.GenerateConfig(
        adapter_name="default",
        prompt_template=template,
    )

    def evaluate(
        instruction,
        input="",
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        repetition_penalty=1.1,
        max_new_tokens=128,
        stream_output=False,
    ):
        instruction = instruction.strip()
        if len(instruction) == 0:
            instruction = placeholder_text

        input = input.strip()
        if len(input) == 0:
            input = None

        generation_config.prompts = [(instruction, input)]
        generation_config.temperature = temperature
        generation_config.top_p = top_p
        generation_config.top_k = top_k
        generation_config.repetition_penalty = repetition_penalty

        generate_params = {
            "model": model,
            "tokenizer": tokenizer,
            "configs": [generation_config],
            "max_gen_len": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.

            def generate_with_callback(callback=None, **kwargs):
                moe_peft.generate(stream_callback=callback, **kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    yield output
            return  # early return for stream_output

        # Without streaming
        output = moe_peft.generate(**generate_params)
        yield output["default"][0]

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder=placeholder_text,
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=1, value=1, label="Temperature"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.9, label="Sampling Top-P"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Sampling Top-K"
            ),
            gr.components.Slider(
                minimum=0, maximum=2, value=1.1, label="Repetition Penalty"
            ),
            gr.components.Slider(
                minimum=1,
                maximum=model.config_.max_seq_len_,
                step=1,
                value=1024,
                label="Max Tokens",
            ),
            gr.components.Checkbox(label="Stream Output", value=True),
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="MoE-PEFT LLM Evaluator",
        description="Evaluate language models and LoRA weights",  # noqa: E501
    ).queue().launch(server_name=server_name, share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
