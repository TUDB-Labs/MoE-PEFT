import gc
import os

import torch

from .common import BasicExecutor
from .cpu import CPUExecutor
from .cuda import CUDAExecutor
from .mps import MPSExecutor

executor_dict = {
    "CUDA": CUDAExecutor,
    "MPS": MPSExecutor,
    "CPU": CPUExecutor,
}


def _init_executor():
    env = os.getenv("MOE_PEFT_EXECUTOR_TYPE")
    if env is not None:
        env = env.upper()
        if env not in executor_dict:
            raise ValueError(f"Assigning unknown executor type {env}")
        return executor_dict[env]()
    elif torch.cuda.is_available():
        return CUDAExecutor()
    elif torch.backends.mps.is_available():
        return MPSExecutor()
    else:
        return CPUExecutor()


executor: BasicExecutor = _init_executor()


class no_cache(object):
    def __enter__(self):
        executor.empty_cache()
        gc.collect()
        return self

    def __exit__(self, type, value, traceback):
        executor.empty_cache()
        gc.collect()


__all__ = [
    "BasicExecutor",
    "CUDAExecutor",
    "MPSExecutor",
    "CPUExecutor",
    "executor",
    "no_cache",
]
