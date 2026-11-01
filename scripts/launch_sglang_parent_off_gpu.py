#!/usr/bin/env python3
"""Run sglang.launch_server with the HTTP parent process off the GPU.

The parent never runs a model, yet importing transformers (torchao probes Triton),
sgl_kernel and flashinfer each ask for device properties, and the first one creates a
CUDA context that holds 454 MiB of the card. That is most of what one speculative draft
token costs at 48 GB. The answers are constants on this card, so the parent gets them
without a context. The scheduler is a spawned child and sees the real torch.cuda.
"""
import runpy
import sys
import types

import torch

PROPERTIES = types.SimpleNamespace(
    name="NVIDIA GeForce RTX 4090",
    major=8,
    minor=9,
    multi_processor_count=128,
    total_memory=47 * 2**30,
)


def keep_parent_off_gpu():
    torch.cuda._lazy_init = lambda: None
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_capability = lambda device=None: (PROPERTIES.major, PROPERTIES.minor)
    torch.cuda.get_device_properties = lambda device=None: PROPERTIES
    torch.cuda.get_device_name = lambda device=None: PROPERTIES.name


if __name__ == "__main__":
    keep_parent_off_gpu()
    sys.argv[0] = "sglang.launch_server"
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)
