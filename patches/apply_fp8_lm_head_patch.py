#!/usr/bin/env python3
"""Env-gated online FP8 (W8A8-dynamic) quantization for the LM head.

Installs a small quant method into vLLM and hooks
CompressedTensorsConfig.get_quant_method so that the lm_head module
(prefix == "lm_head") gets it when VLLM_FP8_LM_HEAD=1.

The LM head weight is quantized to FP8 e4m3 at load time (per-tensor),
halving the bytes read per decode step. Activations are quantized
dynamically per token. Idempotent; writes .bak next to patched files.

Run with the vLLM venv python.
"""
import pathlib
import shutil
import sys

VLLM = pathlib.Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/jovyan/infer/venv017/lib/python3.12/site-packages/vllm"
)
CT = VLLM / "model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
UTILS = VLLM / "model_executor/layers/quantization/utils"
TARGET = UTILS / "fp8_lm_head.py"
MODEL = VLLM / "model_executor/models/qwen3_5.py"

MODULE_SRC = '''# SPDX-License-Identifier: Apache-2.0
# Installed by patches/apply_fp8_lm_head_patch.py
"""Online FP8 (W8A8-dynamic) quantization for the LM head."""

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

FP8_MAX = 448.0
_CHUNK = 4096
logger = init_logger(__name__)


class Fp8LmHeadMethod(UnquantizedEmbeddingMethod):
    """Per-tensor FP8 weights, per-token dynamic FP8 activations."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight.data
        n, k = w.shape

        amax = torch.zeros((), dtype=torch.float32, device=w.device)
        for i in range(0, n, _CHUNK):
            amax = torch.maximum(amax, w[i : i + _CHUNK].abs().amax().float())
        scale = (amax / FP8_MAX).clamp(min=1e-12)

        # Keep the native [N, K] layout; cutlass needs stride(0) == 1 on the
        # transposed [K, N] view passed to the GEMM.
        w_nk = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
        inv = 1.0 / scale
        for i in range(0, n, _CHUNK):
            w_nk[i : i + _CHUNK] = (w[i : i + _CHUNK].float() * inv).clamp(-FP8_MAX, FP8_MAX)

        layer.register_parameter("weight_fp8", Parameter(w_nk.t(), requires_grad=False))
        layer.register_parameter(
            "weight_scale_fp8", Parameter(scale.reshape(1).clone(), requires_grad=False)
        )
        layer.register_parameter("weight", None)
        del w
        torch.cuda.empty_cache()
        logger.info(
            "FP8 lm_head: quantized %dx%d weight to e4m3 (scale=%.6g)",
            n, k, scale.item(),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        xq, xs = ops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
        out = ops.cutlass_scaled_mm(
            xq,
            layer.weight_fp8,
            scale_a=xs,
            scale_b=layer.weight_scale_fp8,
            out_dtype=x.dtype,
            bias=bias,
        )
        return out.view(*x.shape[:-1], layer.weight_fp8.shape[1])
'''

HOOK = (
    '        if os.environ.get("VLLM_FP8_LM_HEAD", "0") == "1" and prefix.endswith("lm_head"):\n'
    "            from vllm.model_executor.layers.quantization.utils.fp8_lm_head"
    " import Fp8LmHeadMethod\n"
    "            return Fp8LmHeadMethod()\n"
)

SIG_END = ') -> "QuantizeMethodBase | None":\n'


def main() -> int:
    TARGET.write_text(MODULE_SRC)
    print(f"wrote {TARGET}")

    src = CT.read_text()
    if "VLLM_FP8_LM_HEAD" in src:
        if 'prefix == "lm_head"' in src:
            src = src.replace('prefix == "lm_head"', 'prefix.endswith("lm_head")')
            CT.write_text(src)
            print(f"repaired hook condition in {CT}")
        else:
            print(f"{CT} already patched")
    else:
        _patch_compressed_tensors(src)
    _patch_model()
    return 0


def _patch_compressed_tensors(src: str) -> None:
    if "\nimport os\n" not in src:
        if "\nimport torch" in src:
            src = src.replace("\nimport torch", "\nimport os\n\nimport torch", 1)
        else:
            src = "import os\n" + src

    start = src.index("    def get_quant_method(\n")
    end = src.index(SIG_END, start) + len(SIG_END)
    shutil.copy2(CT, CT.with_suffix(".py.bak"))
    src = src[:end] + HOOK + src[end:]
    CT.write_text(src)
    print(f"patched {CT} (backup: {CT.name}.bak)")


def _patch_model() -> None:
    src = MODEL.read_text()
    old = (
        "                self.lm_head = ParallelLMHead(\n"
        "                    config.vocab_size,\n"
        "                    config.hidden_size,\n"
        '                    prefix=maybe_prefix(prefix, "lm_head"),\n'
        "                )"
    )
    new = (
        "                self.lm_head = ParallelLMHead(\n"
        "                    config.vocab_size,\n"
        "                    config.hidden_size,\n"
        '                    prefix=maybe_prefix(prefix, "lm_head"),\n'
        "                    quant_config=self.quant_config,\n"
        "                )"
    )
    if new in src:
        print(f"{MODEL} already patched")
        return
    if old not in src:
        raise SystemExit("lm_head construction pattern not found in qwen3_5.py")
    shutil.copy2(MODEL, MODEL.with_suffix(".py.bak"))
    MODEL.write_text(src.replace(old, new, 1))
    print(f"patched {MODEL} (backup: {MODEL.name}.bak)")


if __name__ == "__main__":
    raise SystemExit(main())
