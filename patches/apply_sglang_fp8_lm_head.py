#!/usr/bin/env python3
"""Online block FP8 for the BF16 lm_head of Qwen3.8-27B-FP8 (SGLang).

The official checkpoint leaves lm_head in BF16: 2.5GB read at every decode step,
twice per step under DFlash because the draft borrows the target head. Profiling
puts it at 11-13% of a step. With SGLANG_FP8_LM_HEAD=1 the head is quantized to
FP8 with [128, 128] scale blocks at load time and goes through the tuned W8A8
GEMM. SGLang already dispatches any non-dense head through `quant_method.apply`,
under CUDA graphs and in DFlash's candidate top-k.

This changes target logits, so it has to pass IFBench like any quantization.
SGLANG_FP8_LM_HEAD=0 (the default) leaves the head alone.

Apply with the python of the SGLang env:
  /root/infer/sg22/bin/python patches/apply_sglang_fp8_lm_head.py
"""
import importlib.util
from pathlib import Path
import sys

MODULE = '''"""Online block FP8 for a BF16 lm_head. Installed by patches/apply_sglang_fp8_lm_head.py."""
import logging
import os

import torch
from torch.nn import Parameter

from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

logger = logging.getLogger(__name__)

BLOCK = 128
FP8_MAX = 448.0
QUANT_CHUNK_ROWS = 1024


class Fp8LmHeadMethod:
    def apply(self, layer, x, bias=None):
        q_input, x_scale = per_token_group_quant_fp8(x.view(-1, x.shape[-1]), BLOCK)
        output = w8a8_block_fp8_matmul(
            q_input,
            layer.weight,
            x_scale,
            layer.weight_scale_inv,
            [BLOCK, BLOCK],
            output_dtype=x.dtype,
        )
        return output.view(*x.shape[:-1], -1)


def maybe_quantize_lm_head(model):
    if os.getenv("SGLANG_FP8_LM_HEAD", "0") != "1":
        return
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16 or weight.ndim != 2:
        return
    N, K = weight.shape
    if N % BLOCK or K % BLOCK:
        return
    quantized = torch.empty_like(weight.data, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        N // BLOCK, K // BLOCK, device=weight.device, dtype=torch.float32
    )
    for start in range(0, N // BLOCK, QUANT_CHUNK_ROWS):
        rows = min(QUANT_CHUNK_ROWS, N // BLOCK - start)
        part = weight.data[start * BLOCK : (start + rows) * BLOCK].float()
        blocks = part.reshape(rows, BLOCK, K // BLOCK, BLOCK)
        scale = (blocks.abs().amax(dim=(1, 3)) / FP8_MAX).clamp_min(1e-12)
        full = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        quantized[start * BLOCK : (start + rows) * BLOCK] = (part / full).to(
            torch.float8_e4m3fn
        )
        scales[start : start + rows] = scale
    head.weight = Parameter(quantized, requires_grad=False)
    head.weight_scale_inv = Parameter(scales, requires_grad=False)
    head.quant_method = Fp8LmHeadMethod()
    logger.info("Quantized lm_head to block FP8: %s", tuple(head.weight.shape))
'''

HOOK_OLD = (
    "                quant_method.process_weights_after_loading(module)\n"
    "\n"
    "    @staticmethod\n"
    "    def restore_weights_before_loading(model, target_device):\n"
)
HOOK_NEW = (
    "                quant_method.process_weights_after_loading(module)\n"
    "        from sglang.srt.layers.quantization.fp8_lm_head import maybe_quantize_lm_head\n"
    "\n"
    "        maybe_quantize_lm_head(model)\n"
    "\n"
    "    @staticmethod\n"
    "    def restore_weights_before_loading(model, target_device):\n"
)


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        print("sglang not found in this python", file=sys.stderr)
        return 1
    root = Path(spec.origin).parent
    (root / "srt/layers/quantization/fp8_lm_head.py").write_text(MODULE)
    loader = root / "srt/model_loader/loader.py"
    text = loader.read_text()
    if "maybe_quantize_lm_head" in text:
        print(f"already patched: {loader}")
        return 0
    if text.count(HOOK_OLD) != 1:
        print(f"expected one match in {loader}", file=sys.stderr)
        return 1
    loader.write_text(text.replace(HOOK_OLD, HOOK_NEW))
    print(f"patched: {loader}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
