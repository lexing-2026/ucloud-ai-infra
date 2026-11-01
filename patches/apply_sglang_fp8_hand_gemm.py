#!/usr/bin/env python3
"""Hand-scheduled FP8 GEMM for sm_89 (SGLang).

Installs patches/fp8_hand_sm89.py next to the Triton kernels, precompiles its kernels
(ptxas needs about 40 s per kernel, so nothing is compiled on the request path), moves the
FP8 weights to tile first storage after loading and makes `w8a8_block_fp8_matmul_triton`
try the hand kernel first.

With SGLANG_FP8_HAND_GEMM=1 every FP8 linear except lm_head goes to the hand kernels, and a
large MLP call uses the fused gate_up + SiLU + quant kernel (SGLANG_FP8_HAND_FUSED=0 turns
only that off). lm_head stays on Triton. Needs patches/apply_sglang_fp8_lm_head.py first.

Apply with the python of the SGLang env:
  /root/infer/sg22/bin/python patches/apply_sglang_fp8_hand_gemm.py
"""
import importlib.util
from pathlib import Path
import shutil
import sys

HOOK_OLD = (
    "    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)\n"
    "\n"
    "    block_n, block_k = block_size\n"
)
HOOK_NEW = (
    "    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)\n"
    "\n"
    "    if fp8_hand_sm89.matmul(A, B, C, As, Bs, M, N, K, block_size):\n"
    "        return C\n"
    "\n"
    "    block_n, block_k = block_size\n"
)
IMPORT_OLD = "from sglang.srt.layers import deep_gemm_wrapper\n"
IMPORT_NEW = (
    "from sglang.kernels.ops.gemm import fp8_hand_sm89\n"
    "from sglang.srt.layers import deep_gemm_wrapper\n"
)
MLP_OLD = (
    "        x,\n"
    "    ):\n"
    "        gate_up, _ = self.gate_up_proj(x)\n"
    "        if self._enable_silu_fp4_quant_fusion and not isinstance(gate_up, tuple):\n"
)
MLP_NEW = (
    "        x,\n"
    "    ):\n"
    "        from sglang.kernels.ops.gemm import fp8_hand_sm89\n"
    "\n"
    "        fused = fp8_hand_sm89.fused_gate_up(self.gate_up_proj, x)\n"
    "        if fused is not None:\n"
    "            x, _ = self.down_proj(fused)\n"
    "            return x\n"
    "        gate_up, _ = self.gate_up_proj(x)\n"
    "        if self._enable_silu_fp4_quant_fusion and not isinstance(gate_up, tuple):\n"
)


LOADER_OLD = "        maybe_quantize_lm_head(model)\n"
LOADER_NEW = (
    "        maybe_quantize_lm_head(model)\n"
    "        from sglang.kernels.ops.gemm import fp8_hand_sm89\n"
    "\n"
    "        fp8_hand_sm89.relayout(model)\n"
)


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        print("sglang not found in this python", file=sys.stderr)
        return 1
    root = Path(spec.origin).parent
    target = root / "kernels/ops/gemm/fp8_hand_sm89.py"
    shutil.copyfile(Path(__file__).with_name("fp8_hand_sm89.py"), target)

    kernel = root / "kernels/ops/gemm/fp8_kernel.py"
    text = kernel.read_text()
    if "fp8_hand_sm89" in text:
        print(f"already patched: {kernel}")
    else:
        if text.count(HOOK_OLD) != 1 or text.count(IMPORT_OLD) != 1:
            print(f"expected one match in {kernel}", file=sys.stderr)
            return 1
        kernel.write_text(text.replace(HOOK_OLD, HOOK_NEW).replace(IMPORT_OLD, IMPORT_NEW))
        print(f"patched: {kernel}")

    mlp = root / "srt/models/qwen2_moe.py"
    text = mlp.read_text()
    if "fp8_hand_sm89" in text:
        print(f"already patched: {mlp}")
    else:
        if text.count(MLP_OLD) != 1:
            print(f"expected one match in {mlp}", file=sys.stderr)
            return 1
        mlp.write_text(text.replace(MLP_OLD, MLP_NEW))
        print(f"patched: {mlp}")

    loader = root / "srt/model_loader/loader.py"
    text = loader.read_text()
    if "fp8_hand_sm89" in text:
        print(f"already patched: {loader}")
    else:
        if text.count(LOADER_OLD) != 1:
            print(f"expected one match in {loader}", file=sys.stderr)
            return 1
        loader.write_text(text.replace(LOADER_OLD, LOADER_NEW))
        print(f"patched: {loader}")

    module = importlib.util.spec_from_file_location("fp8_hand_sm89", target)
    fp8_hand_sm89 = importlib.util.module_from_spec(module)
    module.loader.exec_module(fp8_hand_sm89)
    print("precompiled:", fp8_hand_sm89.precompile())
    return 0


if __name__ == "__main__":
    sys.exit(main())
