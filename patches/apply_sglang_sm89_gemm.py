#!/usr/bin/env python3
"""Let SGLang use its SWAP_AB/SPLIT_K FP8 GEMM tiles on sm89 (RTX 4090).

SGLang only takes `_w8a8_block_fp8_matmul_hopper` on sm90, but nothing in that
Triton kernel is sm90 specific; the gate just keeps the tuned tiles off other
GPUs. With tiles tuned by scripts/tune_fp8_gemm.py in place, the config itself
is what asks for SWAP_AB/SPLIT_K, so the gate can go.

Apply with the python of the SGLang env, after apply_rtx4090_fp8_gemm_configs.py:
  /root/infer/sg22/bin/python patches/apply_sglang_sm89_gemm.py
"""
import importlib.util
from pathlib import Path
import sys

OLD = "hopper_tuned = get_platform().is_sm90 and ("
NEW = "hopper_tuned = ("


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        print("sglang not found in this python", file=sys.stderr)
        return 1
    target = Path(spec.origin).parent / "kernels/ops/gemm/fp8_kernel.py"
    text = target.read_text()
    if NEW in text:
        print(f"already patched: {target}")
        return 0
    if text.count(OLD) != 1:
        print(f"expected one match in {target}", file=sys.stderr)
        return 1
    target.write_text(text.replace(OLD, NEW))
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
