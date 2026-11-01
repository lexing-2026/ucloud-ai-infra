#!/usr/bin/env python3
"""Install tile configs for the W8A8 block FP8 GEMM on RTX 4090.

Neither vLLM nor SGLang ships a config for this device, so every Qwen3.8-27B
shape falls back to the default tile (BLOCK_M=64, 4 warps), and small-M decode
pays for 64 wrapped rows per tile. The JSONs in fp8_gemm_rtx4090/ were tuned on
RTX 4090 by scripts/tune_fp8_gemm.py and change tiling only, not the math.
SWAP_AB/SPLIT_K entries also need patches/apply_sglang_sm89_gemm.py on SGLang.

Apply with the python of the target env; --revert removes the files:
  /usr/local/miniconda3/envs/py312/bin/python patches/apply_rtx4090_fp8_gemm_configs.py
"""
import importlib.util
from pathlib import Path
import shutil
import sys

SOURCE = Path(__file__).resolve().parent / "fp8_gemm_rtx4090"
ENGINES = {
    # package -> (candidate configs dirs, how the engine spells block_shape)
    # SGLang only: vLLM's kernel rejects the SWAP_AB/SPLIT_K keys these files carry.
    # SGLang moved the configs between 0.5.15 and 0.5.22.
    "sglang": (
        ["kernels/ops/gemm/configs", "srt/layers/quantization/configs"],
        "[128, 128]",
    ),
}


def main() -> int:
    revert = "--revert" in sys.argv[1:]
    installed = 0
    for package, (rel_dirs, block_shape) in ENGINES.items():
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            continue
        root = Path(spec.origin).parent
        target = next((root / d for d in rel_dirs if (root / d).is_dir()), None)
        if target is None:
            print(f"skip {package}: no configs dir under {root}")
            continue
        for src in sorted(SOURCE.glob("*.json")):
            dst = target / src.name.replace("[128,128]", block_shape)
            if revert:
                dst.unlink(missing_ok=True)
            else:
                shutil.copyfile(src, dst)
            installed += 1
            print(("removed " if revert else "installed ") + str(dst))
    if not installed:
        print("no sglang found in this python", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
