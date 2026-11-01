#!/usr/bin/env python3
"""Cut the per K step scale work in SGLang's Triton FP8 GEMM.

Both W8A8 block kernels run `dot(a, b) * a_s * b_s` over the whole fp32 tile for
every K block. A tile that sits inside one weight scale block has a single b_s,
so the two multiplies collapse into one factor per M row and the tile takes one
FMA per element. The result differs from the old kernel only by fp32 rounding;
scripts/fp8_gemm_factor_check.py compares the two paths in one process.

SGLANG_FP8_ONE_BS=0 turns the new path off at runtime.

Apply with the python of the SGLang env, after apply_sglang_sm89_gemm.py:
  /root/infer/sg22/bin/python patches/apply_sglang_fp8_factor.py
"""
import importlib.util
from pathlib import Path
import re
import sys

MARK = "ONE_BS_PER_TILE"

SIGNATURE_PLAIN = "    needs_masking: tl.constexpr,\n):\n"
SIGNATURE_HOPPER = "    SPLIT_K: tl.constexpr = 1,\n):\n"
SIGNATURE_NEW = "    ONE_BS_PER_TILE: tl.constexpr = False,\n):\n"

BS_POINTER_OLD = "    Bs_ptrs = Bs + offs_bsn * stride_Bs_n\n"
BS_POINTER_NEW = (
    BS_POINTER_OLD + "    Bs_ptr = Bs + (pid_n * BLOCK_SIZE_N // group_n) * stride_Bs_n\n"
)
FIRST_TILE_OLD = "    Bs_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_Bs_k\n"
FIRST_TILE_NEW = (
    FIRST_TILE_OLD + "    Bs_ptr += (first_tile // n_tiles_k_per_group_k) * stride_Bs_k\n"
)
SCALE_LOAD_OLD = "        a_s = tl.load(As_ptrs)\n        b_s = tl.load(Bs_ptrs)\n"
SCALE_LOAD_NEW = (
    "        a_s = tl.load(As_ptrs)\n"
    "        if ONE_BS_PER_TILE:\n"
    "            a_s = a_s * tl.load(Bs_ptr)\n"
    "        else:\n"
    "            b_s = tl.load(Bs_ptrs)\n"
)
STEP_OLD = "        Bs_ptrs += scale_step_k * stride_Bs_k\n"
STEP_NEW = STEP_OLD + "        Bs_ptr += scale_step_k * stride_Bs_k\n"

ACC_PLAIN_OLD = "        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]\n"
ACC_PLAIN_NEW = (
    "        if ONE_BS_PER_TILE:\n"
    "            accumulator += tl.dot(a, b) * a_s[:, None]\n"
    "        else:\n"
    "            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]\n"
)
ACC_HOPPER_OLD = (
    "        if SWAP_AB:\n"
    "            accumulator += (\n"
    "                tl.dot(tl.trans(b), tl.trans(a)) * b_s[:, None] * a_s[None, :]\n"
    "            )\n"
    "        else:\n"
    "            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]\n"
)
ACC_HOPPER_NEW = (
    "        if ONE_BS_PER_TILE:\n"
    "            if SWAP_AB:\n"
    "                accumulator += tl.dot(tl.trans(b), tl.trans(a)) * a_s[None, :]\n"
    "            else:\n"
    "                accumulator += tl.dot(a, b) * a_s[:, None]\n"
    "        elif SWAP_AB:\n"
    "            accumulator += (\n"
    "                tl.dot(tl.trans(b), tl.trans(a)) * b_s[:, None] * a_s[None, :]\n"
    "            )\n"
    "        else:\n"
    "            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]\n"
)

CALL_OLD = "        **config,\n        needs_masking=needs_masking,\n    )\n"
CALL_NEW = (
    "        **config,\n        needs_masking=needs_masking,\n        **one_scale_kwargs,\n    )\n"
)
GRID_OLD = "    needs_masking = bool(K % config[\"BLOCK_SIZE_K\"] != 0)\n\n    def grid(META):\n"
GRID_NEW = (
    "    needs_masking = bool(K % config[\"BLOCK_SIZE_K\"] != 0)\n\n"
    "    # One weight scale per tile lets the K loop take one fmul per row.\n"
    "    one_scale_kwargs = {}\n"
    "    if kernel in (_w8a8_block_fp8_matmul, _w8a8_block_fp8_matmul_hopper):\n"
    "        one_scale_kwargs[\"ONE_BS_PER_TILE\"] = (\n"
    "            _FP8_ONE_BS and block_n % config[\"BLOCK_SIZE_N\"] == 0\n"
    "        )\n\n"
    "    def grid(META):\n"
)
GLOBAL_OLD = "logger = logging.getLogger(__name__)\n"
GLOBAL_NEW = GLOBAL_OLD + "_FP8_ONE_BS = os.getenv(\"SGLANG_FP8_ONE_BS\", \"1\") != \"0\"\n"


def replace_once(text: str, old: str, new: str, where: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{where}: expected one match of {old[:48]!r}, got {text.count(old)}")
    return text.replace(old, new)


def section(text: str, name: str) -> tuple[int, int]:
    start = text.index(f"def {name}(")
    following = re.search(r"^@triton\.jit$", text[start:], re.MULTILINE)
    return start, start + following.start()


def edit_kernel(text: str, name: str, pairs: list[tuple[str, str]]) -> str:
    start, end = section(text, name)
    body = text[start:end]
    for old, new in pairs:
        body = replace_once(body, old, new, name)
    return text[:start] + body + text[end:]


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        print("sglang not found in this python", file=sys.stderr)
        return 1
    target = Path(spec.origin).parent / "kernels/ops/gemm/fp8_kernel.py"
    text = target.read_text()
    if MARK in text:
        print(f"already patched: {target}")
        return 0
    text = edit_kernel(
        text,
        "_w8a8_block_fp8_matmul",
        [
            (SIGNATURE_PLAIN, SIGNATURE_PLAIN[:-4] + SIGNATURE_NEW),
            (BS_POINTER_OLD, BS_POINTER_NEW),
            (SCALE_LOAD_OLD, SCALE_LOAD_NEW),
            (STEP_OLD, STEP_NEW),
            (ACC_PLAIN_OLD, ACC_PLAIN_NEW),
        ],
    )
    text = edit_kernel(
        text,
        "_w8a8_block_fp8_matmul_hopper",
        [
            (SIGNATURE_HOPPER, SIGNATURE_HOPPER[:-4] + SIGNATURE_NEW),
            (BS_POINTER_OLD, BS_POINTER_NEW),
            (FIRST_TILE_OLD, FIRST_TILE_NEW),
            (SCALE_LOAD_OLD, SCALE_LOAD_NEW),
            (STEP_OLD, STEP_NEW),
            (ACC_HOPPER_OLD, ACC_HOPPER_NEW),
        ],
    )
    text = replace_once(text, CALL_OLD, CALL_NEW, "w8a8_block_fp8_matmul_triton")
    text = replace_once(text, GRID_OLD, GRID_NEW, "w8a8_block_fp8_matmul_triton")
    text = replace_once(text, GLOBAL_OLD, GLOBAL_NEW, "module")
    target.write_text(text)
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
