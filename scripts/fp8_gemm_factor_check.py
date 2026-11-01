#!/usr/bin/env python3
"""Check the one-factor FP8 GEMM path against the old path in one process.

Runs the real w8a8_block_fp8_matmul_triton on every tuned M anchor, the serving
batch sizes and unseen M. The new path must match within --max-err; timing is
alternated so clock drift hits both sides.
"""
import argparse
import json
import time

import torch

from sglang.kernels.ops.gemm import fp8_kernel

BLOCK = 128
FP8_MAX = 448.0
SHAPES = [(34816, 5120), (16384, 5120), (14336, 5120), (5120, 17408), (5120, 6144)]
M_LIST = [1, 5, 7, 8, 9, 10, 13, 16, 20, 32, 33, 40, 64, 72, 100, 256, 300, 1024, 4096, 5000, 8192]


def quant(x, rows, cols):
    R, C = x.shape
    s = x.abs().reshape(R // rows, rows, C // cols, cols).amax(dim=(1, 3)) / FP8_MAX
    s = s.clamp_min(1e-12)
    full = s.repeat_interleave(rows, 0).repeat_interleave(cols, 1)
    return (x / full).to(torch.float8_e4m3fn), s


def run(A, B, As, Bs, one_bs):
    fp8_kernel._FP8_ONE_BS = one_bs
    return fp8_kernel.w8a8_block_fp8_matmul_triton(A, B, As, Bs, [BLOCK, BLOCK], torch.bfloat16)


def timed(fn, reps):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t)
    out.sort()
    return out[len(out) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=9)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-err", type=float, default=1e-3)
    a = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    worst = 0.0
    for N, K in SHAPES:
        w = torch.randn(N, K, device=dev) * 0.02
        B, Bs = quant(w, BLOCK, BLOCK)
        for M in M_LIST:
            x = torch.randn(M, K, device=dev)
            A, As = quant(x, 1, BLOCK)
            ref = run(A, B, As, Bs, False).float()
            new = run(A, B, As, Bs, True).float()
            err = ((ref - new).norm() / ref.norm()).item()
            worst = max(worst, err)
            ratios = []
            for _ in range(a.rounds):
                told = timed(lambda: run(A, B, As, Bs, False), a.reps)
                tnew = timed(lambda: run(A, B, As, Bs, True), a.reps)
                ratios.append(told / tnew)
            ratios.sort()
            print(json.dumps(dict(N=N, K=K, M=M, err=round(err, 7), old_ms=round(told * 1e3, 4), new_ms=round(tnew * 1e3, 4),
                                  gain=round(ratios[len(ratios) // 2], 3), ok=err <= a.max_err)), flush=True)
            del x, A, As, ref, new
        del w, B, Bs
        torch.cuda.empty_cache()
    print(json.dumps(dict(worst_err=worst, passed=worst <= a.max_err)), flush=True)
    print("CHECK_END", flush=True)


if __name__ == "__main__":
    main()
