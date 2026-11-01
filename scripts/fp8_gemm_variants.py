#!/usr/bin/env python3
"""Same-process comparison of FP8 block GEMM kernel structures on sm89.

reference : sglang _w8a8_block_fp8_matmul_hopper
v1_factor : exact; one scale factor per K step, so the tile takes one FMA per element
v2_outer  : single scale per token and per N block; K loop is pure MMA, scale once after it
"""
import argparse
import json
import time

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.fp8_kernel import _w8a8_block_fp8_matmul_hopper

BLOCK = 128
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


@triton.jit
def _v1_factor(
    A, B, C, As, Bs, M, N, K, group_n, group_k,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_As_m, stride_As_k, stride_Bs_k, stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SWAP_AB: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    As_ptrs = As + offs_am * stride_As_m
    Bs_ptr = Bs + (pid_n * BLOCK_SIZE_N // group_n) * stride_Bs_n

    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        factor = tl.load(As_ptrs) * tl.load(Bs_ptr)
        if SWAP_AB:
            accumulator += tl.dot(tl.trans(b), tl.trans(a)) * factor[None, :]
        else:
            accumulator += tl.dot(a, b) * factor[:, None]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += stride_As_k
        Bs_ptr += stride_Bs_k
    if SWAP_AB:
        accumulator = tl.trans(accumulator)
    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def _v2_outer(
    A, B, C, As, Bs, M, N, K, group_n,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SWAP_AB: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if SWAP_AB:
            accumulator = tl.dot(tl.trans(b), tl.trans(a), accumulator)
        else:
            accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if SWAP_AB:
        accumulator = tl.trans(accumulator)
    a_s = tl.load(As + offs_am)
    b_s = tl.load(Bs + offs_bn // group_n)
    c = (accumulator * a_s[:, None] * b_s[None, :]).to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def quant_block(x, rows_per_block, cols_per_block):
    """Max-abs FP8 scale per block; rows_per_block=1 is per token."""
    R, C_ = x.shape
    bx = x.abs().unfold(0, rows_per_block, rows_per_block) if rows_per_block > 1 else x.abs().unsqueeze(-1)
    bx = bx.unfold(1, cols_per_block, cols_per_block).reshape(R // rows_per_block, C_ // cols_per_block, -1)
    s = (bx.amax(-1) / FP8_MAX).clamp_min(1e-12)
    full = s.repeat_interleave(rows_per_block, 0).repeat_interleave(cols_per_block, 1)
    return (x / full).to(FP8), s


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="8192x34816x5120,8192x5120x17408,8192x16384x5120,8192x5120x6144,16x34816x5120,16x5120x17408")
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--tiles", default="64x128x128x4x3,128x128x128x4x3,128x256x128x8x3,64x256x256x4x3,128x128x256x8x2,256x128x128x8x3")
    a = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    base = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=32, num_warps=4, num_stages=3, SWAP_AB=True)

    for shape in a.shapes.split(","):
        M, N, K = map(int, shape.split("x"))
        x = torch.randn(M, K, device=dev) * torch.exp(torch.randn(M, 1, device=dev))
        w = torch.randn(N, K, device=dev) * 0.02
        A, As = quant_block(x, 1, BLOCK)          # [M, K/128]
        B, Bs = quant_block(w, BLOCK, BLOCK)      # [N/128, K/128]
        A2, As2 = quant_block(x, 1, K)            # [M, 1]
        B2, Bs2 = quant_block(w, BLOCK, K)        # [N/128, 1]
        Bs2 = Bs2[:, 0]; As2 = As2[:, 0]
        C = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        C1 = C.clone(); C2 = C.clone()
        mac = M * N * K

        def grid(c):
            return triton.cdiv(M, c["BLOCK_SIZE_M"]) * triton.cdiv(N, c["BLOCK_SIZE_N"])

        def ref():
            _w8a8_block_fp8_matmul_hopper[(grid(base), 1)](
                A, B, C, As, Bs, M, N, K, BLOCK, BLOCK,
                A.stride(-2), A.stride(-1), B.stride(1), B.stride(0), C.stride(-2), C.stride(-1),
                As.stride(-2), As.stride(-1), Bs.stride(1), Bs.stride(0), **base, needs_masking=False,
            )

        def v1():
            _v1_factor[(grid(base),)](
                A, B, C1, As, Bs, M, N, K, BLOCK, BLOCK,
                A.stride(-2), A.stride(-1), B.stride(1), B.stride(0), C1.stride(-2), C1.stride(-1),
                As.stride(-2), As.stride(-1), Bs.stride(1), Bs.stride(0), **base,
            )

        def v2(cfg):
            _v2_outer[(grid(cfg),)](
                A2, B2, C2, As2, Bs2, M, N, K, BLOCK,
                A2.stride(-2), A2.stride(-1), B2.stride(1), B2.stride(0), C2.stride(-2), C2.stride(-1),
                **cfg,
            )

        ref(); ref(); v1(); v1(); v2(base); v2(base)
        t0 = timed(ref, a.reps)
        t1 = timed(v1, a.reps)
        t2 = timed(lambda: v2(base), a.reps)
        t0b = timed(ref, a.reps)
        err1 = ((C.float() - C1.float()).norm() / C.float().norm()).item()
        truth = x @ w.T
        e_ref = ((C.float() - truth).norm() / truth.norm()).item()
        e_v2 = ((C2.float() - truth).norm() / truth.norm()).item()
        print(json.dumps(dict(shape=shape, ref_ms=round(t0, 3), ref_again_ms=round(t0b, 3), ref_TMACs=round(mac / t0 / 1e9),
                              v1_ms=round(t1, 3), v1_gain=round(t0 / t1, 3), v2_ms=round(t2, 3), v2_gain=round(t0 / t2, 3),
                              v1_vs_ref=round(err1, 7), rel_err_ref=round(e_ref, 5), rel_err_v2=round(e_v2, 5))), flush=True)

        if M >= 1024:
            for tile in a.tiles.split(","):
                bm, bn, bk, w_, s = map(int, tile.split("x"))
                for swap in (True, False):
                    cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=32, num_warps=w_, num_stages=s, SWAP_AB=swap)
                    try:
                        v2(cfg); t = timed(lambda: v2(cfg), max(5, a.reps // 3))
                        print(json.dumps(dict(shape=shape, tile=tile, swap=swap, v2_ms=round(t, 3), gain=round(t0 / t, 3))), flush=True)
                    except Exception as e:
                        print(json.dumps(dict(shape=shape, tile=tile, swap=swap, error=str(e)[:90])), flush=True)
        del A, B, A2, B2, C, C1, C2, x, w, truth
        torch.cuda.empty_cache()
    print("VAR_END", flush=True)


if __name__ == "__main__":
    main()
