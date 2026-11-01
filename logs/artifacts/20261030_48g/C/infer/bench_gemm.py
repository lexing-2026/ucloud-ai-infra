"""Benchmark the actual SGLang block-FP8 GEMM on the 5 Qwen3.8-27B weight shapes.

Reports the selected tuned config, correctness vs a dequantized reference, and
both TFLOPS and weight-streaming GB/s (decode is bandwidth-bound).
"""
import json, torch
from sglang.kernels.ops.gemm.fp8_kernel import (
    w8a8_block_fp8_matmul, get_w8a8_block_fp8_configs, deep_gemm_wrapper)

dev = "cuda"
BLOCK = [128, 128]

SHAPES = [(34816, 5120), (16384, 5120), (14336, 5120), (5120, 17408), (5120, 6144)]
MS = [5, 9, 18, 72, 144, 8192]

print("deepgemm_enabled", getattr(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", None), flush=True)


def ref_matmul(A, B, As, Bs):
    M, K = A.shape
    N = B.shape[0]
    Ad = (A.float().view(M, K // 128, 128) * As[..., None]).reshape(M, K)
    Bd = (B.float().view(N // 128, 128, K // 128, 128) * Bs[:, None, :, None]).reshape(N, K)
    return Ad @ Bd.T


def timeit(fn, warmup=8, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


rows = []
for N, K in SHAPES:
    B = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    Bs = (torch.rand(N // 128, K // 128, device=dev) * 0.02 + 0.01).float()
    cfg = get_w8a8_block_fp8_configs(N, K, 128, 128)
    for M in MS:
        A = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
        As = (torch.rand(M, K // 128, device=dev) * 0.02 + 0.01).float()
        out = w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16)
        err = None
        if M <= 18:
            ref = ref_matmul(A, B, As, Bs)
            err = round(((out.float() - ref).abs().mean() / ref.abs().mean()).item(), 5)
        ms = timeit(lambda: w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16),
                    iters=60 if M <= 144 else 15)
        macs = M * N * K
        wbytes = N * K  # fp8 weights dominate
        chosen = cfg[min(cfg.keys(), key=lambda x: abs(x - M))] if cfg else "default"
        row = dict(N=N, K=K, M=M, ms=round(ms, 4),
                   tflops=round(2 * macs / ms / 1e9, 1),
                   weight_gbs=round(wbytes / ms / 1e6, 1),
                   rel_err=err, cfg={k: chosen[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N",
                                                           "BLOCK_SIZE_K", "GROUP_SIZE_M",
                                                           "num_warps", "num_stages", "SPLIT_K", "SWAP_AB")
                                     if k in chosen} if isinstance(chosen, dict) else chosen)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del A, As, out
    del B, Bs
print("BENCH_GEMM_END")
