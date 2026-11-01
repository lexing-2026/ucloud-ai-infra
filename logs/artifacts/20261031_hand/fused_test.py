"""Fused gate_up GEMM + SiLU + quant against the Triton chain it replaces."""
import ctypes, json, sys, time, torch
import torch.nn.functional as F
from hand_lib import *
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from gen_fused import gen_fused, smem_bytes as fsmem


class Fused:
    def __init__(self, K, stages=2, gap=5, NB=4, cps=8, bpos=5, G=64, opt=1):
        self.fn = build(gen_fused(K, stages, gap, NB, cps, bpos), "fp8_gemm_silu_quant", None, opt)
        self.smem = fsmem(stages); self.G = G; self.NB = NB
        cu.cuFuncSetCacheConfig(self.fn, 1)
        rc = cu.cuFuncSetAttribute(self.fn, 8, ctypes.c_int(self.smem)); assert rc == 0, rc

    def __call__(self, A, B, As, Bs):
        M, K = A.numel() // A.shape[-1], A.shape[-1]
        N = B.shape[0]
        Q = A.new_empty((M, N // 2), dtype=torch.float8_e4m3fn)
        Qs = A.new_empty((M, N // 2 // 128), dtype=torch.float32)
        blocks = ((M + 63) // 64) * (N // 256)
        grid = (blocks + self.NB - 1) // self.NB
        vals = [ctypes.c_uint64(x.data_ptr()) for x in (A, B, Q, As, Bs, Qs)] + [ctypes.c_uint32(v) for v in (M, N, self.G)]
        ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        s = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = cu.cuLaunchKernel(self.fn, grid, 1, 1, 256, 1, 1, self.smem, s, ptrs, None); assert rc == 0, rc
        return Q, Qs


def chain(A, B, As, Bs):
    c = w8a8_block_fp8_matmul_triton(A, B, As, Bs, [128, 128], torch.bfloat16)
    g, u = c.chunk(2, -1)
    y = F.silu(g) * u
    return per_token_group_quant_fp8(y, 128) + (y,)


f = Fused(5120)
print("built", flush=True)
for N, K, M in [(2048, 5120, 64), (2048, 5120, 200), (4096, 5120, 300), (34816, 5120, 1024)]:
    A, B, As, Bs = make(N, K, M)
    Q, Qs = f(A, B, As, Bs); torch.cuda.synchronize()
    q, s, y = chain(A, B, As, Bs)
    Dq = Q.float() * Qs.repeat_interleave(128, -1)
    Dr = q.float() * s.repeat_interleave(128, -1)
    print(json.dumps(dict(N=N, M=M, q_vs_y=((Dq - y.float()).abs().mean() / y.float().abs().mean()).item(),
                          ref_vs_y=((Dr - y.float()).abs().mean() / y.float().abs().mean()).item(),
                          scale_rel=((Qs - s).abs() / s).max().item(), nan=int(Dq.isnan().sum()), inf=int(Dq.isinf().sum()))), flush=True)
print("FUSED_CHECK_END", flush=True)
