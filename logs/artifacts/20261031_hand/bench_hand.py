"""Triton vs hand-scheduled FP8 GEMM: correctness against Triton, then A/B timing."""
import ctypes, json, os, subprocess, sys, tempfile, time, torch
from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton
sys.path.insert(0, "/root/infer")
from gen_fp8_gemm import gen, smem_bytes

PTXAS = "/root/infer/sg22/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"
CUOBJ = "/usr/local/cuda/bin/cuobjdump"
torch.cuda.init(); torch.zeros(1, device="cuda")
cu = ctypes.CDLL("libcuda.so.1")
ctx = ctypes.c_void_p(); dev = ctypes.c_int()
cu.cuDeviceGet(ctypes.byref(dev), 0); cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev); cu.cuCtxSetCurrent(ctx)


def build(src, name, keep=None):
    d = tempfile.mkdtemp(); p, o = f"{d}/k.ptx", f"{d}/k.cubin"
    open(p, "w").write(src)
    r = subprocess.run([PTXAS, "-arch", "sm_89", "-m64", p, "-o", o], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-1500:]); raise SystemExit(1)
    if keep:
        subprocess.run(f"{CUOBJ} -sass {o} > {keep}", shell=True)
    img = open(o, "rb").read(); mod = ctypes.c_void_p(); fn = ctypes.c_void_p()
    rc = cu.cuModuleLoadData(ctypes.byref(mod), img); assert rc == 0, rc
    rc = cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()); assert rc == 0, rc
    return fn


class Hand:
    def __init__(self, K, stages=2, interleave=True, gap=3, G=32, keep=None):
        self.fn = build(gen(K, stages, interleave, gap), "fp8_gemm_hand", keep)
        self.smem = smem_bytes(stages); self.G = G
        cu.cuFuncSetCacheConfig(self.fn, 1)
        rc = cu.cuFuncSetAttribute(self.fn, 8, ctypes.c_int(self.smem))
        assert rc == 0, ("attr", rc, self.smem)

    def __call__(self, A, B, As, Bs):
        M, K = A.numel() // A.shape[-1], A.shape[-1]
        N = B.shape[0]
        C = A.new_empty(A.shape[:-1] + (N,), dtype=torch.bfloat16)
        grid = ((M + 63) // 64) * (N // 128)
        vals = [ctypes.c_uint64(x.data_ptr()) for x in (A, B, C, As, Bs)]
        vals += [ctypes.c_uint32(v) for v in (M, N, K, self.G)]
        ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        s = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = cu.cuLaunchKernel(self.fn, grid, 1, 1, 128, 1, 1, self.smem, s, ptrs, None)
        assert rc == 0, rc
        return C


def make(N, K, M):
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    Bs = (torch.rand(N // 128, K // 128, device="cuda") * 0.02 + 0.01).float()
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    As = (torch.rand(M, K // 128, device="cuda") * 0.02 + 0.01).float()
    return A, B, As, Bs


def ms(f, reps=9):
    f(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2] * 1e3


SHAPES = [(34816, 5120), (16384, 5120), (14336, 5120), (5120, 17408), (5120, 6144)]
if __name__ == "__main__":
    for (N, K, M) in [(2048, 512, 64), (5120, 6144, 128), (5120, 6144, 200), (14336, 5120, 257)]:
        h = Hand(K, keep="/root/infer/hand.sass")
        A, B, As, Bs = make(N, K, M)
        r = w8a8_block_fp8_matmul_triton(A, B, As, Bs, [128, 128], torch.bfloat16).float()
        o = h(A, B, As, Bs).float(); torch.cuda.synchronize()
        d = (o - r).abs()
        print(json.dumps(dict(N=N, K=K, M=M, max_abs=d.max().item(), rel=(d.mean() / r.abs().mean()).item(),
                              nan=int(o.isnan().sum()), bad=int((d > r.abs().clamp(min=1e-3) * 0.02).sum()))), flush=True)
    print("CHECK_END", flush=True)
    variants = {"s2": dict(stages=2), "s3": dict(stages=3), "s2_tail": dict(stages=2, interleave=False)}
    for N, K in SHAPES:
        A, B, As, Bs = make(N, K, 8192)
        base = ms(lambda: w8a8_block_fp8_matmul_triton(A, B, As, Bs, [128, 128], torch.bfloat16))
        row = dict(N=N, K=K, triton_ms=round(base, 3))
        for nm, kw in variants.items():
            x = Hand(K, **kw)
            t = ms(lambda: x(A, B, As, Bs))
            row[nm] = round(t, 3); row[nm + "_x"] = round(base / t, 3)
        row["tflops_triton"] = round(2 * 8192 * N * K / base / 1e9, 1)
        print(json.dumps(row), flush=True)
        del A, B, As, Bs
    print("BENCH_END", flush=True)
