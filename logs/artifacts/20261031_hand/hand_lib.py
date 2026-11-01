"""Triton vs hand-scheduled FP8 GEMM: correctness against Triton, then A/B timing."""
import ctypes, hashlib, json, os, subprocess, sys, tempfile, time, torch
from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton
sys.path.insert(0, "/root/infer")
from gen_fp8_gemm import gen, smem_bytes

PTXAS = "/root/infer/sg22/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"
CUOBJ = "/usr/local/cuda/bin/cuobjdump"
torch.cuda.init(); torch.zeros(1, device="cuda")
cu = ctypes.CDLL("libcuda.so.1")
ctx = ctypes.c_void_p(); dev = ctypes.c_int()
cu.cuDeviceGet(ctypes.byref(dev), 0); cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev); cu.cuCtxSetCurrent(ctx)


CACHE = "/root/infer/cubin_cache"
os.makedirs(CACHE, exist_ok=True)


def compile_ptx(src, opt=3):
    """Returns a cubin path; safe to call from a thread pool."""
    h = hashlib.sha256((src + str(opt)).encode()).hexdigest()[:24]
    o = f"{CACHE}/{h}.cubin"
    if os.path.exists(o):
        return o
    d = tempfile.mkdtemp(); p = f"{d}/k.ptx"
    open(p, "w").write(src)
    r = subprocess.run([PTXAS, "-arch", "sm_89", "-m64", f"-O{opt}", p, "-o", o + ".tmp"], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-1500:]); raise SystemExit(1)
    os.replace(o + ".tmp", o)
    return o


def build(src, name, keep=None, opt=3):
    o = compile_ptx(src, opt)
    if keep:
        subprocess.run(f"{CUOBJ} -sass {o} > {keep}", shell=True)
    img = open(o, "rb").read(); mod = ctypes.c_void_p(); fn = ctypes.c_void_p()
    rc = cu.cuModuleLoadData(ctypes.byref(mod), img); assert rc == 0, rc
    rc = cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()); assert rc == 0, rc
    return fn


class Hand:
    def __init__(self, K, stages=2, gap=5, BM=128, NB=8, interleave=True, G=None, keep=None, dbg=False, cps=3, bpos=0, opt=3):
        self.dbg = dbg; self.BM = BM; self.NB = NB
        self.fn = build(gen(K, stages, gap, BM, NB, interleave, dbg=dbg, cps=cps, bpos=bpos), "fp8_gemm_hand", keep, opt)
        self.smem = smem_bytes(stages, BM); self.G = G or 2048 // BM
        cu.cuFuncSetCacheConfig(self.fn, 1)
        rc = cu.cuFuncSetAttribute(self.fn, 8, ctypes.c_int(self.smem))
        assert rc == 0, ("attr", rc, self.smem)

    def __call__(self, A, B, As, Bs):
        M, K = A.numel() // A.shape[-1], A.shape[-1]
        N = B.shape[0]
        C = A.new_empty(A.shape[:-1] + (N,), dtype=torch.bfloat16)
        BM = self.BM
        blocks = ((M + BM - 1) // BM) * (N // 128)
        grid = (blocks + self.NB - 1) // self.NB
        vals = [ctypes.c_uint64(x.data_ptr()) for x in (A, B, C, As, Bs)]
        vals += [ctypes.c_uint32(v) for v in (M, N, K, self.G)]
        if self.dbg:
            self.D = torch.zeros(grid * 4, device="cuda", dtype=torch.int32)
            vals.append(ctypes.c_uint64(self.D.data_ptr()))
        ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        s = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = cu.cuLaunchKernel(self.fn, grid, 1, 1, BM * 2, 1, 1, self.smem, s, ptrs, None)
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


