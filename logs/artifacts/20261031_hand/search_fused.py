"""Random search over schedule and walk knobs, scored by sustained ms at the power cap."""
import json, random, subprocess, sys, threading, time, torch
from concurrent.futures import ThreadPoolExecutor
from hand_lib import *

def smi(stop, out):
    while not stop.is_set():
        r = subprocess.run("nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.split(",")
        try: out.append((float(r[0]), float(r[1])))
        except Exception: pass
        time.sleep(0.2)

def soak(f, sec=2.0, warm=0.6):
    t = time.time()
    while time.time() - t < warm:
        for _ in range(16):
            f()
        torch.cuda.synchronize()
    stop = threading.Event(); out = []; th = threading.Thread(target=smi, args=(stop, out)); th.start()
    t = time.time(); n = 0
    while time.time() - t < sec:
        for _ in range(16):
            f()
        n += 16
        torch.cuda.synchronize()
    el = time.time() - t; stop.set(); th.join()
    return dict(ms=round(el / n * 1e3, 3), mhz=sorted(x[0] for x in out)[len(out) // 2], w=sorted(x[1] for x in out)[len(out) // 2])

import torch.nn.functional as F
from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.layers.activation import silu_and_mul
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



N, K = 34816, 5120
A, B, As, Bs = make(N, K, 8192)
h = Hand(K, BM=128, gap=5, cps=8, bpos=5, opt=1, NB=4, G=32)
def base_chain():
    c = w8a8_block_fp8_matmul_triton(A, B, As, Bs, [128, 128], torch.bfloat16)
    return per_token_group_quant_fp8(silu_and_mul(c), 128)
def hand_chain():
    c = h(A, B, As, Bs)
    return per_token_group_quant_fp8(silu_and_mul(c), 128)
print(json.dumps(dict(triton_chain=soak(base_chain, 2.5))), flush=True)
print(json.dumps(dict(hand_chain=soak(hand_chain, 2.5))), flush=True)
random.seed(int(sys.argv[1]))
seen = {}
while len(seen) < int(sys.argv[2]):
    kw = dict(gap=random.choice([3, 4, 5, 6]), cps=random.choice([4, 6, 8, 10]), bpos=random.choice([3, 4, 5, 6]),
              opt=random.choice([1, 1, 2]), NB=random.choice([1, 2, 4, 8]), G=random.choice([32, 64, 128]))
    seen[json.dumps(kw, sort_keys=True)] = kw
def comp(kw):
    return compile_ptx(gen_fused(K, 2, kw["gap"], kw["NB"], kw["cps"], kw["bpos"]), kw["opt"])
with ThreadPoolExecutor(18) as ex:
    list(ex.map(comp, seen.values()))
res = []
for i, (key, kw) in enumerate(seen.items()):
    x = Fused(K, 2, kw["gap"], kw["NB"], kw["cps"], kw["bpos"], kw["G"], kw["opt"])
    r = soak(lambda: x(A, B, As, Bs))
    res.append((r["ms"], key))
    print(json.dumps(dict(r=r, kw=kw)), flush=True)
res.sort()
print("TOP", json.dumps(res[:6]), flush=True)
print(json.dumps(dict(hand_chain=soak(hand_chain, 2.5))), flush=True)
print("SEARCH_END", flush=True)
