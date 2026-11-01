"""Sweep hand-kernel knobs; every timing is paired with a Triton run on the same buffers."""
import json, subprocess, threading, time, torch
from hand_lib import *

def smi(stop, out):
    while not stop.is_set():
        r = subprocess.run("nvidia-smi --query-gpu=clocks.sm,power.draw,clocks.mem --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.split(",")
        try: out.append((float(r[0]), float(r[1])))
        except Exception: pass
        time.sleep(0.15)

def soak(f, sec=4.0):
    stop = threading.Event(); out = []; th = threading.Thread(target=smi, args=(stop, out)); th.start()
    t = time.time(); n = 0
    while time.time() - t < sec:
        f(); n += 1
    torch.cuda.synchronize(); el = time.time() - t; stop.set(); th.join()
    v = sorted(out)[len(out) // 2]; p = sorted(x[1] for x in out)[len(out) // 2]
    return dict(ms=round(el / n * 1e3, 3), mhz=v[0], watt=p)

SHAPES = [(34816, 5120), (5120, 17408)]
# correctness of the persistent walk, including a partial last block and a short last CTA
for (N, K, M) in [(5120, 6144, 128), (5120, 6144, 200), (14336, 5120, 257), (14336, 5120, 300), (1024, 5120, 8192)]:
    for kw in (dict(), dict(BM=64)):
        h = Hand(K, **kw)
        A, B, As, Bs = make(N, K, M)
        r = w8a8_block_fp8_matmul_triton(A, B, As, Bs, [128, 128], torch.bfloat16).float()
        o = h(A, B, As, Bs).float(); torch.cuda.synchronize()
        d = (o - r).abs()
        print(json.dumps(dict(N=N, K=K, M=M, kw=kw, rel=(d.mean() / r.abs().mean()).item(), nan=int(o.isnan().sum()), bad=int((d > r.abs().clamp(min=1e-3) * 0.02).sum()))), flush=True)
for N, K in SHAPES:
    A, B, As, Bs = make(N, K, 8192)
    tri = lambda: w8a8_block_fp8_matmul_triton(A, B, As, Bs, [128, 128], torch.bfloat16)
    print(json.dumps(dict(N=N, K=K, soak_tri=soak(tri))), flush=True)
    for name, kw in [("b128_nb8", dict()), ("b128_nb16", dict(NB=16)), ("b128_nb4", dict(NB=4)), ("b128_nb32", dict(NB=32)), ("b128_nb8_g3", dict(gap=3)),
                     ("b64_nb8", dict(BM=64)), ("b64_nb1", dict(BM=64, NB=1))]:
        x = Hand(K, **kw)
        print(json.dumps(dict(N=N, K=K, v=name, soak=soak(lambda: x(A, B, As, Bs), 2.5))), flush=True)
    del A, B, As, Bs
print("SWEEP_END", flush=True)
