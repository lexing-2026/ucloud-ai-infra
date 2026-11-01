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

CANDS = {"c1": dict(BM=128, gap=5, cps=8, bpos=5, opt=1, NB=4, G=32),
         "c2": dict(BM=128, gap=3, cps=6, bpos=6, opt=1, NB=1, G=64),
         "c3": dict(BM=128, gap=3, cps=6, bpos=4, opt=1, NB=1, G=32)}
tot = {k: 0.0 for k in CANDS}; tot["tri"] = 0.0
W = {(34816, 5120): 64, (16384, 5120): 48, (14336, 5120): 16, (5120, 17408): 64, (5120, 6144): 64}
for (N, K), n in W.items():
    A, B, As, Bs = make(N, K, 8192)
    tri = lambda: w8a8_block_fp8_matmul_triton(A, B, As, Bs, [128, 128], torch.bfloat16)
    row = dict(N=N, K=K)
    r = [soak(tri)["ms"]]
    for nm, kw in CANDS.items():
        x = Hand(K, **kw)
        o = x(A, B, As, Bs).float(); ref = tri().float(); torch.cuda.synchronize()
        err = ((o - ref).abs().mean() / ref.abs().mean()).item()
        s = soak(lambda: x(A, B, As, Bs))
        row[nm] = s["ms"]; row[nm + "_mhz"] = s["mhz"]; row[nm + "_err"] = float(f"{err:.1e}")
        tot[nm] += s["ms"] * n
    r.append(soak(tri)["ms"])
    row["tri"] = round(sum(r) / 2, 3); tot["tri"] += row["tri"] * n
    print(json.dumps(row), flush=True)
    del A, B, As, Bs
print(json.dumps({k: round(v, 1) for k, v in tot.items()}), flush=True)
print("VERIFY_END", flush=True)
