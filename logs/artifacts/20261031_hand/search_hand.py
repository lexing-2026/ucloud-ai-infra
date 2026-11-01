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

N, K = int(sys.argv[1]), int(sys.argv[2])
random.seed(int(sys.argv[3]))
A, B, As, Bs = make(N, K, 8192)
tri = lambda: w8a8_block_fp8_matmul_triton(A, B, As, Bs, [128, 128], torch.bfloat16)
print(json.dumps(dict(ref=soak(tri, 2.5))), flush=True)
seen = {}
fix = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
while len(seen) < int(sys.argv[4]):
    BM = random.choice([64, 128, 128])
    kw = dict(BM=BM, gap=random.choice([3, 4, 4, 5]), cps=random.choice([2, 4, 5, 6, 8]), bpos=random.choice([2, 3, 4, 5, 6]),
              opt=random.choice([0, 1, 1, 2]), NB=random.choice([1, 2, 4]), G={64: random.choice([32, 64, 128]), 128: random.choice([16, 32, 64])}[BM])
    kw.update(fix)
    seen[json.dumps(kw, sort_keys=True)] = kw

def comp(kw):
    try:
        return compile_ptx(gen(K, 2, kw["gap"], kw["BM"], kw["NB"], True, cps=kw["cps"], bpos=kw["bpos"]), kw["opt"])
    except SystemExit:
        return None

t0 = time.time()
with ThreadPoolExecutor(18) as ex:
    done = list(ex.map(comp, seen.values()))
print("compiled", sum(x is not None for x in done), round(time.time() - t0), flush=True)
res = []
for i, (key, kw) in enumerate(seen.items()):
    if done[i] is None:
        continue
    x = Hand(K, **kw)
    r = soak(lambda: x(A, B, As, Bs))
    res.append((r["ms"], key))
    print(json.dumps(dict(r=r, kw=kw)), flush=True)
    if i % 12 == 11:
        print(json.dumps(dict(ref=soak(tri, 1.5))), flush=True)
res.sort()
print("TOP", json.dumps(res[:8]), flush=True)
print("SEARCH_END", flush=True)
