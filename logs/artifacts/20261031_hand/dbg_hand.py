import json, sys, torch
from hand_lib import *
N, K = 34816, 5120
A, B, As, Bs = make(N, K, 8192)
KT = K // 128
for name, kw in [("b64", dict(gap=5)), ("b128", dict(gap=5, BM=128, G=16))]:
    h = Hand(K, dbg=True, keep=f"/root/infer/{name}.sass", **kw)
    for _ in range(3):
        h(A, B, As, Bs); torch.cuda.synchronize()
    d = h.D.view(-1, 4).cpu().numpy().astype("int64")
    print(json.dumps(dict(v=name, ctas=len(d), pro=round(float(d[:, 0].mean())), loop_per_tile=round(float(d[:, 1].mean() / KT)), store=round(float(d[:, 2].mean())),
                          wait=round(float((d[:, 3] & 0xffff).mean() / KT), 1), bar=round(float(((d[:, 3] >> 16) & 0xffff).mean() / KT), 1))), flush=True)
print("DBG_END")
