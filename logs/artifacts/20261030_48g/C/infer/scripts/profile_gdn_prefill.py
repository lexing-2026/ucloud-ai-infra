#!/usr/bin/env python3
"""Time each Triton GDN prefill stage at the real Qwen3.8-27B shape, in one process."""
import argparse
import json
import time
from collections import defaultdict

import torch

from sglang.kernels.ops.attention.fla import chunk as C

STAGES = [
    "l2norm_fwd",
    "chunk_local_cumsum",
    "chunk_gated_delta_rule_fwd_intra",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_fwd_o",
]


def timed(name, fn, totals):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        totals[name] += time.perf_counter() - start
        return out

    return wrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--q-heads", type=int, default=16)
    p.add_argument("--v-heads", type=int, default=48)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--layers", type=int, default=48)
    a = p.parse_args()

    dev = "cuda"
    torch.manual_seed(0)
    T = a.tokens
    q = torch.randn(1, T, a.q_heads, a.dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, a.q_heads, a.dim, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, a.v_heads, a.dim, device=dev, dtype=torch.bfloat16)
    g = -torch.nn.functional.softplus(torch.randn(1, T, a.v_heads, device=dev))
    beta = torch.sigmoid(torch.randn(1, T, a.v_heads, device=dev))
    state = torch.zeros(1, a.v_heads, a.dim, a.dim, device=dev)
    idx = torch.tensor([0], device=dev, dtype=torch.int64)
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)

    totals = defaultdict(float)
    for name in STAGES:
        setattr(C, name, timed(name, getattr(C, name), totals))

    def once():
        state.zero_()
        return C.chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
            initial_state_indices=idx, cu_seqlens=cu, head_first=False,
            use_qk_l2norm_in_kernel=True, inplace_update=True,
        )

    for _ in range(a.warmup):
        once()
    totals.clear()
    walls = []
    for _ in range(a.reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        once()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - start)
    walls.sort()
    med = walls[len(walls) // 2]
    out = {
        "tokens": T,
        "wall_ms_p50": round(med * 1e3, 3),
        "wall_ms_min": round(walls[0] * 1e3, 3),
        "layers_x_wall_s": round(med * a.layers, 3),
        "stages_ms": {n.replace("chunk_", ""): round(v / a.reps * 1e3, 3) for n, v in totals.items()},
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
