#!/usr/bin/env python3
"""Tune Triton tiles for the W8A8 block FP8 GEMM on RTX 4090 (sm89).

Workflow borrowed from ForgeKernel (github.com/OpenBMB/ForgeKernel): the default
tile is the reference; each candidate is judged for correctness against it and
timed beside it in the same process; a candidate only wins if it stays correct
and beats the current best by more than the noise floor, then it becomes the
reference for the next round.

Why it matters: `offs_am = (... + arange(BLOCK_SIZE_M)) % M` pads small M by
wrapping rows, so decode (M=5 under MTP4) pays for BLOCK_SIZE_M rows per tile.
SWAP_AB (transpose the MMA so M can drop to 8) and SPLIT_K (parallelize K when
N/M give too few blocks for the SMs) already exist in SGLang but are gated to
sm90; the tuner lifts that gate, and patches/apply_sglang_sm89_gemm.py does the
same for serving.

  python scripts/tune_fp8_gemm.py --out patches/fp8_gemm_rtx4090_tuned
  # next generation: current best is the reference, random restarts escape it
  python scripts/tune_fp8_gemm.py --evolve-from patches/fp8_gemm_rtx4090 --restarts 30
"""
import argparse
import itertools
import json
import os
import random
import statistics
import time

import torch

# (N, K) of every FP8 linear in Qwen3.8-27B, from the engine's config lookups.
SHAPES = [(34816, 5120), (16384, 5120), (14336, 5120), (5120, 17408), (5120, 6144)]
# Decode under MTP4 at batch 1/8/2/4, then chunked prefill. Most useful first.
# Engines pick the nearest M, so prefill also needs a low anchor (see below).
GRID_M = [5, 40, 10, 20, 8192]
PREFILL_M = 8192
PREFILL_ANCHOR_M = 256
BLOCK = 128
DEFAULT = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": BLOCK,
    "BLOCK_SIZE_K": BLOCK,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 3,
}
SEARCH = {
    "SWAP_AB": [False, True],
    "SPLIT_K": [1, 2, 4, 8],
    "BLOCK_SIZE_M": [8, 16, 32, 64, 128, 256],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128, 256],
    "GROUP_SIZE_M": [1, 16, 32, 64],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4, 5, 6],
}
MAX_RELATIVE_ERROR = 1e-3


class Sm90Gate:
    """Report sm90 so SGLang accepts SWAP_AB/SPLIT_K configs on sm89."""

    is_sm90 = True

    def __init__(self, platform):
        self._platform = platform

    def __getattr__(self, name):
        return getattr(self._platform, name)


def make_inputs(m, n, k):
    device = "cuda"
    a = (torch.randn(m, k, device=device) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device=device) * 0.5).to(torch.float8_e4m3fn)
    a_s = torch.rand(m, (k + BLOCK - 1) // BLOCK, device=device) + 0.5
    b_s = torch.rand((n + BLOCK - 1) // BLOCK, (k + BLOCK - 1) // BLOCK, device=device) + 0.5
    return a, b, a_s, b_s


def run_kernel(kernel_module, config, inputs):
    kernel_module.get_w8a8_block_fp8_configs = lambda *_: {1: config}
    a, b, a_s, b_s = inputs
    return kernel_module.w8a8_block_fp8_matmul_triton(
        a, b, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
    )


def timed(kernel_module, config, inputs, budget_s=1.5, warmup=2):
    for _ in range(warmup):
        run_kernel(kernel_module, config, inputs)
    samples = []
    deadline = time.perf_counter() + budget_s
    while time.perf_counter() < deadline or len(samples) < 5:
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_kernel(kernel_module, config, inputs)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def is_correct(kernel_module, config, inputs, expected):
    got = run_kernel(kernel_module, config, inputs).float()
    denom = expected.abs().mean().clamp_min(1e-6)
    return ((got - expected).abs().mean() / denom).item() <= MAX_RELATIVE_ERROR


def descend(kernel_module, start, start_ms, inputs, expected, m, noise, tried):
    best, best_ms = start, start_ms
    improved = True
    while improved:
        improved = False
        for key, values in SEARCH.items():
            for value in values:
                candidate = {**best, key: value}
                ms = evaluate(kernel_module, candidate, inputs, expected, m, tried)
                if ms is not None and ms < best_ms * (1 - noise):
                    best, best_ms = candidate, ms
                    improved = True
                    print(f"  promote {ms:.3f}ms {candidate}", flush=True)
    return best, best_ms


def evaluate(kernel_module, candidate, inputs, expected, m, tried):
    """Median ms of a correct, untried candidate; None otherwise."""
    fingerprint = json.dumps(candidate, sort_keys=True)
    if fingerprint in tried or candidate["BLOCK_SIZE_M"] > max(m, 16) * 2:
        return None
    tried.add(fingerprint)
    try:
        if not is_correct(kernel_module, candidate, inputs, expected):
            print(f"  reject incorrect {candidate}", flush=True)
            return None
        return timed(kernel_module, candidate, inputs) * 1e3
    except Exception as error:  # a tile Triton cannot compile
        print(f"  skip {candidate}: {str(error)[:80]}", flush=True)
        return None


def coordinate_descent(kernel_module, m, n, k, start=None, restarts=0, noise=0.015):
    inputs = make_inputs(m, n, k)
    expected = run_kernel(kernel_module, DEFAULT, inputs).float()
    reference = start or DEFAULT
    reference_ms = timed(kernel_module, reference, inputs) * 1e3
    tried = {json.dumps(reference, sort_keys=True)}
    best, best_ms = descend(
        kernel_module, dict(reference), reference_ms, inputs, expected, m, noise, tried
    )
    rng = random.Random(n * 31 + k * 7 + m)
    for _ in range(restarts):
        candidate = {key: rng.choice(values) for key, values in SEARCH.items()}
        ms = evaluate(kernel_module, candidate, inputs, expected, m, tried)
        if ms is not None and ms < best_ms * 1.1:
            found, found_ms = descend(
                kernel_module, candidate, ms, inputs, expected, m, noise, tried
            )
            if found_ms < best_ms:
                best, best_ms = found, found_ms
    # Re-time both in one run so the verdict does not rest on the search's clock.
    reference_ms = timed(kernel_module, reference, inputs) * 1e3
    best_ms = timed(kernel_module, best, inputs) * 1e3
    return best, reference_ms, best_ms, len(tried)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/infer/patches/fp8_gemm_rtx4090_tuned")
    parser.add_argument("--module", default="sglang.kernels.ops.gemm.fp8_kernel")
    parser.add_argument("--evolve-from", help="dir of tuned JSONs to use as reference")
    parser.add_argument("--restarts", type=int, default=0)
    parser.add_argument("--grid-m", default=",".join(map(str, GRID_M)))
    args = parser.parse_args()

    import importlib

    kernel_module = importlib.import_module(args.module)
    gate = Sm90Gate(kernel_module.get_platform())
    kernel_module.get_platform = lambda: gate
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    os.makedirs(args.out, exist_ok=True)

    tables = {shape: {} for shape in SHAPES}
    parents = {}
    if args.evolve_from:
        for n, k in SHAPES:
            name = f"N={n},K={k},device_name={device_name},dtype=fp8_w8a8,block_shape=[128,128].json"
            with open(os.path.join(args.evolve_from, name)) as handle:
                parents[(n, k)] = json.load(handle)
            tables[(n, k)] = dict(parents[(n, k)])
    for m in map(int, args.grid_m.split(",")):
        for n, k in SHAPES:
            print(f"== N={n} K={k} M={m}", flush=True)
            start = parents.get((n, k), {}).get(str(m))
            best, reference_ms, best_ms, tried = coordinate_descent(
                kernel_module, m, n, k, start=start, restarts=args.restarts
            )
            tables[(n, k)][str(m)] = best
            if m == PREFILL_M:
                # Without this, M around 4096 is nearer to 40 and takes a decode tile.
                tables[(n, k)][str(PREFILL_ANCHOR_M)] = best
            line = {
                "N": n, "K": k, "M": m, "reference_ms": round(reference_ms, 4),
                "best_ms": round(best_ms, 4), "speedup": round(reference_ms / best_ms, 3),
                "tried": tried, "config": best,
            }
            print(json.dumps(line), flush=True)
            with open(os.path.join(args.out, "tuning_log.ndjson"), "a") as handle:
                handle.write(json.dumps(line) + "\n")
            torch.cuda.empty_cache()
            name = f"N={n},K={k},device_name={device_name},dtype=fp8_w8a8,block_shape=[128,128].json"
            with open(os.path.join(args.out, name), "w") as handle:
                json.dump(tables[(n, k)], handle, indent=2)


if __name__ == "__main__":
    main()
