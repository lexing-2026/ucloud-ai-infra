#!/usr/bin/env python3
"""Closed-loop benchmark shared by vLLM and SGLang so A/B numbers are comparable.

Follows the official track-A口径: prompts are random non-special token IDs sent
to /v1/completions, ignore_eos, temperature 0, warmup excluded. TPOT per request
is (e2e - ttft) / (osl - 1), then percentiles over requests, as docs/eval does.
This is NOT the official AIPerf harness: use it for A/B, not for the final report.

Default mode isolates the cache (fresh seed and cache_salt), like the official
harness. --shared-prefix N models real traffic instead (same system prompt or
document, different question): every prompt shares N leading tokens under one
salt, warmup primes the cache, and cached_tokens are reported. Keep the two modes
in separate tables.

  python scripts/bench_ab.py --base-url http://127.0.0.1:8000 --tag t0 -c 1 -n 4
  python scripts/bench_ab.py --tag cache --shared-prefix 7168 -c 1,4 -n 8
"""
import argparse
import asyncio
import json
import random
import statistics
import time
import uuid

import aiohttp
from transformers import AutoTokenizer


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))]


def make_prompts(pool, count, length, seed, prefix=()):
    rng = random.Random(seed)
    return [list(prefix) + rng.choices(pool, k=length - len(prefix)) for _ in range(count)]


async def one_request(session, args, prompt, salt):
    body = {
        "model": args.served,
        "prompt": prompt,
        "max_tokens": args.osl,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }
    start = time.perf_counter()
    ttft = None
    prompt_tokens = completion_tokens = cached_tokens = None
    async with session.post(f"{args.base_url}/v1/completions", json=body) as resp:
        if resp.status != 200:
            return {"error": f"{resp.status}: {(await resp.text())[:300]}"}
        async for raw in resp.content:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            if ttft is None and any(c.get("text") for c in chunk.get("choices", [])):
                ttft = time.perf_counter() - start
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens")
    e2e = time.perf_counter() - start
    if ttft is None or not completion_tokens or completion_tokens < 2:
        return {"error": f"bad stream: ttft={ttft} osl={completion_tokens}"}
    return {
        "ttft": ttft,
        "e2e": e2e,
        "tpot": (e2e - ttft) / (completion_tokens - 1),
        "isl": prompt_tokens,
        "osl": completion_tokens,
        "cached": cached_tokens,
    }


async def closed_loop(args, prompts, concurrency, salt):
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    queue = list(enumerate(prompts))
    results = []

    async def worker(session):
        while queue:
            _, prompt = queue.pop(0)
            results.append(await one_request(session, args, prompt, salt))

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        start = time.perf_counter()
        await asyncio.gather(*(worker(session) for _ in range(concurrency)))
        return results, time.perf_counter() - start


def summarize(args, concurrency, results, wall):
    good = [r for r in results if "error" not in r]
    errors = [r["error"] for r in results if "error" in r]
    summary = {
        "tag": args.tag,
        "c": concurrency,
        "requests": len(results),
        "ok": len(good),
        "wall_s": round(wall, 2),
    }
    if good:
        out_tokens = sum(r["osl"] for r in good)
        for key, scale, name in (("ttft", 1e3, "ttft"), ("tpot", 1e3, "tpot"), ("e2e", 1, "e2e")):
            values = [r[key] * scale for r in good]
            summary[f"{name}_p50"] = round(statistics.median(values), 2)
            summary[f"{name}_p95"] = round(percentile(values, 95), 2)
        summary["out_tok_s"] = round(out_tokens / wall, 2)
        summary["isl"] = sorted({r["isl"] for r in good})[:4]
        summary["osl"] = sorted({r["osl"] for r in good})[:4]
        cached = [r["cached"] for r in good if r["cached"] is not None]
        if cached:
            summary["cached_p50"] = statistics.median(cached)
    if errors:
        summary["errors"] = errors[:3]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/model/ModelScope/Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--served", default="qwen3.8-27b-fp8")
    parser.add_argument("--tag", default="unlabeled")
    parser.add_argument("-c", "--concurrency", default="1", help="comma separated")
    parser.add_argument("-n", "--num", type=int, default=4, help="requests per level")
    parser.add_argument("--isl", type=int, default=8192)
    parser.add_argument("--osl", type=int, default=512)
    parser.add_argument("--shared-prefix", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", default="/root/infer/logs/bench_ab.ndjson")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special = set(tokenizer.all_special_ids)
    pool = [i for i in range(tokenizer.vocab_size) if i not in special]

    for concurrency in (int(c) for c in args.concurrency.split(",")):
        seed = random.SystemRandom().randrange(2**31)
        prefix = random.Random(seed).choices(pool, k=args.shared_prefix)
        warmup = make_prompts(pool, max(4, concurrency), args.isl, seed, prefix)
        measured = make_prompts(pool, args.num, args.isl, seed + 1, prefix)
        shared_salt = uuid.uuid4().hex
        salts = (shared_salt, shared_salt) if prefix else (uuid.uuid4().hex, shared_salt)
        asyncio.run(closed_loop(args, warmup, concurrency, salts[0]))
        results, wall = asyncio.run(closed_loop(args, measured, concurrency, salts[1]))
        summary = summarize(args, concurrency, results, wall)
        summary["seed"] = seed
        summary["shared_prefix"] = args.shared_prefix
        line = json.dumps(summary, ensure_ascii=False)
        print(line, flush=True)
        with open(args.out, "a") as handle:
            handle.write(line + "\n")


if __name__ == "__main__":
    main()
