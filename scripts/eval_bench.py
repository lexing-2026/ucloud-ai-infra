#!/usr/bin/env python3
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from transformers import AutoTokenizer

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen3.8-27b-awq"
TOK = AutoTokenizer.from_pretrained(
    "/tmp/infer/models/qwen38-awq", trust_remote_code=True
)


def build_prompt(task: str, rec: dict, max_ctx_tokens: int) -> str:
    if task == "mmlu":
        options = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(rec["choices"])
        )
        return (
            "Answer the following multiple-choice question. "
            "Output only the letter A, B, C, or D.\n\n"
            f"Question: {rec['question']}\n{options}\nAnswer:"
        )
    if task == "longbench":
        ctx = rec["context"]
        ids = TOK.encode(ctx, add_special_tokens=False)
        if len(ids) > max_ctx_tokens:
            ctx = TOK.decode(ids[:max_ctx_tokens], skip_special_tokens=True)
        options = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(rec["choices"])
        )
        return (
            "Read the long context and answer the multiple-choice question. "
            "Output only the letter A, B, C, or D.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {rec['question']}\n{options}\nAnswer:"
        )
    if task == "gsm8k":
        return (
            "Solve the problem step by step. "
            "Put the final numeric answer on its own line as: #### <number>\n\n"
            f"Question: {rec['question']}"
        )
    raise ValueError(task)


def request_one(task: str, rec: dict, max_ctx_tokens: int, timeout: int) -> dict:
    prompt = build_prompt(task, rec, max_ctx_tokens)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512 if task == "gsm8k" else 16,
    }
    if task == "gsm8k":
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "reasoning_effort": "low",
        }
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.time()
    r = requests.post(URL, json=payload, timeout=timeout)
    latency = time.time() - t0
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "") + "\n" + (msg.get("reasoning") or "")
    return {
        "latency_s": latency,
        "text": text,
        "usage": data.get("usage"),
        "request_tokens": len(TOK.encode(prompt, add_special_tokens=False)),
    }


def parse_letter(text: str) -> str | None:
    m = re.search(r"(?:answer|Answer)\s*[:：]?\s*\(?([ABCD])\)?", text)
    if m:
        return m.group(1)
    found = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text.upper())
    return found[-1] if found else None


def parse_number(text: str) -> float | None:
    m = re.findall(r"####\s*([+-]?\d[\d,.]*)", text)
    raw = m[-1] if m else None
    if raw is None:
        nums = re.findall(r"[+-]?\d[\d,.]*", text)
        raw = nums[-1] if nums else None
    if raw is None:
        return None
    raw = raw.replace(",", "").rstrip(".")
    try:
        return float(raw)
    except ValueError:
        return None


def gold_number(answer: str) -> float | None:
    m = re.search(r"####\s*([+-]?\d[\d,.]*)", answer)
    raw = m.group(1) if m else answer
    raw = raw.replace(",", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def score(task: str, rec: dict, text: str) -> tuple[bool, str | None]:
    if task in ("mmlu", "longbench"):
        pred = parse_letter(text)
        return pred == rec["answer"], pred
    pred = parse_number(text)
    gold = gold_number(rec["answer"])
    return pred is not None and gold is not None and abs(pred - gold) < 1e-6, (
        None if pred is None else str(pred)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["mmlu", "gsm8k", "longbench"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-context-tokens", type=int, default=26000)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l]
    if args.limit:
        records = records[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    ok = 0

    def run(rec):
        try:
            rr = request_one(args.task, rec, args.max_context_tokens, args.timeout)
            correct, pred = score(args.task, rec, rr["text"])
            return {
                "id": rec.get("id"),
                "task": args.task,
                "gold": rec.get("answer"),
                "pred": pred,
                "correct": correct,
                "latency_s": rr["latency_s"],
                "request_tokens": rr["request_tokens"],
                "usage": rr["usage"],
                "text": rr["text"][-800:],
                "error": None,
            }
        except Exception as e:
            return {
                "id": rec.get("id"),
                "task": args.task,
                "gold": rec.get("answer"),
                "pred": None,
                "correct": False,
                "latency_s": None,
                "request_tokens": None,
                "usage": None,
                "text": "",
                "error": repr(e),
            }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run, rec) for rec in records]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            results.append(r)
            ok += int(r["correct"])
            if i % 10 == 0 or i == len(futures):
                print(f"{args.task}: {i}/{len(futures)} done, acc={ok/i:.3f}", flush=True)

    # preserve input order
    order = {rec.get("id"): i for i, rec in enumerate(records)}
    results.sort(key=lambda r: order.get(r["id"], 10**9))
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    errors = sum(1 for r in results if r["error"])
    print(
        json.dumps(
            {
                "task": args.task,
                "total": total,
                "correct": ok,
                "accuracy": ok / total if total else None,
                "errors": errors,
                "out": str(out),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
