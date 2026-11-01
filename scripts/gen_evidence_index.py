#!/usr/bin/env python3
"""Generate docs/EVIDENCE_INDEX.md: every official-harness run shipped in the
submission, with the numbers read straight out of its own summary.json.

Judges can match each table in docs/BENCHMARK_REPORT.md against a run directory
without trusting our transcription. Run it against the submission tree:

    python3 scripts/gen_evidence_index.py --root submit
"""

import argparse
import json
import pathlib

METRICS = ("prompt_level_strict", "inst_level_strict", "prompt_level_loose")


def per_request_count(run: pathlib.Path) -> int:
    total = 0
    for f in sorted(run.glob("performance/c*/profile_export.jsonl")):
        with f.open("rb") as fh:
            total += sum(1 for line in fh if line.strip())
    return total


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def row(rel, data):
    conf = data.get("configuration", {})
    perf = data.get("performance") or []
    acc = data.get("accuracy") or {}
    thr = {p["concurrency"]: p for p in perf}
    errors = sum(p.get("error_requests", 0) for p in perf)
    is_quality = bool(acc) and not perf
    workload = "IFBench" if is_quality else f"{conf.get('isl')}/{conf.get('osl')}"
    requests = conf.get("accuracy_limit", conf.get("perf_requests")) if is_quality \
        else conf.get("perf_requests")
    warmup = "—" if is_quality else conf.get("warmup_requests")
    cells = [
        f"`{rel}`",
        (data.get("started_at") or "")[:16].replace("T", " "),
        conf.get("mode", "perf"),
        workload,
        requests,
        warmup,
    ]
    for c in (1, 2, 4, 8):
        p = thr.get(c)
        cells.append(fmt(p["output_token_throughput"]) if p else "—")
    for c in (1, 8):
        p = thr.get(c)
        cells.append(fmt(p["metrics"]["time_to_first_token"]["p50"] / 1000, 3) if p else "—")
    for c in (1, 8):
        p = thr.get(c)
        cells.append(fmt(p["metrics"]["inter_token_latency"]["p50"], 2) if p else "—")
    metrics = acc.get("metrics") or {}
    scores = " / ".join(
        f"{m}={metrics[m]['score']:.4f}" for m in METRICS if m in metrics
    )
    acc_err = (acc.get("execution_summary") or {}).get("errored")
    cells += [
        str(int(errors)) if perf else ("—" if acc_err is None else str(acc_err)),
        scores or "—",
        data.get("status", "—"),
    ]
    return [str(c) for c in cells]


def native_quality_rows(root: pathlib.Path, covered: set):
    """Runs whose per-question phase shipped but not the harness summary.json."""
    rows = []
    for report in sorted(root.rglob("accuracy/native/reports/*/*.json")):
        run = report.parents[4]
        if str(run) in covered or not report.name.endswith(".json"):
            continue
        try:
            data = json.loads(report.read_text())
        except (ValueError, OSError):
            continue
        metrics = {m["identity"]["name"]: m for m in data.get("metrics") or []}
        if "prompt_level_strict" not in metrics:
            continue
        exec_sum = data.get("execution_summary") or {}
        scores = " / ".join(
            f"{name}={metrics[name]['score']:.4f}"
            for name in METRICS
            if name in metrics
        )
        rows.append([str(c) for c in [
            f"`{run.relative_to(root)}`",
            "—",
            "accuracy",
            "IFBench",
            exec_sum.get("requested", "—"),
            "—",
            *["—"] * 4,
            *["—"] * 4,
            str(exec_sum.get("errored", "—")),
            scores,
            "完整" if not exec_sum.get("incomplete") else "异常",
        ]])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="submit", help="submission tree to scan")
    ap.add_argument("--out", help="markdown output path (default <root>/docs/EVIDENCE_INDEX.md)")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    out = pathlib.Path(args.out) if args.out else root / "docs" / "EVIDENCE_INDEX.md"

    rows = []
    covered = set()
    for summary in sorted(root.rglob("summary.json")):
        rel = summary.parent.relative_to(root)
        if not str(rel).startswith("logs/artifacts/"):
            continue
        try:
            data = json.loads(summary.read_text())
        except (ValueError, OSError):
            continue
        if "configuration" not in data:
            continue
        covered.add(str(summary.parent))
        rows.append(row(str(rel), data) + [str(per_request_count(summary.parent))])

    rows += [r + ["—"] for r in native_quality_rows(root, covered)]
    rows.sort(key=lambda r: r[0])

    header = [
        "运行目录", "开始 (UTC)", "模式", "ISL/OSL", "请求", "预热",
        "c1 tok/s", "c2", "c4", "c8",
        "c1 TTFT P50 (s)", "c8 TTFT P50 (s)", "c1 TPOT P50 (ms)", "c8 TPOT P50 (ms)",
        "错误", "质量分", "状态", "逐请求记录",
    ]
    align = ["---"] * len(header)

    lines = [
        "# 原始数据索引",
        "",
        "本表由 `scripts/gen_evidence_index.py` 从每个运行目录自带的 `summary.json` 读出，",
        "不是手工誊写：任何一个数字都能在该目录的 `performance/c*/profile_export.jsonl`",
        "（逐请求记录）里重算。`docs/BENCHMARK_REPORT.md` 第 6 节按主题指向这些目录。",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += [
        "",
        f"共 {len(rows)} 个运行目录。",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"{out}: {len(rows)} runs")


if __name__ == "__main__":
    main()
