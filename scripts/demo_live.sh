#!/usr/bin/env bash
# 现场演示脚本：只依赖 curl、python3、nvidia-smi，不需要评测工具的虚拟环境。
# 服务必须已经在跑（scripts/run_sglang_48g.sh）。
#
#   bash scripts/demo_live.sh              # 默认 127.0.0.1:30000
#   PORT=8000 SERVED=my-model bash scripts/demo_live.sh
#
# 精确的官方口径数字用 docs/eval/run.sh，这个脚本只负责"看得见跑得起来"。
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
SERVED=${SERVED:-qwen3.8-27b-fp8}
BASE="http://${HOST}:${PORT}"
PROMPT=${PROMPT:-用一句话解释投机解码为什么能降低单请求的 TPOT。}

echo "== 1. 服务与模型 =="
curl -sf "$BASE/v1/models" | python3 -c '
import json, sys
for m in json.load(sys.stdin)["data"]:
    print("   served-model-name:", m["id"])
'

echo
echo "== 2. 显存与进程 =="
nvidia-smi --query-gpu=name,memory.used,memory.total,power.draw,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed "s/^/   /"

echo
echo "== 3. 流式对话（首字延迟与逐字输出可见）=="
PROMPT="$PROMPT" BASE="$BASE" SERVED="$SERVED" python3 - <<'PY'
import json, os, time, urllib.request

prompt = os.environ["PROMPT"]
body = json.dumps({
    "model": os.environ["SERVED"],
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 256,
    "temperature": 0,
    "stream": True,
}).encode()
req = urllib.request.Request(
    os.environ["BASE"] + "/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"})

t0 = time.monotonic()
ttft = None
n = 0
print("> " + prompt)
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"]
        piece = delta.get("content") or ""
        if piece:
            if ttft is None:
                ttft = time.monotonic() - t0
            print(piece, end="", flush=True)
            n += 1
total = time.monotonic() - t0
print()
print(f"   首字 {ttft * 1000:.0f} ms | 输出 {n} 个流式片段 | "
      f"端到端 {total:.2f} s | 平均 {n / max(total - ttft, 1e-6):.1f} tok/s")
PY

echo
echo "== 4. 服务侧统计（并发、KV 占用、投机接受率）=="
if curl -sf "$BASE/metrics" >/dev/null 2>&1; then
  curl -s "$BASE/metrics" | grep -E '^sglang:(num_running_reqs|num_used_tokens|token_usage|spec_accept_length|spec_accept_rate)' |
    sed "s/^/   /" || echo "   metrics 里还没有投机/并发计数（服务空闲）"
else
  echo "   未开 metrics。需要时：EXTRA_ARGS='--enable-metrics --enable-mfu-metrics' bash scripts/run_sglang_48g.sh"
fi

echo
echo "== 5. 下一步：官方口径 =="
cat <<EOF
   正式性能口径（每档 100 请求，实测 22 分钟）：
     bash docs/eval/run.sh --base-url $BASE/v1 --model $SERVED --mode perf
   性能 + IFBench 质量（实测约 70 分钟）：
     bash docs/eval/run.sh --base-url $BASE/v1 --model $SERVED --mode all
   快速（每档 4 请求，演示够用，约 2–4 分钟）：
     bash docs/eval/run.sh --base-url $BASE/v1 --model $SERVED --mode perf --perf-requests 4
EOF
