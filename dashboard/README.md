# 演示看板：Prometheus + Grafana

抓取 SGLang 的 `/metrics`，在 Grafana 上看吞吐、TTFT、TPOT、队列与投机接受率。看板是可选演示。

服务端带上 metrics：

```bash
EXTRA_ARGS="--enable-metrics --enable-mfu-metrics" bash scripts/run_sglang_48g.sh
```

起看板：

```bash
cd dashboard
sed -e 's/{{SGLANG_HOST}}/<服务地址>/' -e 's/{{SGLANG_PORT}}/30000/' prometheus.yml.template > prometheus.yml
docker compose up -d
```

Grafana 在 `http://localhost:3000`（匿名免登录），Prometheus 在 `http://localhost:9091`。停止用 `docker compose down`。

造流量：

```bash
bash docs/eval/run.sh --base-url http://<服务地址>:30000/v1 --model qwen3.8-27b-fp8 --mode perf --perf-requests 4
```
