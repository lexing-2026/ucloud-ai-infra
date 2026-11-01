# 现场演示与备用录屏

演示用的服务配置与提交配置完全一致（`scripts/run_sglang_48g.sh` 不带参数），不另外准备"演示专用配置"。

## 1. 前置：服务在跑

```bash
bash scripts/setup_sglang_48g.sh        # 只需一次，建 /root/infer/sg22 并打补丁
bash scripts/run_sglang_48g.sh          # 首次约 3 分钟（权重加载 + CUDA 图捕获）
```

就绪判据：

```bash
curl -s http://127.0.0.1:30000/v1/models | python3 -m json.tool
```

看到 `"id": "qwen3.8-27b-fp8"` 即可。服务日志里对应的一行是 `The server is fired up and ready to roll!`。

## 2. 现场 3 分钟动线

`bash scripts/demo_live.sh` 会把下面 1–4 步串起来跑，只需要终端 + `curl` + `python3`，不依赖评测工具的虚拟环境。看板（第 5 步）需要另外开 Grafana。

| 时间 | 操作 | 屏上内容 | 讲解要点 |
|---|---|---|---|
| 0:00–0:20 | `demo_live.sh` 第 1、2 步 | `/v1/models`；`nvidia-smi` 显存 47338 MiB、GPU 上只有调度进程 | 单卡 48GB 约束达标；HTTP 父进程不占 GPU，省回 454 MiB |
| 0:20–1:10 | 第 3 步：流式对话 | 回答逐字出现，末尾打印首字延迟与平均 tok/s | 服务是标准 OpenAI 协议，任何客户端可接；`temperature=0` 下结果可复现 |
| 1:10–1:40 | 第 4 步：`/metrics` | `num_running_reqs`、`token_usage`、`spec_accept_length`、`spec_accept_rate` | 投机解码接受率在线可见，不是"宣称加速" |
| 1:40–2:40 | `docs/eval/run.sh --mode perf --perf-requests 4` | 官方 AIPerf 逐档打印 c1/2/4/8 吞吐 | 口径与官方评测一致：8192/512、每档 8 预热、换 `cache_salt` |
| 2:40–3:00 | Grafana 看板 | 吞吐、TTFT/TPOT、队列与显存曲线 | 长稳与并发行为可观测 |

`--perf-requests 4` 一轮约 2–4 分钟。小样本只做"数字对得上量级"的现场验证，报告里的正式数字是每档 100 请求，见 [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) §2。

## 3. 每条命令的预期输出

### 3.1 `scripts/demo_live.sh`

```
== 1. 服务与模型 ==
   served-model-name: qwen3.8-27b-fp8

== 2. 显存与进程 ==
   NVIDIA GeForce RTX 4090, 47338 MiB, 49140 MiB, xxx W, xxx %
   <pid>, 46378 MiB

== 3. 流式对话（首字延迟与逐字输出可见）==
> 用一句话解释投机解码为什么能降低单请求的 TPOT。
   ...（回答）
   首字 xxx ms | 输出 nnn 个流式片段 | 端到端 x.xx s | 平均 xx.x tok/s
```

显存两行取自提交配置的实际快照（`logs/artifacts/20261031_final/C/logs/z13_apps.txt`：GPU 上一个进程 46378 MiB、总占用 47338 MiB）。

- 这条 prompt 只有几十字符，首字应在几百毫秒内。8192 输入的 TTFT P50 是 1.569s，两者不可混比（短输入数据见 BENCHMARK_REPORT §4）。
- "平均 tok/s"按流式片段计数，片段与 token 不是一一对应，只用于观感；权威的单请求速度看 TPOT。

### 3.2 官方口径

```bash
bash docs/eval/run.sh --base-url http://127.0.0.1:30000/v1 --model qwen3.8-27b-fp8 --mode perf --perf-requests 4
```

结果目录写进 `outputs/<时间戳>/`，`report.md` 与 `summary.json` 各档吞吐的期望值（每档 100 请求，C 机）：

| 并发 | 期望 tok/s | 小样本容忍 |
|---:|---:|---|
| 1 | 138.99 | ±5% |
| 2 | 190.98 | ±5% |
| 4 | 234.42 | ±5% |
| 8 | 254.39 | ±5% |

跨卡差异约 4%，见 [REPRODUCTION.md](REPRODUCTION.md) §6。

完整评测（含 IFBench）：

```bash
bash docs/eval/run.sh --base-url http://127.0.0.1:30000/v1 --model qwen3.8-27b-fp8 --mode all
```

每档 100 请求实测 21分37秒（`logs/artifacts/20261031_hand/C/outputs/perf_final2/summary.json` 的 `started_at`/`ended_at`），加 IFBench 100 题后约 70 分钟（IFBench 单跑实测 47.5 分钟）。不适合现场等待，用录屏或已落盘的 `logs/artifacts/`。

### 3.2.1 图片与 Agent 客户端

演示图片或接 Agent 客户端时用这条启动，提交配置和官方成绩都不用它：

```bash
VISION=1 CONTEXT=240000 EXTRA_ARGS="--enable-metrics --allow-auto-truncate --api-key <KEY>" \
  bash scripts/run_sglang_48g.sh
```

- `VISION=1` 保留视觉塔（多占 0.92 GB），图片在 CPU 上预处理，因为 HTTP 父进程不建 CUDA 上下文。
- `CONTEXT=240000` 配合 Agent 的长会话。默认 65536 时，prompt 加 `max_tokens` 超限会返回 400，且历史不缩短就一直失败。客户端的 `contextWindow` 不要超过这个值。
- `--allow-auto-truncate` 在接近上限时自动缩短 `max_tokens`，不再直接返回 400。

### 3.3 看板

```bash
cd dashboard
sed -e 's/{{SGLANG_HOST}}/<服务地址>/' -e 's/{{SGLANG_PORT}}/30000/' prometheus.yml.template > prometheus.yml
docker compose up -d    # Prometheus :9091，Grafana :3000，匿名免登录
```

服务侧要带 metrics：`EXTRA_ARGS="--enable-metrics --enable-mfu-metrics" bash scripts/run_sglang_48g.sh`。没开时 `demo_live.sh` 第 4 步会打印提示并跳过，不影响其余步骤。

## 4. 备用录屏

### 4.1 内容清单（3 分钟，按 §2 的分镜）

- [ ] 开头一屏：`nvidia-smi` 显示 1× RTX 4090 48GB、显存 47338 MiB、GPU 上只有一个进程
- [ ] `bash scripts/run_sglang_48g.sh` 的启动横幅（`sglang: ctx=65536 mem=0.92 kv=fp8_e4m3 attn=flashinfer ... spec=dflash`）和 `max_total_num_tokens=283423` 一行
- [ ] `demo_live.sh` 第 1–3 步：模型名、显存、流式对话逐字输出与首字延迟
- [ ] `/metrics` 里的 `spec_accept_rate`，说明加速来源可在线核验
- [ ] 官方 `docs/eval/run.sh --mode perf` 跑完 c1 与 c8 两档的输出版面
- [ ] Grafana 看板的吞吐/TTFT/TPOT 曲线，c1→c8 并发爬升过程
- [ ] 长上下文：一次 60000 输入请求成功返回（对应 BENCHMARK_REPORT §4 的稳定性行）
- [ ] 结尾一屏：IFBench `prompt_level_strict=0.83` 与报告表格

### 4.2 录制

```bash
# 有麦（推荐，讲解直接录进音轨）
ffmpeg -y -f x11grab -framerate 30 -video_size 1920x1080 -i "$DISPLAY" \
       -f alsa -i default -c:v libx264 -preset veryfast -crf 23 -c:a aac \
       docs/defense/demo_3min.mp4
# 无麦版本：去掉 -f alsa/-i default/-c:a aac，讲解留给现场或字幕
```

Wayland 会话下 `x11grab` 不可用，改用 OBS 的 PipeWire 屏幕捕获。录完校验时长与分辨率：

```bash
ffprobe -v error -show_entries format=duration -show_entries stream=width,height,codec_name docs/defense/demo_3min.mp4
```

文件命名 `docs/defense/demo_3min.mp4`。

## 5. 故障预案

| 现象 | 处置 |
|---|---|
| 远程 GPU 不可达 / 公网端口未放行 | 现场改放 §4 的录屏；`docs/eval` 的历史 `summary.json` 直接在浏览器里展示（`logs/artifacts/20261031_hand/C/outputs/perf_final2/`） |
| 启动 OOM | 降 `MEM_FRAC=0.90` 或 `DRAFT_TOKENS=8`；报告里的最终配置是 0.92 + draft 13，演示降级要口头说明 |
| `spec=dflash` 加载草稿失败 | `SPEC=mtp bash scripts/run_sglang_48g.sh`，MTP4 是同机 69.62 tok/s 的上一档配置，不是最终配置 |
| FlashInfer 版本检查报错 | 已由 `FLASHINFER_DISABLE_VERSION_CHECK=1` 绕过；若仍失败，退 `ATTN=triton`，c8 吞吐约低 9% |
| `/metrics` 404 | 服务没带 `--enable-metrics`；`demo_live.sh` 会自动跳过第 4 步 |
| 评测工具缺依赖 | `bash docs/eval/run.sh --mode check` 只检查不安装；非交互模式不会自动装包 |
| 显卡被上一个进程占住 | 同一时间只允许一个服务占卡；先停旧服务再启动（`scripts/` 里不代做 kill，避免误杀） |
