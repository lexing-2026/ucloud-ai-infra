# 部署与配置说明

所有命令在本仓库根目录执行；仓库放在 `/root/infer` 时脚本的默认路径全部生效，放到别处则用 `VENV`、`LOGDIR` 等环境变量覆盖（脚本里都是 `${VAR:-默认}` 形式）。

## 1. 硬件与系统

| 项 | 值 |
|---|---|
| GPU | 1 × RTX 4090 **48 GB**（49140 MiB），sm_89，功耗上限 450 W |
| 驱动 / CUDA | driver 610.57.04，CUDA 13.2（容器内 UMD 13.3） |
| CPU / 内存 | AMD EPYC 7A23，192 线程（cgroup 限 20），90 GB |
| 平台 | 优云智算 Compshare GPU pod；镜像 Ubuntu 22.04 + CUDA 13.2 + Python 3.12.11 + torch 2.11.0+cu130 |
| 存储 | `/` 与 `/root` 为 overlay（重建即丢）；`/model` 是平台共享模型库（持久） |

单卡，`--tp-size 1`。

## 2. 软件版本

| 组件 | 版本 |
|---|---|
| 推理引擎 | SGLang 0.5.22 |
| torch | 2.14.1+cu130 |
| flashinfer | 0.7.1（cubin 需绕过版本检查，见 §5） |
| Python | 3.12.11（venv `--system-site-packages`，复用镜像的 torch） |
| 评测工具 | AIPerf 0.13.0、EvalScope 1.13.0（`docs/eval/run.sh` 自建隔离环境） |
| 手写 kernel 工具链 | 镜像自带 `ptxas`（CUDA 13.2），打补丁时 `-O1` 预编译为 `.cubin` |

完整依赖清单：`logs/artifacts/20261030_48g/official/pip_freeze_sg22.txt`（207 行，venv 用 `--system-site-packages` 复用镜像的 torch，因此清单只含 venv 内实际安装的包；镜像侧版本见 `logs/artifacts/20261030_48g/REBUILD.md`）。

## 3. 环境构建（两步）

```bash
bash scripts/setup_sglang_48g.sh    # 建 venv、装引擎、打全部补丁、装官方评测工具、下载草稿权重
bash scripts/run_sglang_48g.sh      # 启动服务，端口 30000，首次约 3 分钟（含 CUDA 图捕获）
```

`setup_sglang_48g.sh` 做的事，按顺序：

1. 用镜像的 `/usr/local/miniconda3/envs/py312/bin/python` 建 `--system-site-packages` 的 venv（默认 `/root/infer/sg22`），只装 `sglang==0.5.22`，torch 复用镜像，避免 5 GB 重复下载。
2. 依次打 7 个补丁（顺序有意义）：`apply_rtx4090_fp8_gemm_configs.py`（安装调好的 4090 tile）→ `apply_sglang_sm89_gemm.py`（解除 `SWAP_AB`/`SPLIT_K` 的 sm90 限制，tile 配置依赖它）→ `apply_sglang_fp8_lm_head.py` → `apply_sglang_fp8_hand_gemm.py`（生成并 `ptxas -O1` 预编译手写 GEMM 与融合 MLP 的 cubin）→ `apply_sglang_language_only.py`（让 Qwen3.5/3.8 走 `--language-model-only`）→ `apply_sglang_gdn_prefill.py`（GDN prefill 去拷贝并融合 conv、归一化、o、norm 量化，须打在未改过的 GDN 文件上）→ `apply_sglang_dflash_replayssm.py`（让 DFlash 走 GDN ReplaySSM 验证）。
3. 为官方评测工具建两个隔离 venv（`aiperf-0.13.0`、`evalscope-1.13.0`）并下载 nltk 数据。
4. 从 ModelScope 拉 DSpark 草稿权重（DFlash2 在平台 `/model` 里已有），校验 `model.safetensors` 的 sha256。提交配置用不到它，重跑或没有外网时用 `DSPARK_DOWNLOAD=0` 跳过这一步。

国内镜像与代理：脚本默认走 `mirrors.aliyun.com/pypi/simple` 并清掉环境代理——平台默认代理只有 1–2 MB/s，实测差异很大。

冒烟：

```bash
curl -s http://127.0.0.1:30000/v1/models
bash scripts/demo_live.sh           # 模型、显存、流式对话、投机接受率
```

## 4. 提交配置逐项说明

`run_sglang_48g.sh` 不带参数即提交配置。每一项都对应一次实测取舍，证据在 [Benchmark 报告](BENCHMARK_REPORT.md) 与 [优化方案](OPTIMIZATION_REPORT.md)。

| 参数 | 值 | 为什么是这个值 |
|---|---|---|
| `--speculative-algorithm DFLASH` + `--speculative-num-draft-tokens 13` | DFlash2 草稿，draft 13 | 同机 100 请求四档合计 draft 10/12/13 = 768/778/785 |
| `--speculative-draft-model-quantization fp8` | FP8 草稿 | 只改提议，目标模型仍逐 token 验证；自定义块级 FP8 与它打平，已删 |
| `--enable-linear-replayssm-spec` | ReplaySSM 验证 | 紧凑窗口代替每 draft token 一份完整 GDN 状态，KV 从 123585 涨到 283423 token；需要 `patches/apply_sglang_dflash_replayssm.py` |
| `SGLANG_FP8_HAND_GEMM=1` | 手写 sm_89 GEMM + 融合 MLP | 权重转 tile 优先存储，所有 M 走手写 kernel，融合 kernel 在 M≥1024 生效 |
| `SGLANG_FP8_LM_HEAD=1` | 在线 FP8 lm_head | decode 每步少读一半 head；c2–c8 约 +1~2.3%，IFBench 不变，还省出 1.27 GB 给 KV |
| `--speculative-draft-window-size 2048` | 草稿窗口 | 100 请求无可测收益（±0.6% 内，两台机顺序相反）；保留只因 IFBench 与 soak 都在其上完成 |
| `--attention-backend flashinfer` | FlashInfer | 同口径 c8 比 Triton 高约 9% |
| `--kv-cache-dtype fp8_e4m3`（草稿 KV 同） | FP8 KV | KV 与草稿 KV 各省一半显存，换来 draft 深度 |
| `--max-mamba-cache-size 40` | 40 槽 | 默认 15 槽时每请求占 5 槽，日志一行 warning 就把并发静默限到 3，c8 TTFT 到 24 s |
| `--max-running-requests 8` | 8 路 | 与官方最高并发一致 |
| `--mem-fraction-static 0.92` | 0.92 | 0.925 以上 prefill 瞬时约 2.2 GB 空闲不够，启动 OOM |
| `--language-model-only` | 不加载视觉塔 | 省 0.92 GB；文本输入下 MRoPE 与 1D RoPE 等价。处理图片用 `VISION=1`，图片在 CPU 上预处理 |
| `--context-length 65536` | 64K | 官方 harness 对 `max-model-len<=32768` 判为配置异常 |
| `--chunked-prefill-size 8192` | 8192 | chunk 4096 让 c1 TTFT 从 1.655 s 变差到 1.708 s |
| `--cuda-graph-backend-prefill disabled` | 关 prefill 图 | 给草稿 KV 让显存；开回来启动即 OOM，且 GPU 忙碌已 99.4% |
| `--mamba-radix-cache-strategy extra_buffer` | extra_buffer | `extra_buffer_lazy` 在投机下断言失败；`no_buffer` 无整体优势且要放弃 overlap 调度 |
| `--reasoning-parser qwen3` / `--tool-call-parser qwen3_coder` | 解析思考与工具调用 | 与模型的推理/代码输出格式匹配 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 减少碎片 | 显存余量本来就小 |
| `scripts/launch_sglang_parent_off_gpu.py` | HTTP 父进程不建 CUDA 上下文 | torchao / sgl_kernel / flashinfer 在 import 时都会查询设备，白白占 454 MiB |
| `FLASHINFER_DISABLE_VERSION_CHECK=1` | 绕过 cubin 版本检查 | 镜像的 flashinfer-cubin 无 0.7.1 对应版本 |

改这些开关跑对照实验：`SPEC=off|mtp|dflash|dspark`、`MEM_FRAC`、`KV`、`ATTN`、`MAX_MAMBA`、`CHUNK`、`DRAFT_TOKENS`、`DRAFT_WINDOW`、`DRAFT_QUANT`、`REPLAY_SPEC`、`VISION`、`PREFILL_GRAPH` 等都是环境变量，`bash scripts/run_sglang_48g.sh --help` 之外另有 `EXTRA_ARGS` 兜底。

## 5. 评测与压测

```bash
# 官方口径：AIPerf 8192/512 × c1/2/4/8（每档 100 请求 + 8 预热）+ EvalScope IFBench limit=100
bash docs/eval/run.sh --base-url http://127.0.0.1:30000/v1 --model qwen3.8-27b-fp8 --mode all

# 只检查依赖，不请求模型
bash docs/eval/run.sh --mode check

# 快速回归（每档 4 请求，只用于确认服务活着）
bash docs/eval/run.sh --base-url http://127.0.0.1:30000/v1 --model qwen3.8-27b-fp8 --mode perf --perf-requests 4
```

`docs/eval/` 是赛题提供的官方评测工具，原样未改；结果目录结构见任一 `logs/artifacts/**/perf_*`。同机 A/B 类对比用 `scripts/bench_ab.py`（随机 token-ID 输入，只用于同表内部比较，不可与官方默认文本模式的数字混比——口径差异见 Benchmark 报告的说明）。

## 6. 可观测性

- 服务端：SGLang `/metrics`（Prometheus 格式），`dashboard/` 提供 docker-compose + Prometheus + Grafana，启动方式见 `dashboard/README.md`；`dashboard/.env.example` 填服务地址即可，不需要任何密钥（SGLang 默认不校验，`API_KEY=EMPTY`）。
- 客户端：每轮评测的 `summary.json` / `report.md` / `profile_export.jsonl`（逐请求）都落盘，`docs/EVIDENCE_INDEX.md` 汇总。
- 长稳：soak 期间每 30 秒采一次 `nvidia-smi`，采样文件与日志同名（`z13s_nvsmi.log`）。

## 7. 已知环境问题

| 现象 | 处理 |
|---|---|
| 装包只有 1–2 MB/s | 脚本内已清代理并走阿里云镜像；`UV_CACHE_DIR` 可指到持久盘 |
| flashinfer cubin 版本检查失败 | `FLASHINFER_DISABLE_VERSION_CHECK=1`（脚本已导出） |
| 并发被限到 3 | 是 Mamba 槽不足，不是调度器 bug；`--max-mamba-cache-size 40` |
| 提高 `--mem-fraction-static` 后启动 OOM | prefill 瞬时需要约 2.2 GB 空闲，0.92 是这套配置的实际上限 |
| 容器内 `nvidia-smi -lgc/-ac` 返回成功但不生效 | 无权限，功耗墙 450 W 是真实上限，未做任何改时钟 |
| A 机出现过 `cuInit` NO_DEVICE | 平台侧 `[GPU requires reset]`，stop/start 换卡恢复，与软件配置无关 |
