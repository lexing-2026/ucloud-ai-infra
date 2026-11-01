# 单卡 RTX 4090 运行 Qwen3.8-27B：推理优化

UCloud × 上海大学 AI 黑客松 · 赛道 A（AI Infra）。

在 1 张 RTX 4090 48GB 上用 SGLang 服务官方 `Qwen3.8-27B-FP8`。核心优化有四块：

1. **手写 sm_89 FP8 算子**：用 PTX 手写 GEMM，并把 gate_up GEMM、SiLU、激活量化融合成一个 kernel，比调好的 Triton kernel 快 8–9%，单层 MLP 前半段再快 15.7%。
2. **ForgeKernel 流程调 tile**：为 sm89 调出专用 tile（含投机解码的验证批），启用 SGLang 原本只给 sm90 用的 `SWAP_AB` / `SPLIT_K`。
3. **GDN 少搬数据**：这些 kernel 都顶着显存带宽，时间跟字节数走。prefill 去掉 144 次拷贝，把 `o`、归一化、量化并进前一个 kernel，单层从 3.07 ms 降到 2.165 ms，norm 加量化从 0.501 ms 降到 0.283 ms；投机验证改用 ReplaySSM，不再为每个 draft token 写一份完整状态。
4. **省显存换投机深度**：FP8 草稿、在线 FP8 lm_head、去掉视觉塔、让 HTTP 父进程离开 GPU，把省出的显存换成 DFlash2 draft 13。

官方 AIPerf 8192/512，每档 100 请求，零错误：c1/2/4/8 = **138.99 / 190.98 / 234.42 / 254.39 tok/s**；IFBench `prompt_level_strict` **0.83**（门槛 0.76）。同口径（官方 harness、每档 100 请求）下 SGLang 0.5.22 出厂配置为 56.46 / 91.10 / 129.22 / 161.29，即 **2.17× / 1.87× / 1.61× / 1.48×**。原样 vLLM 0.25.1 的 18.28 / 36.71 / 64.25 / 64.46 是每档 20 请求的口径，只作引擎横向比较，不与上表换算。

## 环境

| 项 | 值 |
|---|---|
| GPU | 1 × RTX 4090 48GB（49140 MiB），driver 610.57.04 |
| 平台镜像 | 优云智算 `SGLang v0.5.15`（Ubuntu 22.04，CUDA 13.2，Python 3.12） |
| 推理引擎 | SGLang 0.5.22，torch 2.14.1+cu130，flashinfer 0.7.1 |
| 模型 | `Qwen/Qwen3.8-27B-FP8`，平台路径 `/model/ModelScope/Qwen/Qwen3.8-27B-FP8`（含 `mtp.safetensors`） |
| 依赖清单 | `logs/artifacts/20261030_48g/official/pip_freeze_sg22.txt` |

## 一条命令复现

把本仓库放到 `/root/infer`，然后：

```bash
bash scripts/setup_sglang_48g.sh       # 建 /root/infer/sg22，装 tile 与补丁，装官方评测工具
bash scripts/run_sglang_48g.sh         # 启动服务，端口 30000，首次约 3 分钟（含图捕获）
```

冒烟：

```bash
curl -s http://127.0.0.1:30000/v1/models
```

官方评测：

```bash
bash docs/eval/run.sh --base-url http://127.0.0.1:30000/v1 --model qwen3.8-27b-fp8 --mode all
```

## 五分钟验收

```bash
bash scripts/demo_live.sh      # 模型名、显存、流式对话、投机接受率
```

结果量级对得上 `docs/DEMO.md` §3 即可；正式数字用 `--mode perf --perf-requests 4`（约 2–4 分钟）。

## 最终配置

`scripts/run_sglang_48g.sh` 不带参数时使用：

| 参数 | 值 | 作用 |
|---|---|---|
| `--speculative-algorithm DFLASH --speculative-num-draft-tokens 13` | DFlash2 draft 13 | 同机 100 请求四档合计最高（draft 10/12/13 = 768/778/785） |
| `--speculative-draft-model-quantization fp8` | FP8 草稿 | 只改提议，目标模型仍逐 token 验证；草稿读权重更快 |
| `--enable-linear-replayssm-spec` | ReplaySSM 验证 | 紧凑窗口代替每个 draft token 一份完整 GDN 状态，KV 从 123585 涨到 283423 token |
| `SGLANG_FP8_HAND_GEMM=1` | 手写 GEMM 与融合 MLP | 权重转 tile 优先存储，所有 M 走手写 kernel，M≥1024 走融合 kernel |
| `SGLANG_FP8_LM_HEAD=1` | 在线 FP8 lm_head | decode 每步少读一半 lm_head；同机 c2–c8 约 +2%，IFBench 不变 |
| `--speculative-draft-window-size 2048` | | 收益在噪声内，因 IFBench 与 soak 均在此配置上完成而保留 |
| `--attention-backend flashinfer` | | 比 Triton 的 c8 吞吐高 9% |
| `--kv-cache-dtype fp8_e4m3` | | KV 和草稿 KV 都省一半显存 |
| `--max-mamba-cache-size 40` | | 不设时 Mamba 池把并发静默限到 3 |
| `--max-running-requests 8` | | 与官方最高并发一致 |
| `--mem-fraction-static 0.92` | | prefill 瞬时要约 2.2 GB 空闲，再高就 OOM |
| `--language-model-only` | | 不加载视觉塔，省 0.92 GB；要处理图片时用 `VISION=1`，见 [docs/DEMO.md](docs/DEMO.md) |
| `scripts/launch_sglang_parent_off_gpu.py` | | HTTP 父进程不建 CUDA 上下文，省 454 MiB |
| `--context-length 65536` | | 官方 harness 要求 >32768 |
| `--chunked-prefill-size 8192` | | |
| `--cuda-graph-backend-prefill disabled` | | 关 prefill 图，给草稿 KV 让显存 |
| `--mamba-radix-cache-strategy extra_buffer` | | 保留 overlap 调度与前缀缓存 |

## 数据与可核验性

- 报告里每一个吞吐、TTFT、TPOT、质量分都能在 `logs/artifacts/` 找到对应运行目录；[docs/EVIDENCE_INDEX.md](docs/EVIDENCE_INDEX.md) 由脚本从各运行的 `summary.json` 自动生成，逐条给出目录、时间、口径、四档数字、错误数与逐请求记录所在文件。
- `VERSION` 记录模型来源、引擎/torch/flashinfer 精确版本、依赖清单行数与 sha256，以及仓库内脚本与补丁的逐文件 sha256。
- 提交配置在 C 机连续跑 1000 请求 43.5 分钟，零错误，196.37 tok/s，显存启动后恒定 47338 MiB。

## 仓库结构

```text
scripts/setup_sglang_48g.sh            环境重建
scripts/run_sglang_48g.sh              服务启动（最终配置）
scripts/launch_sglang_parent_off_gpu.py  让 HTTP 父进程离开 GPU
scripts/demo_live.sh                   现场演示串联（只依赖 curl/python3/nvidia-smi）
scripts/tune_fp8_gemm.py               FP8 GEMM tile 调优器
scripts/bench_ab.py                    vLLM / SGLang 通用闭环压测，含共享前缀模式
patches/fp8_gemm_rtx4090/              4090 tile 配置（调优产物）
patches/apply_rtx4090_fp8_gemm_configs.py  安装 tile
patches/apply_sglang_sm89_gemm.py      解除 SWAP_AB/SPLIT_K 的 sm90 限制
patches/fp8_hand_sm89.py               手写 sm_89 GEMM 与融合 MLP 的 PTX 生成器、运行时
patches/apply_sglang_fp8_hand_gemm.py  安装上一项并预编译 .cubin
patches/apply_sglang_language_only.py  允许 Qwen3.5 使用 --language-model-only
patches/apply_sglang_gdn_prefill.py    GDN prefill 去拷贝并融合 conv、归一化、o、norm 量化
patches/apply_sglang_dflash_replayssm.py  让 DFlash 走 GDN ReplaySSM 验证
patches/apply_sglang_fp8_lm_head.py    在线 FP8 lm_head（环境变量开关）
docs/MODEL_SELECTION.md                模型选型说明
docs/DEPLOYMENT.md                     部署与配置说明
docs/BENCHMARK_REPORT.md               Benchmark 测试报告
docs/OPTIMIZATION_REPORT.md            优化方案说明与性能优化分析
docs/REPRODUCTION.md                   完整复验材料包
docs/DEMO.md                           现场演示与备用录屏
docs/EVIDENCE_INDEX.md                 原始数据索引（脚本生成）
docs/eval/                             官方评测工具
docs/defense/                          答辩 PPT（tex / pdf / html）
dashboard/                             AIPerf + Prometheus + Grafana 演示看板
logs/artifacts/20261030_48g/official/  官方评测原始结果
logs/artifacts/20261031_hand/          手写算子微基准、搜索记录与最终运行结果
```
